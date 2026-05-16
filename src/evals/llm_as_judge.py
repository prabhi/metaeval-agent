"""
LLM-as-Judge Eval Generator

Given an AgentManifest, this module:
1. Synthesizes diverse input prompts for the agent
2. Defines rubric criteria based on the agent's purpose
3. Generates an evaluator function that calls an LLM judge
4. Returns an EvalArtifact ready for LangSmith upload
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult, run_evaluator
from pydantic import BaseModel, Field

from src.utils.manifest import AgentManifest


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class JudgeTestCase(BaseModel):
    input_prompt: str = Field(description="The user message to send to the target agent")
    evaluation_criteria: list[str] = Field(
        description="List of criteria to evaluate the output against (binary pass/fail each)"
    )
    context: str = Field(default="", description="Optional context or background for this test case")


class JudgeEvalSuite(BaseModel):
    rubric_description: str = Field(description="High-level description of what quality means for this agent")
    test_cases: list[JudgeTestCase]
    scoring_dimensions: list[str] = Field(
        description="The overarching dimensions being assessed (accuracy, safety, helpfulness, etc.)"
    )


@dataclass
class EvalArtifact:
    eval_type: str
    dataset_name: str
    test_cases: list[dict]
    evaluator_fn: Callable
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

JUDGE_GENERATION_PROMPT = """You are an expert AI evaluation engineer.

Given this agent manifest, generate a comprehensive LLM-as-judge evaluation suite.

Agent Manifest:
{manifest_summary}

Requirements:
- Generate 15 diverse test cases covering: happy paths, edge cases, ambiguous inputs, scope boundaries
- Define 4-6 binary evaluation criteria per test case (pass=1, fail=0)
- The criteria should reflect what "good" looks like for THIS specific agent
- Include test cases that probe potential failure modes identified in the manifest

Respond with a JSON object matching the JudgeEvalSuite schema.
"""


def generate_llm_judge_evals(manifest: AgentManifest, llm: ChatOpenAI | None = None) -> EvalArtifact:
    """
    Auto-generate an LLM-as-judge eval suite from an AgentManifest.

    Args:
        manifest: The scanned agent's manifest
        llm: Optional LLM override (defaults to gpt-4o)

    Returns:
        EvalArtifact with test cases and evaluator function
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    structured_llm = llm.with_structured_output(JudgeEvalSuite)

    prompt = JUDGE_GENERATION_PROMPT.format(manifest_summary=manifest.summary())
    suite: JudgeEvalSuite = structured_llm.invoke(prompt)

    test_cases = [
        {
            "input": tc.input_prompt,
            "evaluation_criteria": tc.evaluation_criteria,
            "context": tc.context,
        }
        for tc in suite.test_cases
    ]

    evaluator_fn = _build_judge_evaluator(suite.rubric_description, suite.scoring_dimensions, llm)

    return EvalArtifact(
        eval_type="llm_as_judge",
        dataset_name=f"{manifest.agent_name}_judge_evals",
        test_cases=test_cases,
        evaluator_fn=evaluator_fn,
        metadata={
            "rubric": suite.rubric_description,
            "scoring_dimensions": suite.scoring_dimensions,
            "num_test_cases": len(test_cases),
        },
    )


def _build_judge_evaluator(rubric: str, dimensions: list[str], llm: ChatOpenAI) -> Callable:
    """
    Build a LangSmith-compatible evaluator function that uses an LLM judge.
    The judge scores each criterion as 0 or 1 and returns the mean.
    """

    JUDGE_EVAL_PROMPT = """You are an impartial AI evaluator.

Rubric: {rubric}

User Input: {input}
Agent Output: {output}
Evaluation Criteria: {criteria}

For each criterion, output 1 (pass) or 0 (fail).
Then output an overall_score (mean of all criteria) and a brief explanation.

Respond as JSON: {{"scores": {{"criterion_name": 0_or_1, ...}}, "overall_score": float, "explanation": str}}
"""

    @run_evaluator
    def judge_evaluator(run, example) -> EvaluationResult:
        criteria = example.outputs.get("evaluation_criteria", dimensions)
        response = llm.invoke(
            JUDGE_EVAL_PROMPT.format(
                rubric=rubric,
                input=run.inputs.get("input", ""),
                output=run.outputs.get("output", str(run.outputs)),
                criteria=json.dumps(criteria),
            )
        )
        try:
            result = json.loads(response.content)
            score = result.get("overall_score", 0.0)
            explanation = result.get("explanation", "")
        except (json.JSONDecodeError, AttributeError):
            score = 0.0
            explanation = "Judge response parse error"

        return EvaluationResult(
            key="llm_judge_score",
            score=score,
            comment=explanation,
        )

    return judge_evaluator
