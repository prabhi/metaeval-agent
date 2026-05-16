"""
Trajectory / Tool-Use Eval Generator

Tests whether the agent invokes the RIGHT tools in the RIGHT ORDER.

For each tool in the manifest, we generate:
  - Single-tool invocation tests (does the agent USE this tool at all?)
  - Multi-tool sequence tests (does the agent chain tools correctly?)
  - Tool-avoidance tests (does the agent NOT use a tool when inappropriate?)

Scoring:
  - Exact match: full trajectory matches expected
  - Partial match: Levenshtein distance between tool sequences
  - Set match: correct tools called, order ignored
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult, run_evaluator
from pydantic import BaseModel, Field

from src.utils.manifest import AgentManifest, ToolSpec


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TrajectoryTestCase(BaseModel):
    input_prompt: str = Field(description="User message that should trigger a specific tool sequence")
    expected_tool_sequence: list[str] = Field(
        description="Ordered list of tool names the agent should call"
    )
    allow_extra_tools: bool = Field(
        default=True,
        description="Whether additional tool calls beyond the expected sequence are OK"
    )
    test_category: str = Field(
        description="single_tool | multi_tool | tool_avoidance | error_recovery"
    )
    rationale: str = Field(description="Why this sequence is expected for this input")


class TrajectoryEvalSuite(BaseModel):
    test_cases: list[TrajectoryTestCase]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

TRAJECTORY_GENERATION_PROMPT = """You are an expert AI evaluation engineer specializing in agentic systems.

Given this agent's tool list, generate trajectory evaluation test cases.

Agent: {agent_name}
Tools available:
{tools_list}

Generate test cases in these categories:
1. single_tool: Input that should invoke exactly ONE specific tool (2 cases per tool)
2. multi_tool: Input that requires a SEQUENCE of tools (5 cases minimum)
3. tool_avoidance: Input where the agent should NOT call any tool (3 cases)
4. error_recovery: Input where a tool will fail and the agent must recover (2 cases)

For each test case, specify the EXACT expected tool call sequence as a list.
Be specific — the expected_tool_sequence should reflect real tool chaining logic.

Respond as a JSON object matching the TrajectoryEvalSuite schema.
"""


def generate_trajectory_evals(manifest: AgentManifest, llm: ChatOpenAI | None = None) -> "EvalArtifact":
    """
    Auto-generate trajectory evals from the agent's tool list.
    """
    from src.evals.llm_as_judge import EvalArtifact

    if llm is None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    if not manifest.tools:
        return EvalArtifact(
            eval_type="trajectory",
            dataset_name=f"{manifest.agent_name}_trajectory_evals",
            test_cases=[],
            evaluator_fn=_trajectory_evaluator,
            metadata={"warning": "No tools found in manifest — trajectory evals skipped"},
        )

    structured_llm = llm.with_structured_output(TrajectoryEvalSuite)

    tools_list = "\n".join(
        f"  - {t.name}: {t.description[:100]}" for t in manifest.tools
    )

    prompt = TRAJECTORY_GENERATION_PROMPT.format(
        agent_name=manifest.agent_name,
        tools_list=tools_list,
    )

    suite: TrajectoryEvalSuite = structured_llm.invoke(prompt)

    test_cases = [
        {
            "input": tc.input_prompt,
            "expected_trajectory": tc.expected_tool_sequence,
            "allow_extra_tools": tc.allow_extra_tools,
            "test_category": tc.test_category,
            "rationale": tc.rationale,
        }
        for tc in suite.test_cases
    ]

    return EvalArtifact(
        eval_type="trajectory",
        dataset_name=f"{manifest.agent_name}_trajectory_evals",
        test_cases=test_cases,
        evaluator_fn=_trajectory_evaluator,
        metadata={"num_test_cases": len(test_cases), "tools_scanned": [t.name for t in manifest.tools]},
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _levenshtein(s1: list[str], s2: list[str]) -> int:
    """Edit distance between two tool sequences."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


@run_evaluator
def _trajectory_evaluator(run, example) -> EvaluationResult:
    """
    Score a trajectory by comparing actual tool calls to expected sequence.

    Looks for tool calls in run.outputs under keys:
      - "tool_calls": [{"name": "tool_name", ...}, ...]
      - "intermediate_steps": [(AgentAction, observation), ...]
    """
    expected: list[str] = example.outputs.get("expected_trajectory", [])
    allow_extra: bool = example.outputs.get("allow_extra_tools", True)

    # Extract actual tool calls from run output
    actual: list[str] = []
    outputs = run.outputs or {}

    if "tool_calls" in outputs:
        actual = [tc.get("name", "") for tc in outputs["tool_calls"]]
    elif "intermediate_steps" in outputs:
        actual = [step[0].tool for step in outputs["intermediate_steps"] if hasattr(step[0], "tool")]

    if not expected:
        return EvaluationResult(key="trajectory_score", score=1.0, comment="No expected trajectory defined")

    # Exact match
    if actual == expected:
        return EvaluationResult(key="trajectory_score", score=1.0, comment="Exact trajectory match")

    # Set match (order-insensitive)
    if set(actual) == set(expected):
        score = 0.75
        comment = "Correct tools called but in wrong order"
    else:
        # Partial match via normalized Levenshtein
        max_len = max(len(actual), len(expected), 1)
        edit_dist = _levenshtein(actual, expected)
        score = max(0.0, 1.0 - edit_dist / max_len)
        comment = f"Partial match. Expected: {expected}, Got: {actual}, edit_distance={edit_dist}"

    # Penalize extra tool calls if not allowed
    if not allow_extra and len(actual) > len(expected):
        extra_penalty = 0.1 * (len(actual) - len(expected))
        score = max(0.0, score - extra_penalty)
        comment += f" | Extra tools penalty applied"

    return EvaluationResult(key="trajectory_score", score=round(score, 3), comment=comment)
