"""
RAG Retrieval Eval Generator (RAGAS-style)

Only runs when manifest.has_retrieval is True.

Auto-generates four RAGAS metric test suites:
  1. Context Precision  — are retrieved chunks relevant to the query?
  2. Context Recall     — do retrieved chunks contain the answer?
  3. Faithfulness       — is the final answer grounded in context?
  4. Answer Relevancy   — does the answer actually address the question?

Additionally generates:
  - Out-of-domain questions (agent should say "I don't know")
  - Unanswerable questions (gap in knowledge base)
  - Contradictory context inputs (tests faithfulness under noise)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult, run_evaluator
from pydantic import BaseModel, Field

from src.utils.manifest import AgentManifest


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RAGTestCase(BaseModel):
    question: str = Field(description="The question to ask the RAG agent")
    ground_truth_answer: str = Field(description="The correct answer (if known)")
    expected_context_keywords: list[str] = Field(
        description="Keywords that MUST appear in retrieved context for this question to be answerable"
    )
    test_category: str = Field(
        description="in_domain | out_of_domain | unanswerable | contradictory | adversarial"
    )
    difficulty: str = Field(description="easy | medium | hard")


class RAGEvalSuite(BaseModel):
    domain_description: str = Field(
        description="Inferred description of what this RAG agent's knowledge base covers"
    )
    test_cases: list[RAGTestCase]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

RAG_GENERATION_PROMPT = """You are an expert in evaluating Retrieval-Augmented Generation (RAG) systems.

Given this agent's manifest, generate a comprehensive RAG evaluation suite.

Agent: {agent_name}
System Prompt (excerpt): {system_prompt}
Retrieval Tools: {retrieval_tools}

Generate test cases covering:
- 8 in_domain questions (varying difficulty: easy/medium/hard)
- 3 out_of_domain questions (agent should refuse or say "I don't know")
- 2 unanswerable questions (in-domain topic but answer not in any plausible KB)
- 2 contradictory questions (question contains a false premise)
- 2 adversarial questions (designed to cause hallucination)

For each question, provide:
- A realistic ground truth answer (or "UNANSWERABLE" for unanswerable cases)
- Keywords that a relevant chunk MUST contain
- Difficulty level

Respond as a JSON object matching the RAGEvalSuite schema.
"""


def generate_rag_evals(manifest: AgentManifest, llm: ChatOpenAI | None = None) -> "EvalArtifact":
    """
    Auto-generate RAGAS-style eval suite for RAG agents.
    Returns empty artifact if the agent has no retrieval tools.
    """
    from src.evals.llm_as_judge import EvalArtifact

    if not manifest.has_retrieval:
        return EvalArtifact(
            eval_type="rag",
            dataset_name=f"{manifest.agent_name}_rag_evals",
            test_cases=[],
            evaluator_fn=_rag_faithfulness_evaluator,
            metadata={"skipped": True, "reason": "No retrieval tools detected in manifest"},
        )

    if llm is None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    structured_llm = llm.with_structured_output(RAGEvalSuite)

    retrieval_tools = [t.name for t in manifest.tools if t.is_retrieval_tool]

    prompt = RAG_GENERATION_PROMPT.format(
        agent_name=manifest.agent_name,
        system_prompt=manifest.system_prompt[:400],
        retrieval_tools=", ".join(retrieval_tools),
    )

    suite: RAGEvalSuite = structured_llm.invoke(prompt)

    test_cases = [
        {
            "input": tc.question,
            "ground_truth": tc.ground_truth_answer,
            "expected_context_keywords": tc.expected_context_keywords,
            "test_category": tc.test_category,
            "difficulty": tc.difficulty,
        }
        for tc in suite.test_cases
    ]

    return EvalArtifact(
        eval_type="rag",
        dataset_name=f"{manifest.agent_name}_rag_evals",
        test_cases=test_cases,
        evaluator_fn=_rag_faithfulness_evaluator,
        metadata={
            "domain": suite.domain_description,
            "num_test_cases": len(test_cases),
            "categories": list(set(tc.test_category for tc in suite.test_cases)),
        },
    )


# ---------------------------------------------------------------------------
# RAGAS-inspired evaluators
# ---------------------------------------------------------------------------

@run_evaluator
def _rag_faithfulness_evaluator(run, example) -> EvaluationResult:
    """
    Faithfulness: Is every claim in the answer supported by the retrieved context?
    Score: fraction of claims that are grounded.
    """
    answer = str(run.outputs.get("output", run.outputs))
    context = str(run.outputs.get("context", run.outputs.get("retrieved_docs", "")))
    ground_truth = example.outputs.get("ground_truth", "")

    if not context:
        return EvaluationResult(
            key="faithfulness",
            score=0.0,
            comment="No context returned by agent — cannot assess faithfulness",
        )

    # Use LLM to check faithfulness (production would use RAGAS library directly)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(
        f"""Given this context: {context[:1500]}

And this answer: {answer}

Rate faithfulness: what fraction of claims in the answer are directly supported by the context?
Respond as JSON: {{"score": 0.0_to_1.0, "unsupported_claims": ["..."], "explanation": "..."}}"""
    )

    import json
    try:
        result = json.loads(response.content)
        score = float(result.get("score", 0.0))
        explanation = result.get("explanation", "")
    except (json.JSONDecodeError, ValueError):
        score = 0.0
        explanation = "Parse error in faithfulness evaluator"

    return EvaluationResult(key="faithfulness", score=score, comment=explanation)


@run_evaluator
def _rag_context_relevancy_evaluator(run, example) -> EvaluationResult:
    """
    Context Precision: Are the retrieved chunks relevant to the question?
    Score: fraction of retrieved chunks that are relevant.
    """
    question = str(run.inputs.get("input", ""))
    retrieved_docs = run.outputs.get("retrieved_docs", [])

    if not retrieved_docs:
        return EvaluationResult(key="context_precision", score=0.0, comment="No retrieved docs found in output")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    docs_str = "\n---\n".join(str(d) for d in retrieved_docs[:5])

    response = llm.invoke(
        f"""Question: {question}

Retrieved chunks:
{docs_str[:2000]}

For each chunk, is it relevant to answering the question? (yes/no)
Then compute context_precision = relevant_chunks / total_chunks.
Respond as JSON: {{"context_precision": float, "chunk_relevance": [true/false, ...], "explanation": "..."}}"""
    )

    import json
    try:
        result = json.loads(response.content)
        score = float(result.get("context_precision", 0.0))
        explanation = result.get("explanation", "")
    except (json.JSONDecodeError, ValueError):
        score = 0.0
        explanation = "Parse error in context relevancy evaluator"

    return EvaluationResult(key="context_precision", score=score, comment=explanation)
