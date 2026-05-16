"""
MetaEval Agent — Main LangGraph StateGraph

This is the orchestrator. It wires together 5 nodes:
  1. scanner_node    — extracts AgentManifest from target agent source
  2. analyzer_node   — builds a RiskProfile from the manifest
  3. suggester_node  — ranks and justifies eval suggestions
  4. builder_node    — generates all 4 eval type artifacts
  5. uploader_node   — registers datasets in LangSmith

Run via CLI:
  python -m src.agents.metaeval_agent --agent-path ./src/agents/example_target_agent.py

Or import and call:
  from src.agents.metaeval_agent import run_metaeval
  result = run_metaeval(agent_source_code=source, agent_name="my-agent")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.evals.adversarial import generate_adversarial_evals
from src.evals.llm_as_judge import EvalArtifact, generate_llm_judge_evals
from src.evals.rag_evals import generate_rag_evals
from src.evals.trajectory import generate_trajectory_evals
from src.tools.langsmith_uploader import upload_eval_artifacts
from src.utils.manifest import AgentManifest, extract_manifest_from_source
from src.utils.risk_profile import RiskProfile, RiskProfileSchema


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class MetaEvalState(TypedDict):
    # Input
    target_agent_path: str
    target_agent_source: str
    agent_name: str

    # Node outputs
    agent_manifest: AgentManifest | None
    risk_profile: RiskProfile | None
    eval_suggestions: list[dict]
    eval_artifacts: list[EvalArtifact]

    # Uploader output
    langsmith_results: dict

    # Control
    messages: Annotated[list[BaseMessage], add_messages]
    error: str | None


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=temperature)


# ---------------------------------------------------------------------------
# Node 1: Scanner
# ---------------------------------------------------------------------------

def scanner_node(state: MetaEvalState) -> dict:
    """
    Read the target agent's source code and extract its AgentManifest.
    Supports: file path, inline source code, or '-' for stdin.
    """
    source = state.get("target_agent_source", "")
    path = state.get("target_agent_path", "")
    agent_name = state.get("agent_name", "unknown-agent")

    if not source and path:
        try:
            source = Path(path).read_text()
            agent_name = Path(path).stem
        except FileNotFoundError:
            return {
                "error": f"File not found: {path}",
                "messages": [AIMessage(content=f"❌ Scanner: file not found at '{path}'")],
            }

    if not source:
        return {
            "error": "No source code provided",
            "messages": [AIMessage(content="❌ Scanner: no source code or file path provided")],
        }

    manifest = extract_manifest_from_source(source, agent_name=agent_name)

    summary_msg = (
        f"✅ Scanner complete for '{manifest.agent_name}'\n"
        f"   Type: {manifest.agent_type} | Model: {manifest.llm_model}\n"
        f"   Tools: {[t.name for t in manifest.tools]}\n"
        f"   Nodes: {manifest.graph_nodes}\n"
        f"   Has retrieval: {manifest.has_retrieval}\n"
        f"   Warnings: {manifest.scan_warnings or 'none'}"
    )

    return {
        "agent_manifest": manifest,
        "agent_name": manifest.agent_name,
        "messages": [AIMessage(content=summary_msg)],
        "error": None,
    }


# ---------------------------------------------------------------------------
# Node 2: Analyzer
# ---------------------------------------------------------------------------

ANALYZER_PROMPT = """You are a senior AI safety and reliability engineer.

Analyze this agent manifest and produce a structured risk profile.
Identify ALL relevant failure dimensions — be thorough, not conservative.

Agent Manifest:
{manifest_summary}

Output a RiskProfileSchema JSON with:
- All risk dimensions (at least 5, maximum 10)
- Severity: HIGH (likely to fail, high impact), MEDIUM, or LOW
- Which eval types apply: llm_as_judge, trajectory, rag, adversarial
- Top concern: the single most important failure mode to test
- Recommended eval priority order
"""


def analyzer_node(state: MetaEvalState) -> dict:
    """Use LLM to analyze the manifest and build a RiskProfile."""
    manifest: AgentManifest = state.get("agent_manifest")
    if not manifest:
        return {"error": "No manifest available", "messages": [AIMessage(content="❌ Analyzer: missing manifest")]}

    llm = _get_llm()
    structured_llm = llm.with_structured_output(RiskProfileSchema)

    prompt = ANALYZER_PROMPT.format(manifest_summary=manifest.summary())
    schema: RiskProfileSchema = structured_llm.invoke(prompt)
    risk_profile = RiskProfile.from_schema(schema)

    summary_msg = (
        f"✅ Analyzer complete\n"
        f"   Top concern: {risk_profile.top_concern}\n"
        f"   HIGH risk dimensions: {[d.name for d in risk_profile.high_risk_dimensions()]}\n"
        f"   Eval priority: {risk_profile.recommended_eval_priority}"
    )

    return {
        "risk_profile": risk_profile,
        "messages": [AIMessage(content=summary_msg)],
    }


# ---------------------------------------------------------------------------
# Node 3: Suggester
# ---------------------------------------------------------------------------

class EvalSuggestion(BaseModel):
    eval_type: str = Field(description="llm_as_judge | trajectory | rag | adversarial")
    title: str = Field(description="Short descriptive title for this eval")
    rationale: str = Field(description="Why this specific eval matters for this agent")
    priority: int = Field(description="1 (highest) to 10 (lowest)")
    estimated_test_cases: int = Field(description="How many test cases to generate")
    implementation_hint: str = Field(description="Key implementation detail to get right")


class EvalSuggestionList(BaseModel):
    suggestions: list[EvalSuggestion]


SUGGESTER_PROMPT = """You are an AI evaluation strategist.

Given this risk profile, produce a ranked list of concrete eval suggestions.

{risk_profile_summary}

For each HIGH and MEDIUM risk dimension, suggest 2-3 specific evals.
Each suggestion should be concrete and actionable — not generic.
Rank by: (severity × coverage × ease_of_implementation).

Respond as EvalSuggestionList JSON.
"""


def suggester_node(state: MetaEvalState) -> dict:
    """Generate and rank specific eval suggestions from the risk profile."""
    risk_profile: RiskProfile = state.get("risk_profile")
    if not risk_profile:
        return {"error": "No risk profile", "messages": [AIMessage(content="❌ Suggester: missing risk profile")]}

    llm = _get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(EvalSuggestionList)

    prompt = SUGGESTER_PROMPT.format(risk_profile_summary=risk_profile.summary_for_llm())
    result: EvalSuggestionList = structured_llm.invoke(prompt)

    suggestions = sorted(
        [s.model_dump() for s in result.suggestions],
        key=lambda x: x["priority"]
    )

    summary_msg = (
        f"✅ Suggester complete — {len(suggestions)} evals suggested\n"
        + "\n".join(
            f"   [{s['priority']}] {s['eval_type']}: {s['title']}"
            for s in suggestions[:5]
        )
        + ("\n   ..." if len(suggestions) > 5 else "")
    )

    return {
        "eval_suggestions": suggestions,
        "messages": [AIMessage(content=summary_msg)],
    }


# ---------------------------------------------------------------------------
# Node 4: Builder
# ---------------------------------------------------------------------------

def builder_node(state: MetaEvalState) -> dict:
    """
    Dispatch to all 4 eval generators and collect EvalArtifacts.
    Always builds all 4 types; skips gracefully when not applicable.
    """
    manifest: AgentManifest = state["agent_manifest"]
    llm = _get_llm(temperature=0.2)

    artifacts: list[EvalArtifact] = []

    print("\n🔨 Builder: generating eval artifacts...")

    judge_artifact = generate_llm_judge_evals(manifest, llm=llm)
    artifacts.append(judge_artifact)
    print(f"   ✓ LLM-as-judge: {len(judge_artifact.test_cases)} test cases")

    traj_artifact = generate_trajectory_evals(manifest, llm=llm)
    artifacts.append(traj_artifact)
    print(f"   ✓ Trajectory: {len(traj_artifact.test_cases)} test cases")

    rag_artifact = generate_rag_evals(manifest, llm=llm)
    artifacts.append(rag_artifact)
    skipped = "(skipped — no retrieval)" if rag_artifact.metadata.get("skipped") else f"{len(rag_artifact.test_cases)} test cases"
    print(f"   ✓ RAG evals: {skipped}")

    adv_artifact = generate_adversarial_evals(manifest, llm=llm)
    artifacts.append(adv_artifact)
    print(f"   ✓ Adversarial: {len(adv_artifact.test_cases)} test cases")

    total = sum(len(a.test_cases) for a in artifacts)
    summary_msg = (
        f"✅ Builder complete — {total} total test cases across {len(artifacts)} eval types\n"
        f"   judge={len(judge_artifact.test_cases)} | "
        f"trajectory={len(traj_artifact.test_cases)} | "
        f"rag={len(rag_artifact.test_cases)} | "
        f"adversarial={len(adv_artifact.test_cases)}"
    )

    return {
        "eval_artifacts": artifacts,
        "messages": [AIMessage(content=summary_msg)],
    }


# ---------------------------------------------------------------------------
# Node 5: Uploader
# ---------------------------------------------------------------------------

def uploader_node(state: MetaEvalState) -> dict:
    """Upload all eval artifacts to LangSmith."""
    artifacts: list[EvalArtifact] = state.get("eval_artifacts", [])
    agent_name: str = state.get("agent_name", "unknown")

    if not artifacts:
        return {
            "langsmith_results": {},
            "messages": [AIMessage(content="⚠️ Uploader: no artifacts to upload")],
        }

    # Check for LangSmith API key
    if not os.getenv("LANGSMITH_API_KEY"):
        mock_results = {
            "project_name": f"metaeval-{agent_name}-dry-run",
            "note": "LANGSMITH_API_KEY not set — dry run mode. Set the key to upload to LangSmith.",
            "dataset_ids": {a.eval_type: f"mock-{a.eval_type}-id" for a in artifacts if a.test_cases},
        }
        return {
            "langsmith_results": mock_results,
            "messages": [AIMessage(content=f"⚠️ Dry run mode (no LANGSMITH_API_KEY). Would upload {len(artifacts)} datasets.")],
        }

    results = upload_eval_artifacts(
        artifacts=artifacts,
        agent_name=agent_name,
        run_evals=False,  # Set True to immediately run the agent under eval
    )

    summary_msg = (
        f"✅ Uploader complete\n"
        f"   Project: {results['project_name']}\n"
        f"   Datasets created: {list(results['dataset_ids'].keys())}\n"
        f"   Skipped (no test cases): {results.get('skipped', [])}"
    )

    return {
        "langsmith_results": results,
        "messages": [AIMessage(content=summary_msg)],
    }


# ---------------------------------------------------------------------------
# Error routing
# ---------------------------------------------------------------------------

def should_continue(state: MetaEvalState) -> str:
    if state.get("error"):
        return "end"
    return "continue"


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(MetaEvalState)

    graph.add_node("scanner", scanner_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("suggester", suggester_node)
    graph.add_node("builder", builder_node)
    graph.add_node("uploader", uploader_node)

    graph.add_edge(START, "scanner")

    # Conditional edges: stop on error, continue otherwise
    graph.add_conditional_edges(
        "scanner",
        should_continue,
        {"continue": "analyzer", "end": END},
    )
    graph.add_conditional_edges(
        "analyzer",
        should_continue,
        {"continue": "suggester", "end": END},
    )
    graph.add_edge("suggester", "builder")
    graph.add_edge("builder", "uploader")
    graph.add_edge("uploader", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

metaeval_graph = build_graph()


def run_metaeval(
    agent_source_code: str = "",
    agent_path: str = "",
    agent_name: str = "unknown-agent",
) -> MetaEvalState:
    """
    Run the full MetaEval pipeline on a target agent.

    Args:
        agent_source_code: Raw Python source code of the agent
        agent_path: Path to the agent's Python file (alternative to source_code)
        agent_name: Human-readable name for the agent

    Returns:
        Final MetaEvalState with all artifacts and LangSmith results
    """
    initial_state: MetaEvalState = {
        "target_agent_path": agent_path,
        "target_agent_source": agent_source_code,
        "agent_name": agent_name,
        "agent_manifest": None,
        "risk_profile": None,
        "eval_suggestions": [],
        "eval_artifacts": [],
        "langsmith_results": {},
        "messages": [HumanMessage(content=f"Scan and evaluate agent: {agent_name or agent_path}")],
        "error": None,
    }

    final_state = metaeval_graph.invoke(initial_state)
    return final_state


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    app = typer.Typer()
    console = Console()

    @app.command()
    def main(
        agent_path: str = typer.Argument(..., help="Path to the target agent Python file"),
        agent_name: str = typer.Option("", "--name", "-n", help="Override agent name"),
    ):
        console.print(Panel.fit("🤖 MetaEval Agent", subtitle="Auto-generating evals for your agent"))

        final_state = run_metaeval(agent_path=agent_path, agent_name=agent_name or Path(agent_path).stem)

        if final_state.get("error"):
            console.print(f"[red]Error: {final_state['error']}[/red]")
            raise typer.Exit(1)

        # Print suggestions table
        suggestions = final_state.get("eval_suggestions", [])
        if suggestions:
            table = Table(title="Eval Suggestions (ranked by priority)")
            table.add_column("Priority", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Title")
            table.add_column("Rationale")
            for s in suggestions[:8]:
                table.add_row(str(s["priority"]), s["eval_type"], s["title"], s["rationale"][:60] + "...")
            console.print(table)

        # Print artifact summary
        artifacts = final_state.get("eval_artifacts", [])
        for a in artifacts:
            status = "✓" if a.test_cases else "⚠ skipped"
            console.print(f"  {status} {a.eval_type}: {len(a.test_cases)} test cases → '{a.dataset_name}'")

        ls_results = final_state.get("langsmith_results", {})
        if ls_results.get("project_name"):
            console.print(f"\n[green]LangSmith project:[/green] {ls_results['project_name']}")

    app()
