"""
Integration tests for MetaEval Agent.

Tests the scanner and manifest extraction without requiring API keys.
Full end-to-end tests require OPENAI_API_KEY and LANGSMITH_API_KEY.
"""

import textwrap
import pytest
from src.utils.manifest import extract_manifest_from_source, AgentManifest, ToolSpec


# ---------------------------------------------------------------------------
# Scanner / Manifest Tests
# ---------------------------------------------------------------------------

SAMPLE_AGENT_SOURCE = textwrap.dedent("""
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, START, END

    system_prompt = "You are a helpful assistant for AcmeCorp customer support."

    @tool
    def retrieve_product_docs(query: str) -> str:
        \"\"\"Search and retrieve product documentation from the knowledge base.\"\"\"
        return "doc content"

    @tool
    def get_order_status(order_id: str) -> str:
        \"\"\"Fetch the current status of a customer order.\"\"\"
        return "Shipped"

    llm_model = "gpt-4o-mini"

    graph = StateGraph(dict)
    graph.add_node("agent", lambda s: s)
    graph.add_node("tools", lambda s: s)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "tools")
    graph.add_edge("tools", END)
""")


def test_scanner_extracts_tools():
    manifest = extract_manifest_from_source(SAMPLE_AGENT_SOURCE, agent_name="test-agent")
    tool_names = [t.name for t in manifest.tools]
    assert "retrieve_product_docs" in tool_names
    assert "get_order_status" in tool_names


def test_scanner_detects_retrieval_tool():
    manifest = extract_manifest_from_source(SAMPLE_AGENT_SOURCE, agent_name="test-agent")
    assert manifest.has_retrieval is True
    retrieval_tools = [t for t in manifest.tools if t.is_retrieval_tool]
    assert len(retrieval_tools) >= 1
    assert retrieval_tools[0].name == "retrieve_product_docs"


def test_scanner_extracts_graph_nodes():
    manifest = extract_manifest_from_source(SAMPLE_AGENT_SOURCE, agent_name="test-agent")
    assert "agent" in manifest.graph_nodes
    assert "tools" in manifest.graph_nodes


def test_scanner_extracts_system_prompt():
    manifest = extract_manifest_from_source(SAMPLE_AGENT_SOURCE, agent_name="test-agent")
    assert "AcmeCorp" in manifest.system_prompt


def test_scanner_detects_langgraph_type():
    manifest = extract_manifest_from_source(SAMPLE_AGENT_SOURCE, agent_name="test-agent")
    assert manifest.agent_type == "langgraph"


def test_scanner_generates_hash():
    manifest = extract_manifest_from_source(SAMPLE_AGENT_SOURCE, agent_name="test-agent")
    assert len(manifest.source_code_hash) == 12


def test_scanner_handles_syntax_error():
    bad_source = "def foo(: this is not valid python"
    manifest = extract_manifest_from_source(bad_source, agent_name="broken-agent")
    assert len(manifest.scan_warnings) > 0
    assert "SyntaxError" in manifest.scan_warnings[0]


def test_scanner_warns_on_no_tools():
    no_tools_source = textwrap.dedent("""
        system_prompt = "You are an agent."
        llm_model = "gpt-4o"
    """)
    manifest = extract_manifest_from_source(no_tools_source, agent_name="no-tools-agent")
    assert any("No @tool" in w for w in manifest.scan_warnings)


def test_manifest_summary_is_string():
    manifest = extract_manifest_from_source(SAMPLE_AGENT_SOURCE, agent_name="test-agent")
    summary = manifest.summary()
    assert isinstance(summary, str)
    assert "test-agent" in summary


# ---------------------------------------------------------------------------
# Trajectory Evaluator Tests (no API key needed)
# ---------------------------------------------------------------------------

def test_levenshtein_distance():
    from src.evals.trajectory import _levenshtein
    assert _levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert _levenshtein(["a", "b"], ["a", "b", "c"]) == 1
    assert _levenshtein([], ["a"]) == 1
    assert _levenshtein(["a"], []) == 1
    assert _levenshtein(["a", "b"], ["b", "a"]) == 2


# ---------------------------------------------------------------------------
# Risk Profile Tests
# ---------------------------------------------------------------------------

def test_risk_profile_high_filter():
    from src.utils.risk_profile import RiskProfile, RiskDimension
    profile = RiskProfile(
        agent_name="test",
        dimensions=[
            RiskDimension("hallucination", "HIGH", "reason", ["llm_as_judge"]),
            RiskDimension("tool_misuse", "MEDIUM", "reason", ["trajectory"]),
            RiskDimension("minor_issue", "LOW", "reason", ["llm_as_judge"]),
        ],
        top_concern="hallucination",
    )
    high = profile.high_risk_dimensions()
    assert len(high) == 1
    assert high[0].name == "hallucination"


def test_risk_profile_all_eval_types():
    from src.utils.risk_profile import RiskProfile, RiskDimension
    profile = RiskProfile(
        agent_name="test",
        dimensions=[
            RiskDimension("A", "HIGH", "r", ["llm_as_judge", "trajectory"]),
            RiskDimension("B", "MEDIUM", "r", ["rag", "adversarial"]),
        ],
    )
    all_types = profile.get_all_eval_types()
    assert set(all_types) == {"llm_as_judge", "trajectory", "rag", "adversarial"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
