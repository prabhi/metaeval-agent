"""
AgentManifest — the schema extracted by the Scanner node.

MetaEval Agent scans a target agent's source code or module and
produces this structured manifest. Everything downstream (risk
analysis, eval generation) reasons from this object.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolSpec:
    """Represents a single tool the target agent can call."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    is_retrieval_tool: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "is_retrieval_tool": self.is_retrieval_tool,
        }


@dataclass
class AgentManifest:
    """
    Full schema of a scanned agent. Produced by the Scanner node and
    consumed by every subsequent node in the MetaEval graph.
    """
    agent_name: str
    agent_type: str                     # "langgraph" | "langchain" | "custom"
    llm_model: str                      # e.g. "gpt-4o", "claude-3-5-sonnet"
    system_prompt: str
    tools: list[ToolSpec] = field(default_factory=list)
    graph_nodes: list[str] = field(default_factory=list)
    graph_edges: list[tuple[str, str]] = field(default_factory=list)
    has_retrieval: bool = False
    output_schema: dict = field(default_factory=dict)
    source_code: str = ""
    source_code_hash: str = ""
    scan_warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.source_code and not self.source_code_hash:
            self.source_code_hash = hashlib.sha256(
                self.source_code.encode()
            ).hexdigest()[:12]
        self.has_retrieval = any(t.is_retrieval_tool for t in self.tools)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "llm_model": self.llm_model,
            "system_prompt": self.system_prompt[:500] + "..." if len(self.system_prompt) > 500 else self.system_prompt,
            "tools": [t.to_dict() for t in self.tools],
            "graph_nodes": self.graph_nodes,
            "graph_edges": [list(e) for e in self.graph_edges],
            "has_retrieval": self.has_retrieval,
            "output_schema": self.output_schema,
            "source_code_hash": self.source_code_hash,
            "scan_warnings": self.scan_warnings,
        }

    def summary(self) -> str:
        """Human-readable one-liner for LLM context windows."""
        tools_str = ", ".join(t.name for t in self.tools) or "none"
        return (
            f"Agent '{self.agent_name}' ({self.agent_type}) using {self.llm_model}. "
            f"Tools: [{tools_str}]. Nodes: {self.graph_nodes}. "
            f"Has retrieval: {self.has_retrieval}. "
            f"System prompt excerpt: {self.system_prompt[:200]!r}"
        )


def extract_manifest_from_source(source_code: str, agent_name: str = "unknown") -> AgentManifest:
    """
    Parse Python source code with AST to extract agent metadata.
    Falls back gracefully when code is too dynamic to inspect statically.
    """
    warnings: list[str] = []
    tools: list[ToolSpec] = []
    graph_nodes: list[str] = []
    graph_edges: list[tuple[str, str]] = []
    system_prompt = ""
    llm_model = "unknown"
    agent_type = "custom"

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return AgentManifest(
            agent_name=agent_name,
            agent_type="unknown",
            llm_model="unknown",
            system_prompt="",
            source_code=source_code,
            scan_warnings=[f"SyntaxError during parse: {e}"],
        )

    # Walk AST to extract signals
    for node in ast.walk(tree):
        # Detect tool definitions via @tool decorator
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                dec_name = ""
                if isinstance(decorator, ast.Name):
                    dec_name = decorator.id
                elif isinstance(decorator, ast.Attribute):
                    dec_name = decorator.attr
                if dec_name == "tool":
                    docstring = ast.get_docstring(node) or ""
                    retrieval_keywords = {"retrieve", "search", "fetch", "query", "vectorstore", "rag"}
                    is_retrieval = any(k in docstring.lower() or k in node.name.lower() for k in retrieval_keywords)
                    tools.append(ToolSpec(
                        name=node.name,
                        description=docstring,
                        is_retrieval_tool=is_retrieval,
                    ))

        # Detect LangGraph nodes via .add_node(...)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_node":
                agent_type = "langgraph"
                if node.args and isinstance(node.args[0], ast.Constant):
                    graph_nodes.append(node.args[0].value)

            # Detect .add_edge(...)
            if isinstance(func, ast.Attribute) and func.attr == "add_edge":
                if len(node.args) >= 2:
                    src = node.args[0].value if isinstance(node.args[0], ast.Constant) else "?"
                    dst = node.args[1].value if isinstance(node.args[1], ast.Constant) else "?"
                    graph_edges.append((src, dst))

        # Detect system prompt in string assignments
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and "system" in target.id.lower():
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        system_prompt = node.value.value

        # Detect LLM model strings
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            model_keywords = ["gpt-4", "gpt-3.5", "claude", "gemini", "llama", "mistral"]
            if any(k in node.value.lower() for k in model_keywords):
                llm_model = node.value

    if not tools:
        warnings.append("No @tool-decorated functions found — tool list may be incomplete.")
    if not system_prompt:
        warnings.append("No system_prompt variable detected — prompt analysis will be limited.")

    return AgentManifest(
        agent_name=agent_name,
        agent_type=agent_type,
        llm_model=llm_model,
        system_prompt=system_prompt,
        tools=tools,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        source_code=source_code,
        scan_warnings=warnings,
    )
