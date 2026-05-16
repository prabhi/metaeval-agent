"""
RiskProfile — output of the Analyzer node.

The Analyzer LLM call reads the AgentManifest and identifies
which failure dimensions are most likely and most impactful
for THIS specific agent. Each dimension maps to one or more
eval types that MetaEval will auto-generate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic models (used with LLM structured output)
# ---------------------------------------------------------------------------

class RiskDimensionSchema(BaseModel):
    name: str = Field(description="Short name for this risk dimension")
    severity: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="How likely and impactful this failure mode is"
    )
    reason: str = Field(
        description="1-2 sentence explanation of why this is a risk for this specific agent"
    )
    applicable_eval_types: list[str] = Field(
        description="Which eval types address this risk: llm_as_judge, trajectory, rag, adversarial"
    )


class RiskProfileSchema(BaseModel):
    agent_name: str
    dimensions: list[RiskDimensionSchema] = Field(
        description="All identified risk dimensions, sorted HIGH → LOW"
    )
    top_concern: str = Field(
        description="The single most important thing to evaluate for this agent"
    )
    recommended_eval_priority: list[str] = Field(
        description="Ordered list of eval types to implement first"
    )


# ---------------------------------------------------------------------------
# Dataclasses (internal representation)
# ---------------------------------------------------------------------------

@dataclass
class RiskDimension:
    name: str
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    reason: str
    applicable_eval_types: list[str]

    @classmethod
    def from_schema(cls, schema: RiskDimensionSchema) -> "RiskDimension":
        return cls(
            name=schema.name,
            severity=schema.severity,
            reason=schema.reason,
            applicable_eval_types=schema.applicable_eval_types,
        )


@dataclass
class RiskProfile:
    agent_name: str
    dimensions: list[RiskDimension] = field(default_factory=list)
    top_concern: str = ""
    recommended_eval_priority: list[str] = field(default_factory=list)

    @classmethod
    def from_schema(cls, schema: RiskProfileSchema) -> "RiskProfile":
        return cls(
            agent_name=schema.agent_name,
            dimensions=[RiskDimension.from_schema(d) for d in schema.dimensions],
            top_concern=schema.top_concern,
            recommended_eval_priority=schema.recommended_eval_priority,
        )

    def high_risk_dimensions(self) -> list[RiskDimension]:
        return [d for d in self.dimensions if d.severity == "HIGH"]

    def get_all_eval_types(self) -> list[str]:
        """Deduplicated list of all required eval types across all dimensions."""
        seen = set()
        result = []
        for dim in self.dimensions:
            for et in dim.applicable_eval_types:
                if et not in seen:
                    seen.add(et)
                    result.append(et)
        return result

    def summary_for_llm(self) -> str:
        lines = [f"Risk Profile for '{self.agent_name}':", f"Top concern: {self.top_concern}"]
        for dim in self.dimensions:
            lines.append(
                f"  [{dim.severity}] {dim.name}: {dim.reason} "
                f"(evals: {', '.join(dim.applicable_eval_types)})"
            )
        return "\n".join(lines)
