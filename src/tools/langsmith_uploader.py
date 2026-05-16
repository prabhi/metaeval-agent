"""
LangSmith Uploader

Registers eval datasets and experiments in LangSmith.
Each EvalArtifact becomes a named dataset in a dedicated project.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

from langsmith import Client

if TYPE_CHECKING:
    from src.evals.llm_as_judge import EvalArtifact


def upload_eval_artifacts(
    artifacts: list["EvalArtifact"],
    agent_name: str,
    project_prefix: str = "metaeval",
    run_evals: bool = False,
    target_agent_callable=None,
) -> dict:
    """
    Upload all eval artifacts to LangSmith.

    Args:
        artifacts: List of EvalArtifact objects from the builder node
        agent_name: Name of the target agent (used in project/dataset naming)
        project_prefix: Prefix for the LangSmith project name
        run_evals: If True, immediately run the target agent over each dataset
        target_agent_callable: The agent function to evaluate (required if run_evals=True)

    Returns:
        dict with dataset_ids, project_url, experiment_urls
    """
    client = Client()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"{project_prefix}-{agent_name}-{timestamp}"

    results = {
        "project_name": project_name,
        "dataset_ids": {},
        "experiment_urls": {},
        "skipped": [],
    }

    for artifact in artifacts:
        if not artifact.test_cases:
            results["skipped"].append(artifact.eval_type)
            continue

        # Create dataset
        dataset = client.create_dataset(
            dataset_name=artifact.dataset_name,
            description=f"Auto-generated {artifact.eval_type} evals for {agent_name} by MetaEval Agent",
            data_type="kv",
        )
        results["dataset_ids"][artifact.eval_type] = str(dataset.id)

        # Upload examples
        examples = [
            {
                "inputs": {"input": tc.get("input", tc)},
                "outputs": {k: v for k, v in tc.items() if k != "input"},
            }
            for tc in artifact.test_cases
        ]
        client.create_examples(dataset_id=dataset.id, examples=examples)

        print(f"  ✓ Uploaded {len(examples)} {artifact.eval_type} test cases → dataset '{artifact.dataset_name}'")

        # Optionally run evals
        if run_evals and target_agent_callable:
            try:
                results_obj = client.run_on_dataset(
                    dataset_name=artifact.dataset_name,
                    llm_or_chain_factory=target_agent_callable,
                    evaluation=artifact.evaluator_fn,
                    project_name=project_name,
                    verbose=False,
                )
                experiment_url = f"https://smith.langchain.com/projects/{project_name}"
                results["experiment_urls"][artifact.eval_type] = experiment_url
                print(f"  ✓ Ran {artifact.eval_type} evals → {experiment_url}")
            except Exception as e:
                print(f"  ✗ Failed to run {artifact.eval_type} evals: {e}")

    return results
