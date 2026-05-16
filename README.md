# MetaEval Agent 🤖

> **The agent that evaluates agents.** Give it any LangGraph agent and it auto-generates a full eval suite — no test cases written by hand.

![MetaEval Agent pipeline](docs/blog-hero.svg)

## What it does

MetaEval Agent scans a target agent's source code and automatically produces:

- **LLM-as-Judge evals** — rubric-graded output quality, accuracy, and safety
- **Trajectory evals** — did the agent call the right tools in the right order?
- **RAG evals** — faithfulness, context precision, and recall (RAGAS-style)
- **Adversarial evals** — prompt injection, jailbreaks, malformed inputs

Everything is uploaded to **LangSmith** as runnable datasets.

---

## API Keys

MetaEval Agent uses two external services. Here's what each key is for and where to get it.

### 1. OpenAI API Key — required for all LLM-powered evals

Used by: the Analyzer node (risk profiling), Suggester node (eval generation), LLM-as-Judge evaluator, RAG faithfulness evaluator, and the Adversarial eval generator.

**Get your key:** https://platform.openai.com/api-keys

```bash
export OPENAI_API_KEY="sk-..."
```

> **Which model is used?** The pipeline defaults to `gpt-4o` for generation (Analyzer, Suggester, Builder) and `gpt-4o-mini` for evaluation scoring (judge calls at test time). You can override either in `src/agents/metaeval_agent.py` by changing the `_get_llm()` helper.

### 2. LangSmith API Key — required to upload datasets and run experiments

Used by: the Uploader node, which registers each eval suite as a named dataset in your LangSmith project and optionally runs your target agent against it.

**Get your key:** https://smith.langchain.com → Settings → API Keys

```bash
export LANGSMITH_API_KEY="ls__..."
export LANGSMITH_PROJECT="metaeval"   # optional, defaults to "default"
```

> **No LangSmith key?** The pipeline runs in **dry-run mode** automatically — it generates all eval artifacts locally and prints a summary, but skips the upload step. This is useful for inspecting the generated test cases without needing a LangSmith account.

### Using a `.env` file (recommended)

Create a `.env` file in the project root (it is already in `.gitignore` so it will never be committed):

```bash
# .env
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=metaeval
```

Then load it before running:

```bash
# Option A — use python-dotenv (already in requirements.txt)
python -c "from dotenv import load_dotenv; load_dotenv()"

# Option B — source it directly in your shell
set -a && source .env && set +a
```

### Key requirements by eval type

| Eval type | OpenAI key | LangSmith key |
|-----------|-----------|---------------|
| Scanner (AST analysis) | ❌ not needed | ❌ not needed |
| Analyzer — risk profiling | ✅ required | ❌ not needed |
| Suggester — ranked evals | ✅ required | ❌ not needed |
| LLM-as-Judge generation | ✅ required | ❌ not needed |
| LLM-as-Judge scoring | ✅ required | ✅ for upload |
| Trajectory evals | ✅ required | ✅ for upload |
| RAG evals (RAGAS-style) | ✅ required | ✅ for upload |
| Adversarial — LLM cases | ✅ required | ✅ for upload |
| Adversarial — hardcoded cases | ❌ not needed | ✅ for upload |
| Unit tests (`pytest tests/`) | ❌ not needed | ❌ not needed |

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/your-handle/metaeval-agent
cd metaeval-agent
pip install -r requirements.txt

# 2. Set API keys (see above)
export OPENAI_API_KEY="sk-..."
export LANGSMITH_API_KEY="ls__..."

# 3. Scan your own agent
python -m src.agents.metaeval_agent ./your_agent.py --name "my-agent"

# 4. Or try the included example target agent
python -m src.agents.metaeval_agent ./src/agents/example_target_agent.py

# 5. Run unit tests — no API keys needed
pytest tests/ -v
```

### Programmatic usage

```python
from src.agents.metaeval_agent import run_metaeval

# From a file path
result = run_metaeval(agent_path="./my_agent.py", agent_name="my-agent")

# From source code directly
with open("./my_agent.py") as f:
    source = f.read()

result = run_metaeval(agent_source_code=source, agent_name="my-agent")

# Inspect results
print(result["eval_suggestions"])       # ranked eval proposals
print(result["eval_artifacts"])         # generated test cases
print(result["langsmith_results"])      # dataset IDs and project URL
```

---

## Stack

- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent orchestration (5-node StateGraph)
- [LangSmith](https://smith.langchain.com/) — eval dataset registration and experiment tracking
- [RAGAS](https://docs.ragas.io/) — RAG evaluation metrics
- [OpenAI](https://platform.openai.com/) — LLM backbone (gpt-4o / gpt-4o-mini)
- Python 3.11+

## Project structure

```
metaeval-agent/
├── src/
│   ├── agents/
│   │   ├── metaeval_agent.py        # Main LangGraph StateGraph (5 nodes)
│   │   └── example_target_agent.py  # Sample agent to scan
│   ├── evals/
│   │   ├── llm_as_judge.py          # Needs OPENAI_API_KEY
│   │   ├── trajectory.py            # Needs OPENAI_API_KEY
│   │   ├── rag_evals.py             # Needs OPENAI_API_KEY
│   │   └── adversarial.py           # Needs OPENAI_API_KEY (LLM cases only)
│   ├── tools/
│   │   └── langsmith_uploader.py    # Needs LANGSMITH_API_KEY
│   └── utils/
│       ├── manifest.py              # AST-based scanner — no keys needed
│       └── risk_profile.py
├── tests/
│   └── test_metaeval.py             # 12 unit tests — no keys needed
├── .env                             # ← create this, never commit it
└── requirements.txt
```

## Read more

Full write-up on Medium: _[https://medium.com/@prabhi/the-agent-that-evaluates-agents-building-a-self-scaling-ai-test-harness-3e51f914d646]_

---

MIT License
