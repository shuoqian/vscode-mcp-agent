# VS Code Intelligent Issue & Signal Tracking Agent

An autonomous agent that monitors the [microsoft/vscode](https://github.com/microsoft/vscode) GitHub repository, filters open issues against user-specified criteria, verifies candidates through evidence-gated reasoning, cross-checks against long-term memory to avoid duplicate alerts, and delivers a fact-checked executive summary — with a human reviewer in the loop before anything is remembered across sessions.

## Problem

Maintainers, extension authors, and power users of large repositories like `microsoft/vscode` waste hours manually triaging thousands of open issues to find the subset relevant to a specific subsystem, platform, and symptom. A naive LLM-based tool for this task fails in two specific ways this project addresses directly: it can silently return topically wrong results when its search strategy guesses wrong, and it can hallucinate plausible-sounding root causes not actually present in the source issues.

## Architecture

A 6-node [LangGraph](https://github.com/langchain-ai/langgraph) state machine:

```
Planner (Tree-of-Thought query strategy selection)
   → Fetch & Vector Filter (GitHub search + ChromaDB cross-session dedup)
   → Process Issues (ReAct, evidence-required)
   → Verify & Reflect (confidence gate, ≤2 retries, else abstain)
   → Final Summary (Report Writer Agent)
   → Report Critic (Report Verifier Agent — independent, can veto)
   → Human review gate (gates long-term memory commit)
```

Full design rationale — including the Tree-of-Thought search strategy, the RAG/vector memory design, the multi-agent generator/verifier split, and the safety/guardrails/evaluation plan — is in [`DESIGN_DOC.md`](./DESIGN_DOC.md).

**Key design properties:**
- **Read-only.** The agent only calls GitHub's search endpoint and its own local vector store — no write/comment/close capability exists, even adversarially.
- **Abstains rather than guesses.** If confidence is too low after retries, the agent reports nothing rather than a weak result.
- **Independently fact-checked.** A second agent audits every draft report against source evidence before it's shown to you, with authority to veto the narrative entirely.
- **Human-gated memory.** Every report is reviewed by a human before its issues are committed to long-term memory; rejecting a report means those issues stay eligible to resurface next run.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd vscode-mcp-agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install langgraph langchain-groq httpx rich python-dotenv
   ```

4. **Setup API Key:**
   then edit .env and set:
   GITHUB_TOKEN=<your GitHub personal access token>
   GROQ_API_KEY=<your Groq API key>

A GitHub token isn't strictly required for public read access but raises your rate limit significantly — recommended for repeated use.

## Usage

```bash
python main.py
```

You'll be prompted for:
- Target OS (e.g., `Windows`)
- Target VS Code module (e.g., `Terminal`)
- Symptom/interest area (e.g., `performance-specific`)
- Issue type (e.g., `Bug`)
- Time range in days (e.g., `7`)

The agent will then show its Tree-of-Thought branch selection, fetch and filter issues, run evidence-gated verification, and display a fact-checked report (or an explicit abstention). You'll be asked to approve, reject, or skip review of the result — this decides whether the run's issues are committed to long-term memory.

## Evaluation

Every run appends one record to `agent_metrics.jsonl` (created on first run) capturing confidence, verification outcome, latency, and your review decision. To compute aggregate metrics across runs:

```python
import pandas as pd
df = pd.read_json("agent_metrics.jsonl", lines=True)

print("Groundedness rate:", (df["flagged_claims_count"] == 0).mean())
print("Veto rate:", df["report_verified"].eq(False).mean())
print("Abstention rate:", df["is_abstained"].mean())
print("Human approval rate:", df["human_feedback"].eq("approved").mean())
print("Avg latency (s):", df["total_latency_sec"].mean())
```

## Repository Structure

```
vscode-mcp-agent/
├── README.md                 # This file
├── DESIGN_DOC.md              # Full design doc: architecture, ToT, RAG, multi-agent, safety
├── main.py                    # CLI entry point, human review gate, metrics logging
├── agent/
│   ├── state.py               # LangGraph AgentState schema
│   ├── tools.py                # GitHub API, ChromaDB, label-matching utilities
│   └── workflow.py             # All 6 graph nodes + LLM orchestration
├── agent_metrics.jsonl         # Persisted per-run evaluation log (created at runtime)
├── vscode_labels_cache.json    # Cached repo label taxonomy (created at runtime)
├── requirements.txt
└── .env.example
```

## Known Limitations

- No sanitization of untrusted GitHub issue text before it's used in LLM prompts (a prompt-injection surface — see `DESIGN_DOC.md` Section 12.1).
- No labeled evaluation set yet for formal precision/calibration/dedup-accuracy metrics.
- Escalation on repeated abstentions/vetoes across runs is currently manual (read `agent_metrics.jsonl` yourself), not automated.
- Scoped to a single repository (`microsoft/vscode`) and a single LLM provider (Groq).

See `DESIGN_DOC.md` Section 9 for the full limitations and next-steps discussion.

## License
MIT