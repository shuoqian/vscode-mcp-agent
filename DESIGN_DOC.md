# VS Code Intelligent Issue & Signal Tracking Agent - Design Document

**Course:** CMU Agentic AI Capstone | **Target Repository:** microsoft/vscode
**Status:** Implementation Phase (Module 3, updated)
**Disclaimer:** Some of the content is created by learning through Gemini 3.6 flash

## 1. Selected Agent, Problem Statement, & Target User
### Problem & Core Agent Concept
The **VS Code Intelligent Signal Tracking Agent** is a smart helper designed to cut through information overload in the `microsoft/vscode` repository. Maintainers and power users waste hours reading through thousands of issues to find specific bugs or performance regressions. This agent automates the discovery, filtering, and summarization of these signals based on modular user inputs, and remembers what it has already reported so it doesn't repeat itself across runs.

### Target User
*   **VS Code Maintainers & Extension Authors:** Who need fast, non-duplicate alerts about critical bugs and specific module regressions.
*   **Power Developers & Software Engineers:** Who track specific features (like terminal, perf, MCP) and need reliable summaries of recent activity.

## 2. Why Simple Prompts and Basic LLMs Fail
*   **No Live Data Access:** Standard models cannot query the live GitHub API to see what was reported in the last 24 hours.
*   **Context Bloat:** Feeding 30+ raw issue bodies into a single prompt confuses the LLM and causes hallucinations.
*   **Brittle Search Logic:** A single guessed label (e.g. `"terminal"`) often isn't enough — repos like `microsoft/vscode` fragment the same concept across multiple real labels (`terminal`, `workbench-terminal`, `terminal-suggest`), so naive single-label queries silently under-match.
*   **Lack of State Management:** Complex filtering (OS check, module match, symptom analysis, cross-session deduplication) requires a stateful workflow that a single prompt cannot reliably maintain.

## 3. Environment & Interactive Setup
The agent connects real-time GitHub data **and** long-term vector memory to a stateful reasoning loop:

```
[GitHub REST API] ──┐
                     ├──► [LangGraph Agent State] ──► [Interactive Terminal (Rich)]
[ChromaDB Vector Memory] ──┘
```

*   **Data Sources:** GitHub REST API (`/search/issues`) for live issue retrieval; a local persistent ChromaDB collection for cross-session memory.
*   **Reasoning Engine:** Groq (Llama 3.3 70B, JSON mode) for high-speed taxonomy classification, planning, and per-issue analysis.
*   **State Management:** LangGraph `AgentState` (TypedDict) stores raw issues, categorized hard/soft label groups, the candidate pool, and the multi-step execution plan.
*   **User Interface:** A command-line interface built with `Rich` for formatted prompts, status panels, and executive summaries.

## 4. Agent Actions & Step-by-Step Cycle
The agent follows a 5-node state machine logic:

1.  **Planner Node:** Analyzes user criteria (OS, Module, Symptom, Issue Type) and classifies candidate labels into categories:
    *   `module`, `type`, and `os` are treated as **hard filters** — each category can resolve to multiple real repo labels (OR'd together, e.g. `label:"terminal","workbench-terminal"`), and categories are AND'd against each other.
    *   `symptom` is treated as a **soft/semantic-only** signal — it is never used to filter the GitHub query, only to judge relevance later.
    *   Generates exactly 2 verification tasks per issue: (1) module + issue-type check, (2) symptom check.
2.  **Fetch & Vector Filter Node:** Retrieves live issues from GitHub using the categorized query, with **progressive relaxation** (dropping `os` → `module` → `type` in that order, one whole category at a time) if the fully-constrained query returns nothing. Every surviving issue is then checked against ChromaDB — issues that are semantic near-matches of something reported in a **past run** are dropped here, before ReAct ever sees them.
3.  **Process Issues (ReAct) Node:** Iteratively runs each remaining issue through the 2 verification tasks, requiring a quoted evidence snippet per task and explicitly rejecting tangential matches. An issue is only pooled if it passes both tasks.
4.  **Verify & Reflect Node:** Audits the candidate pool's average confidence against a 0.80 threshold. If it fails, it triggers a **Dynamic Reflection Loop** (max 2 retries) with injected critique feedback before **abstaining** rather than delivering a low-confidence report.
5.  **Final Summary Node:** Synthesizes the verified pool into a bulleted executive report, then commits the newly-reported issues into ChromaDB so future runs can retrieve against them.

## 5. How Feedback Guides the Agent
*   **Short-Loop System Feedback:** The `verify_and_reflect` node evaluates whether the candidate pool clears the confidence threshold. If not, the agent receives "Critique Feedback" and re-runs ReAct (up to 2 retries) before abstaining.
*   **Long-Loop Cross-Session Feedback:** ChromaDB vector memory acts as persistent feedback across runs — it tells the Fetch node what's already been reported, so repeat runs only surface new signals. (Full retrieval design in Section 9.)
*   **Structured User Context:** The agent's strategy is initialized using specific fields (Module, Symptom, Issue Type) rather than a single sentence, allowing for more precise planning.

## 6. User Interaction
The system runs as an interactive Python CLI.

### 1. Terminal Startup Questions
```text
[?] Target OS Environment [Windows]: Windows
[?] Target VS Code Module (e.g. Agent, Terminal, User Interface, MCP, etc) [Terminal]: Terminal
[?] Describe your interests / symptoms (any, performance-specific or memory-specific) [performance-specific]: performance-specific
[?] Target Issue Type (e.g. New Feature, Bug, etc) [Bug]: Bug
[?] Time range in days (e.g., 1, 7, 30) [7]: 7
```

### 2. Real-Time Execution Logging (actual run output)
```text
├─► [Constructed Query Preview]: repo:microsoft/vscode is:issue state:open created:>=2026-08-01 label:"terminal" label:"bug"
├─► Hard Label Groups (OR within category, AND across): {'module': ['terminal'], 'type': ['bug'], 'os': []}
├─► Soft Labels (judge context only, symptom): ['performance']

[Fetch Node] ℹ️ Effective hard label groups: {'module': ['terminal'], 'type': ['bug']}
[Fetch Result] Successfully fetched 1 issue(s).
└─► [Memory Screen Result] 1 issue(s) remaining (0 duplicate(s) suppressed).

[ReAct Loop 1/1: Issue #329539] "agent host terminal does not survive on reload"
├─► [Task 1] ✅ PASSED (Confidence: 0.90, Evidence: 'terminal, bug')
├─► [Task 2] ❌ DISCARDED (Passed: False, Conf: 0.0, Evidence: "no mention of 'performance'")

📊 [REFLECTION] Pool Confidence: 0.00 (Threshold: 0.80) → Retry #1 → Retry #2 → [ABSTENTION TRIGGERED]
```
Note: unlike an earlier draft of this doc, there is no separate per-issue "deduplication task" inside the ReAct loop — cross-session deduplication happens once, upfront, in the Fetch & Vector Filter Node via ChromaDB, not as a third ReAct task.

## 7. Consideration of Using Model Context Protocol (MCP)
The architecture is designed to be **MCP-Ready**. While currently integrated, the components are logically decoupled:
*   **MCP Host:** The LangGraph workflow controller.
*   **GitHub MCP Server:** Wraps API logic for standardized issue retrieval.
*   **Memory MCP Server:** Wraps ChromaDB for standardized cross-session semantic deduplication.

This decoupling allows the core reasoning agent to be hosted in different environments (CLI, Claude Desktop, Cursor) while reusing the same toolsets.

**Current Status:** ChromaDB-based long-term memory is already implemented and running (see Section 9) — but as a **direct Python library integration**, not yet behind a formal MCP Server boundary. Similarly, GitHub access is a direct `httpx`/`requests` call, not an MCP tool call. Transitioning both to a formal MCP Client/Server architecture is slated for a future module; today's "MCP-Ready" claim refers to the logical decoupling in the code's structure, not a working MCP transport.

## 8. Core Design Reasoning
To solve the problem of information overload in open-source tracking, this agent relies on five specific architectural pillars:

*   **Categorized Hard/Soft Filtering:** Splitting matched labels into `module`/`type`/`os` (hard, AND'd across categories, OR'd within) versus `symptom` (soft, semantic-only) avoids the failure mode of either over-constraining a query into zero results or under-matching by picking only one label per concept.
*   **Reasoning Loop (ReAct & Reflection):** The agent implements a ReAct-based loop to evaluate individual issues, requiring quoted evidence per check. A failed check immediately discards the issue; a passed check contributes to its pooled confidence score. The **Reflection Loop** then audits the pool as a whole, deciding whether to retry or abstain.
*   **Memory Requirements (Short and Long Term):**
    *   **Short-Term Memory:** Managed via `AgentState` to maintain the candidate pool and execution logs during a single session.
    *   **Long-Term Memory:** Implemented via ChromaDB for **cross-session deduplication** — without it, the agent would repeatedly alert the user about the same persistent issue every time it runs. See Section 9 for the full retrieval design.
*   **Tool Grounding:** The GitHub REST API tool resolves the "LLM Knowledge Cutoff" limitation by grounding reasoning in real-time repository state, so the agent never summarizes stale or resolved data.
*   **Comparison to Prompt-Only Approach:** A prompt-only approach fails under **Context Bloat**. This design resolves that by isolating each issue into its own reasoning step and pre-filtering both by taxonomy (labels) and by memory (ChromaDB) before any issue reaches the LLM judge — scaling regardless of how many issues are fetched.

## 9. Retrieval-Augmented Generation (RAG) & Vector Database Design

**Is retrieval required?** Yes. The agent needs to know whether an issue has already been reported in a past run, and that can't be answered from the live GitHub payload alone. It requires semantic retrieval from a persistent store of previously-surfaced issues to prevent duplicate alerts across sessions.

**Integration:** External data source is a local **ChromaDB** persistent collection (`vscode_reported_issues`), populated by the agent's own past outputs.
*   **Read (retrieval):** In the Fetch node, each freshly-fetched issue's `title + body[:300]` is embedded (ChromaDB `DefaultEmbeddingFunction`, `all-MiniLM-L6-v2`) and queried against the collection (`n_results=1`). If nearest-neighbor distance ≤ 0.35, the issue is treated as a duplicate and dropped before it reaches ReAct.
*   **Write (indexing):** In the Final Summary node, after an issue clears the confidence gate and is reported, it's embedded and `upsert`'d into the same collection for future runs to retrieve against.
*   **Data model per stored issue:** id = `issue_<number>`, document = `title + body[:300]` (the embedded text), metadata = `{number, url, timestamp}`.

**Retrieval changing the output — example:** Run 1 reports Issue #310201 ("Terminal input severely laggy after extension update") and stores its embedding. In Run 2, GitHub independently returns Issue #315980 — a differently-worded report of the same regression. Its embedding lands within the distance threshold of #310201's stored embedding, so it's filtered out before ReAct/reporting. Without retrieval, #315980 would have been reported again as if new.

**Key design choices:**
*   **Source:** the agent's own prior outputs only — no external corpus.
*   **Chunking:** one document per issue (title + first 300 chars of body), no further splitting — issues are already short, self-contained units.
*   **Top-k:** `n_results=1` — the question is binary (duplicate or not), so only the nearest neighbor matters.
*   **Metric/threshold:** ChromaDB default L2 distance, ≤ 0.35, tuned empirically.

**Failure mode & mitigation:** The main risk is false-positive suppression — two distinct issues with similar wording could fall within the distance threshold and get incorrectly merged, silently hiding a genuinely new issue from the user. The threshold is kept conservative (biased toward under-suppressing), the embedded text includes body content (not just title) to reduce superficial matches, and every run logs how many issues were suppressed as duplicates so under-reporting is at least visible and investigable.