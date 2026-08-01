# VS Code Intelligent Issue & Signal Tracking Agent - Design Document

**Course:** CMU Agentic AI Capstone | **Target Repository:** microsoft/vscode  
**Status:** Implementation Phase (Module 2 Complete)  
**Disclaimer:**  Some of the content is created by learning through Gemini 3.6 flash  

## 1. Selected Agent, Problem Statement, & Target User
### Problem & Core Agent Concept
The **VS Code Intelligent Signal Tracking Agent** is a smart helper designed to cut through information overload in the `microsoft/vscode` repository. Maintainers and power users waste hours reading through thousands of issues to find specific bugs or performance regressions. This agent automates the discovery, filtering, and summarization of these signals based on modular user inputs.

### Target User
*   **VS Code Maintainers & Extension Authors:** Who need fast, non-duplicate alerts about critical bugs and specific module regressions.
*   **Power Developers & Software Engineers:** Who track specific features (like terminal, perf, MCP) and need reliable summaries of recent activity.

## 2. Why Simple Prompts and Basic LLMs Fail
*   **No Live Data Access:** Standard models cannot query the live GitHub API to see what was reported in the last 24 hours.
*   **Context Bloat:** Feeding 30+ raw issue bodies into a single prompt confuses the LLM and causes hallucinations.
*   **Lack of State Management:** Complex filtering (OS check, module match, symptom analysis, deduplication) requires a stateful workflow that a single prompt cannot reliably maintain.

## 3. Environment & Interactive Setup
The agent connects real-time GitHub data to a stateful reasoning loop:

`[GitHub REST API] ──► [LangGraph Agent State] ──► [Interactive Terminal (Rich)]`

*   **Data Sources:** GitHub REST API (Search endpoint) for live issue retrieval.
*   **Reasoning Engine:** Groq (Llama 3.3 70B) for high-speed planning and analysis.
*   **State Management:** LangGraph `AgentState` (TypedDict) stores raw issues, candidate pool, and the multi-step execution plan.
*   **User Interface:** A command-line interface built with `Rich` for formatted prompts, status panels, and executive summaries.

## 4. Agent Actions & Step-by-Step Cycle
The agent follows a 5-node state machine logic:

1.  **Planner Node:** Analyzes user criteria (OS, Module, Symptom, Issue Type) and categorizes the request:
    *   **Simple Request:** Generates EXACTLY 2 distinct filtering tasks.
    *   **Complicated Request:** Generates EXACTLY 3 distinct filtering tasks.
    *   **Automatic Task:** Appends a mandatory "Deduplicate matching issues" task to the plan.
2.  **Fetch Issues Node:** Retrieves live issues from GitHub based on the user's specified time lookback.
3.  **Process Issues (ReAct) Node:** Iteratively processes each fetched issue against the planner's tasks:
    *   Executes N filtering tasks sequentially (e.g., Task 1: Module/Type, Task 2: OS/Symptoms).
    *   Executes the final Deduplication Task (Task 3 or 4) to ensure uniqueness before adding to the candidate pool.
4.  **Verify & Reflect Node:** Audits the candidate pool. If the results are poor or the pool is empty, it triggers a **Dynamic Reflection Loop** (max 2 retries) to re-process data with corrective feedback.
5.  **Final Summary Node:** Synthesizes the verified pool into a bulleted executive report.

## 5. How Feedback Guides the Agent
*   **Short-Loop System Feedback:** The `verify_and_reflect` node evaluates if the candidate pool is adequate. If "NEEDS_REFINEMENT" is triggered, the agent receives "Critique Feedback" (e.g., "The pool is empty, broaden matching criteria") to adjust logic in the next loop.
*   **Structured User Context:** The agent's strategy is initialized using specific fields (Module, Symptom, Issue Type) rather than a single sentence, allowing for more precise planning.

## 6. User Interaction
The system runs as an interactive Python CLI.

### 1. Terminal Startup Questions
```text
[?] Target OS Environment [Windows]: Windows
[?] Target VS Code Module (e.g. Agent, Terminal, User Interface, MCP, etc) [Agent]: Terminal
[?] Describe your interests / symptoms (any, performance-specific or memory-specific) [performance-specific]: performance-specific
[?] Target Issue Type (e.g. New Feature, Bug, etc) [Bug]: Bug
[?] Time range in days (e.g., 1, 7, 30) [7]: 7
```

### 2. Real-Time Execution Logging (Example for Simple Request)
```text
📋 [PLANNER NODE] Generating Input-Specific Execution Tasks
  📌 Task 1: Verify if the issue belongs to the 'Terminal' module and is a 'Bug'
  📌 Task 2: Check for 'performance-specific' symptoms specifically on Windows
  📌 Task 3: Deduplicate matching issues

⚙️ [EXECUTION NODE] Executing Planned Tasks on 25 Issues
  [ReAct Loop 1/25: Issue #327104] "Terminal input lag on Win11"
  ├─► [Executing Task 1] ✅ PASSED Task 1
  ├─► [Executing Task 2] ✅ PASSED Task 2
  ├─► [Executing Task 3] ✅ PASSED Task 3 (Verified Unique)
  └─► [Action] 📥 Added Issue #327104 to Candidate Pool
```

## 7. Consideration of Using Model Context Protocol (MCP)
The architecture is designed to be **MCP-Ready**. While currently integrated, the components are logically decoupled:
*   **MCP Host:** The LangGraph workflow controller.
*   **GitHub MCP Server:** Wraps API logic for standardized issue retrieval.
*   **Memory MCP Server:** Future integration for ChromaDB vector storage to handle long-term semantic deduplication across sessions.

This decoupling allows the core reasoning agent to be hosted in different environments (CLI, Claude Desktop, Cursor) while reusing the same toolsets.

**Current Status:** The agent is logically decoupled to be MCP-Ready, but currently uses integrated Python tools for GitHub interaction and state management. Transitioning to a formal MCP Client/Server architecture is slated for a future module.

## 8. Core Design Reasoning
To solve the problem of information overload in open-source tracking, this agent relies on four specific architectural pillars:

*   **Reasoning Loop (ReAct & Reflection):** The agent implements a ReAct-based loop to evaluate individual issues. Reasoning steps guide actions by checking specific criteria (e.g., "Does this match the target OS?"). Observations (the LLM's 'YES' or 'NO' response) influence subsequent decisions: a 'NO' observation on a relevance check immediately triggers an action to discard the issue, while a 'YES' observation leads to a deduplication check. Finally, the **Reflection Loop** audits the final candidate pool, deciding whether to re-evaluate the raw data based on observed pool quality.
*   **Memory Requirements (Short and Long Term):** 
    *   **Short-Term Memory:** Managed via `AgentState` to maintain the candidate pool and execution logs during a single session. This is needed because the final synthesis node must process information gathered across multiple independent issue-evaluation loops.
    *   **Long-Term Memory:** Required for **Cross-Session Deduplication**. Without long-term memory (e.g., vector store), the agent would repeatedly alert the user about the same persistent issue every time it runs.
*   **Tool Grounding:** The use of an external **GitHub REST API tool** is necessary to resolve the limitation of "LLM Knowledge Cutoff." By fetching live issues, the agent grounds its reasoning in real-time repository state, ensuring it never summarizes stale or resolved data.
*   **Comparison to Prompt-Only Approach:** A prompt-only approach fails when faced with **Context Bloat** (e.g., trying to read 100 GitHub issues in one context window). This leads to hallucinations and missed details. Our design resolves this failure mode by isolating each issue into its own reasoning step, ensuring high-fidelity analysis that scales regardless of the number of issues fetched.
