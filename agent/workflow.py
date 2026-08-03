import os
import json
from typing import Any, Literal
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.tools import (
    fetch_github_issues_tool,
    check_vector_memory_duplicate,
    save_issues_to_vector_memory
)
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0
)

def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
            else:
                parts.append(str(part))
        return "".join(parts).strip()
    return str(content).strip()

# =====================================================================
# NODE 1: Dynamic Router & Planner Node (Point 1)
# =====================================================================
async def planner_node(state: AgentState):
    target_os = state["target_os"]
    target_module = state["target_module"]
    symptom_type = state["symptom_type"]
    issue_type = state["issue_type"]
    days = state["time_range_days"]

    # Calculate exact start date in Python
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    router_prompt = PromptTemplate.from_template(
        "You are an issue triage planner for microsoft/vscode.\n"
        "Translate user criteria into a structured JSON payload for GitHub Search API.\n"
        "User Criteria:\n"
        "- Module: {target_module}\n"
        "- OS: {target_os}\n"
        "- Symptoms: {symptom_type}\n"
        "- Issue Type: {issue_type}\n"
        "- Cutoff Date (created on or after): {cutoff_date}\n\n"
        "Instructions:\n"
        "1. In 'constructed_query', include `created:>={cutoff_date}` to enforce the time lookback.\n"
        "2. Formulate a single string 'constructed_query' for GitHub search API.\n"
        "3. Provide an array of 2 distinct verification tasks in 'plan_tasks'.\n\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{{\n"
        '  "constructed_query": "repo:microsoft/vscode is:issue state:open created:>={cutoff_date} ...",\n'
        '  "plan_tasks": ["Task 1 description", "Task 2 description"]\n'
        "}}"
    )

    res = await (router_prompt | llm).ainvoke({
        "target_module": target_module,
        "target_os": target_os,
        "symptom_type": symptom_type,
        "issue_type": issue_type,
        "cutoff_date": cutoff_date
    })

    clean_json = extract_text(res.content).replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean_json)
        constructed_query = parsed.get("constructed_query", "")
        plan_tasks = parsed.get("plan_tasks", [])
    except Exception:
        constructed_query = f"repo:microsoft/vscode is:issue state:open \"{target_module}\" \"{symptom_type}\""
        plan_tasks = [
            f"Verify issue belongs to {target_module} and is a {issue_type}",
            f"Check for {symptom_type} symptoms specifically on {target_os}"
        ]

    # Ensure required prefix exists
    if "repo:microsoft/vscode" not in constructed_query:
        constructed_query = f"repo:microsoft/vscode is:issue state:open {constructed_query}"

    plan_tasks.append("Deduplicate matching issues")

    print(f"  ├─► [Taxonomy Translation Query]: {constructed_query}")
    for idx, task in enumerate(plan_tasks, 1):
        print(f"  📌 Task {idx}: {task}")

    return {
        "constructed_query": constructed_query,
        "plan_tasks": plan_tasks,
        "reflection_count": 0,
        "critique_feedback": "",
        "is_abstained": False
    }

# =====================================================================
# NODE 2: Fetch & Cross-Session Vector Memory Node (Point 4)
# =====================================================================
async def fetch_issues_node(state: AgentState):
    """Retrieves open issues and filters out past alerted issues using ChromaDB."""
    query = state["constructed_query"]
    days = state["time_range_days"]
    print(f"\n[Fetch Node] Retrieving issues with query: {query}")

    raw_issues = await fetch_github_issues_tool(query, days)
    print(f"[Fetch Result] Successfully fetched {len(raw_issues)} raw issue(s).")

    # Point 4: Apply ChromaDB Vector Deduplication
    vector_filtered_issues = []
    filtered_out_count = 0

    print("  ├─► [ChromaDB Memory Check] Screening issues against long-term alert history...")
    for issue in raw_issues:
        is_past_duplicate = check_vector_memory_duplicate(
            issue_number=issue["number"],
            issue_title=issue["title"],
            issue_body=issue["body"]
        )
        if is_past_duplicate:
            filtered_out_count += 1
            print(f"  │   └── 🛑 Issue #{issue['number']} filtered out (Previously alerted in past session)")
        else:
            vector_filtered_issues.append(issue)

    print(f"  └─► [Memory Screen Result] {len(vector_filtered_issues)} issue(s) remaining ({filtered_out_count} duplicate(s) suppressed).")

    return {
        "raw_issues": raw_issues,
        "vector_filtered_issues": vector_filtered_issues
    }

# =====================================================================
# NODE 3: Per-Issue ReAct Processing & Confidence Scoring Node
# =====================================================================
async def process_issues_react_node(state: AgentState):
    """Executes verification tasks and computes alignment confidence scores."""
    issues_to_process = state.get("vector_filtered_issues", [])
    plan_tasks = state.get("plan_tasks", [])
    critique_feedback = state.get("critique_feedback", "")

    pooled_issues = []
    total = len(issues_to_process)

    print("\n" + "="*65)
    print(f" ⚙️ [EXECUTION NODE] ReAct Processing on {total} Issue(s)")
    print("="*65)

    eval_prompt = PromptTemplate.from_template(
        "Criteria to verify: {task_description}\n"
        "Critique Feedback: {critique_feedback}\n\n"
        "GitHub Issue Title: {title}\n"
        "GitHub Issue Body:\n{body}\n\n"
        "Evaluate alignment with criteria. Respond ONLY with JSON:\n"
        "{{\n"
        '  "passed": true/false,\n'
        '  "confidence": float between 0.0 and 1.0\n'
        "}}"
    )

    for idx, issue in enumerate(issues_to_process, 1):
        issue_id = f"#{issue['number']}"
        issue_title = issue['title']
        issue_body = issue["body"][:1200]

        print(f"\n[ReAct Loop {idx}/{total}: Issue {issue_id}] \"{issue_title}\"")

        issue_scores = []
        is_discarded = False

        for t_idx, task_desc in enumerate(plan_tasks[:-1], 1):
            print(f"  ├─► [Executing Task {t_idx}: {task_desc}]")
            res = await (eval_prompt | llm).ainvoke({
                "task_description": task_desc,
                "critique_feedback": critique_feedback or "None",
                "title": issue_title,
                "body": issue_body
            })

            clean = extract_text(res.content).replace("```json", "").replace("```", "").strip()
            try:
                eval_data = json.loads(clean)
                passed = eval_data.get("passed", False)
                conf = float(eval_data.get("confidence", 0.5))
            except Exception:
                passed = True
                conf = 0.50

            if not passed:
                print(f"  │   └─ Status: ❌ DISCARDED (Failed Task {t_idx})")
                is_discarded = True
                break

            issue_scores.append(conf)
            print(f"  │   └─ Status: ✅ PASSED (Task Confidence: {conf:.2f})")

        if not is_discarded:
            avg_issue_conf = sum(issue_scores) / len(issue_scores) if issue_scores else 0.80
            print(f"  └─► [Action] 📥 Added Issue {issue_id} (Confidence: {avg_issue_conf:.2f})")
            pooled_issues.append({
                "number": issue["number"],
                "title": issue_title,
                "url": issue["html_url"],
                "body": issue_body,
                "confidence": avg_issue_conf
            })

    # Point 3: Compute aggregate pool confidence score
    if pooled_issues:
        avg_pool_conf = sum(i["confidence"] for i in pooled_issues) / len(pooled_issues)
    else:
        avg_pool_conf = 0.0

    return {
        "pooled_issues": pooled_issues,
        "pool_confidence_score": avg_pool_conf
    }

# =====================================================================
# NODE 4: Verify, Reflect & Abstention Gate Node (Point 3)
# =====================================================================
async def verify_and_reflect_node(state: AgentState):
    """Audits pool quality against the 0.80 confidence threshold for abstention."""
    pooled_issues = state.get("pooled_issues", [])
    pool_confidence = state.get("pool_confidence_score", 0.0)
    reflection_count = state.get("reflection_count", 0) + 1

    CONFIDENCE_THRESHOLD = 0.80

    print("\n" + "="*65)
    print(f" 🔍 [REFLECTION & QUALITY GATE] Pool Confidence Audit (Attempt #{reflection_count})")
    print("="*65)
    print(f"  ├─► Candidate Issues in Pool: {len(pooled_issues)}")
    print(f"  ├─► Calculated Pool Confidence: {pool_confidence:.2f} (Threshold: {CONFIDENCE_THRESHOLD})")

    # Point 3: Abstention Rule Check
    if not pooled_issues or pool_confidence < CONFIDENCE_THRESHOLD:
        print(f"  └─► [QUALITY GATE FAILED] Confidence score ({pool_confidence:.2f}) below threshold {CONFIDENCE_THRESHOLD}.")

        if reflection_count <= 2:
            critique = "Pool confidence is too low or empty. Broaden keywords and retry."
            print(f"  └─► Triggering Reflection Retry #{reflection_count}...")
            return {
                "reflection_count": reflection_count,
                "critique_feedback": critique,
                "is_abstained": False
            }
        else:
            print("  └─► [ABSTENTION PROTOCOL TRIGGERED] Maximum retries exceeded. Abstaining from report generation.")
            return {
                "reflection_count": reflection_count,
                "critique_feedback": "ABSTAIN",
                "is_abstained": True
            }

    print("  └─► [QUALITY GATE PASSED] High confidence verified. Proceeding to report synthesis.")
    return {
        "reflection_count": reflection_count,
        "critique_feedback": "PASSED",
        "is_abstained": False
    }

def route_after_verification(state: AgentState) -> Literal["process_issues_react", "final_summary"]:
    if state.get("is_abstained", False):
        return "final_summary"
    feedback = state.get("critique_feedback", "PASSED")
    if feedback != "PASSED" and state.get("reflection_count", 0) <= 2:
        return "process_issues_react"
    return "final_summary"

# =====================================================================
# NODE 5: Final Summary Synthesis & Memory Commit Node (Point 4)
# =====================================================================
async def final_summary_node(state: AgentState):
    """Synthesizes executive summary and commits new alerts to ChromaDB."""
    pooled = state.get("pooled_issues", [])
    is_abstained = state.get("is_abstained", False)

    print("\n" + "="*65)
    print(f" 📊 [FINAL NODE] Generating Report & Updating Memory")
    print("="*65)

    if is_abstained or not pooled:
        summary_text = (
            "⚠️ [ABSTENTION NOTICE] The agent abstained from generating a summary report.\n"
            "Reason: Matching issues yielded insufficient confidence (< 0.80 threshold).\n"
            "Recommendation: Please consider broadening your target module, symptom keywords, or time range."
        )
        return {"aggregate_summary": summary_text}

    combined_text = "\n".join([
        f"• Issue #{i['number']} ({i['title']}) [Confidence: {i['confidence']:.2f}]\n  URL: {i['url']}\n  Snippet: {i['body'][:300]}..."
        for i in pooled
    ])

    final_prompt = PromptTemplate.from_template(
        "User Interest: {symptom_type} in {target_module} ({target_os})\n\n"
        "Verified Candidate Issues:\n{combined_text}\n\n"
        "Provide a clean executive report highlighting common patterns, affected components, and root causes."
    )

    final_res = await (final_prompt | llm).ainvoke({
        "symptom_type": state["symptom_type"],
        "target_module": state["target_module"],
        "target_os": state["target_os"],
        "combined_text": combined_text
    })

    # Point 4: Commit verified issues into ChromaDB memory
    print("  └─► [Vector Memory Commit] Persisting newly summarized issues into ChromaDB...")
    save_issues_to_vector_memory(pooled)

    return {"aggregate_summary": extract_text(final_res.content)}

# =====================================================================
# WORKFLOW BUILDER
# =====================================================================
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("fetch_issues", fetch_issues_node)
workflow.add_node("process_issues_react", process_issues_react_node)
workflow.add_node("verify_and_reflect", verify_and_reflect_node)
workflow.add_node("final_summary", final_summary_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "fetch_issues")
workflow.add_edge("fetch_issues", "process_issues_react")
workflow.add_edge("process_issues_react", "verify_and_reflect")

workflow.add_conditional_edges(
    "verify_and_reflect",
    route_after_verification,
    {
        "process_issues_react": "process_issues_react",
        "final_summary": "final_summary"
    }
)

workflow.add_edge("final_summary", END)

app_agent = workflow.compile()