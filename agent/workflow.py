import os
import json
from typing import Any, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.tools import fetch_github_issues_tool
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0
)

def extract_text(content: Any) -> str:
    """Safely extracts string text from LLM responses."""
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
# NODE 1: Planner Node
# =====================================================================
async def planner_node(state: AgentState):
    """
    Analyzes user inputs and generates filtering tasks.
    Simple input -> 2 filtering tasks + 1 deduplication task (3 total).
    Complicated input -> 3 filtering tasks + 1 deduplication task (4 total).
    """
    target_os = state["target_os"]
    target_module = state["target_module"]
    symptom_type = state["symptom_type"]
    issue_type = state["issue_type"]

    print("\n" + "="*65)
    print(" 📋 [PLANNER NODE] Generating Input-Specific Execution Tasks")
    print("="*65)

    planner_prompt = PromptTemplate.from_template(
        "You are an issue triage planner for the microsoft/vscode repository. Analyze the following criteria:\n"
        "- Target OS: {target_os}\n"
        "- Target Module: {target_module}\n"
        "- Symptoms: {symptom_type}\n"
        "- Issue Type: {issue_type}\n\n"
        "Your goal is to generate a multi-step verification plan.\n"
        "1. Categorize the request as 'simple' (requires 2 filtering steps) or 'complicated' (requires 3 filtering steps).\n"
        "2. If simple: Return EXACTLY 2 distinct, non-overlapping filtering tasks.\n"
        "3. If complicated: Return EXACTLY 3 distinct, non-overlapping filtering tasks.\n"
        "The tasks must cover all input criteria (OS, Module, Symptoms, Issue Type) across the steps.\n"
        "Example for simple (OS=Windows, Module=Terminal): [\"Verify issue belongs to Terminal and is on Windows\", \"Check for specific symptom match\"]\n"
        "Return ONLY a JSON array of strings. DO NOT include deduplication or summary steps."
    )

    res = await (planner_prompt | llm).ainvoke({
        "target_os": target_os,
        "target_module": target_module,
        "symptom_type": symptom_type,
        "issue_type": issue_type
    })
    
    clean_json = extract_text(res.content).replace("```json", "").replace("```", "").strip()
    try:
        tasks = json.loads(clean_json)
        if not isinstance(tasks, list):
            tasks = [str(tasks)]
        
        # Ensure at least 2 tasks if LLM returns only one
        if len(tasks) < 2:
            tasks = [
                f"Verify issue belongs to {target_module} and is a {issue_type}",
                f"Check for {symptom_type} symptoms specifically on {target_os}"
            ]
    except Exception:
        tasks = [
            f"Verify issue belongs to {target_module} and is a {issue_type}",
            f"Check for {symptom_type} symptoms specifically on {target_os}"
        ]

    # Always append deduplication as the last task
    tasks.append("Deduplicate matching issues")

    for idx, task in enumerate(tasks, 1):
        print(f"  📌 Task {idx}: {task}")

    return {
        "plan_tasks": tasks,
        "reflection_count": 0,
        "critique_feedback": ""
    }

# =====================================================================
# NODE 2: Fetch Issues Node
# =====================================================================
async def fetch_issues_node(state: AgentState):
    """Retrieves open GitHub issues over specified timeframe."""
    days = state["time_range_days"]
    print(f"\n[Fetch] Retrieving open issues created over the past {days} day(s)...")
    
    raw_issues = await fetch_github_issues_tool(days)
    count = len(raw_issues)
    print(f"[Fetch Result] Successfully fetched {count} issue(s).")
    
    return {"raw_issues": raw_issues}

# =====================================================================
# NODE 3: Per-Issue ReAct Processing Node
# =====================================================================
async def process_issues_react_node(state: AgentState):
    """Executes Planner Tasks item-by-item on every fetched issue."""
    raw_issues = state.get("raw_issues", [])
    critique_feedback = state.get("critique_feedback", "")
    plan_tasks = state.get("plan_tasks", ["Filter Task 1", "Filter Task 2", "Deduplicate matching issues"])

    pooled_issues = []
    total = len(raw_issues)

    print("\n" + "="*65)
    print(f" ⚙️ [EXECUTION NODE] Executing Planned Tasks on {total} Issues")
    print("="*65)

    # Generic task execution prompt
    execute_task_prompt = PromptTemplate.from_template(
        "Criteria to verify: {task_description}\n"
        "Reflection Guidance: {critique_feedback}\n\n"
        "GitHub Issue Title: {title}\n"
        "GitHub Issue Body:\n{body}\n\n"
        "Does this issue strictly satisfy the criteria above? Reply strictly 'YES' or 'NO'."
    )

    # Specific deduplication prompt
    dup_prompt = PromptTemplate.from_template(
        "Existing Pooled Issues:\n{pool_titles}\n\n"
        "New Issue Title: {new_title}\n"
        "New Issue Body:\n{new_body}\n\n"
        "Is the new issue a duplicate or covering the exact same bug as any existing issue in the pool? Reply strictly 'YES' or 'NO'."
    )

    for idx, issue in enumerate(raw_issues, 1):
        issue_id = f"#{issue['number']}"
        issue_title = issue['title']
        issue_body = issue["body"][:1200]

        print(f"\n[ReAct Loop {idx}/{total}: Issue {issue_id}] \"{issue_title}\"")

        is_discarded = False
        for t_idx, task_desc in enumerate(plan_tasks, 1):
            is_last = (t_idx == len(plan_tasks))
            print(f"  ├─► [Executing Task {t_idx}: {task_desc}]")

            if not is_last:
                # Filtering Task
                res = await (execute_task_prompt | llm).ainvoke({
                    "task_description": task_desc,
                    "critique_feedback": critique_feedback if critique_feedback else "None",
                    "title": issue_title,
                    "body": issue_body
                })
                if "YES" not in extract_text(res.content).upper():
                    print(f"  │   └─ Status: ❌ DISCARDED (Failed Task {t_idx})")
                    is_discarded = True
                    break
                print(f"  │   └─ Status: ✅ PASSED Task {t_idx}")
            else:
                # Deduplication Task
                is_dup = False
                if pooled_issues:
                    pool_titles_text = "\n".join([f"- #{item['number']}: {item['title']}" for item in pooled_issues])
                    dup_res = await (dup_prompt | llm).ainvoke({
                        "pool_titles": pool_titles_text,
                        "new_title": issue_title,
                        "new_body": issue_body
                    })
                    is_dup = "YES" in extract_text(dup_res.content).upper()

                if is_dup:
                    print(f"  │   └─ Status: ❌ DISCARDED (Failed Task {t_idx} - Duplicate)")
                    is_discarded = True
                else:
                    print(f"  │   └─ Status: ✅ PASSED Task {t_idx} (Verified Unique)")
                    print(f"  └─► [Action] 📥 Added Issue {issue_id} to Candidate Pool")
                    pooled_issues.append({
                        "number": issue["number"],
                        "title": issue_title,
                        "url": issue["html_url"],
                        "body": issue_body
                    })

        if is_discarded:
            continue

    return {"pooled_issues": pooled_issues}

# =====================================================================
# NODE 4: Self-Reflection / Verification Node
# =====================================================================
async def verify_and_reflect_node(state: AgentState):
    """Evaluates candidate pool quality."""
    pooled_issues = state.get("pooled_issues", [])
    reflection_count = state.get("reflection_count", 0) + 1

    print("\n" + "="*65)
    print(f" 🔍 [REFLECTION NODE] Verifying Pool Quality against Plan (Attempt #{reflection_count})")
    print("="*65)

    if reflection_count > 2:
        print("  └─► [Verdict] Max retry threshold reached. Proceeding to final synthesis.")
        return {"reflection_count": reflection_count, "critique_feedback": "PASSED"}

    pooled_summary_text = "\n".join([f"- #{i['number']}: {i['title']}" for i in pooled_issues]) if pooled_issues else "None"

    verify_prompt = PromptTemplate.from_template(
        "User Criteria: OS={target_os}, Module={target_module}, Symptoms={symptom_type}, Type={issue_type}\n"
        "Candidate Issue Pool ({count} issue(s) found):\n{pooled_summary_text}\n\n"
        "Evaluate if the candidate pool is adequate and accurate for answering the user's request.\n"
        "Reply strictly with JSON:\n"
        '{{\n'
        '  "status": "PASSED" or "NEEDS_REFINEMENT",\n'
        '  "critique": "Brief explanation or instructions to broaden/narrow criteria"\n'
        '}}'
    )

    res = await (verify_prompt | llm).ainvoke({
        "target_os": state["target_os"],
        "target_module": state["target_module"],
        "symptom_type": state["symptom_type"],
        "issue_type": state["issue_type"],
        "count": len(pooled_issues),
        "pooled_summary_text": pooled_summary_text
    })

    clean_res = extract_text(res.content).replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean_res)
        status = data.get("status", "PASSED")
        critique = data.get("critique", "")
    except Exception:
        status = "PASSED"
        critique = "Default pass."

    print(f"  └─► [Verdict] Status: {status} | Critique: {critique}")

    return {
        "reflection_count": reflection_count,
        "critique_feedback": critique if status == "NEEDS_REFINEMENT" else "PASSED"
    }

def route_after_verification(state: AgentState) -> Literal["process_issues_react", "final_summary"]:
    feedback = state.get("critique_feedback", "PASSED")
    if feedback != "PASSED" and state.get("reflection_count", 0) <= 2:
        print("\n🔁 [Dynamic Loop Triggered] Routing back to re-evaluate issues...")
        return "process_issues_react"
    return "final_summary"

# =====================================================================
# NODE 5: Final Summary Synthesis Node
# =====================================================================
async def final_summary_node(state: AgentState):
    """Synthesizes final executive report."""
    pooled = state.get("pooled_issues", [])

    print("\n" + "="*65)
    print(f" 📊 [FINAL NODE] Executing Synthesis")
    print("="*65)

    if not pooled:
        return {
            "aggregate_summary": "No relevant open issues matching your specific interest were found."
        }

    combined_text = "\n".join([
        f"• Issue #{i['number']} ({i['title']}) - URL: {i['url']}\n  Snippet: {i['body'][:300]}..."
        for i in pooled
    ])

    final_prompt = PromptTemplate.from_template(
        "User Interest: {symptom_type} in {target_module} ({target_os})\n\n"
        "Verified Candidate Issues:\n{combined_text}\n\n"
        "Provide a clean, bulleted executive summary highlighting common patterns, affected components, and root causes."
    )

    final_res = await (final_prompt | llm).ainvoke({
        "symptom_type": state["symptom_type"],
        "target_module": state["target_module"],
        "target_os": state["target_os"],
        "combined_text": combined_text
    })

    return {"aggregate_summary": extract_text(final_res.content)}

# =====================================================================
# LANGGRAPH WORKFLOW BUILDER
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
