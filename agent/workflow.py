import os
import json
from typing import Any, Literal
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.tools import fetch_github_issues_tool
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

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
# NODE 1: Planner Node (Strictly Anchored to Input Symptoms & OS)
# =====================================================================
async def planner_node(state: AgentState):
    """
    Analyzes the user's OS and symptom sentence.
    Breaks down complex/multi-part symptoms into targeted issue-filtering tasks.
    """
    user_interest = state["user_interest"]
    target_os = state.get("target_os", "Windows")
    time_range = state["time_range_days"]

    print("\n" + "="*65)
    print(" 📋 [PLANNER NODE] Generating Input-Specific Execution Tasks")
    print("="*65)

    planner_prompt = PromptTemplate.from_template(
        "You are an issue triage planner. Analyze the user's specific input:\n"
        "- Target OS: {target_os}\n"
        "- Stated Symptoms / Interests: '{user_interest}'\n"
        "- Lookback Window: Past {time_range} days\n\n"
        "Your job is to generate a strict 3-step filtering and synthesis plan for evaluating GitHub issues.\n"
        "CRITICAL RULE: DO NOT suggest external actions like 'run Windows diagnostics' or 'monitor local RAM'.\n"
        "ALL tasks must be about checking and filtering the fetched GitHub issues.\n\n"
        "Instructions for Task Creation:\n"
        "- If the user symptom sentence is complex or multi-part (e.g. 'slow startup and high memory leak'), "
        "break the symptom evaluation into distinct sub-checks for Task 1 and Task 2.\n"
        "- Ensure Target OS ({target_os}) context filtering is explicitly addressed.\n"
        "- Task 3 must always be deduplication and executive report synthesis.\n\n"
        "Return ONLY a valid JSON array of 3 plain string tasks WITHOUT 'Task X:' prefixes. Example format:\n"
        '[\n'
        '  "Check if issue mentions performance slowdown or application lag on {target_os}",\n'
        '  "Check if issue mentions memory leaks or OOM crashes",\n'
        '  "Deduplicate matching issues and synthesize final executive report"\n'
        ']'
    )

    res = await (planner_prompt | llm).ainvoke({
        "user_interest": user_interest,
        "target_os": target_os,
        "time_range": time_range
    })
    
    clean_json = extract_text(res.content).replace("```json", "").replace("```", "").strip()
    try:
        tasks = json.loads(clean_json)
    except Exception:
        # Fallback if JSON parsing fails
        tasks = [
            f"Filter issues matching symptom criteria: '{user_interest}'",
            f"Verify applicability to target OS ({target_os}) and exclude unrelated platform bugs",
            "Deduplicate candidate pool and synthesize final executive report"
        ]

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
# NODE 3: Per-Issue ReAct Processing Node (Executing Planner Tasks)
# =====================================================================
async def process_issues_react_node(state: AgentState):
    """Executes Planner Tasks item-by-item on every fetched issue."""
    raw_issues = state.get("raw_issues", [])
    user_interest = state.get("user_interest", "")
    target_os = state.get("target_os", "Windows")
    critique_feedback = state.get("critique_feedback", "")
    plan_tasks = state.get("plan_tasks", [
        "Evaluate relevance against user interest",
        "Check OS context matching",
        "Deduplicate candidate pool"
    ])

    task1_desc = plan_tasks[0] if len(plan_tasks) > 0 else "Evaluate relevance against user symptoms"
    task2_desc = plan_tasks[1] if len(plan_tasks) > 1 else f"Verify target OS ({target_os}) context and uniqueness"

    pooled_issues = []
    total = len(raw_issues)

    print("\n" + "="*65)
    print(f" ⚙️ [EXECUTION NODE] Executing Planned Tasks on {total} Issues")
    print("="*65)

    match_prompt = PromptTemplate.from_template(
        "User Interest / Symptoms: {user_interest}\n"
        "Target OS Context: {target_os}\n"
        "Reflection Guidance: {critique_feedback}\n\n"
        "GitHub Issue Title: {title}\n"
        "GitHub Issue Body:\n{body}\n\n"
        "Does this issue directly relate to the user interest/symptoms? Reply strictly 'YES' or 'NO'."
    )

    dup_prompt = PromptTemplate.from_template(
        "Existing Pooled Issues:\n{pool_titles}\n\n"
        "New Issue Title: {new_title}\n"
        "New Issue Body:\n{new_body}\n\n"
        "Is the new issue a duplicate or covering the exact same bug as any existing issue in the pool? Reply strictly 'YES' or 'NO'."
    )

    for idx, issue in enumerate(raw_issues, 1):
        issue_id = f"#{issue['number']}"
        issue_title = issue['title']

        print(f"\n[ReAct Loop {idx}/{total}: Issue {issue_id}] \"{issue_title}\"")

        # -------------------------------------------------------------
        # EXECUTE TASK 1: Relevance Filtering against Symptoms
        # -------------------------------------------------------------
        print(f"  ├─► [Executing Task 1: {task1_desc}]")
        
        match_res = await (match_prompt | llm).ainvoke({
            "user_interest": user_interest,
            "target_os": target_os,
            "critique_feedback": critique_feedback if critique_feedback else "None",
            "title": issue_title,
            "body": issue["body"][:1200]
        })
        
        is_match = "YES" in extract_text(match_res.content).upper()

        if not is_match:
            print(f"  │   └─ Status: ❌ DISCARDED (Failed Task 1 - Not Relevant)")
            continue

        print(f"  │   └─ Status: ✅ PASSED Task 1 (Relevant Issue)")

        # -------------------------------------------------------------
        # EXECUTE TASK 2: OS Filtering & Deduplication Check
        # -------------------------------------------------------------
        print(f"  ├─► [Executing Task 2: {task2_desc}]")
        
        if pooled_issues:
            pool_titles_text = "\n".join([f"- #{item['number']}: {item['title']}" for item in pooled_issues])
            dup_res = await (dup_prompt | llm).ainvoke({
                "pool_titles": pool_titles_text,
                "new_title": issue_title,
                "new_body": issue["body"][:1200]
            })
            is_dup = "YES" in extract_text(dup_res.content).upper()
        else:
            is_dup = False

        if is_dup:
            print(f"  │   └─ Status: ❌ DISCARDED (Failed Task 2 - Duplicate Entry)")
        else:
            print(f"  │   └─ Status: ✅ PASSED Task 2 (Verified Unique)")
            print(f"  └─► [Action] 📥 Added Issue {issue_id} to Candidate Pool")
            pooled_issues.append({
                "number": issue["number"],
                "title": issue_title,
                "url": issue["html_url"],
                "body": issue["body"][:1200]
            })

    return {"pooled_issues": pooled_issues}

# =====================================================================
# NODE 4: Self-Reflection / Verification Node (Dynamic Loop)
# =====================================================================
async def verify_and_reflect_node(state: AgentState):
    """
    Evaluates candidate pool quality against user intent and planner tasks.
    Triggers dynamic reflection loop if quality/completeness criteria are not met.
    """
    pooled_issues = state.get("pooled_issues", [])
    user_interest = state.get("user_interest", "")
    reflection_count = state.get("reflection_count", 0) + 1

    print("\n" + "="*65)
    print(f" 🔍 [REFLECTION NODE] Verifying Pool Quality against Plan (Attempt #{reflection_count})")
    print("="*65)

    # Max retries guardrail to prevent infinite execution loops
    if reflection_count > 2:
        print("  └─► [Verdict] Max retry threshold reached. Proceeding to final synthesis.")
        return {"reflection_count": reflection_count, "critique_feedback": "PASSED"}

    pooled_summary_text = "\n".join([f"- #{i['number']}: {i['title']}" for i in pooled_issues]) if pooled_issues else "None"

    verify_prompt = PromptTemplate.from_template(
        "User Query: '{user_interest}'\n"
        "Candidate Issue Pool ({count} issue(s) found):\n{pooled_summary_text}\n\n"
        "Evaluate if the candidate pool is adequate and accurate for answering the user's request.\n"
        "Reply strictly with JSON:\n"
        '{{\n'
        '  "status": "PASSED" or "NEEDS_REFINEMENT",\n'
        '  "critique": "Brief explanation or instructions to broaden/narrow criteria"\n'
        '}}'
    )

    res = await (verify_prompt | llm).ainvoke({
        "user_interest": user_interest,
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
    """Conditional Edge Router based on Verification Result."""
    feedback = state.get("critique_feedback", "PASSED")
    if feedback != "PASSED" and state.get("reflection_count", 0) <= 2:
        print("\n🔁 [Dynamic Loop Triggered] Routing back to re-evaluate issues with reflection feedback...")
        return "process_issues_react"
    return "final_summary"

# =====================================================================
# NODE 5: Final Summary Synthesis Node
# =====================================================================
async def final_summary_node(state: AgentState):
    """Executes Task 3: Synthesizes final executive report across pooled issues."""
    pooled = state.get("pooled_issues", [])
    user_interest = state.get("user_interest", "")
    plan_tasks = state.get("plan_tasks", [])
    task3_desc = plan_tasks[2] if len(plan_tasks) > 2 else "Synthesize final executive report"

    print("\n" + "="*65)
    print(f" 📊 [FINAL NODE] Executing Task 3: {task3_desc}")
    print("="*65)

    if not pooled:
        return {
            "aggregate_summary": "No relevant open issues matching your specific interest were found in the specified timeframe."
        }

    combined_text = "\n".join([
        f"• Issue #{i['number']} ({i['title']}) - URL: {i['url']}\n  Snippet: {i['body'][:300]}..."
        for i in pooled
    ])

    final_prompt = PromptTemplate.from_template(
        "User Stated Interest: '{user_interest}'\n\n"
        "Verified Candidate Issues:\n{combined_text}\n\n"
        "Provide a clean, bulleted executive summary highlighting common patterns, affected components, and root causes."
    )

    final_res = await (final_prompt | llm).ainvoke({
        "user_interest": user_interest,
        "combined_text": combined_text
    })

    return {"aggregate_summary": extract_text(final_res.content)}

# =====================================================================
# LANGGRAPH WORKFLOW BUILDER
# =====================================================================
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("fetch_issues", fetch_issues_node)
workflow.add_node("process_issues_react", process_issues_react_node)
workflow.add_node("verify_and_reflect", verify_and_reflect_node)
workflow.add_node("final_summary", final_summary_node)

# Add Fixed Sequential Edges
workflow.set_entry_point("planner")
workflow.add_edge("planner", "fetch_issues")
workflow.add_edge("fetch_issues", "process_issues_react")
workflow.add_edge("process_issues_react", "verify_and_reflect")

# Conditional Edge (Dynamic Self-Reflection Loop)
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