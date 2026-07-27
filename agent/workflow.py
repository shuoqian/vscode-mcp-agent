import os
from typing import Any
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.tools import fetch_github_issues_tool
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Initialize Groq LLM (High rate limit, ultra-fast inference)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0
)

def extract_text(content: Any) -> str:
    """Safely extracts plain string text from LLM content."""
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

async def fetch_issues_node(state: AgentState):
    """Initial Step: Fetch ALL open issues within the user-specified time range."""
    days = state["time_range_days"]
    
    log1 = f"[Fetch] Retrieving ALL open issues created in microsoft/vscode over the past {days} day(s)..."
    print(log1)
    
    raw_issues = await fetch_github_issues_tool(days)
    count = len(raw_issues)
    
    log2 = f"[Fetch Result] Successfully fetched total count of {count} issue(s)."
    print(log2)
    
    logs = [log1, log2]
    
    if count > 0:
        first_issue = f"#{raw_issues[0]['number']}"
        last_issue = f"#{raw_issues[-1]['number']}"
        log3 = f"[Fetch Metadata] First fetched issue (newest): {first_issue} | Last fetched issue (oldest): {last_issue}\n"
        print(log3)
        logs.append(log3)
    else:
        log3 = "[Fetch Metadata] No issues were created in this time window.\n"
        print(log3)
        logs.append(log3)
    
    return {"raw_issues": raw_issues, "logs": logs}

async def process_issues_react_node(state: AgentState):
    """
    Optimized Per-Issue ReAct Loop (Powered by Groq):
    - Step 1: Direct Relevance Check with issue title logging
    - Step 2: Deduplicate (Checks against existing pooled issues)
    - Step 3: Action (Adds to pool if relevant & unique)
    """
    raw_issues = state.get("raw_issues", [])
    user_interest = state.get("user_interest", "")
    
    pooled_issues = []
    logs = []

    match_prompt = PromptTemplate.from_template(
        "User Interest: {user_interest}\n\n"
        "GitHub Issue Title: {title}\n"
        "GitHub Issue Body:\n{body}\n\n"
        "Does this issue directly relate to the user's stated interest? Reply strictly with 'YES' or 'NO'."
    )

    dup_prompt = PromptTemplate.from_template(
        "Existing Pooled Issues:\n{pool_titles}\n\n"
        "New Issue Title: {new_title}\n"
        "New Issue Body:\n{new_body}\n\n"
        "Is the new issue a duplicate or covering the exact same problem as any existing issue in the pool? Reply strictly with 'YES' or 'NO'."
    )

    total_issues = len(raw_issues)

    for idx, issue in enumerate(raw_issues, 1):
        issue_id = f"#{issue['number']}"
        issue_title = issue['title']
        
        header_log = f"--- [ReAct Loop Item {idx}/{total_issues}: Issue {issue_id}] ---"
        logs.append(header_log)
        print(header_log)

        # Step 1: Direct Relevance Match with Issue Title Logging
        step1_log = f"[ReAct: Step 1 - Plan {issue_id}] Checking relevance for issue titled: \"{issue_title}\"..."
        logs.append(step1_log)
        print(step1_log)

        match_res = await (match_prompt | llm).ainvoke({
            "user_interest": user_interest,
            "title": issue_title,
            "body": issue["body"][:1500]
        })
        match_text = extract_text(match_res.content)
        is_match = "YES" in match_text.upper()

        if not is_match:
            res_log = f"[ReAct: Step 1 - Result {issue_id}] [DISCARDED] Not relevant.\n"
            logs.append(res_log)
            print(res_log)
            continue
        else:
            res_log = f"[ReAct: Step 1 - Result {issue_id}] [MATCHED] Issue is relevant!"
            logs.append(res_log)
            print(res_log)

        # Step 2: Deduplicate against Pooled Issues
        if pooled_issues:
            step2_log = f"[ReAct: Step 2 - Deduplicate {issue_id}] Checking duplicate status against pool of {len(pooled_issues)} issue(s)..."
            logs.append(step2_log)
            print(step2_log)

            pool_titles_text = "\n".join([f"- #{item['number']}: {item['title']}" for item in pooled_issues])
            dup_res = await (dup_prompt | llm).ainvoke({
                "pool_titles": pool_titles_text,
                "new_title": issue_title,
                "new_body": issue["body"][:1500]
            })
            dup_text = extract_text(dup_res.content)
            is_dup = "YES" in dup_text.upper()
        else:
            is_dup = False

        if is_dup:
            dup_result_log = f"[ReAct: Step 2 - Result {issue_id}] [DISCARDED] Duplicate issue.\n"
            logs.append(dup_result_log)
            print(dup_result_log)
        else:
            uniq_result_log = f"[ReAct: Step 2 - Result {issue_id}] [UNIQUE] Verified unique."
            logs.append(uniq_result_log)
            print(uniq_result_log)
            
            # Step 3: Action (Pool insertion)
            action_log = f"[ReAct: Step 3 - Action {issue_id}] Added to final pool.\n"
            logs.append(action_log)
            print(action_log)

            pooled_issues.append({
                "number": issue["number"],
                "title": issue_title,
                "url": issue["html_url"],
                "body": issue["body"][:1500]
            })

    return {"pooled_issues": pooled_issues, "logs": logs}

async def final_summary_node(state: AgentState):
    """Synthesize final executive summary across all pooled unique issues using Groq."""
    pooled = state.get("pooled_issues", [])
    
    if not pooled:
        return {
            "aggregate_summary": "No relevant open issues matching your specific interest sentence were found in the specified timeframe.",
            "logs": ["\n[Synthesize] Output pool is empty. Final report generated."]
        }

    combined_text = "\n".join([
        f"• Issue #{i['number']} ({i['title']}) - URL: {i['url']}\n  Content: {i['body'][:300]}..."
        for i in pooled
    ])

    final_prompt = PromptTemplate.from_template(
        "The following are unique relevant GitHub issues found for the user's interest: '{user_interest}'.\n\n"
        "{combined_text}\n\n"
        "Provide a clean, bulleted executive summary highlighting common patterns and key findings."
    )

    final_res = await (final_prompt | llm).ainvoke({
        "user_interest": state["user_interest"],
        "combined_text": combined_text
    })

    return {
        "aggregate_summary": extract_text(final_res.content),
        "logs": [f"\n[Synthesize] Successfully synthesized final executive summary from {len(pooled)} pooled issue(s)."]
    }

workflow = StateGraph(AgentState)
workflow.add_node("fetch_issues", fetch_issues_node)
workflow.add_node("process_issues_react", process_issues_react_node)
workflow.add_node("final_summary", final_summary_node)

workflow.set_entry_point("fetch_issues")
workflow.add_edge("fetch_issues", "process_issues_react")
workflow.add_edge("process_issues_react", "final_summary")
workflow.add_edge("final_summary", END)

app_agent = workflow.compile()