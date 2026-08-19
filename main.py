import asyncio
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from agent.workflow import app_agent

console = Console()

def print_header():
    console.print(
        Panel.fit(
            "[bold cyan]VS Code Intelligent Signal Tracking Agent[/bold cyan]\n"
            "[dim]Enhanced with Dynamic Taxonomy Routing, Quality Gates & Vector Memory[/dim]",
            border_style="cyan"
        )
    )

async def main():
    print_header()

    target_os = Prompt.ask(
        "[?] Target OS Environment",
        default="Windows"
    )

    target_module = Prompt.ask(
        "[?] Target VS Code Module (e.g. Agent, Terminal, User Interface, MCP, etc)",
        default="Terminal"
    )

    symptom_type = Prompt.ask(
        "[?] Describe your interests / symptoms (any, performance-specific or memory-specific)",
        default="performance-specific"
    )

    issue_type = Prompt.ask(
        "[?] Target Issue Type (e.g. New Feature, Bug, etc)",
        default="Bug"
    )

    time_range = Prompt.ask(
        "[?] Time range in days (e.g., 1, 7, 30)",
        default="7"
    )

    try:
        days_int = int(time_range)
    except ValueError:
        days_int = 7

    # Initial Agent State
    initial_state = {
        "target_os": target_os,
        "target_module": target_module,
        "symptom_type": symptom_type,
        "issue_type": issue_type,
        "time_range_days": days_int,
        "constructed_query": "",
        "plan_tasks": [],
        "hard_label_groups": {},
        "text_keywords": [],
        "tot_branch_trace": [],
        "hard_labels": [],
        "soft_labels": [],
        "matched_labels": [],
        "raw_issues": [],
        "vector_filtered_issues": [],
        "pooled_issues": [],
        "pool_confidence_score": 0.0,
        "is_abstained": False,
        "reflection_count": 0,
        "critique_feedback": "",
        "aggregate_summary": "",
        "draft_summary": "",
        "source_evidence_text": "",
        "flagged_claims": [],
        "report_verified": True,
        "logs": []
    }

    console.print("\n[bold yellow]--- Executing Agentic Intelligence Workflow ---[/bold yellow]\n")

    try:
        final_state = await app_agent.ainvoke(initial_state)

        is_abstained = final_state.get("is_abstained", False)
        report_verified = final_state.get("report_verified", True)
        flagged_claims = final_state.get("flagged_claims", [])

        if is_abstained:
            border_style, title_style = "yellow", "bold yellow"
        elif not report_verified:
            border_style, title_style = "red", "bold red"
        else:
            border_style, title_style = "green", "bold green"

        verification_note = ""
        if not is_abstained:
            if not report_verified:
                verification_note = "[bold red]Narrative suppressed by Report Verifier Agent — showing raw verified issues only.[/bold red]\n"
            elif flagged_claims:
                verification_note = f"[yellow]Report Verifier Agent removed/hedged {len(flagged_claims)} unsupported claim(s) before delivery.[/yellow]\n"
            else:
                verification_note = "[dim]Report Verifier Agent: fully grounded, no claims flagged.[/dim]\n"

        summary_panel = Panel(
            f"[bold white]Executive Summary:[/bold white]\n"
            f"{verification_note}"
            f"{final_state.get('aggregate_summary', 'No summary generated.')}",
            title=f"[{title_style}][VS Code Alert] {target_module} Issues ({target_os})[/{title_style}]",
            subtitle=(
                f"[dim]Processed: {len(final_state.get('pooled_issues', []))} issue(s) | "
                f"Confidence: {final_state.get('pool_confidence_score', 0.0):.2f} | "
                f"Memory Store: Active[/dim]"
            ),
            border_style=border_style
        )
        console.print("\n", summary_panel)

    except Exception as e:
        console.print(f"\n[bold red][ERROR] Execution failed:[/bold red] {e}")

if __name__ == "__main__":
    asyncio.run(main())