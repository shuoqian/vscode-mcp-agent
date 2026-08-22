import asyncio
import os
import json
import time
from datetime import datetime, timezone
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from agent.workflow import app_agent
from agent.tools import save_issues_to_vector_memory

console = Console()

METRICS_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_metrics.jsonl")


def print_header():
    console.print(
        Panel.fit(
            "[bold cyan]VS Code Intelligent Signal Tracking Agent[/bold cyan]\n"
            "[dim]Enhanced with Dynamic Taxonomy Routing, Quality Gates & Vector Memory[/dim]",
            border_style="cyan"
        )
    )


def append_metrics_record(record: dict) -> None:
    """
    Section 12.3: persisted metrics log. Appends one JSON line per run so
    metrics (groundedness rate, veto rate, abstention rate, latency, human
    approval rate) can be computed in aggregate across many runs, not just
    read off a single run's console output. Best-effort — a logging failure
    should never crash the agent itself.
    """
    try:
        with open(METRICS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        console.print(f"[dim]⚠ Could not write metrics log: {e}[/dim]")


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
        "tot_selected_branch": "",
        "used_unfiltered_fallback": False,
        "node_latencies": {},
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

    run_started_at = time.perf_counter()
    human_feedback = None  # "approved" | "rejected" | "skipped" | None (abstained — nothing to review)
    escalation_reasons = []

    try:
        final_state = await app_agent.ainvoke(initial_state)
        total_latency_sec = round(time.perf_counter() - run_started_at, 3)

        is_abstained = final_state.get("is_abstained", False)
        report_verified = final_state.get("report_verified", True)
        flagged_claims = final_state.get("flagged_claims", [])
        used_unfiltered_fallback = final_state.get("used_unfiltered_fallback", False)
        pooled_issues = final_state.get("pooled_issues", [])

        # ---------------------------------------------------------------
        # Section 12.4: human-intervention criteria. These are the same
        # conditions named in the design doc — surfaced here as visible
        # warnings BEFORE asking the human for a verdict, so their review
        # isn't a blind rubber-stamp.
        # ---------------------------------------------------------------
        if not report_verified:
            escalation_reasons.append("Report Verifier Agent vetoed the narrative (fallback report shown)")
        if used_unfiltered_fallback:
            escalation_reasons.append("Fetch fell back to a fully unfiltered query — no topical label filter actually matched")

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
                f"[dim]Processed: {len(pooled_issues)} issue(s) | "
                f"Confidence: {final_state.get('pool_confidence_score', 0.0):.2f} | "
                f"Latency: {total_latency_sec}s[/dim]"
            ),
            border_style=border_style
        )
        console.print("\n", summary_panel)

        # ---------------------------------------------------------------
        # Human-in-the-loop review (Section 12.4/12.5). Only meaningful
        # when there's actually a report to judge — abstention already
        # means nothing was delivered, so there's nothing to approve.
        # ---------------------------------------------------------------
        if not is_abstained:
            if escalation_reasons:
                console.print(Panel(
                    "\n".join(f"⚠ {r}" for r in escalation_reasons),
                    title="[bold red]Flagged for Human Review[/bold red]",
                    border_style="red"
                ))

            choice = Prompt.ask(
                "\n[bold]Human review:[/bold] Does this report look accurate and useful?",
                choices=["y", "n", "skip"],
                default="skip"
            )
            human_feedback = {"y": "approved", "n": "rejected", "skip": "skipped"}[choice]

            if human_feedback == "rejected":
                console.print(
                    "[yellow]Feedback noted. This run's issues will NOT be committed to long-term memory, "
                    "so they remain eligible to be re-surfaced (and re-reviewed) on the next run.[/yellow]"
                )
            elif pooled_issues:
                console.print("  └─► [Vector Memory Commit] Persisting newly summarized issues into ChromaDB...")
                save_issues_to_vector_memory(pooled_issues)
        else:
            console.print("[dim]No report was generated (abstained) — nothing to review or commit.[/dim]")

        # ---------------------------------------------------------------
        # Section 12.3: persisted metrics log.
        # ---------------------------------------------------------------
        append_metrics_record({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_os": target_os,
            "target_module": target_module,
            "symptom_type": symptom_type,
            "issue_type": issue_type,
            "time_range_days": days_int,
            "is_abstained": is_abstained,
            "report_verified": report_verified,
            "flagged_claims_count": len(flagged_claims),
            "pooled_issues_count": len(pooled_issues),
            "pool_confidence_score": final_state.get("pool_confidence_score", 0.0),
            "used_unfiltered_fallback": used_unfiltered_fallback,
            "tot_selected_branch": final_state.get("tot_selected_branch", ""),
            "node_latencies_sec": final_state.get("node_latencies", {}),
            "total_latency_sec": total_latency_sec,
            "human_feedback": human_feedback,
            "escalation_reasons": escalation_reasons,
        })

    except Exception as e:
        console.print(f"\n[bold red][ERROR] Execution failed:[/bold red] {e}")


if __name__ == "__main__":
    asyncio.run(main())