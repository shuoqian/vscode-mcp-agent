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
            "[bold cyan]VS Code Intelligent Signal Tracking Agent (Gemini Free Mode)[/bold cyan]",
            border_style="cyan"
        )
    )

async def main():
    print_header()
    
    # User inputs
    os_env = Prompt.ask(
        "[?] Target OS Environment", 
        default="Windows"
    )
    
    # Updated default sentence here:
    user_interest = Prompt.ask(
        "[?] Describe your interests / symptoms in a sentence", 
        default="performance issue or memory issue"
    )
    
    time_range = Prompt.ask(
        "[?] Time range in days (e.g., 1, 7, 30)", 
        default="30"
    )

    try:
        days_int = int(time_range)
    except ValueError:
        days_int = 30

    # Initial Agent State
    initial_state = {
        "user_interest": user_interest,
        "time_range_days": days_int,
        "target_os": os_env,
        "raw_issues": [],
        "pooled_issues": [],
        "aggregate_summary": "",
        "logs": []
    }

    console.print("\n[bold yellow]--- Executing Per-Issue ReAct Agent Loop ---[/bold yellow]\n")

    try:
        final_state = await app_agent.ainvoke(initial_state)

        # Print Final Consolidated Alert Box
        summary_panel = Panel(
            f"[bold white]Consolidated Executive Summary:[/bold white]\n"
            f"{final_state.get('aggregate_summary', 'No summary generated.')}",
            title=f"[bold green][VS Code Alert] Live Issues Summary (Past {days_int} Days)[/bold green]",
            subtitle=f"[dim]Data Source: Live GitHub REST API | Processed: {len(final_state.get('pooled_issues', []))} issue(s)[/dim]",
            border_style="green"
        )
        console.print("\n", summary_panel)

    except Exception as e:
        console.print(f"\n[bold red][ERROR] Execution failed:[/bold red] {e}")

if __name__ == "__main__":
    asyncio.run(main())