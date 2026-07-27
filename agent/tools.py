import httpx
import urllib.parse
from datetime import datetime, timedelta, timezone

async def fetch_github_issues_tool(days: int = 30) -> list[dict]:
    """Fetches ALL open GitHub issues created within the specified time range."""
    since_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # Query for ALL open issues in microsoft/vscode created within the time window
    query = f"repo:microsoft/vscode type:issue state:open created:>={since_date}"
    url = f"https://api.github.com/search/issues?q={urllib.parse.quote(query)}&per_page=30&sort=created&order=desc"

    items = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers={"User-Agent": "MCP-Agent"})
        if resp.status_code == 200:
            items = resp.json().get("items", [])

    cleaned_issues = [
        {
            "number": item["number"],
            "title": item["title"],
            "body": item.get("body", "") or "No body provided.",
            "html_url": item.get("html_url")
        }
        for item in items
    ]

    return cleaned_issues