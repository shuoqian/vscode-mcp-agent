import httpx
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB persistent client for cross-session long-term memory
chroma_client = chromadb.PersistentClient(path="./chroma_db_memory")
ef = embedding_functions.DefaultEmbeddingFunction()
memory_collection = chroma_client.get_or_create_collection(
    name="vscode_reported_issues",
    embedding_function=ef
)

async def fetch_github_issues_tool(constructed_query: str, days: int = 30) -> list[dict]:
    """
    Fetches GitHub issues using the dynamically constructed search query.
    Falls back to a broader search if the primary query returns no results.
    """
    url = f"https://api.github.com/search/issues?q={urllib.parse.quote(constructed_query)}&per_page=15&sort=created&order=desc"

    items = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers={"User-Agent": "VSCode-Agentic-Tracer"})
        if resp.status_code == 200:
            items = resp.json().get("items", [])
        else:
            print(f"  [API Warning] Query returned status {resp.status_code}. Executing fallback...")

    # Fallback Mechanism: If zero items found, run a broader keyword query
    if not items:
        since_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        fallback_query = f"repo:microsoft/vscode type:issue state:open created:>={since_date}"
        fallback_url = f"https://api.github.com/search/issues?q={urllib.parse.quote(fallback_query)}&per_page=15&sort=created&order=desc"
        async with httpx.AsyncClient(timeout=15.0) as client:
            fb_resp = await client.get(fallback_url, headers={"User-Agent": "VSCode-Agentic-Tracer"})
            if fb_resp.status_code == 200:
                items = fb_resp.json().get("items", [])

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

def check_vector_memory_duplicate(issue_number: int, issue_title: str, issue_body: str, similarity_threshold: float = 0.85) -> bool:
    """
    Point 4: Long-Term Memory check via ChromaDB.
    Returns True if an issue with similar semantic contents was previously reported.
    """
    if memory_collection.count() == 0:
        return False

    query_text = f"{issue_title}\n{issue_body[:300]}"
    results = memory_collection.query(
        query_texts=[query_text],
        n_results=1
    )

    if results and results["distances"] and len(results["distances"][0]) > 0:
        # ChromaDB DefaultEmbeddingFunction returns L2 distance; convert to similarity proxy or check distance
        # Lower distance = higher similarity
        distance = results["distances"][0][0]
        # Distance <= 0.35 typically corresponds to >= 0.85 cosine similarity for default embeddings
        if distance <= 0.35:
            return True

    return False

def save_issues_to_vector_memory(issues: List[Dict[str, Any]]) -> None:
    """
    Point 4: Vectorizes and persists newly summarized alerts into ChromaDB.
    """
    if not issues:
        return

    documents = []
    ids = []
    metadatas = []

    for issue in issues:
        doc_id = f"issue_{issue['number']}"
        doc_text = f"{issue['title']}\n{issue.get('body', '')[:300]}"

        ids.append(doc_id)
        documents.append(doc_text)
        metadatas.append({
            "number": issue["number"],
            "url": issue.get("url", ""),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    memory_collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )