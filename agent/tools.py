import os
import json
import time
import httpx
import requests
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple
from difflib import get_close_matches
import chromadb
from chromadb.utils import embedding_functions

# --- ChromaDB Vector Memory Initialization ---
MEMORY_DB_PATH = "./chroma_db_memory"
chroma_client = chromadb.PersistentClient(path=MEMORY_DB_PATH)
ef = embedding_functions.DefaultEmbeddingFunction()
memory_collection = chroma_client.get_or_create_collection(
    name="vscode_reported_issues",
    embedding_function=ef
)

# --- Label Cache Configuration ---
CACHE_FILE = Path(__file__).resolve().parent.parent / "vscode_labels_cache.json"
CACHE_TTL_SECONDS = 86400  # 24-hour cache TTL


def clear_vector_memory() -> None:
    """Utility function to wipe ChromaDB vector memory completely."""
    global chroma_client, memory_collection
    try:
        chroma_client.delete_collection(name="vscode_reported_issues")
    except Exception:
        pass
    memory_collection = chroma_client.get_or_create_collection(
        name="vscode_reported_issues",
        embedding_function=ef
    )
    print("  🧹 [Memory Cleared] ChromaDB vector collection has been reset.")


# =====================================================================
# GITHUB LABEL MANAGER CLASS
# =====================================================================
class GitHubLabelManager:
    """Handles fetching, caching, and dynamic taxonomy matching for repository labels."""

    def __init__(self, owner: str = "microsoft", repo: str = "vscode", github_token: str | None = None):
        self.owner = owner
        self.repo = repo
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")

    def fetch_github_labels(self) -> list[str]:
        """Fetches all repository labels via GitHub REST API with pagination."""
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/labels"
        labels = []
        page = 1
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "VSCode-Agentic-Tracer"
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"

        while True:
            try:
                response = requests.get(
                    url,
                    params={"per_page": 100, "page": page},
                    headers=headers,
                    timeout=10
                )
                if response.status_code != 200:
                    print(f"  ⚠️ [Label Manager] HTTP {response.status_code}: Unable to fetch live labels.")
                    break

                data = response.json()
                if not data or not isinstance(data, list):
                    break

                for label_obj in data:
                    labels.append(label_obj["name"])

                if len(data) < 100:
                    break
                page += 1

            except Exception as e:
                print(f"  ⚠️ [Label Manager] Network error while syncing labels: {e}")
                break

        return labels

    def get_valid_labels(self, force_refresh: bool = False) -> list[str]:
        """Retrieves labels from local cache or syncs from GitHub if expired/missing."""
        if not force_refresh and CACHE_FILE.exists():
            try:
                cache_data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                age = time.time() - cache_data.get("timestamp", 0)
                if age < CACHE_TTL_SECONDS and cache_data.get("labels"):
                    print(f"  ℹ️ [Label Manager] Loaded {len(cache_data['labels'])} cached labels from '{CACHE_FILE.name}'")
                    return cache_data["labels"]
            except Exception as e:
                print(f"  ⚠️ [Label Manager] Cache read error ({e}). Re-syncing...")

        print("  🔄 [Label Sync] Fetching current valid labels from GitHub API...")
        labels = self.fetch_github_labels()

        if labels:
            cache_payload = {
                "timestamp": time.time(),
                "owner_repo": f"{self.owner}/{self.repo}",
                "count": len(labels),
                "labels": labels
            }
            CACHE_FILE.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
            print(f"  ✅ [Label Sync] Cached {len(labels)} real labels to '{CACHE_FILE.name}'")
        else:
            print("  ⚠️ [Label Sync] Fallback: Could not sync labels.")

        return labels

    def match_single_label(self, candidate: str, valid_labels: list[str]) -> str | None:
        """
        Validates ONE candidate label string against the real repository label list.
        Returns the exact, correctly-cased label if a confident match is found, else None.
        Used so we can preserve category (module/type/symptom/os) association per label
        instead of losing it in a bulk list operation.
        """
        if not candidate:
            return None
        clean = candidate.strip().lower()
        valid_map = {l.lower(): l for l in valid_labels}

        if clean in valid_map:
            return valid_map[clean]

        fuzzy = get_close_matches(clean, list(valid_map.keys()), n=1, cutoff=0.6)
        if fuzzy:
            return valid_map[fuzzy[0]]

        # substring fallback, but require the candidate to be a reasonably
        # significant chunk of the label (avoids e.g. "os" matching everything)
        for v_lower, original in valid_map.items():
            if len(clean) >= 4 and (clean in v_lower or v_lower in clean):
                return original

        return None

    def find_related_labels(self, keyword: str, valid_labels: list[str], max_matches: int = 8) -> list[str]:
        """
        Returns ALL real repo labels plausibly related to `keyword`, instead of
        just the single best match. This is what lets e.g. module="Terminal"
        pick up "terminal", "workbench-terminal", "terminal-conpty", etc. — any
        of which should count when OR'd together in the query.

        Uses substring matching (either direction) plus fuzzy matching, so it
        catches both naming variants (terminal-suggest) and near-misses
        (perf vs performance). Capped at max_matches to keep queries sane.
        """
        if not keyword:
            return []
        clean = keyword.strip().lower()
        if not clean:
            return []

        matches: list[str] = []

        # Substring match in either direction (only for reasonably specific
        # keywords — skip 1-2 char keywords to avoid matching everything)
        if len(clean) >= 3:
            for label in valid_labels:
                low = label.lower()
                if clean in low or low in clean:
                    matches.append(label)

        # Fuzzy match to catch naming variants substring-matching would miss
        fuzzy_lower = get_close_matches(clean, [l.lower() for l in valid_labels], n=max_matches, cutoff=0.6)
        valid_map = {l.lower(): l for l in valid_labels}
        for f in fuzzy_lower:
            original = valid_map.get(f)
            if original and original not in matches:
                matches.append(original)

        return list(dict.fromkeys(matches))[:max_matches]

    def filter_query_labels(self, candidate_labels: list[str], valid_labels: list[str]) -> tuple[list[str], list[str]]:
        """
        Dynamic Taxonomy Grounding (bulk variant, kept for backward compatibility /
        fallback path when the LLM categorization call fails).
        """
        verified_labels = []
        unmatched_keywords = []

        for item in candidate_labels:
            match = self.match_single_label(item, valid_labels)
            if match:
                verified_labels.append(match)
            else:
                unmatched_keywords.append((item or "").strip().lower())

        unique_labels = list(dict.fromkeys(verified_labels))
        return unique_labels, [k for k in unmatched_keywords if k]


# =====================================================================
# GITHUB ISSUES FETCHING TOOL
# =====================================================================

# Category drop order when a query returns zero results: OS is dropped first
# (least likely to be consistently labeled across all issues), then module
# (may have a broad OR-set already, but if that whole set still misses,
# it's the next most negotiable), then finally "type" (bug/feature) — kept
# until last since it's usually the most reliably-applied label in vscode.
CATEGORY_DROP_ORDER = ["os", "module", "type"]


def _build_query(hard_label_groups: dict[str, list[str]], since_date: str) -> str:
    """
    Builds a GitHub search query where labels WITHIN a category are OR'd
    (label:"a","b") and categories are AND'd (separate label: qualifiers).
    """
    parts = [
        "repo:microsoft/vscode",
        "is:issue",
        "state:open",
        f"created:>={since_date}",
    ]
    # Fixed order for deterministic/readable queries
    for category in ("module", "type", "os"):
        labels = hard_label_groups.get(category) or []
        if not labels:
            continue
        quoted = ",".join(f'"{l}"' for l in labels)
        parts.append(f"label:{quoted}")
    return " ".join(parts)


async def _run_search(query: str) -> list[dict]:
    """Single GitHub search call. Returns [] on any failure or empty result."""
    headers = {"User-Agent": "VSCode-Agentic-Tracer"}
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    url = f"https://api.github.com/search/issues?q={urllib.parse.quote(query)}&per_page=15&sort=created&order=desc"

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json().get("items", [])
            print(f"  ⚠️ [API Warning] Query returned HTTP {resp.status_code} for: {query}")
        except Exception as e:
            print(f"  ⚠️ [Fetch Exception]: {e} for query: {query}")
    return []


async def fetch_github_issues_tool(hard_label_groups: dict[str, list[str]], days: int = 30) -> tuple[list[dict], str, bool]:
    """
    Fetches GitHub issues using PROGRESSIVE CATEGORY relaxation instead of a
    single all-or-nothing query:

      1. Try with ALL categories active, each OR'd internally (module labels
         OR'd together, AND'd against type labels OR'd together, AND'd
         against os labels OR'd together).
      2. If empty, drop one whole category at a time (per CATEGORY_DROP_ORDER
         — os first, then module, then type) and retry.
      3. If still empty with every category dropped, fall back to date+state
         only, flagged as `used_unfiltered_fallback=True` so callers/logs
         know the results are NOT topically filtered.

    Returns: (cleaned_issues, query_actually_used, used_unfiltered_fallback)
    """
    since_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    # Work on a mutable copy so we can drop categories without mutating the
    # caller's state dict.
    active_groups = {k: list(v) for k, v in hard_label_groups.items()}
    query_used = _build_query(active_groups, since_date)
    items = await _run_search(query_used)
    used_unfiltered_fallback = False

    # Progressive relaxation: drop one whole category at a time
    drop_queue = [c for c in CATEGORY_DROP_ORDER if active_groups.get(c)]
    while not items and drop_queue:
        cat_to_drop = drop_queue.pop(0)
        dropped_labels = active_groups.get(cat_to_drop, [])
        active_groups[cat_to_drop] = []
        print(f"  ⚠️ [Query Relaxation] No results — dropping category '{cat_to_drop}' "
              f"(labels: {dropped_labels}) and retrying...")
        query_used = _build_query(active_groups, since_date)
        items = await _run_search(query_used)

    if not items:
        print("  ⚠️ [Query Relaxation] No results even with zero label filters. "
              "Falling back to unfiltered recent issues — treat results as LOW CONFIDENCE / broad.")
        query_used = _build_query({}, since_date)
        items = await _run_search(query_used)
        used_unfiltered_fallback = True

    # Final strict check: an issue must carry AT LEAST ONE label from each
    # category we actually kept active in the query (OR within category,
    # AND across categories) — mirrors the query semantics exactly.
    active_sets = {cat: set(l.lower() for l in labels) for cat, labels in active_groups.items() if labels}
    print(f"  ℹ️ [Fetch Debug] Effective hard label groups: "
          f"{ {k: sorted(v) for k, v in active_sets.items()} or '(none — unfiltered)'}")

    cleaned_issues = []
    for item in items:
        issue_labels = set(
            lbl["name"].lower() if isinstance(lbl, dict) else str(lbl).lower()
            for lbl in item.get("labels", [])
        )

        missing_categories = [
            cat for cat, label_set in active_sets.items()
            if not (issue_labels & label_set)
        ]
        if missing_categories:
            print(f"  🛑 [Fetch Pre-Filter] Dropping Issue #{item['number']} "
                  f"(No label match in required categor{'y' if len(missing_categories)==1 else 'ies'}: {missing_categories})")
            continue

        cleaned_issues.append({
            "number": item["number"],
            "title": item["title"],
            "body": item.get("body", "") or "No body provided.",
            "html_url": item.get("html_url", ""),
            "labels": list(issue_labels)
        })

    return cleaned_issues, query_used, used_unfiltered_fallback


# =====================================================================
# VECTOR MEMORY DEDUPLICATION & STORAGE TOOLS
# =====================================================================
def check_vector_memory_duplicate(issue_number: int, issue_title: str, issue_body: str, distance_threshold: float = 0.35) -> bool:
    """Long-Term Memory check via ChromaDB.

    NOTE: this uses Chroma's default squared-L2 distance on the DefaultEmbeddingFunction's
    embeddings, NOT cosine similarity directly. If those embeddings are (approximately)
    unit-normalized, squared-L2 ≈ 2 * (1 - cosine_similarity), so distance <= 0.35 roughly
    corresponds to cosine_similarity >= ~0.825 — close to, but not exactly, the "0.85 cosine
    similarity" described in the design doc. Treat 0.35 as an empirically-tuned threshold on
    this specific embedding function rather than an exact cosine-similarity equivalent.
    """
    if memory_collection.count() == 0:
        return False

    query_text = f"{issue_title}\n{issue_body[:300]}"
    results = memory_collection.query(
        query_texts=[query_text],
        n_results=1
    )

    if results and results.get("distances") and len(results["distances"][0]) > 0:
        distance = results["distances"][0][0]
        if distance <= distance_threshold:
            return True

    return False


def save_issues_to_vector_memory(issues: List[Dict[str, Any]]) -> None:
    """Vectorizes and persists newly summarized alerts into ChromaDB."""
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