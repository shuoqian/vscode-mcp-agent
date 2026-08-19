import os
import json
import asyncio
from typing import Any, Literal
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.tools import (
    GitHubLabelManager,
    fetch_github_issues_tool,
    estimate_issue_count,
    check_vector_memory_duplicate,
    save_issues_to_vector_memory
)
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# ---------------------------------------------------------------------
# REPORT VERIFIER AGENT — a genuinely separate agent, not another prompt
# on the same "role."
#
# This is deliberately its own ChatGroq instance with its own config,
# distinct from both `llm` (planner/ReAct judge) and `text_llm` (report
# writer, instantiated inside final_summary_node): temperature=0.0 and
# JSON mode, because its job is strict, deterministic fact-checking, not
# fluent prose. It has no stake in the draft report looking good — its
# only objective is to catch claims that aren't grounded in the source
# issue text, and it has real veto power (via `is_reliable`) over what
# the user ultimately sees, not just a cosmetic pass.
# ---------------------------------------------------------------------
critic_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)


def _build_fallback_report(pooled_issues: list[dict]) -> str:
    """
    Deterministic, non-LLM report used ONLY when the Report Verifier Agent
    judges the drafted narrative too unreliable to salvage by editing. No
    LLM involvement here at all — just the raw, already-verified issue data
    — so this fallback can't itself introduce new unsupported claims.
    """
    lines = [
        "# Executive Report (Narrative Suppressed — Verification Failed)",
        "",
        "The drafted narrative report for this run could not be verified against its "
        "source issues and has been withheld to avoid presenting unsupported claims. "
        "Below are the raw, individually-verified candidate issues only:",
        "",
    ]
    for i in pooled_issues:
        lines.append(f"- **Issue #{i['number']}** ({i['title']}) — Confidence: {i['confidence']:.2f} — {i['url']}")
    return "\n".join(lines)


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
# NODE 1: Dynamic Router & Planner Node
# =====================================================================
async def planner_node(state: AgentState):
    target_os = state["target_os"]
    target_module = state["target_module"]
    symptom_type = state["symptom_type"]
    issue_type = state["issue_type"]
    days = state["time_range_days"]

    label_manager = GitHubLabelManager(owner="microsoft", repo="vscode")
    all_repo_labels = label_manager.get_valid_labels()

    # ---------------------------------------------------------------
    # FIX (round 2):
    # - OS is now a HARD filter alongside module + type (per request #1) —
    #   only "symptom" remains a soft/semantic-only filter.
    # - Each category can resolve to MULTIPLE real repo labels, not just
    #   one (per request #2). E.g. module="Terminal" should pick up every
    #   plausible label — "terminal", "workbench-terminal",
    #   "terminal-conpty", etc. — and those are OR'd together in the query
    #   rather than forcing a single best guess.
    #
    # We get candidates from two sources and union them:
    #   (a) the LLM's categorized picks (semantic understanding)
    #   (b) GitHubLabelManager.find_related_labels() — a direct substring/
    #       fuzzy scan of the real label list (catches labels the LLM
    #       didn't think to mention)
    # ---------------------------------------------------------------
    label_matching_prompt = PromptTemplate.from_template(
        "You are a GitHub taxonomy expert for microsoft/vscode.\n"
        "Map the user's search criteria to exact matching labels from the provided repository label list, "
        "and classify EACH matched label into exactly one category. Where multiple real labels plausibly "
        "apply to the same criterion (e.g. several 'terminal'-related labels), include ALL of them, not just one.\n\n"
        "User Criteria:\n"
        "- OS: {target_os}\n"
        "- Module: {target_module}\n"
        "- Symptom/Area: {symptom_type}\n"
        "- Issue Type: {issue_type}\n\n"
        "Available Repository Labels:\n"
        "{labels_subset}\n\n"
        "Instructions:\n"
        "1. Only include labels that literally exist in the provided list.\n"
        "2. Category must be one of: 'module', 'type', 'symptom', 'os'.\n"
        "   - 'module' = which VS Code subsystem/component this is about (e.g. terminal, debug, editor-core)\n"
        "   - 'type' = the kind of issue (e.g. bug, feature-request)\n"
        "   - 'symptom' = performance/memory/crash-style descriptors (e.g. perf, memory-issue)\n"
        "   - 'os' = operating system labels (e.g. windows, macos, linux)\n"
        "3. For criteria with NO matching repository label, put the raw term under 'unmatched_keywords'.\n"
        "4. Return ONLY a valid JSON object, no extra commentary.\n\n"
        "JSON Format:\n"
        "{{\n"
        '  "labels": [{{"label": "terminal", "category": "module"}}, '
        '{{"label": "workbench-terminal", "category": "module"}}, {{"label": "bug", "category": "type"}}],\n'
        '  "unmatched_keywords": ["keyword-1"]\n'
        "}}"
    )

    labels_context = ", ".join(all_repo_labels[:800])

    # category -> set of verified real labels (module/type/os are hard; symptom is soft)
    llm_groups: dict[str, set[str]] = {"module": set(), "type": set(), "os": set(), "symptom": set()}
    unmatched_keywords: list[str] = []

    try:
        res = await (label_matching_prompt | llm).ainvoke({
            "target_os": target_os,
            "target_module": target_module,
            "symptom_type": symptom_type,
            "issue_type": issue_type,
            "labels_subset": labels_context
        })

        clean_json = extract_text(res.content)
        parsed = json.loads(clean_json)

        raw_items = parsed.get("labels", [])
        unmatched_keywords = list(parsed.get("unmatched_keywords", []))

        for item in raw_items:
            raw_label = item.get("label", "")
            category = (item.get("category") or "other").lower()

            verified = label_manager.match_single_label(raw_label, all_repo_labels)
            if not verified:
                unmatched_keywords.append(raw_label)
                continue

            if category in llm_groups:
                llm_groups[category].add(verified)
            else:
                llm_groups.setdefault("symptom", set()).add(verified)

    except Exception as e:
        print(f"⚠️ [Planner Dynamic Filter Exception/Rate Limit]: {e}. Falling back to substring/fuzzy scan only.")
        # LLM call failed entirely — llm_groups stays empty; the substring/fuzzy
        # scan below still runs independently and will populate hard categories.

    # Independent substring/fuzzy scan — runs regardless of LLM success, so we
    # always cast as wide a net as the real label list allows for each category.
    scan_groups = {
        "module": set(label_manager.find_related_labels(target_module, all_repo_labels)),
        "type": set(label_manager.find_related_labels(issue_type, all_repo_labels)),
        "os": set(label_manager.find_related_labels(target_os, all_repo_labels)),
    }

    # ---------------------------------------------------------------
    # TREE-OF-THOUGHT QUERY STRATEGY SELECTION
    #
    # Rather than committing to a single merged label strategy, generate
    # multiple DISTINCT candidate branches, evaluate each with a cheap
    # count-only GitHub call (run in parallel), and pick the branch that
    # actually performs best — instead of only discovering a strategy is
    # bad after a full fetch fails (that reactive fallback still exists
    # downstream in fetch_github_issues_tool as a safety net, but this
    # node now does proactive branch comparison up front).
    #
    #   Branch "narrow"  — one most-confident label per category (highest
    #                       precision, most likely to under-match)
    #   Branch "broad"   — full union of LLM + scan matches per category,
    #                       OR'd together (highest recall via labels)
    #   Branch "keyword" — no label filters at all; falls back to free-text
    #                       search on the raw module/type/os terms (covers
    #                       the case where no reliable labels exist at all)
    #
    # Each branch is scored by its estimated result count: 0 results is
    # disqualifying, a count within a reasonable "sweet spot" window scores
    # highest (favoring precision within that window), and an excessive
    # count is penalized (too broad to trust) but still preferred over zero.
    # ---------------------------------------------------------------
    since_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    def _top_label(llm_set: set[str], scan_set: set[str]) -> list[str]:
        # Prefer the LLM's semantic pick (it has more context) over the raw
        # scan; fall back to the scan's first alphabetical match if the LLM
        # found nothing for this category.
        if llm_set:
            return [sorted(llm_set)[0]]
        if scan_set:
            return [sorted(scan_set)[0]]
        return []

    narrow_groups = {
        "module": _top_label(llm_groups["module"], scan_groups["module"]),
        "type": _top_label(llm_groups["type"], scan_groups["type"]),
        "os": _top_label(llm_groups["os"], scan_groups["os"]),
    }
    broad_groups = {
        "module": sorted(llm_groups["module"] | scan_groups["module"]),
        "type": sorted(llm_groups["type"] | scan_groups["type"]),
        "os": sorted(llm_groups["os"] | scan_groups["os"]),
    }
    keyword_terms = [t for t in (target_module, issue_type, target_os) if t]

    candidate_branches = [
        {"name": "narrow", "hard_label_groups": narrow_groups, "text_keywords": []},
        {"name": "broad", "hard_label_groups": broad_groups, "text_keywords": []},
        {"name": "keyword", "hard_label_groups": {"module": [], "type": [], "os": []}, "text_keywords": keyword_terms},
    ]

    hard_label_groups, text_keywords, tot_trace = await select_query_strategy_tot(candidate_branches, since_date)
    soft_labels = sorted(llm_groups.get("symptom", set()))

    # Task 2 now checks ONLY the symptom — OS is already enforced as a hard
    # label filter upstream, so re-checking it semantically here would be
    # redundant (and was previously causing false negatives when an issue's
    # OS was implied by context but never literally the word "Windows").
    plan_tasks = [
        f"Verify issue belongs to {target_module} and is a {issue_type}",
        f"Check for {symptom_type} symptoms as described by the user",
    ]

    # constructed_query here is just for display/logging — the ACTUAL query used
    # is determined by fetch_github_issues_tool's progressive relaxation, since it
    # may drop whole categories if the full set returns zero results.
    preview_parts = ["repo:microsoft/vscode", "is:issue", "state:open", f"created:>={since_date}"]
    for cat in ("module", "type", "os"):
        labels = hard_label_groups[cat]
        if labels:
            preview_parts.append("label:" + ",".join(f'"{l}"' for l in labels))
    for kw in text_keywords:
        preview_parts.append(f'"{kw.replace("-", " ").strip()}"')
    constructed_query = " ".join(preview_parts)

    print(f"\n  ├─► [Constructed Query Preview]: {constructed_query}")
    print(f"  ├─► Hard Label Groups (OR within category, AND across): {hard_label_groups}")
    print(f"  ├─► Text Keywords (ToT 'keyword' branch only): {text_keywords}")
    print(f"  ├─► Soft Labels (judge context only, symptom): {soft_labels}")
    print(f"  └─► Unmatched Keywords (informational only, not queried): {unmatched_keywords}")

    flat_hard_labels = sorted({l for labels in hard_label_groups.values() for l in labels})

    return {
        "constructed_query": constructed_query,
        "plan_tasks": plan_tasks,
        "hard_label_groups": hard_label_groups,
        "text_keywords": text_keywords,
        "tot_branch_trace": tot_trace,
        "hard_labels": flat_hard_labels,  # backward-compat flat view
        "soft_labels": soft_labels,
        "matched_labels": flat_hard_labels + soft_labels,  # backward-compat display field
        "reflection_count": 0,
        "critique_feedback": "",
        "is_abstained": False
    }


# =====================================================================
# TREE-OF-THOUGHT: query strategy branch generation + evaluation
# =====================================================================

# Result-count "sweet spot" for scoring candidate branches: too few (0) means
# the branch is unusable; too many suggests the filter is too loose to trust
# as topically precise. Both are penalized relative to a count inside this
# window, but a large count is still preferred over zero.
TOT_SWEET_SPOT_MIN = 1
TOT_SWEET_SPOT_MAX = 40


def _score_branch_count(count: int) -> float:
    if count <= 0:
        return -1000.0  # disqualifying — branch would return nothing
    if TOT_SWEET_SPOT_MIN <= count <= TOT_SWEET_SPOT_MAX:
        # Within the sweet spot: prefer smaller (more precise) counts, but
        # any in-range count beats any out-of-range count.
        return 1000.0 - count
    # Too many results to trust as topically precise — still usable, but
    # scored below every in-range branch, worsening the further over it is.
    return -float(count - TOT_SWEET_SPOT_MAX)


async def select_query_strategy_tot(
        branches: list[dict[str, Any]], since_date: str
) -> tuple[dict[str, list[str]], list[str], list[dict[str, Any]]]:
    """
    Tree-of-Thought branch selection for query construction.

    Given several DISTINCT candidate strategies (each a full hard-label-group
    + text-keyword configuration), this:
      1. Builds each branch's query string.
      2. Evaluates all branches IN PARALLEL via a cheap count-only GitHub
         call (estimate_issue_count) — this is the "evaluator" step of ToT.
      3. Scores each branch by its estimated result count.
      4. Selects the highest-scoring branch and returns its configuration.

    This is genuine multi-path exploration-and-compare, not just sequential
    retry-on-failure: all branches are generated and evaluated up front,
    and the choice between them is made by comparing scores, not by trying
    one until it fails and only then trying the next.

    Returns: (winning_hard_label_groups, winning_text_keywords, trace)
    where `trace` is a list of per-branch {name, query, count, score} dicts
    for transparency/logging (and for the design doc / demo).
    """
    async def _evaluate(branch: dict[str, Any]) -> dict[str, Any]:
        from agent.tools import _build_query  # local import avoids polluting module namespace
        query = _build_query(branch["hard_label_groups"], since_date, branch.get("text_keywords", []))
        count = await estimate_issue_count(query)
        score = _score_branch_count(count)
        return {
            "name": branch["name"],
            "hard_label_groups": branch["hard_label_groups"],
            "text_keywords": branch.get("text_keywords", []),
            "query": query,
            "count": count,
            "score": score,
        }

    evaluated = await asyncio.gather(*[_evaluate(b) for b in branches])

    print("\n  🌳 [TREE-OF-THOUGHT] Exploring candidate query strategies in parallel:")
    for b in evaluated:
        print(f"     ├─ Branch '{b['name']}': ~{b['count']} result(s), score={b['score']:.0f}  →  {b['query']}")

    winner = max(evaluated, key=lambda b: b["score"])
    print(f"     └─► Selected branch: '{winner['name']}' (score={winner['score']:.0f}, ~{winner['count']} result(s))")

    return winner["hard_label_groups"], winner["text_keywords"], evaluated


# =====================================================================
# NODE 2: Fetch & Cross-Session Vector Memory Node
# =====================================================================
async def fetch_issues_node(state: AgentState):
    days = state["time_range_days"]
    hard_label_groups = state.get("hard_label_groups", {})
    text_keywords = state.get("text_keywords", [])

    print(f"\n[Fetch Node] Attempting fetch with hard label groups: {hard_label_groups} "
          f"(text_keywords: {text_keywords})")

    raw_issues, query_used, used_unfiltered_fallback = await fetch_github_issues_tool(
        hard_label_groups, days, text_keywords=text_keywords
    )

    print(f"[Fetch Result] Query actually used: {query_used}")
    if used_unfiltered_fallback:
        print("  ⚠️ [Fetch Result] NO label filters matched any issues — results below are UNFILTERED "
              "recent issues and rely entirely on the ReAct judge step to find relevant ones. "
              "Consider widening your time range or checking your module/issue-type wording.")
    print(f"[Fetch Result] Successfully fetched {len(raw_issues)} issue(s).")

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
        "vector_filtered_issues": vector_filtered_issues,
        "constructed_query": query_used,
    }


# =====================================================================
# NODE 3: Per-Issue ReAct Processing Node
# =====================================================================
async def process_issues_react_node(state: AgentState):
    """Executes verification tasks on target issues strictly."""
    issues_to_process = state.get("vector_filtered_issues", [])
    plan_tasks = state.get("plan_tasks", [])
    critique_feedback = state.get("critique_feedback", "")
    soft_labels = state.get("soft_labels", [])
    target_module = state["target_module"]
    target_os = state["target_os"]
    symptom_type = state["symptom_type"]

    pooled_issues = []
    total = len(issues_to_process)

    print("\n" + "="*65)
    print(f" ⚙️ [EXECUTION NODE] ReAct Processing on {total} Issue(s)")
    print("="*65)

    # FIX: require a quoted evidence snippet and explicitly reject tangential/
    # indirect matches, so e.g. "Remote WSL connection" doesn't loosely pass
    # a "belongs to Terminal" check just because WSL touches shells generally.
    eval_prompt = PromptTemplate.from_template(
        "Criteria to verify: {task_description}\n"
        "Additional context labels on this issue (may or may not be relevant): {soft_labels}\n"
        "Critique Feedback: {critique_feedback}\n\n"
        "GitHub Issue Title: {title}\n"
        "GitHub Issue Labels: {labels}\n"
        "GitHub Issue Body:\n{body}\n\n"
        "Does this issue EXPLICITLY and DIRECTLY match the criteria above? "
        "Only answer true if the title, labels, or body directly and unambiguously reference the target. "
        "Tangential, indirect, or 'could be related' matches do NOT count — answer false for those. "
        "You must quote the specific evidence (a short phrase from the title/labels/body) that justifies your answer.\n\n"
        "Respond ONLY with JSON:\n"
        "{{\n"
        '  "passed": true,\n'
        '  "confidence": 0.85,\n'
        '  "evidence": "short quoted phrase or label name that justifies this"\n'
        "}}"
    )

    for idx, issue in enumerate(issues_to_process, 1):
        issue_id = f"#{issue['number']}"
        issue_title = issue['title']
        issue_body = issue["body"][:1200]
        issue_labels = [str(l).lower() for l in issue.get("labels", [])]

        print(f"\n[ReAct Loop {idx}/{total}: Issue {issue_id}] \"{issue_title}\"")

        issue_scores = []
        is_discarded = False

        for t_idx, task_desc in enumerate(plan_tasks, 1):
            print(f"  ├─► [Executing Task {t_idx}: {task_desc}]")
            try:
                await asyncio.sleep(0.3)
                res = await (eval_prompt | llm).ainvoke({
                    "task_description": task_desc,
                    "soft_labels": ", ".join(soft_labels) or "None",
                    "critique_feedback": critique_feedback or "None",
                    "title": issue_title,
                    "labels": ", ".join(issue_labels),
                    "body": issue_body
                })

                clean = extract_text(res.content)
                eval_data = json.loads(clean)
                passed = bool(eval_data.get("passed", False))
                conf = float(eval_data.get("confidence", 0.0))
                evidence = eval_data.get("evidence", "")
            except Exception as err:
                print(f"  │   ⚠️ [LLM Error/Rate-Limit]: {err}")
                # STRICT REQUIREMENT: Error explicitly fails evaluation instead of passing by default
                passed = False
                conf = 0.0
                evidence = ""

            if not passed:
                print(f"  │   └─ Status: ❌ DISCARDED (Failed Task {t_idx} - Passed: {passed}, Conf: {conf}, Evidence: '{evidence}')")
                is_discarded = True
                break

            issue_scores.append(conf)
            print(f"  │   └─ Status: ✅ PASSED (Task Confidence: {conf:.2f}, Evidence: '{evidence}')")

        if not is_discarded:
            avg_issue_conf = sum(issue_scores) / len(issue_scores) if issue_scores else 0.0
            if avg_issue_conf > 0.0:
                print(f"  └─► [Action] 📥 Added Issue {issue_id} (Confidence: {avg_issue_conf:.2f})")
                pooled_issues.append({
                    "number": issue["number"],
                    "title": issue_title,
                    "url": issue["html_url"],
                    "body": issue_body,
                    "confidence": avg_issue_conf
                })

    if pooled_issues:
        avg_pool_conf = sum(i["confidence"] for i in pooled_issues) / len(pooled_issues)
    else:
        avg_pool_conf = 0.0

    return {
        "pooled_issues": pooled_issues,
        "pool_confidence_score": avg_pool_conf
    }


# =====================================================================
# NODE 4: Verify, Reflect & Abstention Gate Node
# =====================================================================
async def verify_and_reflect_node(state: AgentState):
    pooled_issues = state.get("pooled_issues", [])
    pool_confidence = state.get("pool_confidence_score", 0.0)
    reflection_count = state.get("reflection_count", 0) + 1

    CONFIDENCE_THRESHOLD = 0.80

    print("\n" + "="*65)
    print(f" 🔍 [REFLECTION & QUALITY GATE] Pool Confidence Audit (Attempt #{reflection_count})")
    print("="*65)
    print(f"  ├─► Candidate Issues in Pool: {len(pooled_issues)}")
    print(f"  ├─► Calculated Pool Confidence: {pool_confidence:.2f} (Threshold: {CONFIDENCE_THRESHOLD})")

    if not pooled_issues or pool_confidence < CONFIDENCE_THRESHOLD:
        print(f"  └─► [QUALITY GATE FAILED] Confidence score ({pool_confidence:.2f}) below threshold {CONFIDENCE_THRESHOLD}.")

        if reflection_count <= 2:
            critique = "Pool confidence is low or empty. Re-evaluate criteria with higher leniency."
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
# NODE 5: Final Summary Synthesis & Memory Commit Node
# =====================================================================
async def final_summary_node(state: AgentState):
    pooled = state.get("pooled_issues", [])
    is_abstained = state.get("is_abstained", False)

    print("\n" + "="*65)
    print(" 📊 [FINAL NODE] Generating Draft Report & Updating Memory")
    print("="*65)

    if is_abstained or not pooled:
        summary_text = (
            "⚠️ [ABSTENTION NOTICE] The agent abstained from generating a summary report.\n"
            "Reason: Matching issues yielded insufficient confidence (< 0.80 threshold) or API limits/errors prevented verification.\n"
            "Recommendation: Broaden your search criteria or retry after rate limits reset."
        )
        # Abstention has nothing to verify — set aggregate_summary directly
        # and skip the critic node entirely (see route_after_final_summary).
        return {"aggregate_summary": summary_text, "draft_summary": "", "source_evidence_text": ""}

    combined_text = "\n".join([
        f"• Issue #{i['number']} ({i['title']}) [Confidence: {i['confidence']:.2f}]\n  URL: {i['url']}\n  Snippet: {i['body'][:300]}..."
        for i in pooled
    ])

    summary_prompt = PromptTemplate.from_template(
        "User Interest: {symptom_type} in {target_module} ({target_os})\n\n"
        "Verified Candidate Issues:\n{combined_text}\n\n"
        "Provide a clean, well-structured markdown executive report highlighting patterns, affected components, and root causes."
    )

    text_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
    final_res = await (summary_prompt | text_llm).ainvoke({
        "symptom_type": state["symptom_type"],
        "target_module": state["target_module"],
        "target_os": state["target_os"],
        "combined_text": combined_text
    })

    print("  └─► [Vector Memory Commit] Persisting newly summarized issues into ChromaDB...")
    save_issues_to_vector_memory(pooled)

    # NOTE: this is a DRAFT only. aggregate_summary is intentionally left
    # unset here — it's populated downstream by report_critic_node, which
    # has final say over what actually reaches the user.
    return {
        "draft_summary": extract_text(final_res.content),
        "source_evidence_text": combined_text,
    }


# =====================================================================
# NODE 6: Report Verifier Agent — independent fact-check of the draft
# =====================================================================
async def report_critic_node(state: AgentState):
    """
    A second, independent agent (see `critic_llm` above) that audits the
    drafted report against the actual source issue text it was supposed to
    be grounded in. This is a genuine generator/verifier multi-agent
    handoff: the writer (final_summary_node's text_llm) and the verifier
    (critic_llm) are different agent instances with different objectives
    and different sampling configs, and the verifier has real authority to
    override the writer's output — up to and including discarding the
    narrative entirely in favor of a deterministic fallback (see 9.5-style
    reasoning: no single layer's output is trusted unchecked).
    """
    draft = state.get("draft_summary", "")
    pooled = state.get("pooled_issues", [])
    source_evidence = state.get("source_evidence_text", "")

    print("\n" + "="*65)
    print(" 🕵️ [REPORT VERIFIER AGENT] Auditing draft against source evidence")
    print("="*65)

    if not draft or not pooled:
        # Nothing to verify — shouldn't normally be reached since the
        # abstention path skips this node entirely, but guarded defensively.
        return {"aggregate_summary": draft, "flagged_claims": [], "report_verified": True}

    critic_prompt = PromptTemplate.from_template(
        "You are an independent Report Verification Agent. Your ONLY job is to audit a DRAFT "
        "executive report against its SOURCE EVIDENCE (the original GitHub issue excerpts it was "
        "supposed to be based on) and catch any claim in the draft that is NOT actually supported "
        "by the source evidence — including plausible-sounding root-cause explanations, invented "
        "technical details, or generalized recommendations that go beyond what the source material "
        "actually states.\n\n"
        "You did not write the draft and have no stake in it looking good. Your goal is strict, "
        "adversarial accuracy checking, not politeness.\n\n"
        "SOURCE EVIDENCE (the only material any claim in the draft may be grounded in):\n"
        "{source_evidence}\n\n"
        "DRAFT REPORT TO AUDIT:\n"
        "{draft_summary}\n\n"
        "Instructions:\n"
        "1. Identify every claim in the draft stating a cause, pattern, or recommendation NOT "
        "directly traceable to the source evidence above.\n"
        "2. Produce a corrected 'verified_summary': the same report with unsupported claims removed "
        "or rewritten as explicitly-labeled speculation (prefix with 'Possible interpretation "
        "(not confirmed by source):'). Keep everything that IS grounded intact.\n"
        "3. Set 'is_reliable' to false ONLY if the draft is so disconnected from the source evidence "
        "that no coherent, grounded report can be salvaged by editing — this should be rare.\n"
        "4. Return ONLY valid JSON, no extra commentary.\n\n"
        "JSON Format:\n"
        "{{\n"
        '  "flagged_claims": [{{"claim": "short quote or paraphrase of the unsupported claim", '
        '"reason": "why it is not supported by the source evidence"}}],\n'
        '  "is_reliable": true,\n'
        '  "verified_summary": "the corrected markdown report"\n'
        "}}"
    )

    try:
        res = await (critic_prompt | critic_llm).ainvoke({
            "source_evidence": source_evidence,
            "draft_summary": draft,
        })
        parsed = json.loads(extract_text(res.content))
        flagged_claims = parsed.get("flagged_claims", [])
        is_reliable = bool(parsed.get("is_reliable", True))
        verified_summary = parsed.get("verified_summary", "") or draft
    except Exception as e:
        print(f"  ⚠️ [Report Verifier Exception]: {e}. Falling back to deterministic report as a safety measure.")
        flagged_claims = [{"claim": "N/A", "reason": f"Verifier agent call failed: {e}"}]
        is_reliable = False
        verified_summary = ""

    if flagged_claims:
        print(f"  ├─► [Flagged Claims] {len(flagged_claims)} unsupported claim(s) found in draft:")
        for c in flagged_claims:
            print(f"  │     • \"{c.get('claim', '')}\" — {c.get('reason', '')}")
    else:
        print("  ├─► [Flagged Claims] None — draft is fully grounded in source evidence.")

    if not is_reliable:
        print("  └─► [VETO] Verifier judged the draft unsalvageable. Substituting deterministic fallback report.")
        final_text = _build_fallback_report(pooled)
    else:
        print("  └─► [ACCEPTED] Verified summary approved for delivery.")
        final_text = verified_summary

    return {
        "aggregate_summary": final_text,
        "flagged_claims": flagged_claims,
        "report_verified": is_reliable,
    }


def route_after_final_summary(state: AgentState) -> Literal["report_critic", "end"]:
    # Abstention has nothing to verify — go straight to END rather than
    # invoking the critic agent on an empty/notice-only draft.
    if state.get("is_abstained", False):
        return "end"
    return "report_critic"


# =====================================================================
# WORKFLOW BUILDER
# =====================================================================
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("fetch_issues", fetch_issues_node)
workflow.add_node("process_issues_react", process_issues_react_node)
workflow.add_node("verify_and_reflect", verify_and_reflect_node)
workflow.add_node("final_summary", final_summary_node)
workflow.add_node("report_critic", report_critic_node)

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

workflow.add_conditional_edges(
    "final_summary",
    route_after_final_summary,
    {
        "report_critic": "report_critic",
        "end": END
    }
)

workflow.add_edge("report_critic", END)

app_agent = workflow.compile()