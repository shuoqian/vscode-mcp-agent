from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    target_os: str
    user_interest: str
    time_range_days: int
    raw_issues: List[Dict[str, Any]]
    pooled_issues: List[Dict[str, Any]]
    aggregate_summary: str
    logs: List[str]