"""
VisionIQ Shared State Definition for LangGraph Dynamic Orchestration
Maintains session data, query context, routing scores, specialist outputs,
isolated execution logs, and final synthesis results.
"""

from enum import Enum
from typing import TypedDict, Optional, List, Dict, Any, Set


class AgentLifecycleState(str, Enum):
    """
    Explicit lifecycle states for specialist agents in VisionIQ sequential routing.
    Enforces terminal protection, dependency gating, and fault isolation.
    """
    NOT_SELECTED = "NOT_SELECTED"
    WAITING_FOR_DEPENDENCY = "WAITING_FOR_DEPENDENCY"
    ELIGIBLE = "ELIGIBLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


VALID_LIFECYCLE_TRANSITIONS: Dict[AgentLifecycleState, Set[AgentLifecycleState]] = {
    AgentLifecycleState.NOT_SELECTED: {
        AgentLifecycleState.WAITING_FOR_DEPENDENCY,
        AgentLifecycleState.ELIGIBLE,
        AgentLifecycleState.NOT_APPLICABLE,
    },
    AgentLifecycleState.WAITING_FOR_DEPENDENCY: {
        AgentLifecycleState.ELIGIBLE,
        AgentLifecycleState.BLOCKED,
        AgentLifecycleState.NOT_APPLICABLE,
    },
    AgentLifecycleState.ELIGIBLE: {
        AgentLifecycleState.RUNNING,
        AgentLifecycleState.NOT_SELECTED,
    },
    AgentLifecycleState.RUNNING: {
        AgentLifecycleState.COMPLETED,
        AgentLifecycleState.FAILED,
    },
    AgentLifecycleState.COMPLETED: set(),  # Terminal: no duplicate execution
    AgentLifecycleState.FAILED: set(),     # Terminal: failure isolation
    AgentLifecycleState.BLOCKED: set(),    # Terminal: unsatisfied dependency
    AgentLifecycleState.NOT_APPLICABLE: set(),  # Terminal: out-of-scope for query
}


def validate_lifecycle_transition(
    current: AgentLifecycleState,
    target: AgentLifecycleState,
    strict: bool = False
) -> bool:
    """
    Validates whether an agent lifecycle transition is legal.
    If strict=True, raises ValueError on illegal transitions.
    """
    valid_targets = VALID_LIFECYCLE_TRANSITIONS.get(current, set())
    is_valid = target in valid_targets
    if not is_valid and strict:
        raise ValueError(f"Illegal lifecycle transition: {current.value} -> {target.value}")
    return is_valid


class AgentState(TypedDict, total=False):
    # Inputs & Session Metadata
    image_path: str
    user_query: str
    session_id: Optional[str]
    input_type: str  # "image", "video", "camera"
    timeout_seconds: float

    # Intent Classification & Routing Plan
    detected_intent: Optional[str]
    intent_confidence: float
    routing_scores: Dict[str, Dict[str, Any]]
    selected_agents: List[str]
    executed_agents: List[str]
    skipped_agents: List[str]
    failed_agents: List[str]

    # Specialist Agent Raw Outputs
    object_result: Optional[Dict[str, Any]]
    ppe_result: Optional[Dict[str, Any]]
    ocr_result: Optional[Dict[str, Any]]
    defect_result: Optional[Dict[str, Any]]
    scene_result: Optional[Dict[str, Any]]
    evidence_validation_result: Optional[Dict[str, Any]]
    rag_result: Optional[Dict[str, Any]]
    rca_result: Optional[Dict[str, Any]]
    recommendation_result: Optional[Dict[str, Any]]

    # Synthesized Results & Evidence
    findings: List[Dict[str, Any]]
    rag_references: List[Dict[str, Any]]
    synthesis_text: Optional[str]
    evidence_support_score: float
    annotated_path: Optional[str]
    pdf_report_path: Optional[str]
    final_response: Optional[Dict[str, Any]]

    # RAVEN: Risk-Aware Visual Evidence Navigator
    raven_assessment: Optional[Dict[str, Any]]
    raven_decision: Optional[str]  # "ACCEPT", "VERIFY", "ABSTAIN"
    raven_risk_score: Optional[float]
    evidence_completeness: Optional[float]
    conflict_entropy: Optional[float]
    secondary_agents_dispatched: Optional[List[str]]
    raven_dossier: Optional[Dict[str, Any]]
    verification_pass_executed: Optional[bool]
    verification_cycle: Optional[int]

    # Phase 3: Sequential Evidence-Gain Routing & Ledger
    iteration_step: Optional[int]
    max_steps: Optional[int]
    deficit_history: Optional[List[float]]
    current_deficit: Optional[float]
    target_deficit: Optional[float]
    selected_specialist: Optional[str]
    utility_trace: Optional[List[Dict[str, Any]]]
    active_evidence_report: Optional[Dict[str, Any]]
    advisory_absence_flags: Optional[Dict[str, bool]]
    sequential_routing_enabled: Optional[bool]
    agent_lifecycle_states: Optional[Dict[str, str]]
    routing_telemetry: Optional[List[Dict[str, Any]]]
    wallclock_deadline_ms: Optional[float]
    remaining_time_budget_ms: Optional[float]
    stopping_reason: Optional[str]
    routing_strategy: Optional[str]

    # Execution Audit Trail & Latency Telemetry
    execution_log: List[Dict[str, Any]]
    errors: List[str]
    start_time: float
    total_latency_ms: float
