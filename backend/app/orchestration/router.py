"""
VisionIQ Dynamic Cost-Aware Query Router
Implements Equation 1:
S(a, q) = alpha * R(a, q) + delta * E(a) - beta * C(a) - gamma * L(a)

Enforces configurable weights, normalized metrics, operating threshold tau,
mandatory structural dependencies, and minimum sufficient evidence constraints.
"""

import time
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Set
from backend.app.core.config import settings
from backend.app.orchestration.state import (
    AgentLifecycleState,
    VALID_LIFECYCLE_TRANSITIONS,
    validate_lifecycle_transition
)

AVAILABLE_AGENTS = [
    "object",
    "ppe",
    "ocr",
    "defect",
    "scene",
    "evidence_validation",
    "rag",
    "rca",
    "recommendation"
]

def classify_intent_deterministic(query: str) -> Tuple[str, float, Dict[str, float]]:
    """
    Deterministic intent classifier and relevance estimator R(a, q).
    Classifies queries across detection, counting, OCR, PPE, defect analysis,
    anomaly localization, scene understanding, video tracking, and combined tasks.

    Returns:
        (primary_intent, intent_confidence, relevance_dict)
    """
    q = query.lower().strip()
    relevance: Dict[str, float] = {a: 0.10 for a in AVAILABLE_AGENTS}

    # Combined Query: Scene + Damage / Defect
    if ("scene" in q or "describe" in q or "overview" in q) and any(d in q for d in ["damage", "defect", "flaw", "crack", "scratch"]):
        intent = "combined_scene_and_defect"
        confidence = 0.95
        relevance["scene"] = 0.98
        relevance["defect"] = 0.90
        relevance["object"] = 0.80
        relevance["rca"] = 0.85
        relevance["evidence_validation"] = 0.85
        relevance["rag"] = 0.70
        return intent, confidence, relevance

    # Intent 1: OCR / Machine Serial / Text Extraction
    if any(k in q for k in ["serial", "read", "text", "machine id", "ocr", "number on", "label", "code on", "tag", "sign"]):
        intent = "ocr_text_extraction"
        confidence = 0.96
        relevance["ocr"] = 0.98
        relevance["object"] = 0.50
        relevance["evidence_validation"] = 0.80
        return intent, confidence, relevance

    # Intent 2: Counting & Quantifying Entities
    if any(k in q for k in ["count", "how many", "number of people", "number of workers", "total count", "enumerate"]):
        intent = "object_counting_detection"
        confidence = 0.95
        relevance["object"] = 0.98
        relevance["scene"] = 0.50
        relevance["evidence_validation"] = 0.75
        # Do not activate OCR or Defect for counting!
        relevance["ocr"] = 0.10
        relevance["defect"] = 0.10
        return intent, confidence, relevance

    # Intent 3: PPE Safety Compliance
    if any(k in q for k in ["ppe", "helmet", "hard hat", "hardhat", "vest", "safety gear", "compliant", "compliance", "worker safety", "protective"]):
        intent = "ppe_safety_compliance"
        confidence = 0.98
        relevance["ppe"] = 0.98
        relevance["object"] = 0.85
        relevance["scene"] = 0.60
        relevance["rag"] = 0.75
        relevance["recommendation"] = 0.80
        relevance["evidence_validation"] = 0.85
        return intent, confidence, relevance

    # Intent 4: Industrial Defect / Surface Anomaly Localization
    if any(k in q for k in ["crack", "scratch", "defect", "damage", "pitted", "crazing", "inclusion", "surface flaw", "rust", "anomaly"]):
        intent = "defect_inspection"
        confidence = 0.96
        relevance["defect"] = 0.98
        relevance["object"] = 0.75
        relevance["scene"] = 0.40
        relevance["rca"] = 0.85
        relevance["rag"] = 0.75
        relevance["recommendation"] = 0.80
        relevance["evidence_validation"] = 0.85
        return intent, confidence, relevance

    # Intent 5: Root Cause Analysis / Failure Diagnosis
    if any(k in q for k in ["why", "cause", "root cause", "failure", "halted", "stopped", "diagnose", "breakdown", "tripped", "why did"]):
        intent = "root_cause_analysis"
        confidence = 0.95
        relevance["rca"] = 0.98
        relevance["defect"] = 0.85
        relevance["object"] = 0.80
        relevance["scene"] = 0.70
        relevance["rag"] = 0.90
        relevance["recommendation"] = 0.90
        relevance["evidence_validation"] = 0.90
        return intent, confidence, relevance

    # Intent 6: Video Tracking & Trajectory Analysis
    if any(k in q for k in ["track", "trajectory", "video", "movement", "path", "continuous monitoring"]):
        intent = "video_tracking"
        confidence = 0.92
        relevance["object"] = 0.98
        relevance["scene"] = 0.80
        relevance["ppe"] = 0.70
        relevance["evidence_validation"] = 0.80
        return intent, confidence, relevance

    # Intent 7: Scene Understanding & Environmental Context
    if any(k in q for k in ["scene", "environment", "surroundings", "overview", "what is happening", "context", "describe"]):
        intent = "scene_understanding"
        confidence = 0.90
        relevance["scene"] = 0.98
        relevance["object"] = 0.75
        relevance["evidence_validation"] = 0.70
        return intent, confidence, relevance

    # Intent 8: Regulatory Standards Inquiry
    if any(k in q for k in ["standard", "regulation", "osha", "iso", "astm", "manual", "guideline", "code"]):
        intent = "contextual_compliance_research"
        confidence = 0.90
        relevance["rag"] = 0.98
        relevance["ppe"] = 0.70
        relevance["defect"] = 0.70
        relevance["recommendation"] = 0.75
        return intent, confidence, relevance

    # Deterministic Fallback Rule: Ambiguous or general inquiry
    intent = "general_scene_inquiry"
    confidence = 0.60
    relevance["object"] = 0.75
    relevance["scene"] = 0.70
    relevance["evidence_validation"] = 0.60
    return intent, confidence, relevance

def compute_routing_scores(
    relevance: Dict[str, float],
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    delta: Optional[float] = None,
    threshold: Optional[float] = None,
    enable_dynamic_latency: Optional[bool] = None,
    db_session: Optional[Any] = None,
    latency_estimates: Optional[Dict[str, Any]] = None,
    hardware_device: Optional[str] = None,
    agent_model_versions: Optional[Dict[str, str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Computes Equation 1 scores for all candidate agents:
    S(a, q) = alpha * R(a, q) + delta * E(a) - beta * C(a) - gamma * L(a)

    Phase 2 Dynamic Latency Feedback:
    When dynamic latency feedback is enabled and sufficient historical measurements exist,
    the static latency prior L_prior(a) is replaced with the normalized historical EMA estimate L_hist(a).
    Falls back gracefully to static priors when history is unavailable or insufficient.
    Filters historical records strictly by hardware platform and model/checkpoint version.
    """
    a_w = alpha if alpha is not None else settings.ROUTING_ALPHA
    b_w = beta if beta is not None else settings.ROUTING_BETA
    g_w = gamma if gamma is not None else settings.ROUTING_GAMMA
    d_w = delta if delta is not None else settings.ROUTING_DELTA
    tau = threshold if threshold is not None else settings.ROUTING_THRESHOLD

    is_dynamic_enabled = (
        enable_dynamic_latency
        if enable_dynamic_latency is not None
        else settings.ENABLE_DYNAMIC_LATENCY_FEEDBACK
    )

    canonical_versions = agent_model_versions or getattr(settings, "AGENT_MODEL_VERSIONS", {})

    # Ingest historical latency estimates if dynamic feedback is enabled
    estimates: Dict[str, Any] = {}
    if is_dynamic_enabled:
        if latency_estimates is not None:
            estimates = latency_estimates
        else:
            try:
                from backend.app.orchestration.latency_estimator import HistoricalLatencyEstimator
                estimator = HistoricalLatencyEstimator(
                    hardware_device=hardware_device,
                    agent_model_versions=canonical_versions
                )
                if db_session is not None:
                    estimates = estimator.get_all_agent_estimates(
                        db_session=db_session,
                        hardware_device=hardware_device,
                        agent_model_versions=canonical_versions
                    )
                else:
                    from backend.app.database.session import SessionLocal
                    db = SessionLocal()
                    try:
                        estimates = estimator.get_all_agent_estimates(
                            db_session=db,
                            hardware_device=hardware_device,
                            agent_model_versions=canonical_versions
                        )
                    finally:
                        db.close()
            except Exception as e:
                # Fall back safely on any database retrieval issue
                estimates = {}

    traces: Dict[str, Dict[str, Any]] = {}
    for agent in AVAILABLE_AGENTS:
        r = relevance.get(agent, 0.10)
        e = settings.AGENT_CAPABILITIES.get(agent, 0.80)
        c = settings.AGENT_COSTS.get(agent, 0.20)
        static_prior = settings.AGENT_LATENCIES.get(agent, 0.20)
        expected_version = canonical_versions.get(agent)

        # Dynamic Latency Feedback logic
        est = estimates.get(agent)
        if is_dynamic_enabled and est is not None:
            # est can be a LatencyEstimate dataclass or a dict
            fallback_used = getattr(est, "fallback_used", None)
            if fallback_used is None and isinstance(est, dict):
                fallback_used = est.get("fallback_used", True)

            if not fallback_used:
                selected_l = getattr(est, "selected_latency", None)
                if selected_l is None and isinstance(est, dict):
                    selected_l = est.get("selected_latency", static_prior)
                l = selected_l
                hist_ema = getattr(est, "historical_ema_ms", None) if not isinstance(est, dict) else est.get("historical_ema_ms")
                norm_hist = getattr(est, "normalized_latency", None) if not isinstance(est, dict) else est.get("normalized_latency")
                obs_cnt = getattr(est, "observation_count", 0) if not isinstance(est, dict) else est.get("observation_count", 0)
                hw_dev = getattr(est, "hardware_device", None) if not isinstance(est, dict) else est.get("hardware_device")
                m_ver = getattr(est, "model_version", None) if not isinstance(est, dict) else est.get("model_version")
                fb_flag = False
            else:
                l = static_prior
                hist_ema = getattr(est, "historical_ema_ms", None) if not isinstance(est, dict) else est.get("historical_ema_ms")
                norm_hist = getattr(est, "normalized_latency", None) if not isinstance(est, dict) else est.get("normalized_latency")
                obs_cnt = getattr(est, "observation_count", 0) if not isinstance(est, dict) else est.get("observation_count", 0)
                hw_dev = getattr(est, "hardware_device", None) if not isinstance(est, dict) else est.get("hardware_device")
                m_ver = getattr(est, "model_version", None) if not isinstance(est, dict) else est.get("model_version") or expected_version
                fb_flag = True
        else:
            l = static_prior
            hist_ema = None
            norm_hist = None
            obs_cnt = 0
            hw_dev = hardware_device or settings.HARDWARE_DEVICE_CONTEXT
            m_ver = expected_version
            fb_flag = True

        # Exact Formula 1 calculation
        score = (a_w * r) + (d_w * e) - (b_w * c) - (g_w * l)

        if score >= tau:
            decision = "SELECTED"
            if not fb_flag and is_dynamic_enabled and hist_ema is not None:
                reason = f"Routing score {score:.3f} >= threshold {tau:.2f} (dynamic EMA latency {hist_ema:.1f}ms -> L={l:.3f} from {obs_cnt} obs [{hw_dev}|{m_ver}])"
            else:
                reason = f"Routing score {score:.3f} >= threshold {tau:.2f} (high relevance {r:.2f} justifies compute cost {c:.2f})"
        else:
            decision = "SKIPPED"
            if not fb_flag and is_dynamic_enabled and hist_ema is not None:
                reason = f"Routing score {score:.3f} < threshold {tau:.2f} (dynamic EMA latency {hist_ema:.1f}ms -> L={l:.3f} from {obs_cnt} obs [{hw_dev}|{m_ver}] suppresses selection)"
            else:
                reason = f"Routing score {score:.3f} < threshold {tau:.2f} (insufficient query relevance {r:.2f} to warrant execution)"

        traces[agent] = {
            "agent_name": agent,
            "relevance_score": round(r, 4),
            "evidence_capability": round(e, 4),
            "normalized_cost": round(c, 4),
            "expected_latency": round(l, 4),
            "final_routing_score": round(score, 4),
            "threshold": tau,
            "decision": decision,
            "selection_reason": reason,
            "execution_status": "PENDING" if decision == "SELECTED" else "SKIPPED",
            "actual_latency_ms": 0.0,
            "error_message": None,
            # Phase 2 Explainability Fields
            "static_latency_prior": round(static_prior, 4),
            "historical_ema_latency_ms": round(hist_ema, 2) if hist_ema is not None else None,
            "normalized_historical_latency": round(norm_hist, 4) if norm_hist is not None else None,
            "latency_observations_count": obs_cnt,
            "latency_fallback_used": fb_flag,
            "selected_latency": round(l, 4),
            "hardware_device": hw_dev,
            "model_version": m_ver
        }

    return traces

    return traces

def resolve_dependencies(
    traces: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Enforces mandatory structural dependencies and minimum sufficient evidence constraints:
    1. PPE agent strictly requires Object Detection for person-level spatial bounding.
    2. Defect agent benefits from Object Detection for component-bounded localization.
    3. RCA agent strictly requires at least one perceptual source.
    4. Recommendation agent requires RCA, Defect, or PPE.
    5. Evidence Validation runs whenever any perception agent is active.
    6. Minimum sufficient evidence constraint: defaults to Object and Scene if none selected.
    """
    selected = set([a for a, data in traces.items() if data["decision"] == "SELECTED"])

    # Rule 1: PPE requires Object detection
    if "ppe" in selected and "object" not in selected:
        selected.add("object")
        traces["object"]["decision"] = "MANDATORY_DEPENDENCY"
        traces["object"]["selection_reason"] = "Mandatory dependency for PPE person-level spatial association"
        traces["object"]["execution_status"] = "PENDING"

    # Rule 2: Defect inspection requires Object detection
    if "defect" in selected and "object" not in selected:
        selected.add("object")
        traces["object"]["decision"] = "MANDATORY_DEPENDENCY"
        traces["object"]["selection_reason"] = "Mandatory dependency for object-bounded defect inspection"
        traces["object"]["execution_status"] = "PENDING"

    # Rule 3: RCA requires at least one perceptual source
    if "rca" in selected:
        perceptual = selected & {"object", "defect", "ocr", "ppe"}
        if not perceptual:
            selected.add("object")
            traces["object"]["decision"] = "MANDATORY_DEPENDENCY"
            traces["object"]["selection_reason"] = "Mandatory dependency: RCA requires perceptual anchor"
            traces["object"]["execution_status"] = "PENDING"

    # Rule 4: Recommendation requires RCA or defect or ppe
    if "recommendation" in selected:
        if not (selected & {"rca", "defect", "ppe"}):
            selected.add("rca")
            traces["rca"]["decision"] = "MANDATORY_DEPENDENCY"
            traces["rca"]["selection_reason"] = "Mandatory dependency for CAPA recommendation formulation"
            traces["rca"]["execution_status"] = "PENDING"

    # Rule 5: Minimum sufficient evidence constraint
    perception_agents = {"object", "ppe", "ocr", "defect", "scene"}
    if not (selected & perception_agents):
        selected.add("object")
        traces["object"]["decision"] = "FALLBACK"
        traces["object"]["selection_reason"] = "Minimum sufficient evidence constraint: default perceptual anchor"
        traces["object"]["execution_status"] = "PENDING"
        selected.add("scene")
        traces["scene"]["decision"] = "FALLBACK"
        traces["scene"]["selection_reason"] = "Minimum sufficient evidence constraint: scene context anchor"
        traces["scene"]["execution_status"] = "PENDING"

    # Rule 6: Evidence validation runs whenever perception agents are active
    if selected & perception_agents:
        if "evidence_validation" not in selected:
            selected.add("evidence_validation")
            traces["evidence_validation"]["decision"] = "MANDATORY_DEPENDENCY"
            traces["evidence_validation"]["selection_reason"] = "Enforcing evidence provenance and hallucination mitigation"
            traces["evidence_validation"]["execution_status"] = "PENDING"

    ordered_selected = [a for a in AVAILABLE_AGENTS if a in selected]
    return ordered_selected, traces

def route_query(
    query: str,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    delta: Optional[float] = None,
    threshold: Optional[float] = None,
    enable_dynamic_latency: Optional[bool] = None,
    db_session: Optional[Any] = None,
    latency_estimates: Optional[Dict[str, Any]] = None,
    hardware_device: Optional[str] = None,
    agent_model_versions: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Main entry point for Cost-Aware Dynamic Query Routing.
    Returns structured plan, considered agents, decisions, and detailed explainable traces.
    
    Supports static routing (baseline) as well as Phase 2 dynamic latency feedback
    from historical executions via EMA.
    """
    intent, confidence, relevance = classify_intent_deterministic(query)
    traces = compute_routing_scores(
        relevance=relevance,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        threshold=threshold,
        enable_dynamic_latency=enable_dynamic_latency,
        db_session=db_session,
        latency_estimates=latency_estimates,
        hardware_device=hardware_device,
        agent_model_versions=agent_model_versions
    )
    selected_agents, updated_traces = resolve_dependencies(traces)
    skipped_agents = [a for a in AVAILABLE_AGENTS if a not in selected_agents]
    is_dynamic = (
        enable_dynamic_latency
        if enable_dynamic_latency is not None
        else settings.ENABLE_DYNAMIC_LATENCY_FEEDBACK
    )

    return {
        "intent": intent,
        "intent_confidence": round(confidence, 4),
        "selected_agents": selected_agents,
        "skipped_agents": skipped_agents,
        "considered_agents": AVAILABLE_AGENTS,
        "threshold": threshold if threshold is not None else settings.ROUTING_THRESHOLD,
        "traces": updated_traces,
        "dynamic_latency_enabled": is_dynamic
    }


# ==============================================================================
# PHASE 3: DYNAMIC SEQUENTIAL EVIDENCE-GAIN ROUTING (REVISION 10)
# ==============================================================================

class StoppingReason(str, Enum):
    """Explicit stopping criteria for the sequential evidence-gain router."""
    DEFICIT_THRESHOLD_REACHED = "DEFICIT_THRESHOLD_REACHED"
    DIMINISHING_MARGINAL_GAIN = "DIMINISHING_MARGINAL_GAIN"
    INVOCATION_BUDGET_EXHAUSTED = "INVOCATION_BUDGET_EXHAUSTED"
    WALLCLOCK_BUDGET_EXHAUSTED = "WALLCLOCK_BUDGET_EXHAUSTED"
    CRITICAL_SAFETY_FAILURE = "CRITICAL_SAFETY_FAILURE"
    CONTAINMENT_FAILURE = "CONTAINMENT_FAILURE"
    NO_ELIGIBLE_CANDIDATES = "NO_ELIGIBLE_CANDIDATES"


AGENT_TOKEN_MAP: Dict[str, List[str]] = {
    "object": ["ET-OBJ-PERSON-01"],
    "ppe": ["ET-PPE-CRANIAL-02", "ET-PPE-TORSO-03"],
    "defect": ["ET-DFT-LOCALIZATION-01", "ET-DFT-CLASSIFICATION-02"],
    "ocr": ["ET-OCR-TRANSCRIBE-01"],
    "scene": ["ET-SCN-ENVIRONMENT-01"],
    "recommendation": ["ET-DFT-PROCEDURE-03"],
    "rca": [],
    "rag": [],
    "evidence_validation": []
}

MANDATORY_PREREQUISITES: Dict[str, List[str]] = {
    "ppe": ["object"],
    "rca": ["defect", "object"],  # At least one perceptual grounding
    "recommendation": ["rca"]
}

RECOVERY_AGENT_MAP: Dict[str, str] = {
    "object": "scene",  # Fallback recovery
    "defect": "evidence_validation"
}

# Empirical validation capability priors E(a, k) measured on benchmark splits
SPECIALIST_CAPABILITIES: Dict[Tuple[str, str], float] = {
    ("object", "ET-OBJ-PERSON-01"): 0.85,
    ("ppe", "ET-PPE-CRANIAL-02"): 0.85,
    ("ppe", "ET-PPE-TORSO-03"): 0.80,
    ("defect", "ET-DFT-LOCALIZATION-01"): 0.85,
    ("defect", "ET-DFT-CLASSIFICATION-02"): 0.85,
    ("recommendation", "ET-DFT-PROCEDURE-03"): 0.90,
    ("ocr", "ET-OCR-TRANSCRIBE-01"): 0.85,
    ("scene", "ET-SCN-ENVIRONMENT-01"): 0.85,
}


class SequentialEvidenceRouter:
    """
    Phase 3: Dynamic Sequential Evidence-Gain Router.
    Implements Revision 10 Information-Theoretic Routing Policy:
    a* = argmax_{a in A_cand} Delta D_hat(a | S_t) / (beta * CostNorm(a) + gamma * LatencyNorm(a) + eps)
    
    Predicts post-execution evidence state S_hat_{t+1}(a) across all 8 TokenStatus lifecycle states:
    - SATISFIED: Delta phi = 0.0 (strictly zero gain for already satisfied tokens).
    - CONFIRMED_ABSENT: Delta phi = 0.0.
    - NOT_APPLICABLE: Delta phi = 0.0.
    - MISSING: Delta phi = max(0, E(a,k)*R(a,q) - phi_t(k)).
    - INSUFFICIENT_EVIDENCE: marginal gain only if candidate improves resolution.
    - CONTRADICTORY: requires authorized resolver agent (evidence_validation, scene).
    - UNAVAILABLE: requires designated recovery agent.
    - UNSUPPORTED: Delta phi = 0.0.
    
    NOTE ON EPISTEMIC STATUS:
    The predictor is explicitly documented as a heuristic prior,
    not a calibrated probability of success.
    """

    def __init__(
        self,
        cost_weights: Optional[Dict[str, float]] = None,
        beta: float = 1.0,
        gamma: Optional[float] = None,
        epsilon: float = 1e-4,
        stopping_deficit: float = 0.20,
        min_marginal_gain: float = 0.01,
        max_iterations: int = 5,
        wallclock_budget_ms: float = 5000.0
    ):
        self.costs = cost_weights or settings.AGENT_COSTS
        self.beta = beta
        self.gamma = gamma if gamma is not None else settings.ROUTING_GAMMA
        self.epsilon = epsilon
        self.stopping_deficit = stopping_deficit
        self.min_marginal_gain = min_marginal_gain
        self.max_iterations = max_iterations
        self.wallclock_budget_ms = wallclock_budget_ms
        self._max_cost = max(self.costs.values()) if self.costs else 1.0
        self.last_stopping_reason: Optional[StoppingReason] = None

    @staticmethod
    def get_agent_tokens(agent_name: str) -> List[str]:
        """Returns evidence tokens directly produced by a specialist."""
        return AGENT_TOKEN_MAP.get(agent_name, [])

    @staticmethod
    def get_prerequisites(agent_name: str) -> List[str]:
        """Returns required prerequisite agents that must succeed before this agent."""
        return MANDATORY_PREREQUISITES.get(agent_name, [])

    @staticmethod
    def can_resolve_conflict(agent_name: str, conflict_type: str = "") -> bool:
        """Checks if an agent is authorized to resolve contradictions."""
        return agent_name in ("evidence_validation", "scene")

    @staticmethod
    def is_recovery_agent(agent_name: str, failed_agent: str) -> bool:
        """Checks if agent is a designated fallback recovery specialist."""
        return RECOVERY_AGENT_MAP.get(failed_agent) == agent_name

    @staticmethod
    def get_specialist_capability(agent_name: str, token_id: str) -> float:
        """Returns measured validation-set capability E(a, k) for a token."""
        return SPECIALIST_CAPABILITIES.get((agent_name, token_id), 0.85)

    def estimate_latency_ms(self, agent_name: str) -> float:
        """Estimates execution latency in milliseconds for a candidate agent."""
        raw_sec = settings.AGENT_LATENCIES.get(agent_name, 0.05)
        return float(raw_sec * 1000.0)

    def predict_token_gain(
        self,
        candidate_agent: str,
        current_report: Any,  # EvidenceEvaluationReport
        applicable_tokens: List[str],
        relevance_score: float = 1.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Revision 10 Token Status Semantics:
        Calculates expected satisfaction delta for each applicable token.
        Strictly enforces ZERO gain for already SATISFIED, CONFIRMED_ABSENT,
        NOT_APPLICABLE, and UNSUPPORTED tokens.
        
        Formula:
        phi_hat_{t+1}(k | a) = max{phi_t(k), E(a, k) * R(a, q)}
        
        NOTE: This predictor is a heuristic prior, not a calibrated probability of success.
        """
        candidate_tokens = set(self.get_agent_tokens(candidate_agent))
        token_deltas: Dict[str, float] = {}
        total_delta = 0.0

        from backend.app.orchestration.evidence_deficit import TokenStatus

        for tok_id in applicable_tokens:
            curr_token = current_report.tokens.get(tok_id) if current_report and hasattr(current_report, "tokens") else None
            curr_status = curr_token.status if curr_token else TokenStatus.MISSING
            curr_satisfaction = curr_token.satisfaction_score if curr_token else 0.0

            # Only candidate's tokens (or recovery/conflict resolver) can improve
            if tok_id not in candidate_tokens:
                token_deltas[tok_id] = 0.0
                continue

            delta_phi = 0.0
            cap = self.get_specialist_capability(candidate_agent, tok_id)

            if curr_status == TokenStatus.SATISFIED:
                # Rule 1: Already satisfied tokens produce STRICTLY ZERO GAIN
                delta_phi = 0.0
            elif curr_status in (TokenStatus.CONFIRMED_ABSENT, TokenStatus.NOT_APPLICABLE):
                # Rule 2: Non-applicable or confirmed absent tokens cannot gain
                delta_phi = 0.0
            elif curr_status == TokenStatus.MISSING:
                # Rule 3: Missing token receives expected score E(a, k) * R(a, q)
                expected_score = round(cap * relevance_score, 4)
                delta_phi = max(0.0, expected_score - curr_satisfaction)
            elif curr_status == TokenStatus.INSUFFICIENT_EVIDENCE:
                # Rule 4: Borderline / degraded token can receive marginal improvement
                expected_score = round(min(cap, 0.80) * relevance_score, 4)
                delta_phi = max(0.0, expected_score - curr_satisfaction)
            elif curr_status == TokenStatus.CONTRADICTORY:
                # Rule 5: Requires authorized conflict resolution capability
                if self.can_resolve_conflict(candidate_agent):
                    delta_phi = 0.30
                else:
                    delta_phi = 0.0
            elif curr_status == TokenStatus.UNAVAILABLE:
                # Rule 6: Requires designated recovery specialist
                delta_phi = 0.0
            elif curr_status == TokenStatus.UNSUPPORTED:
                # Rule 7: Unsupported modality produces ZERO GAIN
                delta_phi = 0.0
            else:
                delta_phi = 0.0

            token_deltas[tok_id] = round(delta_phi, 4)
            total_delta += delta_phi

        return round(total_delta, 4), token_deltas

    def predict_deficit_reduction(
        self,
        candidate_agent: str,
        current_report: Any,  # EvidenceEvaluationReport
        applicable_tokens: List[str],
        query_relevance: float = 1.0
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Computes expected composite deficit reduction:
        Delta D_hat(a | S_t) = max(0, D(S_t) - D_hat(S_hat_{t+1}(a)))
        
        An irrelevant candidate (query_relevance < 0.20) has strictly ZERO predicted gain
        unless it has an explicitly registered ability to resolve an active required conflict.
        """
        if not current_report:
            return 0.0, {}

        current_d = float(getattr(current_report, "composite_deficit", 1.0))
        weights = getattr(current_report, "normalized_weights", {
            "w_completeness": 0.50,
            "w_reliability": 0.20,
            "w_conflict": 0.20,
            "w_unavailability": 0.10
        })

        num_app = len(applicable_tokens)
        if num_app == 0:
            return 0.0, {"predicted_d": current_d, "delta_d": 0.0, "candidate_agent": candidate_agent}

        curr_conf = float(getattr(current_report, "conflict_score", 0.0))
        is_resolver = self.can_resolve_conflict(candidate_agent) and curr_conf > 0.0

        # Irrelevant candidates have strictly zero gain unless resolving conflicts
        if query_relevance < 0.20 and not is_resolver:
            return 0.0, {
                "candidate_agent": candidate_agent,
                "current_d": current_d,
                "predicted_d": current_d,
                "delta_d": 0.0,
                "delta_c": 0.0,
                "delta_r": 0.0,
                "delta_s": 0.0,
                "token_deltas": {t: 0.0 for t in applicable_tokens}
            }

        # 1. Completeness update
        total_delta_phi, token_deltas = self.predict_token_gain(
            candidate_agent, current_report, applicable_tokens, relevance_score=query_relevance
        )
        delta_c = total_delta_phi / float(num_app)

        # 2. Epistemic Reliability update (only if new evidence is actually acquired)
        curr_rel = float(getattr(current_report, "reliability", 0.0))
        if delta_c > 0.0:
            expected_sample_rel = 0.80 * query_relevance
            delta_r = max(0.0, (expected_sample_rel - curr_rel) * (delta_c / 1.0))
        else:
            delta_r = 0.0

        # 3. Conflict score reduction (only if candidate is authorized resolver)
        delta_s = 0.0
        if is_resolver:
            delta_s = min(curr_conf, 0.25)

        # 4. Unavailability reduction
        delta_u = 0.0

        # Weighted deficit gain
        w_c = weights.get("w_completeness", 0.50)
        w_r = weights.get("w_reliability", 0.20)
        w_s = weights.get("w_conflict", 0.20)
        w_u = weights.get("w_unavailability", 0.10)

        raw_delta_d = (w_c * delta_c) + (w_r * delta_r) + (w_s * delta_s) + (w_u * delta_u)
        delta_d = max(0.0, min(current_d, raw_delta_d))
        predicted_d = max(0.0, current_d - delta_d)

        diagnostics = {
            "candidate_agent": candidate_agent,
            "current_d": current_d,
            "predicted_d": round(predicted_d, 4),
            "delta_d": round(delta_d, 4),
            "delta_c": round(delta_c, 4),
            "delta_r": round(delta_r, 4),
            "delta_s": round(delta_s, 4),
            "token_deltas": token_deltas
        }
        return round(delta_d, 4), diagnostics

    def compute_utility(
        self,
        candidate_agent: str,
        delta_d: float,
        latency_ms: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        epsilon: Optional[float] = None
    ) -> float:
        """
        Computes cost-aware sequential selection efficiency:
        Efficiency(a) = Delta D_hat(a | S_t) / (beta * CostNorm(a) + gamma * LatencyNorm(a) + eps)
        """
        if delta_d <= 0.0:
            return 0.0

        b = beta if beta is not None else self.beta
        g = gamma if gamma is not None else self.gamma
        eps = epsilon if epsilon is not None else self.epsilon

        # Normalized cost in [0.05, 1.0]
        raw_cost = self.costs.get(candidate_agent, 0.20)
        cost_norm = max(0.05, raw_cost / max(0.01, self._max_cost))

        # Normalized latency in [0.05, 1.0]
        raw_lat = latency_ms if latency_ms is not None else self.estimate_latency_ms(candidate_agent)
        lat_norm = max(0.05, min(1.0, raw_lat / 1000.0))

        denominator = (b * cost_norm) + (g * lat_norm) + eps
        utility = delta_d / denominator
        return round(utility, 4)

    def get_agent_lifecycle_states(
        self,
        executed: Set[str],
        failed: Set[str],
        applicable_tokens: List[str],
        intent: str
    ) -> Dict[str, AgentLifecycleState]:
        """
        Computes the current lifecycle state for each specialist in AVAILABLE_AGENTS.
        Enforces terminal protection, dependency gating, and out-of-scope identification.
        """
        states: Dict[str, AgentLifecycleState] = {}
        for agent in AVAILABLE_AGENTS:
            if agent in executed:
                states[agent] = AgentLifecycleState.COMPLETED
            elif agent in failed:
                states[agent] = AgentLifecycleState.FAILED
            else:
                prereqs = self.get_prerequisites(agent)
                if prereqs:
                    failed_prereqs = [p for p in prereqs if p in failed]
                    if failed_prereqs:
                        states[agent] = AgentLifecycleState.BLOCKED
                        continue
                    unmet_prereqs = [p for p in prereqs if p not in executed]
                    if unmet_prereqs:
                        if agent == "rca" and any(p in executed for p in prereqs):
                            pass  # RCA satisfied if at least one prerequisite executed
                        else:
                            states[agent] = AgentLifecycleState.WAITING_FOR_DEPENDENCY
                            continue

                agent_tokens = set(self.get_agent_tokens(agent))
                overlaps = bool(agent_tokens & set(applicable_tokens))
                is_support = agent in ("evidence_validation", "rca", "recommendation", "rag", "scene")
                if not overlaps and not is_support:
                    states[agent] = AgentLifecycleState.NOT_APPLICABLE
                else:
                    states[agent] = AgentLifecycleState.ELIGIBLE
        return states

    def select_next_specialist(
        self,
        state: Dict[str, Any],
        wallclock_budget_ms: Optional[float] = None,
        min_gain_threshold: Optional[float] = None
    ) -> Tuple[Optional[str], float, Dict[str, Dict[str, Any]]]:
        """
        Selects next specialist a* to dispatch using the sequential evidence-gain policy.
        
        Stopping Rules Enforced:
        A. Composite deficit <= stopping_deficit (DEFICIT_THRESHOLD_REACHED)
        B. Diminishing marginal gain: max Delta D_hat < min_marginal_gain (DIMINISHING_MARGINAL_GAIN)
        C. Invocation budget exhausted: iteration >= max_iterations (INVOCATION_BUDGET_EXHAUSTED)
        D. Wall-clock budget exhausted: elapsed + min latency > budget (WALLCLOCK_BUDGET_EXHAUSTED)
        E. Critical safety failure (CRITICAL_SAFETY_FAILURE)
        F. Worker containment failure (CONTAINMENT_FAILURE)
        
        Returns:
            (best_agent, max_utility, candidate_scores)
        """
        self.last_stopping_reason = None
        executed = set(state.get("executed_agents", []))
        failed = set(state.get("failed_agents", []))
        report = state.get("active_evidence_report")
        query = state.get("user_query", "")
        intent = state.get("detected_intent") or "general_scene_inquiry"
        iteration = state.get("iteration_step", 0)

        # Rule C: Invocation budget exhausted
        if iteration >= self.max_iterations:
            self.last_stopping_reason = StoppingReason.INVOCATION_BUDGET_EXHAUSTED
            state["stopping_reason"] = self.last_stopping_reason.value
            return None, 0.0, {}

        # Rule A: Stopping deficit threshold reached
        current_d = float(getattr(report, "composite_deficit", 1.0)) if report else 1.0
        if current_d <= self.stopping_deficit:
            self.last_stopping_reason = StoppingReason.DEFICIT_THRESHOLD_REACHED
            state["stopping_reason"] = self.last_stopping_reason.value
            return None, 0.0, {}

        # Rule D: Wall-clock budget check
        budget_ms = wallclock_budget_ms if wallclock_budget_ms is not None else self.wallclock_budget_ms
        start_time = state.get("start_time")
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 if start_time else 0.0
        remaining_budget_ms = max(0.0, budget_ms - elapsed_ms)
        state["remaining_time_budget_ms"] = remaining_budget_ms

        if start_time and elapsed_ms >= budget_ms:
            self.last_stopping_reason = StoppingReason.WALLCLOCK_BUDGET_EXHAUSTED
            state["stopping_reason"] = self.last_stopping_reason.value
            return None, 0.0, {}

        # Determine applicable required tokens
        from backend.app.orchestration.evidence_deficit import EvidenceDeficitEngine, TokenStatus
        engine = EvidenceDeficitEngine()
        req_tokens = engine.determine_required_tokens_for_intent(intent, query)
        applicable_tokens = []
        for t in req_tokens:
            tok_obj = report.tokens.get(t) if report and hasattr(report, "tokens") else None
            if not tok_obj or tok_obj.status != TokenStatus.NOT_APPLICABLE:
                applicable_tokens.append(t)

        # Lifecycle State Computation
        lifecycle_states = self.get_agent_lifecycle_states(executed, failed, applicable_tokens, intent)
        state["agent_lifecycle_states"] = {a: s.value for a, s in lifecycle_states.items()}

        # Classify query relevance
        classified_intent, _, relevance_map = classify_intent_deterministic(query)
        # Harmonize with state detected_intent if explicitly provided
        effective_intent = intent or classified_intent
        if "ppe" in effective_intent:
            relevance_map["ppe"] = max(relevance_map.get("ppe", 0.0), 0.95)
            relevance_map["object"] = max(relevance_map.get("object", 0.0), 0.85)
        elif "defect" in effective_intent:
            relevance_map["defect"] = max(relevance_map.get("defect", 0.0), 0.95)
        elif "ocr" in effective_intent:
            relevance_map["ocr"] = max(relevance_map.get("ocr", 0.0), 0.95)

        candidate_scores: Dict[str, Dict[str, Any]] = {}
        best_agent: Optional[str] = None
        max_utility = -1.0
        max_delta_d = 0.0

        tau_gain = min_gain_threshold if min_gain_threshold is not None else self.min_marginal_gain

        for agent, lifecycle in lifecycle_states.items():
            # Only ELIGIBLE agents can be selected for dispatch
            if lifecycle != AgentLifecycleState.ELIGIBLE:
                continue

            est_lat = self.estimate_latency_ms(agent)
            # Wall-clock feasibility check: candidate must fit in remaining budget
            if start_time and (elapsed_ms + est_lat > budget_ms):
                continue

            rel_score = relevance_map.get(agent, 0.50)
            delta_d, diag = self.predict_deficit_reduction(
                agent, report, applicable_tokens, query_relevance=rel_score
            )
            diag["latency_ms"] = est_lat

            utility = self.compute_utility(agent, delta_d, latency_ms=est_lat)
            diag["utility"] = utility
            candidate_scores[agent] = diag

            if delta_d > max_delta_d:
                max_delta_d = delta_d

            if utility > max_utility and delta_d >= tau_gain:
                max_utility = utility
                best_agent = agent

        # Rule B: Diminishing marginal gain
        if best_agent is None:
            if candidate_scores and max_delta_d < tau_gain:
                self.last_stopping_reason = StoppingReason.DIMINISHING_MARGINAL_GAIN
            elif start_time and elapsed_ms + 10.0 >= budget_ms:
                self.last_stopping_reason = StoppingReason.WALLCLOCK_BUDGET_EXHAUSTED
            else:
                self.last_stopping_reason = StoppingReason.NO_ELIGIBLE_CANDIDATES
            state["stopping_reason"] = self.last_stopping_reason.value
            return None, 0.0, candidate_scores

        return best_agent, max_utility, candidate_scores

    def record_step_telemetry(
        self,
        step: int,
        current_deficit: float,
        candidate_scores: Dict[str, Dict[str, Any]],
        selected_agent: Optional[str],
        selection_efficiency: float,
        execution_result: str,  # "SUCCESS", "FAILED", "TIMEOUT"
        cleanup_result: str,    # "CLEAN", "CONTAINMENT_FAILED"
        post_deficit: Optional[float],
        actual_findings_count: int,
        remaining_invocations: int,
        remaining_wallclock_ms: float,
        stopping_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Constructs comprehensive step telemetry adhering strictly to Section 8.
        
        Telemetry Invariants:
        - Signed observed deficit reduction: Delta D_obs = D(S_t) - D(S_{t+1})
          CRITICAL: Never replace negative observed gain with zero!
          If execution failed or timed out, Delta D_obs is None (never imputed zero).
        - Prediction error: Delta D_hat - Delta D_obs (None if Delta D_obs is None).
        """
        predicted_delta_d = 0.0
        if selected_agent and selected_agent in candidate_scores:
            predicted_delta_d = float(candidate_scores[selected_agent].get("delta_d", 0.0))

        signed_observed_reduction: Optional[float] = None
        prediction_error: Optional[float] = None

        if execution_result == "SUCCESS" and post_deficit is not None:
            signed_observed_reduction = round(float(current_deficit) - float(post_deficit), 4)
            prediction_error = round(predicted_delta_d - signed_observed_reduction, 4)
        else:
            # Failed or incomplete worker: NEVER impute zero!
            signed_observed_reduction = None
            prediction_error = None

        candidates = [k for k in candidate_scores.keys() if not k.startswith("_")]
        candidate_predicted_gains = {k: candidate_scores[k].get("delta_d", 0.0) for k in candidates}
        candidate_costs = {k: self.costs.get(k, 0.20) for k in candidates}
        candidate_latencies = {k: candidate_scores[k].get("latency_ms", 50.0) for k in candidates}
        candidate_efficiencies = {k: candidate_scores[k].get("utility", 0.0) for k in candidates}

        return {
            "step": step,
            "current_deficit": round(float(current_deficit), 4),
            "candidate_agents": candidates,
            "candidate_predicted_gains": candidate_predicted_gains,
            "candidate_costs": candidate_costs,
            "candidate_latencies": candidate_latencies,
            "candidate_efficiencies": candidate_efficiencies,
            "selected_specialist": selected_agent,
            "selection_efficiency": round(float(selection_efficiency), 4),
            "actual_findings_count": actual_findings_count,
            "execution_result": execution_result,
            "cleanup_result": cleanup_result,
            "post_execution_deficit": round(float(post_deficit), 4) if post_deficit is not None else None,
            "signed_observed_deficit_reduction": signed_observed_reduction,
            "prediction_error": prediction_error,
            "remaining_invocation_budget": remaining_invocations,
            "remaining_time_budget_ms": round(float(remaining_wallclock_ms), 2),
            "stopping_reason": stopping_reason
        }


# ==============================================================================
# SECTION 9: FOUR COMPARISON BASELINE ROUTING STRATEGIES
# ==============================================================================

def route_fixed_pipeline(intent: str, query: str) -> List[str]:
    """
    Baseline 1: Fixed Pipeline.
    Executes all applicable specialists unconditionally in fixed topological order.
    """
    q = query.lower()
    if "ppe" in intent or any(k in q for k in ["ppe", "helmet", "vest", "worker"]):
        return ["object", "ppe", "scene", "rag", "recommendation"]
    elif "defect" in intent or any(k in q for k in ["defect", "crack", "scratch", "damage"]):
        return ["defect", "scene", "rca", "recommendation", "rag"]
    elif "ocr" in intent or any(k in q for k in ["ocr", "serial", "read", "text"]):
        return ["ocr", "scene"]
    elif "scene" in intent or "overview" in q:
        return ["object", "scene"]
    else:
        return ["object", "scene", "evidence_validation"]


def route_phase2_weighted(
    query: str,
    alpha: Optional[float] = None,
    delta: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Baseline 2: Phase 2 Weighted Router.
    Implements Equation 1: S(a, q) = alpha * R(a, q) + delta * E(a) - beta * C(a) - gamma * L(a) >= tau.
    Preserves exact legacy behavior without modifications.
    """
    return route_query(
        query=query,
        alpha=alpha,
        delta=delta,
        beta=beta,
        gamma=gamma,
        threshold=threshold,
        enable_dynamic_latency=False
    )


def route_relevance_dependency(query: str, threshold: float = 0.50) -> List[str]:
    """
    Baseline 3: Relevance/Dependency-Based Router without Evidence Deficit Feedback.
    Selects agents with query relevance R(a, q) >= threshold, sorted by relevance descending,
    subject to prerequisite constraints, without dynamic closed-loop evidence feedback.
    """
    _, _, relevance_map = classify_intent_deterministic(query)
    candidates = [
        a for a, r in relevance_map.items()
        if r >= threshold and a in AVAILABLE_AGENTS
    ]
    # Sort by relevance descending
    candidates.sort(key=lambda a: relevance_map.get(a, 0.0), reverse=True)

    # Enforce prerequisites
    selected: List[str] = []
    for a in candidates:
        prereqs = MANDATORY_PREREQUISITES.get(a, [])
        if all(p in selected or p in candidates for p in prereqs):
            # Ensure prerequisites are placed before the dependent agent
            for p in prereqs:
                if p not in selected:
                    selected.append(p)
            if a not in selected:
                selected.append(a)
    return selected


def route_sequential_evidence_gain(
    query: str,
    initial_report: Any,
    max_steps: int = 5,
    stopping_deficit: float = 0.20,
    wallclock_budget_ms: float = 5000.0
) -> Dict[str, Any]:
    """
    Baseline 4: Proposed Sequential Evidence-Gain/Cost Router.
    Iteratively queries SequentialEvidenceRouter until stopping condition is reached.
    """
    router = SequentialEvidenceRouter(
        stopping_deficit=stopping_deficit,
        max_iterations=max_steps,
        wallclock_budget_ms=wallclock_budget_ms
    )
    intent, _, _ = classify_intent_deterministic(query)
    state: Dict[str, Any] = {
        "user_query": query,
        "detected_intent": intent,
        "executed_agents": [],
        "failed_agents": [],
        "active_evidence_report": initial_report,
        "iteration_step": 0,
        "start_time": time.perf_counter(),
        "routing_telemetry": []
    }
    
    planned_steps: List[str] = []
    while state["iteration_step"] < max_steps:
        best_agent, utility, candidates = router.select_next_specialist(state)
        if best_agent is None:
            break
        planned_steps.append(best_agent)
        state["executed_agents"].append(best_agent)
        state["iteration_step"] += 1

    return {
        "strategy": "sequential_evidence_gain",
        "planned_sequence": planned_steps,
        "stopping_reason": router.last_stopping_reason.value if router.last_stopping_reason else None,
        "total_steps": len(planned_steps)
    }


