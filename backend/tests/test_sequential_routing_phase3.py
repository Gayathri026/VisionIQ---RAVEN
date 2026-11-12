"""
VisionIQ Phase 3 Dedicated Test Suite:
Sequential Evidence-Gain Routing and RAVEN Arbitration (Revision 10)

Covers:
1. Token status prediction (zero gain on satisfied/non-applicable tokens).
2. Missing and insufficient evidence satisfaction gains.
3. Conflict and recovery resolution eligibility.
4. Composite deficit reduction calculation.
5. Utility scoring cost/latency trade-off.
6. Mandatory prerequisite enforcement (e.g. PPE requires object).
7. Terminal agent protection (no re-execution).
8. Sequential loop stopping conditions (deficit threshold & max iterations).
9. RAVEN safety gate: unvalidated physical-absence veto.
10. RAVEN safety gate: critical conflict veto.
11. End-to-end cyclic LangGraph StateGraph execution.
12. Monotonic deficit reduction trajectory.
13. FastAPI /inspect/sequential endpoint verification.
14. Error handling and fault isolation in sequential dispatch.
"""

import os
import sys
import io
import time
from pathlib import Path
import pytest
import numpy as np
import cv2

# Bootstrap sys.path to ensure backend.app resolves from any test runner root
_cur = Path(__file__).resolve().parent
while _cur.parent != _cur:
    if (_cur / "visioniq").exists():
        if str(_cur / "visioniq") not in sys.path:
            sys.path.insert(0, str(_cur / "visioniq"))
        break
    elif (_cur / "backend").exists() and (_cur / "backend" / "app").exists():
        if str(_cur) not in sys.path:
            sys.path.insert(0, str(_cur))
        break
    _cur = _cur.parent

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.orchestration.evidence_deficit import (
    EvidenceDeficitEngine,
    TokenStatus,
    TOKEN_CATALOG,
    EvaluatedEvidenceToken,
    EvidenceEvaluationReport,
    DeficitWeights
)
from backend.app.orchestration.state import (
    AgentLifecycleState,
    validate_lifecycle_transition
)
from backend.app.orchestration.router import (
    SequentialEvidenceRouter,
    StoppingReason,
    AGENT_TOKEN_MAP,
    MANDATORY_PREREQUISITES,
    route_fixed_pipeline,
    route_phase2_weighted,
    route_relevance_dependency,
    route_sequential_evidence_gain
)
from backend.app.orchestration.graph import (
    sequential_vision_iq_app,
    build_sequential_visioniq_graph
)
from backend.app.raven.engine import RavenEngine
from backend.app.raven.schemas import RavenDecision

client = TestClient(app)


def create_test_image(filename="test_seq_routing.jpg", w=200, h=200, color=(128, 128, 128)) -> str:
    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
    path = os.path.join(settings.UPLOADS_DIR, filename)
    img = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


# ==============================================================================
# 1. TOKEN STATUS PREDICTION & SEMANTICS (REVISION 10)
# ==============================================================================

def test_01_token_status_prediction_zero_gain_on_satisfied():
    """Revision 10: Already SATISFIED tokens must produce strictly ZERO GAIN."""
    engine = EvidenceDeficitEngine()
    router = SequentialEvidenceRouter()

    # Create a report where ET-OBJ-PERSON-01 is already SATISFIED
    report = engine.evaluate(
        query="inspect workers for helmets",
        intent="ppe_safety_compliance",
        findings=[
            {
                "finding_uid": "f_obj_01",
                "agent_source": "object",
                "label": "person",
                "bbox": [0.1, 0.1, 0.5, 0.9],
                "detection_confidence": 0.95
            }
        ],
        context_data={"object_result": {"objects": [{"label": "person", "confidence": 0.95}]}},
        executed_agents=["object"]
    )
    assert report.tokens["ET-OBJ-PERSON-01"].status == TokenStatus.SATISFIED

    # Candidate 'object' agent must predict 0.0 gain for ET-OBJ-PERSON-01
    total_delta, deltas = router.predict_token_gain(
        candidate_agent="object",
        current_report=report,
        applicable_tokens=["ET-OBJ-PERSON-01", "ET-PPE-CRANIAL-02", "ET-PPE-TORSO-03"]
    )
    assert deltas["ET-OBJ-PERSON-01"] == 0.0
    assert total_delta == 0.0


def test_02_token_status_prediction_missing_and_insufficient():
    """Revision 10: MISSING tokens receive expected satisfaction score; INSUFFICIENT yields marginal gain."""
    engine = EvidenceDeficitEngine()
    router = SequentialEvidenceRouter()

    # Initial report with all tokens MISSING
    report = engine.evaluate(
        query="inspect workers for helmets",
        intent="ppe_safety_compliance",
        findings=[],
        context_data={},
        executed_agents=[]
    )
    assert report.tokens["ET-OBJ-PERSON-01"].status == TokenStatus.MISSING

    # Object agent predicts expected score 0.85
    total_delta, deltas = router.predict_token_gain(
        candidate_agent="object",
        current_report=report,
        applicable_tokens=["ET-OBJ-PERSON-01", "ET-PPE-CRANIAL-02"]
    )
    assert deltas["ET-OBJ-PERSON-01"] == pytest.approx(0.85, 0.01)
    assert total_delta == pytest.approx(0.85, 0.01)


def test_03_token_status_prediction_non_applicable_and_absent():
    """Revision 10: NOT_APPLICABLE and CONFIRMED_ABSENT tokens produce ZERO GAIN."""
    router = SequentialEvidenceRouter()
    engine = EvidenceDeficitEngine()

    # Manually configure report with CONFIRMED_ABSENT token
    report = engine.evaluate(
        query="inspect workers",
        intent="ppe_safety_compliance",
        findings=[],
        context_data={},
        executed_agents=[]
    )
    report.tokens["ET-OBJ-PERSON-01"].status = TokenStatus.CONFIRMED_ABSENT

    total_delta, deltas = router.predict_token_gain(
        candidate_agent="object",
        current_report=report,
        applicable_tokens=["ET-OBJ-PERSON-01"]
    )
    assert deltas["ET-OBJ-PERSON-01"] == 0.0
    assert total_delta == 0.0


def test_04_conflict_resolution_eligibility():
    """Revision 10: Conflict reduction is only granted to authorized conflict resolvers."""
    router = SequentialEvidenceRouter()
    assert router.can_resolve_conflict("evidence_validation") is True
    assert router.can_resolve_conflict("scene") is True
    assert router.can_resolve_conflict("object") is False
    assert router.can_resolve_conflict("defect") is False


# ==============================================================================
# 2. DEFICIT REDUCTION & UTILITY COMPUTATION
# ==============================================================================

def test_05_predictive_deficit_drop_calculation():
    """Verifies Delta D_hat is positive for relevant specialists and zero for irrelevant specialists."""
    engine = EvidenceDeficitEngine()
    router = SequentialEvidenceRouter()

    report = engine.evaluate(
        query="inspect steel surface flaws and cracks",
        intent="defect_inspection",
        findings=[],
        context_data={},
        executed_agents=[]
    )
    initial_d = report.composite_deficit

    # Defect agent should predict positive deficit reduction
    delta_d_defect, diag_defect = router.predict_deficit_reduction(
        candidate_agent="defect",
        current_report=report,
        applicable_tokens=["ET-DFT-LOCALIZATION-01", "ET-DFT-CLASSIFICATION-02", "ET-SCN-ENVIRONMENT-01"]
    )
    assert delta_d_defect > 0.10
    assert diag_defect["predicted_d"] < initial_d

    # OCR agent provides zero tokens for a defect inspection query
    delta_d_ocr, diag_ocr = router.predict_deficit_reduction(
        candidate_agent="ocr",
        current_report=report,
        applicable_tokens=["ET-DFT-LOCALIZATION-01", "ET-DFT-CLASSIFICATION-02"]
    )
    assert delta_d_ocr == 0.0
    assert diag_ocr["predicted_d"] == pytest.approx(initial_d, 0.01)


def test_06_utility_cost_latency_tradeoff():
    """Higher cost or latency reduces utility score for the same deficit reduction."""
    router = SequentialEvidenceRouter()
    delta_d = 0.30

    # Low cost agent (scene: cost 0.10) vs high cost agent (defect: cost 0.35)
    u_scene = router.compute_utility("scene", delta_d, latency_ms=2.0)
    u_defect = router.compute_utility("defect", delta_d, latency_ms=50.0)

    # Lower cost and lower latency yields higher utility
    assert u_scene > u_defect
    assert u_scene > 0.0
    assert u_defect > 0.0


# ==============================================================================
# 3. PREREQUISITES & TERMINAL INVARIANTS
# ==============================================================================

def test_07_prerequisite_enforcement():
    """PPE specialist cannot run until Object specialist has executed."""
    engine = EvidenceDeficitEngine()
    router = SequentialEvidenceRouter()

    report = engine.evaluate(
        query="inspect worker hardhat compliance",
        intent="ppe_safety_compliance",
        findings=[],
        context_data={},
        executed_agents=[]
    )

    state_before_object = {
        "user_query": "inspect worker hardhat compliance",
        "detected_intent": "ppe_safety_compliance",
        "executed_agents": [],
        "failed_agents": [],
        "active_evidence_report": report,
        "iteration_step": 0
    }

    best_agent, _, candidate_scores = router.select_next_specialist(state_before_object)
    # PPE must NOT be in eligible candidates because 'object' has not executed yet
    assert "ppe" not in candidate_scores
    assert best_agent != "ppe"

    # Now simulate object has executed
    state_after_object = {
        "user_query": "inspect worker hardhat compliance",
        "detected_intent": "ppe_safety_compliance",
        "executed_agents": ["object"],
        "failed_agents": [],
        "active_evidence_report": report,
        "iteration_step": 1
    }
    next_agent, _, candidate_scores_after = router.select_next_specialist(state_after_object)
    # Now ppe is eligible and has positive utility
    assert "ppe" in candidate_scores_after
    assert candidate_scores_after["ppe"]["utility"] > 0.0


def test_08_terminal_agent_protection():
    """Terminal agents cannot be re-executed in the same pass."""
    engine = EvidenceDeficitEngine()
    router = SequentialEvidenceRouter()

    report = engine.evaluate(
        query="inspect worker hardhat compliance",
        intent="ppe_safety_compliance",
        findings=[],
        context_data={},
        executed_agents=["object", "ppe", "scene"]
    )

    state = {
        "user_query": "inspect worker hardhat compliance",
        "detected_intent": "ppe_safety_compliance",
        "executed_agents": ["object", "ppe", "scene"],
        "failed_agents": [],
        "active_evidence_report": report,
        "iteration_step": 3
    }
    next_agent, _, _ = router.select_next_specialist(state)
    assert next_agent not in ["object", "ppe", "scene"]


def test_09_sequential_loop_stopping_condition_deficit_threshold():
    """Router terminates when composite deficit falls below target stopping deficit."""
    router = SequentialEvidenceRouter(stopping_deficit=0.20)
    engine = EvidenceDeficitEngine()

    # Mock an evaluation report with very low deficit (D = 0.15 <= 0.20)
    report = engine.evaluate(
        query="inspect worker",
        intent="general_scene_inquiry",
        findings=[
            {"finding_uid": "f1", "agent_source": "object", "label": "person", "bbox": [0.1, 0.1, 0.4, 0.8], "detection_confidence": 0.98},
            {"finding_uid": "f2", "agent_source": "scene", "label": "scene_metrics", "attributes": {"mean_luminance": 120.0}}
        ],
        context_data={
            "object_result": {"objects": [{"label": "person", "confidence": 0.98}]},
            "scene_result": {"lighting": "adequate", "direct_observations": ["well lit"]}
        },
        executed_agents=["object", "scene"]
    )
    # Force low deficit for test
    report.composite_deficit = 0.12

    state = {
        "user_query": "inspect worker",
        "detected_intent": "general_scene_inquiry",
        "executed_agents": ["object"],
        "failed_agents": [],
        "active_evidence_report": report,
        "iteration_step": 1
    }
    best_agent, utility, _ = router.select_next_specialist(state)
    # Should stop because D <= 0.20
    assert best_agent is None
    assert utility == 0.0


def test_10_sequential_loop_stopping_condition_max_iterations():
    """Router terminates when maximum iterations are reached."""
    router = SequentialEvidenceRouter(max_iterations=3)
    state = {
        "user_query": "inspect worker",
        "detected_intent": "general_scene_inquiry",
        "executed_agents": [],
        "failed_agents": [],
        "active_evidence_report": None,
        "iteration_step": 3  # Max reached
    }
    best_agent, utility, _ = router.select_next_specialist(state)
    assert best_agent is None


# ==============================================================================
# 4. RAVEN REVISION 10 SAFETY GATES
# ==============================================================================

def test_11_raven_safety_gate_unvalidated_absence_override():
    """Revision 10: Unvalidated advisory physical absence prevents ACCEPT, forcing VERIFY."""
    raven = RavenEngine(tau_accept=0.35)

    findings = [
        {
            "finding_uid": "f_worker_01",
            "agent_source": "object",
            "label": "person",
            "bbox": [0.2, 0.2, 0.6, 0.8],
            "detection_confidence": 0.95
        },
        {
            "finding_uid": "f_ppe_01",
            "agent_source": "ppe",
            "label": "Compliant Worker: Helmet & Vest",
            "bbox": [0.2, 0.2, 0.6, 0.8],
            "detection_confidence": 0.95
        }
    ]
    context_data = {
        "object_result": {"objects": [{"label": "person", "confidence": 0.95}], "person_count": 1},
        "ppe_result": {
            "compliant_count": 1,
            "non_compliant_count": 0,
            "person_evaluations": [{"person_id": "P-01", "has_helmet": True, "has_vest": True, "is_occluded": False}]
        }
    }

    # Case A: Without advisory absence flag -> ACCEPT
    assessment_normal = raven.evaluate(
        query="check worker safety",
        intent="ppe_safety_compliance",
        findings=findings,
        context_data=context_data,
        executed_agents=["object", "ppe"]
    )
    assert assessment_normal.decision == RavenDecision.ACCEPT

    # Case B: With unvalidated advisory absence flag -> Overridden to VERIFY
    assessment_advisory = raven.evaluate(
        query="check worker safety",
        intent="ppe_safety_compliance",
        findings=findings,
        context_data=context_data,
        executed_agents=["object", "ppe"],
        advisory_absence_flags={"ET-PPE-CRANIAL-02": True}
    )

    assert assessment_advisory.decision != RavenDecision.ACCEPT
    assert assessment_advisory.decision == RavenDecision.VERIFY
    assert "SAFETY GATE OVERRIDE" in assessment_advisory.decision_rationale


def test_12_raven_safety_gate_critical_conflict_override():
    """Revision 10: Critical spatial conflict (severity >= 0.50) forces ABSTAIN."""
    raven = RavenEngine(tau_accept=0.40)

    findings = [
        {
            "finding_uid": "f_defect",
            "agent_source": "defect",
            "label": "Confirmed Defect: Scratches",
            "bbox": [0.3, 0.3, 0.5, 0.5],
            "detection_confidence": 0.90
        },
        {
            "finding_uid": "f_ocr",
            "agent_source": "ocr",
            "label": "Text: SERIAL-9999",
            "bbox": [0.31, 0.31, 0.49, 0.49],  # Major spatial overlap with semantic contradiction
            "detection_confidence": 0.92
        }
    ]
    context_data = {
        "defect_result": {"defects_detected": 1},
        "ocr_result": {"detected_text": "SERIAL-9999"}
    }

    assessment = raven.evaluate(
        query="inspect steel plate",
        intent="defect_inspection",
        findings=findings,
        context_data=context_data,
        executed_agents=["defect", "ocr"]
    )
    # High conflict forces ABSTAIN
    assert assessment.decision == RavenDecision.ABSTAIN
    assert assessment.conflict_entropy > 0.0


# ==============================================================================
# 5. END-TO-END CYCLIC LANGGRAPH EXECUTION
# ==============================================================================

def test_13_end_to_end_sequential_cyclic_graph_execution():
    """Executes sequential_vision_iq_app through multiple cyclic steps."""
    img_path = create_test_image("test_sequential_e2e.jpg")
    initial_state = {
        "session_id": "seq-e2e-session-001",
        "image_path": img_path,
        "user_query": "Inspect visible personnel for PPE helmet compliance",
        "input_type": "image",
        "timeout_seconds": 15.0,
        "max_steps": 4,
        "target_deficit": 0.20
    }

    final_state = sequential_vision_iq_app.invoke(initial_state)
    assert final_state is not None
    assert "final_response" in final_state

    resp = final_state["final_response"]
    assert resp["session_id"] == "seq-e2e-session-001"
    assert resp["status"] in ("completed", "partial")

    # Check that sequential execution occurred
    assert "deficit_history" in resp
    assert len(resp["deficit_history"]) >= 2  # Baseline D0 plus at least one update
    assert "utility_trace" in resp
    assert resp["iteration_step"] >= 1

    # Check RAVEN assessment exists
    assert resp["raven_decision"] in ["ACCEPT", "VERIFY", "ABSTAIN"]
    assert resp["raven_risk_score"] is not None


def test_14_fastapi_inspect_sequential_endpoint():
    """Tests the dedicated POST /inspect/sequential FastAPI endpoint."""
    img = np.full((100, 100, 3), 130, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)

    response = client.post(
        "/inspect/sequential",
        files={"file": ("test_seq_api.jpg", encoded.tobytes(), "image/jpeg")},
        data={
            "query": "Check workers for protective equipment",
            "target_deficit": 0.25,
            "max_steps": 3
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "findings" in data
    assert "raven_decision" in data
    assert data["status"] in ("completed", "partial")


# ==============================================================================
# 6. COMPREHENSIVE SECTION 10 UNIT & INTEGRATION TESTS
# ==============================================================================

def test_15_deficit_weights_and_normalization():
    """Section 10: Deficit calculation weights must sum to 1.0 and normalize to [0, 1]."""
    weights = DeficitWeights(w_completeness=0.50, w_reliability=0.20, w_conflict=0.20, w_unavailability=0.10)
    norm = weights.normalized()
    assert sum(norm.values()) == pytest.approx(1.0, 1e-6)
    assert norm["w_completeness"] == pytest.approx(0.50, 1e-4)
    assert norm["w_reliability"] == pytest.approx(0.20, 1e-4)
    assert norm["w_conflict"] == pytest.approx(0.20, 1e-4)
    assert norm["w_unavailability"] == pytest.approx(0.10, 1e-4)

    # All components strictly in [0, 1]
    engine = EvidenceDeficitEngine(weights=weights)
    report = engine.evaluate(
        query="inspect steel surface flaws",
        intent="defect_inspection",
        findings=[],
        context_data={},
        executed_agents=[]
    )
    assert 0.0 <= report.composite_deficit <= 1.0
    assert 0.0 <= report.completeness <= 1.0
    assert 0.0 <= report.reliability <= 1.0
    assert 0.0 <= report.conflict_score <= 1.0
    assert 0.0 <= report.unavailability <= 1.0


def test_16_multi_token_specialist_prediction():
    """Section 10: Multi-token specialists (e.g. ppe, defect) compute joint satisfaction improvement."""
    engine = EvidenceDeficitEngine()
    router = SequentialEvidenceRouter()

    # Provide localized person so PPE tokens are applicable and MISSING
    report = engine.evaluate(
        query="inspect worker PPE compliance",
        intent="ppe_safety_compliance",
        findings=[
            {
                "finding_uid": "f_person_01",
                "label": "person",
                "agent_source": "object",
                "bbox": [0.1, 0.1, 0.5, 0.9],
                "detection_confidence": 0.95
            }
        ],
        context_data={"object_result": {"objects": [{"label": "person", "confidence": 0.95}]}},
        executed_agents=["object"]
    )
    assert report.tokens["ET-PPE-CRANIAL-02"].status == TokenStatus.MISSING
    assert report.tokens["ET-PPE-TORSO-03"].status == TokenStatus.MISSING

    # PPE specialist covers both ET-PPE-CRANIAL-02 and ET-PPE-TORSO-03
    total_delta, deltas = router.predict_token_gain(
        candidate_agent="ppe",
        current_report=report,
        applicable_tokens=["ET-OBJ-PERSON-01", "ET-PPE-CRANIAL-02", "ET-PPE-TORSO-03", "ET-SCN-ENVIRONMENT-01"]
    )
    assert "ET-PPE-CRANIAL-02" in deltas
    assert "ET-PPE-TORSO-03" in deltas
    assert deltas["ET-PPE-CRANIAL-02"] > 0.0
    assert deltas["ET-PPE-TORSO-03"] > 0.0
    assert total_delta == pytest.approx(deltas["ET-PPE-CRANIAL-02"] + deltas["ET-PPE-TORSO-03"], 1e-4)


def test_17_unavailable_and_unsupported_evidence_gain():
    """Section 10: Unavailable and unsupported tokens yield 0.0 gain unless recovery agent."""
    router = SequentialEvidenceRouter()
    engine = EvidenceDeficitEngine()

    report = engine.evaluate(
        query="inspect workers",
        intent="ppe_safety_compliance",
        findings=[],
        context_data={},
        executed_agents=[],
        failed_agents=["object"]
    )
    report.tokens["ET-OBJ-PERSON-01"].status = TokenStatus.UNAVAILABLE

    # Standard candidate cannot recover UNAVAILABLE token
    total_delta, deltas = router.predict_token_gain(
        candidate_agent="object",
        current_report=report,
        applicable_tokens=["ET-OBJ-PERSON-01"]
    )
    assert deltas["ET-OBJ-PERSON-01"] == 0.0

    # Unsupported tokens yield strictly 0.0
    report.tokens["ET-OBJ-PERSON-01"].status = TokenStatus.UNSUPPORTED
    total_delta_unsupp, deltas_unsupp = router.predict_token_gain(
        candidate_agent="object",
        current_report=report,
        applicable_tokens=["ET-OBJ-PERSON-01"]
    )
    assert deltas_unsupp["ET-OBJ-PERSON-01"] == 0.0


def test_18_lifecycle_state_transitions_legal_and_illegal():
    """Section 10: Explicit agent lifecycle state transitions must be validated."""
    # Legal transitions
    assert validate_lifecycle_transition(AgentLifecycleState.NOT_SELECTED, AgentLifecycleState.ELIGIBLE) is True
    assert validate_lifecycle_transition(AgentLifecycleState.NOT_SELECTED, AgentLifecycleState.WAITING_FOR_DEPENDENCY) is True
    assert validate_lifecycle_transition(AgentLifecycleState.NOT_SELECTED, AgentLifecycleState.NOT_APPLICABLE) is True
    assert validate_lifecycle_transition(AgentLifecycleState.WAITING_FOR_DEPENDENCY, AgentLifecycleState.ELIGIBLE) is True
    assert validate_lifecycle_transition(AgentLifecycleState.WAITING_FOR_DEPENDENCY, AgentLifecycleState.BLOCKED) is True
    assert validate_lifecycle_transition(AgentLifecycleState.ELIGIBLE, AgentLifecycleState.RUNNING) is True
    assert validate_lifecycle_transition(AgentLifecycleState.RUNNING, AgentLifecycleState.COMPLETED) is True
    assert validate_lifecycle_transition(AgentLifecycleState.RUNNING, AgentLifecycleState.FAILED) is True

    # Illegal transitions (terminal agents cannot re-run, running cannot jump to eligible)
    assert validate_lifecycle_transition(AgentLifecycleState.COMPLETED, AgentLifecycleState.RUNNING) is False
    assert validate_lifecycle_transition(AgentLifecycleState.FAILED, AgentLifecycleState.RUNNING) is False
    assert validate_lifecycle_transition(AgentLifecycleState.BLOCKED, AgentLifecycleState.RUNNING) is False
    assert validate_lifecycle_transition(AgentLifecycleState.RUNNING, AgentLifecycleState.ELIGIBLE) is False

    with pytest.raises(ValueError, match="Illegal lifecycle transition"):
        validate_lifecycle_transition(AgentLifecycleState.COMPLETED, AgentLifecycleState.RUNNING, strict=True)


def test_19_diminishing_marginal_gain_stopping_rule():
    """Section 10: Router stops with DIMINISHING_MARGINAL_GAIN when max Delta D_hat < tau_gain."""
    router = SequentialEvidenceRouter(min_marginal_gain=0.50)  # Unattainably high gain threshold
    engine = EvidenceDeficitEngine()

    report = engine.evaluate(
        query="inspect worker hardhat compliance",
        intent="ppe_safety_compliance",
        findings=[],
        context_data={},
        executed_agents=["object"]
    )
    state = {
        "user_query": "inspect worker hardhat compliance",
        "detected_intent": "ppe_safety_compliance",
        "executed_agents": ["object"],
        "failed_agents": [],
        "active_evidence_report": report,
        "iteration_step": 1
    }

    best_agent, utility, candidates = router.select_next_specialist(state)
    assert best_agent is None
    assert router.last_stopping_reason == StoppingReason.DIMINISHING_MARGINAL_GAIN
    assert state.get("stopping_reason") == "DIMINISHING_MARGINAL_GAIN"


def test_20_wallclock_budget_exhaustion_stopping_rule():
    """Section 10: Router stops with WALLCLOCK_BUDGET_EXHAUSTED when time budget is exceeded."""
    router = SequentialEvidenceRouter(wallclock_budget_ms=10.0)  # Very tight 10ms budget
    engine = EvidenceDeficitEngine()

    report = engine.evaluate(
        query="inspect worker hardhat compliance",
        intent="ppe_safety_compliance",
        findings=[],
        context_data={},
        executed_agents=[]
    )
    # Simulate execution that started 50ms ago
    state = {
        "user_query": "inspect worker hardhat compliance",
        "detected_intent": "ppe_safety_compliance",
        "executed_agents": [],
        "failed_agents": [],
        "active_evidence_report": report,
        "iteration_step": 0,
        "start_time": time.perf_counter() - 0.050  # 50ms elapsed
    }

    best_agent, utility, candidates = router.select_next_specialist(state)
    assert best_agent is None
    assert router.last_stopping_reason == StoppingReason.WALLCLOCK_BUDGET_EXHAUSTED
    assert state.get("stopping_reason") == "WALLCLOCK_BUDGET_EXHAUSTED"


def test_21_signed_observed_deficit_reduction_allows_negative():
    """
    Section 10: Never replace negative observed gain with zero!
    If a specialist introduces a severe conflict or lowers reliability,
    Delta D_obs = D(S_t) - D(S_{t+1}) < 0 must be preserved as signed negative value.
    """
    router = SequentialEvidenceRouter()

    # Pre-execution deficit D_t = 0.30
    current_d = 0.30
    # Post-execution deficit D_{t+1} = 0.45 (increased due to contradiction!)
    post_d = 0.45

    candidate_scores = {
        "defect": {"delta_d": 0.15, "latency_ms": 30.0, "utility": 1.2}
    }

    telemetry = router.record_step_telemetry(
        step=1,
        current_deficit=current_d,
        candidate_scores=candidate_scores,
        selected_agent="defect",
        selection_efficiency=1.2,
        execution_result="SUCCESS",
        cleanup_result="CLEAN",
        post_deficit=post_d,
        actual_findings_count=2,
        remaining_invocations=3,
        remaining_wallclock_ms=4500.0
    )

    # Observed reduction must be strictly negative (-0.15), NEVER clipped to 0.0
    assert telemetry["signed_observed_deficit_reduction"] == -0.15
    assert telemetry["signed_observed_deficit_reduction"] < 0.0
    # Prediction error = Delta D_hat - Delta D_obs = 0.15 - (-0.15) = 0.30
    assert telemetry["prediction_error"] == pytest.approx(0.30, 1e-4)


def test_22_failed_specialist_telemetry_never_imputes_zero():
    """Section 10: For failed specialist executions, observed gain must be None, NEVER imputed as 0.0."""
    router = SequentialEvidenceRouter()

    candidate_scores = {
        "object": {"delta_d": 0.25, "latency_ms": 120.0, "utility": 1.5}
    }

    telemetry = router.record_step_telemetry(
        step=0,
        current_deficit=0.80,
        candidate_scores=candidate_scores,
        selected_agent="object",
        selection_efficiency=1.5,
        execution_result="FAILED",
        cleanup_result="CLEAN",
        post_deficit=None,
        actual_findings_count=0,
        remaining_invocations=4,
        remaining_wallclock_ms=4800.0,
        stopping_reason=None
    )

    # Must be None, never 0.0
    assert telemetry["signed_observed_deficit_reduction"] is None
    assert telemetry["prediction_error"] is None
    assert telemetry["execution_result"] == "FAILED"


def test_23_prediction_error_telemetry():
    """Section 10: Accurate prediction error calculation Delta D_hat - Delta D_obs."""
    router = SequentialEvidenceRouter()
    candidate_scores = {
        "scene": {"delta_d": 0.20, "latency_ms": 5.0, "utility": 2.0}
    }

    # Pre-deficit 0.60, Post-deficit 0.45 -> Delta D_obs = 0.15
    telemetry = router.record_step_telemetry(
        step=1,
        current_deficit=0.60,
        candidate_scores=candidate_scores,
        selected_agent="scene",
        selection_efficiency=2.0,
        execution_result="SUCCESS",
        cleanup_result="CLEAN",
        post_deficit=0.45,
        actual_findings_count=1,
        remaining_invocations=3,
        remaining_wallclock_ms=4900.0
    )
    # Predicted 0.20, observed 0.15 -> prediction error = 0.05
    assert telemetry["signed_observed_deficit_reduction"] == 0.15
    assert telemetry["prediction_error"] == pytest.approx(0.05, 1e-4)


def test_24_worker_timeout_and_cleanup_failure():
    """Section 10: Isolated execution telemetry tracks worker TIMEOUT and cleanup results."""
    router = SequentialEvidenceRouter()
    candidate_scores = {
        "defect": {"delta_d": 0.30, "latency_ms": 500.0, "utility": 0.8}
    }

    telemetry = router.record_step_telemetry(
        step=2,
        current_deficit=0.75,
        candidate_scores=candidate_scores,
        selected_agent="defect",
        selection_efficiency=0.8,
        execution_result="TIMEOUT",
        cleanup_result="CONTAINMENT_FAILED",
        post_deficit=None,
        actual_findings_count=0,
        remaining_invocations=1,
        remaining_wallclock_ms=0.0,
        stopping_reason="CONTAINMENT_FAILURE"
    )
    assert telemetry["execution_result"] == "TIMEOUT"
    assert telemetry["cleanup_result"] == "CONTAINMENT_FAILED"
    assert telemetry["stopping_reason"] == "CONTAINMENT_FAILURE"
    assert telemetry["signed_observed_deficit_reduction"] is None


def test_25_four_baseline_comparison_strategies():
    """Section 10: All four baseline routing strategies execute on the same query."""
    query = "Inspect steel plate surface flaws and cracks"
    intent = "defect_inspection"

    # Baseline 1: Fixed pipeline
    fixed_agents = route_fixed_pipeline(intent=intent, query=query)
    assert "defect" in fixed_agents
    assert "scene" in fixed_agents

    # Baseline 2: Phase 2 weighted router
    phase2_res = route_phase2_weighted(query=query)
    assert "defect" in phase2_res["selected_agents"]
    assert phase2_res["threshold"] == settings.ROUTING_THRESHOLD

    # Baseline 3: Relevance/dependency router
    relevance_agents = route_relevance_dependency(query=query, threshold=0.50)
    assert "defect" in relevance_agents

    # Baseline 4: Proposed sequential evidence-gain router
    engine = EvidenceDeficitEngine()
    initial_report = engine.evaluate(
        query=query,
        intent=intent,
        findings=[],
        context_data={},
        executed_agents=[]
    )
    seq_plan = route_sequential_evidence_gain(
        query=query,
        initial_report=initial_report,
        max_steps=3,
        stopping_deficit=0.20
    )
    assert seq_plan["strategy"] == "sequential_evidence_gain"
    assert len(seq_plan["planned_sequence"]) >= 1
    assert "defect" in seq_plan["planned_sequence"]

