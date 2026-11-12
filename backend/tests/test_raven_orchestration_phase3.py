import io
import os
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.orchestration.graph import (
    vision_iq_app, raven_verification_node, secondary_verification_node,
    escalation_node, route_after_raven
)
from backend.app.orchestration.state import AgentState
from backend.app.raven.engine import RavenEngine
from backend.app.raven.schemas import RavenDecision
from backend.app.raven.calibration import calibrate_confidence, compute_spatial_uncertainty
from backend.app.raven.conflict import CrossAgentConflictEvaluator

client = TestClient(app)

def create_synthetic_image(filename="test_phase3.jpg", width=200, height=200, color=(140, 140, 140)) -> str:
    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
    img_path = os.path.join(settings.UPLOADS_DIR, filename)
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(img_path, img)
    return img_path

# -------------------------------------------------------------
# 1. Calibration and Uncertainty Propagation Tests
# -------------------------------------------------------------

def test_specialist_confidence_calibration_bounds():
    # Overconfident neural predictions must be rescaled
    raw_high = 0.999
    cal_high = calibrate_confidence(raw_high)
    assert cal_high <= 0.98
    assert cal_high >= 0.70

    # Low confidence predictions must not be exaggerated
    raw_low = 0.20
    cal_low = calibrate_confidence(raw_low)
    assert cal_low < 0.30

def test_spatial_uncertainty_resolution_dependency():
    # Small crop (low pixel area) has higher epistemic uncertainty
    small_box = [10.0, 10.0, 30.0, 30.0]  # 20x20 px
    u_small = compute_spatial_uncertainty(small_box, [400, 400])

    large_box = [10.0, 10.0, 150.0, 150.0]  # 140x140 px
    u_large = compute_spatial_uncertainty(large_box, [400, 400])

    assert u_small > u_large
    assert 0.0 <= u_small <= 1.0
    assert 0.0 <= u_large <= 1.0

# -------------------------------------------------------------
# 2. Cross-Agent Conflict Evaluator Unit Tests
# -------------------------------------------------------------

def test_cross_agent_discrepancy_defect_and_ocr_overlap():
    findings = [
        {
            "finding_uid": "F-DFT-01",
            "agent_source": "defect",
            "label": "Confirmed Defect: Scratches",
            "bbox": [50.0, 50.0, 120.0, 90.0],
            "detection_confidence": 0.82
        },
        {
            "finding_uid": "F-OCR-01",
            "agent_source": "ocr",
            "label": "Text: 'SERIAL-8921'",
            "bbox": [55.0, 52.0, 115.0, 88.0],
            "detection_confidence": 0.91
        }
    ]
    context_data = {
        "defect_result": {"defects_detected": 1},
        "ocr_result": {"detected_text": "SERIAL-8921"}
    }
    executed_agents = ["defect", "ocr"]

    report = CrossAgentConflictEvaluator.evaluate(findings, context_data, executed_agents)
    assert report.conflict_entropy > 0.0
    assert len(report.discrepancies) >= 1
    assert any("SPATIAL" in d["type"] or "CONFLICT" in d["type"] for d in report.discrepancies)

# -------------------------------------------------------------
# 3. LangGraph Dynamic Branching & Tri-State Decision Tests
# -------------------------------------------------------------

def test_route_after_raven_decision_paths():
    # ACCEPT -> cognitive_agents
    state_accept: AgentState = {
        "verification_cycle": 0,
        "raven_decision": "ACCEPT"
    }
    assert route_after_raven(state_accept) == "cognitive_agents"

    # VERIFY (cycle 0) -> secondary_verification
    state_verify_initial: AgentState = {
        "verification_cycle": 0,
        "raven_decision": "VERIFY"
    }
    assert route_after_raven(state_verify_initial) == "secondary_verification"

    # VERIFY (cycle 1) -> bounded, routes to cognitive_agents
    state_verify_bounded: AgentState = {
        "verification_cycle": 1,
        "raven_decision": "VERIFY"
    }
    assert route_after_raven(state_verify_bounded) == "cognitive_agents"

    # ABSTAIN -> escalation
    state_abstain: AgentState = {
        "verification_cycle": 0,
        "raven_decision": "ABSTAIN"
    }
    assert route_after_raven(state_abstain) == "escalation"

def test_secondary_verification_node_bounds_and_dispatches():
    img_path = create_synthetic_image("test_sec_ver.jpg")
    state: AgentState = {
        "user_query": "Verify surface condition and labels",
        "detected_intent": "defect",
        "image_path": img_path,
        "verification_cycle": 0,
        "findings": [
            {
                "finding_uid": "F-01",
                "agent_source": "defect",
                "label": "Heuristic Cue",
                "bbox": [20.0, 20.0, 60.0, 60.0],
                "detection_confidence": 0.45
            }
        ],
        "executed_agents": ["defect"],
        "execution_log": [],
        "raven_assessment": {
            "secondary_agents_suggested": ["scene"]
        }
    }

    res = secondary_verification_node(state)
    assert res["verification_cycle"] == 1
    assert res["verification_pass_executed"] is True
    assert "scene" in res["executed_agents"]
    assert "secondary_verification" in [entry["agent_name"] for entry in res["execution_log"]]

def test_escalation_node_sets_flags():
    state: AgentState = {
        "findings": [
            {
                "finding_uid": "F-01",
                "human_review_required": False
            }
        ]
    }
    res = escalation_node(state)
    assert res["human_review_required"] is True
    assert res["safety_clearance_withheld"] is True
    assert all(f["human_review_required"] for f in res["findings"])

# -------------------------------------------------------------
# 4. End-to-End LangGraph Multi-Agent Orchestration with RAVEN
# -------------------------------------------------------------

def test_end_to_end_graph_execution_with_raven_node():
    img_path = create_synthetic_image("test_e2e_graph.jpg")
    initial_state = {
        "session_id": "test-e2e-graph-sess",
        "image_path": img_path,
        "user_query": "Detect all visible workers and verify compliance",
        "input_type": "image",
        "timeout_seconds": 15.0
    }

    final_state = vision_iq_app.invoke(initial_state)
    assert "final_response" in final_state
    final_resp = final_state["final_response"]

    # Verify RAVEN metrics are present and populated
    assert "raven_decision" in final_resp
    assert final_resp["raven_decision"] in ["ACCEPT", "VERIFY", "ABSTAIN"]
    assert "raven_risk_score" in final_resp
    assert isinstance(final_resp["raven_risk_score"], float)
    assert "evidence_completeness" in final_resp
    assert isinstance(final_resp["evidence_completeness"], float)
    assert "conflict_entropy" in final_resp
    assert isinstance(final_resp["conflict_entropy"], float)

    # Verify execution log includes raven_verification trace
    trace_names = [t["agent_name"] for t in final_resp.get("traces", [])]
    assert "raven_verification" in trace_names

def test_end_to_end_api_inspection_returns_raven_assessment():
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    
    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("api_raven_test.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")},
        data={"query": "Inspect PPE and detect violations"}
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_id"] is not None
    assert data["raven_decision"] in ["ACCEPT", "VERIFY", "ABSTAIN"]
    assert data["raven_risk_score"] is not None
    assert data["raven_assessment"] is not None
    assert "risk_score" in data["raven_assessment"]
    assert "decision" in data["raven_assessment"]
    assert "evidence_completeness" in data["raven_assessment"]
    assert "conflict_entropy" in data["raven_assessment"]
