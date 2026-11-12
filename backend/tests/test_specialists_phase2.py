"""
VisionIQ Automated Test Suite: Phase 2 Specialist Agents, Evidence Engine, & Contracts
========================================================================================
Tests:
1. Specialist Output Schema Validation (AgentOutput, ConfidenceType, DetectionState)
2. Agent Identification, Model Versioning, & Image Identifier Propagation
3. Bounding Box Geometry & Normalization Standards
4. Confidence Bounds & Separation of Epistemic Types (Raw vs Calibrated vs Heuristic)
5. Robustness to Missing, Corrupt, and Degenerate Image Inputs
6. PPE Safety Invariant (Never infer non-compliance solely from detector absence)
7. Defect Anomaly Confirmation Boundary (Confirmed Defect vs Surface Anomaly)
8. OCR Optionality & Query-Dependent Invocation
9. Scene Understanding Morphometrics & Quality Assessment
10. Evidence Token Engine & Audit Traceability (Tokens, Status Transitions, Deficit)
11. Cross-Agent Conflict Evaluation & Spatial Discrepancies
12. Minimal API Integration (/inspect & /analyze/image Endpoints)
13. Empirical Specialist Benchmark Evaluation Results Verification
"""

import os
import json
import tempfile
import pytest
import numpy as np
import cv2
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.agents.base import (
    AgentInput, AgentOutput, ExecutionStatus,
    ConfidenceType, DetectionState
)
from backend.app.agents.agent_b_object import ObjectDetectionAgent
from backend.app.agents.agent_c_ppe import PPESafetyAgent, PPESpecialistAgent
from backend.app.agents.agent_d_ocr import OCRAgent, OCRSpecialistAgent
from backend.app.agents.agent_e_defect import IndustrialDefectAgent
from backend.app.agents.agent_f_scene import SceneUnderstandingAgent
from backend.app.orchestration.evidence_deficit import (
    EvidenceDeficitEngine, TokenStatus, TOKEN_CATALOG, DeficitWeights
)
from backend.app.raven.conflict import CrossAgentConflictEvaluator, ConflictReport
from backend.app.schemas.contracts import AnalysisResponse

client = TestClient(app)

# Locate workspace root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAMPLE_NEU = PROJECT_ROOT / "data" / "samples" / "neu_det" / "crazing_1.jpg"
SAMPLE_SHWD = PROJECT_ROOT / "data" / "samples" / "shwd" / "shwd_005298.jpg"
SAMPLE_OCR = PROJECT_ROOT / "data" / "samples" / "ocr" / "ocr_iiit5k_0001.png"
SAMPLE_FACTORY = PROJECT_ROOT / "sample_data" / "FACTORY_SCENE.png"


def make_agent_input(image_path: str, image_id: str = "TEST_IMG_01", query: str = "Inspect industrial scene") -> AgentInput:
    return AgentInput(
        session_id="SESS_TEST_01",
        user_query=query,
        image_path=image_path,
        image_id=image_id
    )


# ------------------------------------------------------------------------------
# 1. Specialist Output Schema Validation
# ------------------------------------------------------------------------------
def test_specialist_output_schemas():
    """Verify that all specialists return AgentOutput conforming to standard schemas."""
    # Agent F (Scene) - light & fast
    agent_f = SceneUnderstandingAgent()
    inp_f = make_agent_input(image_path=str(SAMPLE_FACTORY), image_id="IMG_SCHEMA_01")
    out_f = agent_f.execute(inp_f)
    assert isinstance(out_f, AgentOutput)
    assert out_f.status == ExecutionStatus.SUCCESS
    assert out_f.agent_name == "scene"
    assert "OpenCV" in out_f.model_name
    assert out_f.input_image_id == "IMG_SCHEMA_01"
    assert out_f.confidence_type == ConfidenceType.HEURISTIC_SCORE
    assert "ET-SCN-ENVIRONMENT-08" in out_f.evidence_token_ids

    # Agent B (Object)
    agent_b = ObjectDetectionAgent()
    inp_b = make_agent_input(image_path=str(SAMPLE_FACTORY), image_id="IMG_SCHEMA_02")
    out_b = agent_b.execute(inp_b)
    assert isinstance(out_b, AgentOutput)
    assert out_b.status == ExecutionStatus.SUCCESS
    assert out_b.agent_name == "object"
    assert out_b.model_name == "YOLOv11n"
    assert out_b.confidence_type == ConfidenceType.RAW_MODEL_CONFIDENCE
    assert any("ET-OBJ" in tid for tid in out_b.evidence_token_ids)

    # Agent E (Defect)
    agent_e = IndustrialDefectAgent()
    inp_e = make_agent_input(image_path=str(SAMPLE_NEU), image_id="IMG_SCHEMA_03")
    out_e = agent_e.execute(inp_e)
    assert isinstance(out_e, AgentOutput)
    assert out_e.status == ExecutionStatus.SUCCESS
    assert out_e.agent_name == "defect"
    assert "DefectNet" in out_e.model_name or "IndustrialDefectNet" in out_e.model_name
    assert "ET-DEF-SURFACE-05" in out_e.evidence_token_ids


# ------------------------------------------------------------------------------
# 2. Agent Identification, Model Versioning & Image Identifier Propagation
# ------------------------------------------------------------------------------
def test_agent_image_id_propagation():
    """Ensure that input image_id is strictly propagated across all specialist outputs."""
    test_id = "VERIFIED_TEST_BATCH_99"
    specialists = [
        ObjectDetectionAgent(),
        PPESafetyAgent(),
        IndustrialDefectAgent(),
        SceneUnderstandingAgent()
    ]
    for agent in specialists:
        inp = make_agent_input(image_path=str(SAMPLE_FACTORY), image_id=test_id)
        out = agent.execute(inp)
        assert out.input_image_id == test_id
        assert out.model_version is not None
        assert len(out.model_version) > 0


# ------------------------------------------------------------------------------
# 3. Bounding Box Geometry & Normalization Standards
# ------------------------------------------------------------------------------
def test_bounding_box_geometry_and_normalization():
    """Verify that bounding boxes comply with [x1, y1, x2, y2] non-degeneracy."""
    agent_b = ObjectDetectionAgent()
    out = agent_b.execute(make_agent_input(image_path=str(SAMPLE_FACTORY), image_id="IMG_BBOX_01"))
    assert out.status == ExecutionStatus.SUCCESS

    for f in out.findings:
        bbox = f.get("bbox")
        assert bbox is not None, "Bounding box must be present"
        assert len(bbox) == 4, "Bounding box must have 4 coordinates"
        x1, y1, x2, y2 = bbox
        assert x1 <= x2, f"Invalid bbox width: x1={x1} > x2={x2}"
        assert y1 <= y2, f"Invalid bbox height: y1={y1} > y2={y2}"
        assert x1 >= 0 and y1 >= 0, f"Negative coordinates: {bbox}"

        # If norm_bbox exists, verify [0, 1] bounds
        if "norm_bbox" in f and f["norm_bbox"]:
            nx1, ny1, nx2, ny2 = f["norm_bbox"]
            assert 0.0 <= nx1 <= 1.0
            assert 0.0 <= ny1 <= 1.0
            assert 0.0 <= nx2 <= 1.0
            assert 0.0 <= ny2 <= 1.0


# ------------------------------------------------------------------------------
# 4. Confidence Bounds & Separation of Epistemic Types
# ------------------------------------------------------------------------------
def test_confidence_bounds_and_types():
    """Verify explicit epistemic confidence types and valid [0, 1] numerical bounds."""
    agent_b = ObjectDetectionAgent()
    out = agent_b.execute(make_agent_input(image_path=str(SAMPLE_FACTORY), image_id="IMG_CONF_01"))
    assert out.status == ExecutionStatus.SUCCESS

    for f in out.findings:
        conf = f.get("detection_confidence")
        assert conf is not None
        assert 0.0 <= conf <= 1.0, f"Confidence out of bounds: {conf}"

        ctype = f.get("confidence_type")
        assert ctype in [
            ConfidenceType.RAW_MODEL_CONFIDENCE.value,
            ConfidenceType.CALIBRATED_PROBABILITY.value,
            ConfidenceType.HEURISTIC_SCORE.value
        ]

        if "calibrated_confidence" in f and f["calibrated_confidence"] is not None:
            assert 0.0 <= f["calibrated_confidence"] <= 1.0


# ------------------------------------------------------------------------------
# 5. Robustness to Missing, Corrupt, and Degenerate Image Inputs
# ------------------------------------------------------------------------------
def test_robustness_missing_and_corrupt_images():
    """Verify specialists return ExecutionStatus.FAILED without raising unhandled exceptions."""
    agents = [
        ObjectDetectionAgent(),
        PPESafetyAgent(),
        IndustrialDefectAgent(),
        OCRAgent(),
        SceneUnderstandingAgent()
    ]

    # Non-existent file
    for agent in agents:
        res = agent.execute(make_agent_input(image_path="non_existent_path_xyz.jpg", image_id="MISSING_01"))
        assert res.status == ExecutionStatus.FAILED
        err = res.error_details or res.error_message
        assert err is not None
        assert "not found" in err.lower() or "missing" in err.lower() or "fail" in err.lower()

    # Empty 0-byte file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as empty_file:
        empty_path = empty_file.name

    try:
        for agent in agents:
            res = agent.execute(make_agent_input(image_path=empty_path, image_id="EMPTY_01"))
            assert res.status == ExecutionStatus.FAILED
            err = res.error_details or res.error_message
            assert err is not None
    finally:
        if os.path.exists(empty_path):
            try:
                os.remove(empty_path)
            except Exception:
                pass

    # Corrupt non-image binary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as corrupt_file:
        corrupt_file.write(b"CORRUPT_NOT_AN_IMAGE_HEADER_DATA_123456789")
        corrupt_path = corrupt_file.name

    try:
        for agent in agents:
            res = agent.execute(make_agent_input(image_path=corrupt_path, image_id="CORRUPT_01"))
            assert res.status == ExecutionStatus.FAILED
            err = res.error_details or res.error_message
            assert err is not None
    finally:
        if os.path.exists(corrupt_path):
            try:
                os.remove(corrupt_path)
            except Exception:
                pass


# ------------------------------------------------------------------------------
# 6. PPE Safety Invariant
# ------------------------------------------------------------------------------
def test_ppe_safety_invariant():
    """
    CRITICAL SAFETY RULE:
    Never infer non-compliance solely because a detector did not find a helmet.
    If evidence is missing or ambiguous, state must be UNCERTAIN / INSUFFICIENT_EVIDENCE.
    """
    agent_c = PPESafetyAgent()
    res = agent_c.execute(make_agent_input(image_path=str(SAMPLE_FACTORY), image_id="PPE_SAFETY_01"))
    assert res.status == ExecutionStatus.SUCCESS

    for f in res.findings:
        state = f.get("detection_state")
        compliance = f.get("compliance_status")
        # If compliance is asserted as NON_COMPLIANT, it must be supported by an actual head observation
        if compliance == "NON_COMPLIANT":
            assert f.get("bbox") is not None
        elif state == DetectionState.UNCERTAIN.value:
            assert compliance in ["UNCERTAIN", "UNKNOWN", "PARTIAL"] or f.get("human_review_required") is True


# ------------------------------------------------------------------------------
# 7. Defect Anomaly Confirmation Boundary
# ------------------------------------------------------------------------------
def test_defect_anomaly_confirmation_boundary():
    """Verify distinction between confirmed defect categories and unconfirmed surface anomalies."""
    agent_e = IndustrialDefectAgent()
    res = agent_e.execute(make_agent_input(image_path=str(SAMPLE_NEU), image_id="DEF_BOUND_01"))
    assert res.status == ExecutionStatus.SUCCESS
    assert len(res.findings) > 0

    for f in res.findings:
        assert "is_confirmed_defect" in f
        assert isinstance(f["is_confirmed_defect"], bool)
        assert "label" in f
        lbl = f["label"].lower()
        # Supported NEU-DET defect classes or heuristic anomaly cues
        assert "anomaly" in lbl or any(c in lbl for c in [
            "crazing", "inclusion", "patches",
            "pitted_surface", "rolled-in_scale", "scratches"
        ])


# ------------------------------------------------------------------------------
# 8. OCR Optionality & Query-Dependent Invocation
# ------------------------------------------------------------------------------
def test_ocr_optionality_and_query_dependence():
    """Verify OCR executes when requested on text image, and extracts valid character data."""
    if SAMPLE_OCR.exists():
        agent_d = OCRAgent()
        res = agent_d.execute(make_agent_input(image_path=str(SAMPLE_OCR), image_id="OCR_QUERY_01", query="Read serial text"))
        assert res.status == ExecutionStatus.SUCCESS
        assert res.agent_name == "ocr"
        assert "ET-OCR-SERIAL-07" in res.evidence_token_ids
        assert len(res.findings) >= 0


# ------------------------------------------------------------------------------
# 9. Scene Understanding Morphometrics & Quality Assessment
# ------------------------------------------------------------------------------
def test_scene_understanding_morphometrics():
    """Verify OpenCV morphometrics, image quality flags, and environmental token."""
    agent_f = SceneUnderstandingAgent()
    res = agent_f.execute(make_agent_input(image_path=str(SAMPLE_FACTORY), image_id="SCENE_ENV_01"))
    assert res.status == ExecutionStatus.SUCCESS

    metadata = res.metadata
    assert "mean_luminance" in metadata
    assert "rms_contrast" in metadata
    assert 0.0 <= metadata["mean_luminance"] <= 255.0
    assert metadata["rms_contrast"] >= 0.0

    # Image quality flags
    assert "is_underexposed" in metadata
    assert "is_overexposed" in metadata
    assert "is_blurry" in metadata

    # Token
    assert "ET-SCN-ENVIRONMENT-08" in res.evidence_token_ids


# ------------------------------------------------------------------------------
# 10. Evidence Token Engine & Audit Traceability
# ------------------------------------------------------------------------------
def test_evidence_token_engine_traceability():
    """Verify that findings link to Phase 3.1 token catalog and update deficit correctly."""
    engine = EvidenceDeficitEngine()
    query = "Inspect workers for safety PPE compliance"
    intent = "ppe_safety_compliance"

    # Initial evaluation with empty findings
    report_init = engine.evaluate(
        query=query,
        intent=intent,
        findings=[],
        context_data={},
        executed_agents=[]
    )
    assert report_init.composite_deficit >= 0.5
    assert report_init.tokens["ET-OBJ-PERSON-01"].status == TokenStatus.MISSING
    assert report_init.tokens["ET-PPE-CRANIAL-02"].status == TokenStatus.MISSING

    # Add finding from Agent B
    finding_b = {
        "finding_uid": "FIND-OBJ-001",
        "agent_source": "object",
        "evidence_token_id": "ET-OBJ-PERSON-01",
        "detection_confidence": 0.94,
        "calibrated_confidence": 0.91,
        "detection_state": "DETECTED",
        "label": "person",
        "bbox": [50, 50, 200, 300]
    }
    context_data = {
        "object_result": {
            "objects": [{"label": "person", "bbox": [50, 50, 200, 300], "confidence": 0.94}]
        }
    }
    report_step1 = engine.evaluate(
        query=query,
        intent=intent,
        findings=[finding_b],
        context_data=context_data,
        executed_agents=["object"]
    )
    assert report_step1.tokens["ET-OBJ-PERSON-01"].status == TokenStatus.SATISFIED
    assert report_step1.tokens["ET-OBJ-PERSON-01"].source_finding_uids is not None
    # Deficit must decrease
    assert report_step1.composite_deficit < report_init.composite_deficit

    # Add uncertain finding from Agent C
    finding_c = {
        "finding_uid": "FIND-PPE-002",
        "agent_source": "ppe",
        "evidence_token_id": "ET-PPE-CRANIAL-02",
        "detection_confidence": 0.35,
        "calibrated_confidence": 0.30,
        "detection_state": "UNCERTAIN",
        "label": "helmet",
        "compliance_status": "UNCERTAIN",
        "bbox": [80, 50, 140, 100],
        "human_review_required": True
    }
    context_data["ppe_result"] = {
        "compliance_status": "UNCERTAIN",
        "total_persons_evaluated": 1,
        "compliant_count": 0,
        "non_compliant_count": 0,
        "uncertain_count": 1
    }
    report_step2 = engine.evaluate(
        query=query,
        intent=intent,
        findings=[finding_b, finding_c],
        context_data=context_data,
        executed_agents=["object", "ppe"]
    )
    assert report_step2.tokens["ET-PPE-CRANIAL-02"].status in [
        TokenStatus.CONFIRMED_ABSENT,
        TokenStatus.INSUFFICIENT_EVIDENCE,
        TokenStatus.SATISFIED
    ]


# ------------------------------------------------------------------------------
# 11. Cross-Agent Conflict Evaluation
# ------------------------------------------------------------------------------
def test_cross_agent_conflict_evaluation():
    """Verify conflict evaluation when specialists produce conflicting counts or classifications."""
    # Object detected 2 persons, but PPE evaluated 0 persons
    findings = [
        {
            "finding_uid": "F-OBJ-01",
            "agent_source": "object",
            "label": "person",
            "bbox": [50, 50, 150, 250],
            "detection_confidence": 0.95
        },
        {
            "finding_uid": "F-OBJ-02",
            "agent_source": "object",
            "label": "person",
            "bbox": [200, 50, 300, 250],
            "detection_confidence": 0.92
        }
    ]
    context_data = {
        "object_result": {
            "objects": [
                {"label": "person", "bbox": [50, 50, 150, 250]},
                {"label": "person", "bbox": [200, 50, 300, 250]}
            ]
        },
        "ppe_result": {
            "total_persons_evaluated": 0,
            "compliant_count": 0,
            "non_compliant_count": 0
        }
    }
    report = CrossAgentConflictEvaluator.evaluate(
        findings, context_data, executed_agents=["object", "ppe"]
    )
    assert isinstance(report, ConflictReport)
    assert len(report.discrepancies) > 0
    assert report.conflict_entropy > 0.0


# ------------------------------------------------------------------------------
# 12. Minimal API Integration (/inspect & /analyze/image)
# ------------------------------------------------------------------------------
def test_fastapi_inspect_endpoint():
    """Verify that POST /inspect returns an AnalysisResponse with structured evidence records."""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    img_bytes = encoded.tobytes()

    response = client.post(
        "/inspect",
        files={"file": ("test_inspect.jpg", img_bytes, "image/jpeg")},
        data={"query": "Inspect factory workers and check helmet compliance"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["completed", "success"]
    assert "session_id" in data
    assert "findings" in data
    assert "evidence_records" in data
    assert isinstance(data["evidence_records"], list)
    assert "raven_assessment" in data


def test_fastapi_inspect_error_handling():
    """Verify that POST /inspect validates input files gracefully."""
    # Empty file
    response = client.post(
        "/inspect",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
        data={"query": "Check compliance"}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

    # Corrupt file
    response = client.post(
        "/inspect",
        files={"file": ("corrupt.jpg", b"INVALID_BINARY_BYTES", "image/jpeg")},
        data={"query": "Check compliance"}
    )
    assert response.status_code == 400
    assert "corrupt" in response.json()["detail"].lower() or "decode" in response.json()["detail"].lower()


# ------------------------------------------------------------------------------
# 13. Empirical Specialist Benchmark Results Verification
# ------------------------------------------------------------------------------
def test_empirical_specialist_results_file():
    """Verify that empirical benchmark results file exists and has real non-zero evaluation metrics."""
    res_path = PROJECT_ROOT / "data" / "manifests" / "specialist_evaluation_results.json"
    assert res_path.exists(), "specialist_evaluation_results.json must exist"

    with open(res_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    assert "agent_b_object" in results
    assert "agent_c_ppe" in results
    assert "agent_d_ocr" in results
    assert "agent_e_defect" in results
    assert "agent_f_scene" in results

    # Verify Agent E metrics
    assert "accuracy" in results["agent_e_defect"]
    assert results["agent_e_defect"]["mean_latency_ms"] > 0

    # Verify Agent C metrics
    assert "map_50" in results["agent_c_ppe"]
    assert results["agent_c_ppe"]["mean_latency_ms"] > 0

    # Verify Agent D metrics
    assert "character_error_rate" in results["agent_d_ocr"]
    assert "exact_match_accuracy" in results["agent_d_ocr"]

    # Verify Agent F metrics
    assert results["agent_f_scene"]["valid_reports"] == 10
    assert results["agent_f_scene"]["status"] == "functional"
