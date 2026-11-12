import io
import os
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from backend.app.schemas.contracts import (
    AgentStatus, EvidenceRecord, AnalysisResponse,
    FindingContract, AgentTraceContract, EvidenceRetrievalResponse,
    build_evidence_records
)
from backend.app.database.session import SessionLocal
from backend.app.database.repository import AnalysisRepository
from backend.app.database.models import AnalysisSession

client = TestClient(app)

def create_valid_test_image_bytes(width=100, height=100, color=(128, 128, 128)) -> bytes:
    img = np.full((height, width, 3), color, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    return encoded.tobytes()

# -------------------------------------------------------------
# 1. Schema & Common Evidence Model Tests
# -------------------------------------------------------------

def test_agent_status_enumeration():
    assert AgentStatus.SELECTED == "SELECTED"
    assert AgentStatus.EXECUTED == "EXECUTED"
    assert AgentStatus.SKIPPED == "SKIPPED"
    assert AgentStatus.FAILED == "FAILED"
    assert AgentStatus.PARTIAL == "PARTIAL"

def test_evidence_record_validation():
    rec = EvidenceRecord(
        evidence_id="EV-OBJ-01",
        inspection_id="INS-100",
        query_id="Q-100",
        agent_name="object",
        agent_status=AgentStatus.EXECUTED,
        model_id="YOLO11n",
        model_version="v1.0.0",
        input_image_ref="storage/uploads/test.jpg",
        predicted_class="person",
        object_id="P-01",
        bbox=[10.0, 20.0, 80.0, 90.0],
        raw_confidence=0.95,
        calibrated_confidence=0.7037,
        processing_time_ms=35.5,
        evidence_source="vision_model",
        provenance={"detector": "Ultralytics YOLO"}
    )
    assert rec.evidence_id == "EV-OBJ-01"
    assert rec.inspection_id == "INS-100"
    assert rec.agent_status == "EXECUTED"
    assert rec.calibrated_confidence == 0.7037
    assert rec.bbox == [10.0, 20.0, 80.0, 90.0]

def test_analysis_response_aliases_and_derived_fields():
    finding = FindingContract(
        finding_uid="F-PPE-01",
        agent_source="ppe",
        label="Missing Hard Hat",
        bbox=[15.0, 25.0, 65.0, 95.0],
        detection_confidence=0.88,
        observed_evidence="Worker head area lacks required ANSI hard hat.",
        recommendation="Equip worker with Class E industrial hard hat.",
        human_review_required=True
    )
    trace = AgentTraceContract(
        agent_name="ppe",
        relevance_score=0.95,
        evidence_capability=0.90,
        normalized_cost=0.20,
        expected_latency=0.22,
        final_routing_score=0.62,
        decision="SELECTED",
        execution_status="SUCCESS",
        actual_latency_ms=48.2
    )

    resp = AnalysisResponse(
        session_id="SESS-TEST-001",
        created_at="2026-11-11T12:00:00Z",
        user_query="Inspect PPE compliance",
        input_type="image",
        input_path="storage/uploads/test.jpg",
        selected_agents=["ppe"],
        executed_agents=["ppe"],
        skipped_agents=["defect"],
        failed_agents=[],
        status="completed",
        total_latency_ms=120.5,
        findings=[finding],
        traces=[trace]
    )

    # Verify alias mappings
    assert resp.inspection_id == "SESS-TEST-001"
    assert resp.query == "Inspect PPE compliance"
    assert resp.human_review_status is True
    assert len(resp.detections) == 1
    assert resp.detections[0]["label"] == "Missing Hard Hat"
    assert resp.recommendations == ["Equip worker with Class E industrial hard hat."]
    assert resp.execution_timing["total_latency_ms"] == 120.5
    assert resp.execution_timing["ppe_ms"] == 48.2

def test_build_evidence_records_helper():
    findings = [
        {
            "finding_uid": "F-01",
            "agent_source": "defect",
            "label": "Surface Scratch",
            "bbox": [5.0, 5.0, 30.0, 40.0],
            "detection_confidence": 0.85,
            "is_confirmed_defect": True
        },
        {
            "finding_uid": "F-02",
            "agent_source": "defect",
            "label": "Heuristic Cue",
            "bbox": [50.0, 50.0, 80.0, 80.0],
            "detection_confidence": 0.40,
            "is_confirmed_defect": False,
            "human_review_required": True
        }
    ]
    traces = [
        {
            "agent_name": "defect",
            "execution_status": "SUCCESS",
            "actual_latency_ms": 32.0
        },
        {
            "agent_name": "ocr",
            "execution_status": "SKIPPED",
            "actual_latency_ms": 0.0,
            "selection_reason": "Score 0.12 < threshold 0.45"
        }
    ]

    records = build_evidence_records(
        session_id="SESSION-BUILD-01",
        input_image_ref="test.jpg",
        findings=findings,
        traces=traces
    )

    assert len(records) == 3
    # First is confirmed executed
    assert records[0].agent_status == "EXECUTED"
    assert records[0].predicted_class == "Surface Scratch"
    # Second is unconfirmed / partial
    assert records[1].agent_status == "PARTIAL"
    # Third is skipped trace record
    assert records[2].agent_status == "SKIPPED"
    assert records[2].agent_name == "ocr"

# -------------------------------------------------------------
# 2. Input Validation & Error Handling Tests
# -------------------------------------------------------------

def test_image_upload_validation_unsupported_extension():
    fake_exe = io.BytesIO(b"MZ\x90\x00\x03\x00\x00\x00")
    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("malicious.exe", fake_exe, "application/octet-stream")},
        data={"query": "Analyze image"}
    )
    assert resp.status_code == 400
    assert "Unsupported image file format" in resp.json()["detail"]

def test_image_upload_validation_empty_file():
    empty_file = io.BytesIO(b"")
    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("empty.jpg", empty_file, "image/jpeg")},
        data={"query": "Analyze image"}
    )
    assert resp.status_code == 400
    assert "empty (0 bytes)" in resp.json()["detail"]

def test_image_upload_validation_corrupt_raster():
    corrupt_data = io.BytesIO(b"NOT_A_REAL_IMAGE_DATA_HEADER_CORRUPTED")
    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("corrupt.png", corrupt_data, "image/png")},
        data={"query": "Analyze image"}
    )
    assert resp.status_code == 400
    assert "Invalid or corrupt image file" in resp.json()["detail"]

def test_image_upload_validation_missing_file():
    resp = client.post(
        "/api/v1/analyze/image",
        data={"query": "Analyze image"}
    )
    assert resp.status_code == 400
    assert "No image file provided" in resp.json()["detail"]

# -------------------------------------------------------------
# 3. Dedicated Inspection Endpoints Tests
# -------------------------------------------------------------

def test_ppe_dedicated_endpoint():
    img_bytes = create_valid_test_image_bytes()
    resp = client.post(
        "/api/v1/analyze/ppe",
        files={"file": ("ppe_test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        data={"query": "Verify hard hat and safety vest compliance"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "inspection_id" in data
    assert "ppe" in data["selected_agents"]
    assert "evidence_records" in data
    assert isinstance(data["evidence_records"], list)
    assert "detections" in data

def test_defect_dedicated_endpoint():
    img_bytes = create_valid_test_image_bytes()
    resp = client.post(
        "/api/v1/analyze/defect",
        files={"file": ("defect_test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        data={"query": "Inspect metal surface defects"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "inspection_id" in data
    assert "defect" in data["selected_agents"]
    assert "evidence_records" in data

# -------------------------------------------------------------
# 4. Evidence Retrieval & History Aliases Tests
# -------------------------------------------------------------

def test_evidence_retrieval_endpoint():
    # Run an analysis to generate an active session
    img_bytes = create_valid_test_image_bytes()
    analysis_resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("evidence_test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        data={"query": "Examine objects and environment"}
    )
    assert analysis_resp.status_code == 200
    session_id = analysis_resp.json()["session_id"]

    # Query standardized evidence endpoint
    ev_resp = client.get(f"/api/v1/analysis/{session_id}/evidence")
    assert ev_resp.status_code == 200
    ev_data = ev_resp.json()
    assert ev_data["inspection_id"] == session_id
    assert "evidence_records" in ev_data
    assert "completeness_score" in ev_data
    assert isinstance(ev_data["evidence_records"], list)

    # Query using alias
    alias_resp = client.get(f"/api/v1/inspection/{session_id}/evidence")
    assert alias_resp.status_code == 200
    assert alias_resp.json()["inspection_id"] == session_id

def test_evidence_retrieval_missing_session_404():
    resp = client.get("/api/v1/analysis/nonexistent-session-id-9999/evidence")
    assert resp.status_code == 404

def test_history_and_inspection_details_aliases():
    # History alias
    hist_resp = client.get("/api/v1/inspections?limit=5")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert "sessions" in hist_data
    assert hist_data["status"] == "success"

    if len(hist_data["sessions"]) > 0:
        sess_id = hist_data["sessions"][0]["id"]
        # Inspection details alias
        detail_resp = client.get(f"/api/v1/inspection/{sess_id}")
        assert detail_resp.status_code == 200
        detail_data = detail_resp.json()
        assert detail_data["session"]["id"] == sess_id
        assert "evidence_records" in detail_data["session"]

# -------------------------------------------------------------
# 5. Database Persistence & Invariants Tests
# -------------------------------------------------------------

def test_database_persistence_with_evidence_records():
    db = SessionLocal()
    repo = AnalysisRepository(db)
    test_session_id = f"test-db-persist-{os.urandom(4).hex()}"
    
    mock_ev = [
        {
            "evidence_id": "EV-01",
            "inspection_id": test_session_id,
            "agent_name": "object",
            "agent_status": "EXECUTED",
            "timestamp": "2026-11-11T12:00:00Z",
            "processing_time_ms": 25.0,
            "evidence_source": "vision_model"
        }
    ]

    session = repo.create_session({
        "id": test_session_id,
        "user_query": "DB test query",
        "input_type": "image",
        "input_path": "storage/uploads/test.jpg",
        "selected_agents": ["object"],
        "executed_agents": ["object"],
        "evidence_records": mock_ev
    })

    retrieved = repo.get_inspection(test_session_id)
    assert retrieved is not None
    assert retrieved.id == test_session_id
    assert retrieved.evidence_records == mock_ev

    retrieved_ev = repo.get_evidence_records(test_session_id)
    assert len(retrieved_ev) == 1
    assert retrieved_ev[0]["evidence_id"] == "EV-01"
    db.close()

def test_no_dummy_detections_on_empty_scene():
    """Invariant: Never generate dummy detection results for a real user request."""
    # Create an image that has no persons, defects or text (blank gray square)
    blank_bytes = create_valid_test_image_bytes(width=50, height=50, color=(100, 100, 100))
    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("blank_test.jpg", io.BytesIO(blank_bytes), "image/jpeg")},
        data={"query": "Identify any hazardous defects"}
    )
    assert resp.status_code == 200
    data = resp.json()
    # In a completely blank tiny image, no artificial 'person' or 'defect' should be fabricated
    # If the model finds 0 defects, detections should not contain synthetic fabricated items
    for d in data.get("detections", []):
        assert d.get("label") != "DUMMY_OBJECT"
        assert d.get("label") != "MOCK_DEFECT"
