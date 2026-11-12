"""
Deterministic Safety & Evidence-Validation Test Suite for RAVEN (Phase 4).
Covers all 16 required safety verification scenarios from Section 7 of the specification:

 1. valid_evidence_acceptance (High completeness, verified provenance, low risk -> ACCEPT)
 2. missing_critical_evidence (Mandatory token missing -> ABSTAIN)
 3. missing_optional_evidence (Optional token missing -> still ACCEPT if critical evidence met)
 4. critical_specialist_failure (Critical specialist in failed_agents -> ABSTAIN)
 5. irrelevant_specialist_failure (Non-critical specialist in failed_agents -> does NOT force ABSTAIN)
 6. unresolved_contradiction (Contradiction without resolver -> VERIFY or ABSTAIN by severity)
 7. resolved_contradiction (Conflict resolved by authorized resolver -> allows ACCEPT)
 8. invalid_model_provenance (Tampered/untrusted model version -> ABSTAIN)
 9. invalid_image_provenance (Non-existent or mismatched image path -> ABSTAIN)
10. unvalidated_physical_absence (Detector zero-detection without certificate -> VERIFY)
11. expired_or_revoked_absence_validation (Certificate with past expiry or is_revoked -> ABSTAIN)
12. poor_image_quality (Severe optical blur or pitch black -> ABSTAIN)
13. worker_cleanup_failure (worker_cleanup_failed=True -> ABSTAIN)
14. budget_exhaustion_with_incomplete_evidence (budget_exhausted=True with unsatisfied required tokens -> ABSTAIN)
15. human_escalation_record_generation (HumanReviewRecord & HumanEscalationDossier synthesized with full metadata)
16. safety_gate_precedence_over_low_deficit (Numerical deficit D=0.05 overridden by safety veto to VERIFY or ABSTAIN)
"""

import os
import time
import tempfile
import pytest
import numpy as np
import cv2

from backend.app.raven.engine import RavenEngine
from backend.app.raven.schemas import (
    RavenDecision,
    SiteAbsenceValidationCertificate,
    ConflictRecord,
    HumanReviewRecord,
    HumanEscalationDossier
)
from backend.app.raven.calibration import recompute_clopper_pearson_upper_bound, measure_image_quality


@pytest.fixture
def valid_test_image():
    """Generates a temporary valid sharp synthetic image with sufficient resolution and contrast."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        # Create a 640x480 high-contrast image (checkerboard / structured shapes)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[::40, :, :] = 255
        img[:, ::40, :] = 255
        cv2.putText(img, "VISIONIQ_TEST_CALIBRATION", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(tmp.name, img)
        temp_path = tmp.name

    yield temp_path

    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def blurred_test_image():
    """Generates a severely blurred temporary image failing Laplacian variance threshold (< 5.0)."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        # Apply intense Gaussian blur to eliminate all high-frequency edges
        blurred = cv2.GaussianBlur(img, (99, 99), 30.0)
        cv2.imwrite(tmp.name, blurred)
        temp_path = tmp.name

    yield temp_path

    if os.path.exists(temp_path):
        os.remove(temp_path)


# ==============================================================================
# Scenario 1: Valid Evidence Acceptance
# ==============================================================================
def test_01_valid_evidence_acceptance(valid_test_image):
    """High completeness, verified provenance, and low operational risk must yield ACCEPT."""
    engine = RavenEngine(tau_accept=0.30, tau_escalate=0.65)

    findings = [
        {
            "finding_uid": "F-PPE-01",
            "label": "person",
            "agent_source": "object",
            "detection_confidence": 0.95,
            "bbox": [50.0, 50.0, 180.0, 400.0],
            "human_review_required": False,
            "model_version": "yolov11x_coco_v1.0"
        },
        {
            "finding_uid": "F-PPE-02",
            "label": "hardhat",
            "agent_source": "ppe",
            "detection_confidence": 0.94,
            "bbox": [70.0, 50.0, 150.0, 120.0],
            "human_review_required": False,
            "model_version": "yolov11n_shwd_ppe_v1.0"
        }
    ]
    context_data = {
        "object_result": {
            "objects": [{"label": "person", "bbox": [50, 50, 180, 400], "confidence": 0.95}]
        },
        "ppe_result": {
            "total_persons_evaluated": 1,
            "compliant_count": 1,
            "non_compliant_count": 0,
            "uncertain_count": 0,
            "compliance_status": "COMPLIANT",
            "person_evaluations": [
                {"person_id": "Worker #1", "compliance_status": "COMPLIANT", "is_occluded": False, "uncertainty": 0.05}
            ]
        }
    }

    assessment = engine.evaluate(
        query="Verify worker safety helmet compliance in workcell",
        intent="ppe_safety_compliance",
        findings=findings,
        context_data=context_data,
        executed_agents=["object", "ppe"],
        image_path=valid_test_image
    )

    assert assessment.decision == RavenDecision.ACCEPT
    assert assessment.risk_score < 0.30
    assert assessment.evidence_completeness >= 0.80
    assert assessment.provenance_certified is True
    assert assessment.safety_veto_applied is None
    assert assessment.escalation_dossier is None
    assert assessment.human_review_record is None


# ==============================================================================
# Scenario 2: Missing Critical Evidence
# ==============================================================================
def test_02_missing_critical_evidence(valid_test_image):
    """When mandatory evidence (head protection verification) is missing after PPE execution, ABSTAIN."""
    engine = RavenEngine(tau_accept=0.30, tau_escalate=0.65)

    # Worker body detected, but cranial equipment was never verified (compliant_count=0, non_compliant_count=0)
    findings = [
        {
            "finding_uid": "F-PPE-01",
            "label": "person",
            "agent_source": "object",
            "detection_confidence": 0.92,
            "bbox": [50.0, 50.0, 180.0, 400.0],
            "human_review_required": False,
            "model_version": "yolov11x_coco_v1.0"
        }
    ]
    context_data = {
        "object_result": {
            "objects": [{"label": "person", "bbox": [50, 50, 180, 400], "confidence": 0.92}]
        },
        "ppe_result": {
            "total_persons_evaluated": 1,
            "compliant_count": 0,
            "non_compliant_count": 0,
            "uncertain_count": 1,
            "compliance_status": "UNKNOWN",
            "person_evaluations": []
        }
    }

    assessment = engine.evaluate(
        query="Verify worker safety helmet compliance in workcell",
        intent="ppe_safety_compliance",
        findings=findings,
        context_data=context_data,
        executed_agents=["object", "ppe"],  # ppe executed but failed to satisfy ET-PPE-CRANIAL-02
        image_path=valid_test_image
    )

    assert assessment.decision == RavenDecision.ABSTAIN
    assert assessment.safety_veto_applied == "CRITICAL_EVIDENCE_UNAVAILABLE"
    assert assessment.human_review_record is not None
    assert assessment.escalation_dossier is not None
    assert any("ET-PPE-CRANIAL-02" in item for item in assessment.human_review_record.missing_evidence)


# ==============================================================================
# Scenario 3: Missing Optional Evidence
# ==============================================================================
def test_03_missing_optional_evidence(valid_test_image):
    """Missing optional evidence (high-visibility torso vest) must NOT prevent ACCEPT when mandatory tokens are satisfied."""
    engine = RavenEngine(tau_accept=0.30, tau_escalate=0.65)

    # Person and helmet verified; vest is absent (vest_count=0)
    findings = [
        {
            "finding_uid": "F-PPE-01",
            "label": "person",
            "agent_source": "object",
            "detection_confidence": 0.95,
            "bbox": [50.0, 50.0, 180.0, 400.0],
            "model_version": "yolov11x_coco_v1.0"
        },
        {
            "finding_uid": "F-PPE-02",
            "label": "hardhat",
            "agent_source": "ppe",
            "detection_confidence": 0.93,
            "bbox": [70.0, 50.0, 150.0, 120.0],
            "model_version": "yolov11n_shwd_ppe_v1.0"
        }
    ]
    context_data = {
        "object_result": {
            "objects": [{"label": "person", "bbox": [50, 50, 180, 400], "confidence": 0.95}]
        },
        "ppe_result": {
            "total_persons_evaluated": 1,
            "compliant_count": 1,
            "non_compliant_count": 0,
            "uncertain_count": 0,
            "vest_count": 0,  # Optional vest absent
            "person_evaluations": [
                {"person_id": "Worker #1", "compliance_status": "COMPLIANT", "is_occluded": False, "uncertainty": 0.05}
            ]
        }
    }

    assessment = engine.evaluate(
        query="Verify worker safety helmet compliance in workcell",
        intent="ppe_safety_compliance",
        findings=findings,
        context_data=context_data,
        executed_agents=["object", "ppe"],
        image_path=valid_test_image
    )

    assert assessment.decision == RavenDecision.ACCEPT
    assert assessment.safety_veto_applied is None
    # Vest token is present in repertoire as optional, but its absence did not veto
    vest_token = next(t for t in assessment.evidence_repertoire.tokens if t.token_id == "ET-PPE-TORSO-03")
    assert vest_token.required is False
    assert vest_token.present is False


# ==============================================================================
# Scenario 4: Critical Specialist Failure
# ==============================================================================
def test_04_critical_specialist_failure(valid_test_image):
    """Failure of a critical specialist (defect detector for surface crack inspection) forces ABSTAIN."""
    engine = RavenEngine()

    assessment = engine.evaluate(
        query="Inspect cold-rolled steel surface for fatigue crack defects",
        intent="defect_inspection",
        findings=[],
        context_data={},
        executed_agents=["object"],
        failed_agents=["defect"],  # Critical specialist failed during execution
        image_path=valid_test_image
    )

    assert assessment.decision == RavenDecision.ABSTAIN
    assert assessment.safety_veto_applied == "CRITICAL_SPECIALIST_FAILURE"
    assert "defect" in assessment.failed_critical_specialists
    assert assessment.human_review_record is not None
    assert assessment.escalation_dossier is not None


# ==============================================================================
# Scenario 5: Irrelevant Specialist Failure
# ==============================================================================
def test_05_irrelevant_specialist_failure(valid_test_image):
    """Failure of an irrelevant specialist (OCR in PPE query) must NOT cause ABSTAIN."""
    engine = RavenEngine(tau_accept=0.30, tau_escalate=0.65)

    findings = [
        {
            "finding_uid": "F-PPE-01",
            "label": "person",
            "agent_source": "object",
            "detection_confidence": 0.95,
            "bbox": [50.0, 50.0, 180.0, 400.0],
            "model_version": "yolov11x_coco_v1.0"
        },
        {
            "finding_uid": "F-PPE-02",
            "label": "hardhat",
            "agent_source": "ppe",
            "detection_confidence": 0.94,
            "bbox": [70.0, 50.0, 150.0, 120.0],
            "model_version": "yolov11n_shwd_ppe_v1.0"
        }
    ]
    context_data = {
        "object_result": {
            "objects": [{"label": "person", "bbox": [50, 50, 180, 400], "confidence": 0.95}]
        },
        "ppe_result": {
            "total_persons_evaluated": 1,
            "compliant_count": 1,
            "non_compliant_count": 0,
            "uncertain_count": 0,
            "compliance_status": "COMPLIANT",
            "person_evaluations": [
                {"person_id": "Worker #1", "compliance_status": "COMPLIANT", "is_occluded": False, "uncertainty": 0.05}
            ]
        }
    }

    assessment = engine.evaluate(
        query="Verify worker safety helmet compliance in workcell",
        intent="ppe_safety_compliance",
        findings=findings,
        context_data=context_data,
        executed_agents=["object", "ppe"],
        failed_agents=["ocr"],  # Irrelevant specialist failure
        image_path=valid_test_image
    )

    # Irrelevant failure must NOT trigger CRITICAL_SPECIALIST_FAILURE veto
    assert assessment.safety_veto_applied is None
    assert "ocr" in assessment.failed_irrelevant_specialists
    assert len(assessment.failed_critical_specialists) == 0
    assert assessment.decision == RavenDecision.ACCEPT


# ==============================================================================
# Scenario 6: Unresolved Contradiction
# ==============================================================================
def test_06_unresolved_contradiction(valid_test_image):
    """Severe spatial contradiction between defect and OCR without resolver forces ABSTAIN."""
    engine = RavenEngine()

    findings = [
        {
            "finding_uid": "F-DFT-01",
            "label": "crack",
            "agent_source": "defect",
            "detection_confidence": 0.88,
            "bbox": [100.0, 100.0, 200.0, 160.0],
            "model_version": "yolov11n_neudet_defect_v1.0"
        },
        {
            "finding_uid": "F-OCR-01",
            "label": "SERIAL-1234",
            "agent_source": "ocr",
            "detection_confidence": 0.89,
            "bbox": [105.0, 105.0, 195.0, 155.0],  # Direct IoU > 0.85 spatial overlap
            "model_version": "crnn_iiit5k_ocr_v1.0"
        }
    ]
    context_data = {
        "defect_result": {
            "defects": [{"class": "crack", "bbox": [100.0, 100.0, 200.0, 160.0], "confidence": 0.88}]
        },
        "ocr_result": {
            "detected_text": "SERIAL-1234",
            "bounding_boxes": [[105.0, 105.0, 195.0, 155.0]]
        }
    }

    assessment = engine.evaluate(
        query="Inspect cold-rolled steel surface for defects and serial number",
        intent="defect_inspection",
        findings=findings,
        context_data=context_data,
        executed_agents=["defect", "ocr"],  # No evidence_validation or scene resolver
        image_path=valid_test_image
    )

    assert assessment.decision == RavenDecision.ABSTAIN
    assert assessment.safety_veto_applied == "CRITICAL_CONTRADICTION_UNRESOLVED"
    assert len(assessment.conflict_records) > 0
    assert assessment.conflict_records[0].resolution_status == "UNRESOLVED"
    assert assessment.human_review_record is not None


# ==============================================================================
# Scenario 7: Resolved Contradiction
# ==============================================================================
def test_07_resolved_contradiction(valid_test_image):
    """When an authorized resolver specialist confirms finding, conflict is RESOLVED and allows ACCEPT."""
    engine = RavenEngine(tau_accept=0.35, tau_escalate=0.65)

    findings = [
        {
            "finding_uid": "F-DFT-01",
            "label": "crack",
            "agent_source": "defect",
            "detection_confidence": 0.92,
            "bbox": [100.0, 100.0, 200.0, 160.0],
            "model_version": "yolov11n_neudet_defect_v1.0",
            "is_confirmed_defect": True
        },
        {
            "finding_uid": "F-OCR-01",
            "label": "SERIAL-1234",
            "agent_source": "ocr",
            "detection_confidence": 0.88,
            "bbox": [105.0, 105.0, 195.0, 155.0],
            "model_version": "crnn_iiit5k_ocr_v1.0"
        },
        {
            "finding_uid": "F-VAL-01",
            "label": "Validated flaw morphology",
            "agent_source": "evidence_validation",  # Authorized resolver
            "detection_confidence": 0.96,
            "resolution_for": "F-DFT-01",
            "model_version": "astm_e165_rag_v1.0"
        }
    ]
    context_data = {
        "defect_result": {
            "defects_detected": 1,
            "defect_scan_completed": True,
            "defects": [{"class": "crack", "bbox": [100.0, 100.0, 200.0, 160.0], "confidence": 0.92}]
        },
        "ocr_result": {
            "ocr_scan_completed": True,
            "detected_text": "SERIAL-1234",
            "bounding_boxes": [[105.0, 105.0, 195.0, 155.0]]
        }
    }

    assessment = engine.evaluate(
        query="Inspect cold-rolled steel surface for defects and serial number",
        intent="defect_inspection",
        findings=findings,
        context_data=context_data,
        executed_agents=["defect", "ocr", "evidence_validation"],  # Authorized resolver present
        image_path=valid_test_image
    )

    # Contradiction was resolved by evidence_validation
    assert assessment.safety_veto_applied is None
    assert any(rec.resolution_status == "RESOLVED" for rec in assessment.conflict_records)
    assert assessment.decision == RavenDecision.ACCEPT


# ==============================================================================
# Scenario 8: Invalid Model Provenance
# ==============================================================================
def test_08_invalid_model_provenance(valid_test_image):
    """Untrusted or tampered model version provenance forces ABSTAIN."""
    engine = RavenEngine()

    findings = [
        {
            "finding_uid": "F-TAMPER-01",
            "label": "person",
            "agent_source": "object",
            "detection_confidence": 0.99,
            "model_version": "untrusted_tampered_model_hash_x99",  # Tampered model
            "bbox": [50, 50, 150, 300]
        }
    ]

    assessment = engine.evaluate(
        query="Verify worker safety helmet compliance in workcell",
        intent="ppe_safety_compliance",
        findings=findings,
        context_data={},
        executed_agents=["object"],
        image_path=valid_test_image
    )

    assert assessment.decision == RavenDecision.ABSTAIN
    assert assessment.safety_veto_applied == "INVALID_EVIDENCE_PROVENANCE"
    assert assessment.provenance_certified is False
    assert len(assessment.invalid_findings) > 0


# ==============================================================================
# Scenario 9: Invalid Image Provenance
# ==============================================================================
def test_09_invalid_image_provenance():
    """Missing or inaccessible image path forces immediate ABSTAIN."""
    engine = RavenEngine()

    assessment = engine.evaluate(
        query="Inspect steel sheet for cracks",
        intent="defect_inspection",
        findings=[],
        context_data={},
        executed_agents=["defect"],
        image_path="C:/non_existent/phantom_path/corrupted_image.png"
    )

    assert assessment.decision == RavenDecision.ABSTAIN
    assert assessment.safety_veto_applied == "INVALID_IMAGE_PROVENANCE_OR_QUALITY"
    assert assessment.provenance_certified is False


# ==============================================================================
# Scenario 10: Unvalidated Physical Absence
# ==============================================================================
def test_10_unvalidated_physical_absence(valid_test_image):
    """Advisory physical absence without statistical site certification forces VERIFY."""
    engine = RavenEngine()

    # Detector found zero defects, asserting absence advisory without certificate
    assessment = engine.evaluate(
        query="Inspect steel sheet for cracks",
        intent="defect_inspection",
        findings=[],
        context_data={"defect_result": {"defects_detected": 0}},
        executed_agents=["defect"],
        advisory_absence_flags={"absence_advisory": True},
        absence_certificate=None,  # No certificate provided
        image_path=valid_test_image
    )

    assert assessment.decision == RavenDecision.VERIFY
    assert assessment.safety_veto_applied == "UNVALIDATED_PHYSICAL_ABSENCE"
    assert "scene" in assessment.secondary_agents_suggested or "evidence_re-scan" in assessment.secondary_agents_suggested


# ==============================================================================
# Scenario 11: Expired or Revoked Absence Validation
# ==============================================================================
def test_11_expired_or_revoked_absence_validation(valid_test_image):
    """Expired or revoked site absence certificate triggers safety veto ABSTAIN."""
    engine = RavenEngine()

    # Create certificate that expired 10 minutes ago
    expired_cert = SiteAbsenceValidationCertificate(
        certificate_id="CERT-EXPIRED-01",
        workcell_id="WORKCELL_01",
        camera_config={"camera_id": "CAM_OVERHEAD_01"},
        detector_checkpoint_hash="chk_yolo_v11_baseline_hash",
        operational_subgroup="day_shift_ambient",
        validation_manifest_id="MAN-VAL-2026-001",
        validation_trials_n=100,
        observed_false_negatives_k=0,
        stored_upper_bound=0.0362,
        expiry_timestamp=time.time() - 600,  # Expired
        is_revoked=False
    )

    assessment = engine.evaluate(
        query="Inspect steel sheet for cracks",
        intent="defect_inspection",
        findings=[],
        context_data={},
        executed_agents=["defect"],
        absence_certificate=expired_cert,
        workcell_id="WORKCELL_01",
        image_path=valid_test_image
    )

    assert assessment.decision == RavenDecision.ABSTAIN
    assert assessment.safety_veto_applied == "INVALID_OR_EXPIRED_ABSENCE_CERTIFICATE"

    # Also verify revoked certificate
    revoked_cert = SiteAbsenceValidationCertificate(
        certificate_id="CERT-REVOKED-02",
        workcell_id="WORKCELL_01",
        camera_config={"camera_id": "CAM_OVERHEAD_01"},
        detector_checkpoint_hash="chk_yolo_v11_baseline_hash",
        operational_subgroup="day_shift_ambient",
        validation_manifest_id="MAN-VAL-2026-002",
        validation_trials_n=100,
        observed_false_negatives_k=0,
        stored_upper_bound=0.0362,
        expiry_timestamp=time.time() + 86400,
        is_revoked=True  # Revoked
    )

    assessment_revoked = engine.evaluate(
        query="Inspect steel sheet for cracks",
        intent="defect_inspection",
        findings=[],
        context_data={},
        executed_agents=["defect"],
        absence_certificate=revoked_cert,
        workcell_id="WORKCELL_01",
        image_path=valid_test_image
    )

    assert assessment_revoked.decision == RavenDecision.ABSTAIN
    assert assessment_revoked.safety_veto_applied == "INVALID_OR_EXPIRED_ABSENCE_CERTIFICATE"


# ==============================================================================
# Scenario 12: Poor Image Quality
# ==============================================================================
def test_12_poor_image_quality(blurred_test_image):
    """Severe optical blur (Laplacian variance < 5.0) fails quality gate and forces ABSTAIN."""
    engine = RavenEngine()

    assessment = engine.evaluate(
        query="Inspect steel surface for defects",
        intent="defect_inspection",
        findings=[],
        context_data={},
        executed_agents=["defect"],
        image_path=blurred_test_image
    )

    assert assessment.decision == RavenDecision.ABSTAIN
    assert assessment.safety_veto_applied == "INVALID_IMAGE_PROVENANCE_OR_QUALITY"
    assert assessment.provenance_certified is False


# ==============================================================================
# Scenario 13: Worker Cleanup Failure
# ==============================================================================
def test_13_worker_cleanup_failure(valid_test_image):
    """Failure to clean up isolated worker process triggers worker containment veto ABSTAIN."""
    engine = RavenEngine()

    assessment = engine.evaluate(
        query="Inspect steel surface for defects",
        intent="defect_inspection",
        findings=[],
        context_data={},
        executed_agents=["defect"],
        worker_cleanup_failed=True,  # Execution sandbox containment failure
        image_path=valid_test_image
    )

    assert assessment.decision == RavenDecision.ABSTAIN
    assert assessment.safety_veto_applied == "WORKER_CONTAINMENT_FAILURE"
    assert assessment.provenance_certified is False


# ==============================================================================
# Scenario 14: Budget Exhaustion with Incomplete Evidence
# ==============================================================================
def test_14_budget_exhaustion_with_incomplete_evidence(valid_test_image):
    """Budget exhaustion when mandatory evidence tokens remain unsatisfied forces ABSTAIN."""
    engine = RavenEngine()

    # Repertoire requires defect localization and morphology, but budget ran out before execution
    assessment = engine.evaluate(
        query="Inspect cold-rolled steel surface for inclusion defects",
        intent="defect_inspection",
        findings=[],
        context_data={"defect_result": {}},
        executed_agents=[],  # Router stopped early due to budget
        budget_exhausted=True,
        image_path=valid_test_image
    )

    assert assessment.decision == RavenDecision.ABSTAIN
    assert assessment.safety_veto_applied == "BUDGET_EXHAUSTION_INCOMPLETE_EVIDENCE"
    assert assessment.human_review_record is not None
    assert assessment.escalation_dossier is not None


# ==============================================================================
# Scenario 15: Human Escalation Record Generation
# ==============================================================================
def test_15_human_escalation_record_generation(valid_test_image):
    """Human review record and escalation dossier must contain full forensic details and regulatory clauses."""
    engine = RavenEngine()

    findings = [
        {
            "finding_uid": "F-PPE-UNC-01",
            "label": "Worker #1: Uncertain PPE Compliance (Occluded)",
            "agent_source": "ppe",
            "bbox": [50.0, 50.0, 120.0, 200.0],
            "human_review_required": True,
            "model_version": "yolov11n_shwd_ppe_v1.0"
        }
    ]
    context_data = {
        "ppe_result": {
            "total_persons_evaluated": 1,
            "uncertain_count": 1,
            "person_evaluations": [
                {"person_id": "Worker #1", "compliance_status": "UNCERTAIN", "is_occluded": True, "uncertainty": 0.75}
            ]
        }
    }

    assessment = engine.evaluate(
        query="Are workers wearing required hard hats and safety gear?",
        intent="ppe_safety_compliance",
        findings=findings,
        context_data=context_data,
        executed_agents=["object", "ppe"],
        image_path=valid_test_image
    )

    assert assessment.decision == RavenDecision.ABSTAIN

    # Verify HumanReviewRecord contents
    h_rec = assessment.human_review_record
    assert isinstance(h_rec, HumanReviewRecord)
    assert h_rec.decision == RavenDecision.ABSTAIN.value
    assert len(h_rec.relevant_regions) > 0
    assert h_rec.relevant_regions[0]["bbox"] == [50.0, 50.0, 120.0, 200.0]
    assert len(h_rec.recommended_actions) > 0
    assert "safety guarantee" in h_rec.disclaimer.lower() or "decision support" in h_rec.disclaimer.lower()

    # Verify HumanEscalationDossier contents
    dossier = assessment.escalation_dossier
    assert isinstance(dossier, HumanEscalationDossier)
    assert dossier.urgency_level in ["HIGH", "CRITICAL"]
    assert any("1926" in clause for clause in dossier.regulatory_clauses)
    assert len(dossier.primary_concerns) > 0
    assert len(dossier.recommended_human_actions) > 0


# ==============================================================================
# Scenario 16: Safety Gate Precedence Over Low Deficit
# ==============================================================================
def test_16_safety_gate_precedence_over_low_deficit(valid_test_image):
    """Even when numerical deficit is near zero (D=0.05), a safety gate violation strictly overrides to VERIFY or ABSTAIN."""
    engine = RavenEngine(tau_accept=0.30, tau_escalate=0.65)

    # All evidence tokens appear satisfied
    findings = [
        {
            "finding_uid": "F-DFT-01",
            "label": "crack",
            "agent_source": "defect",
            "detection_confidence": 0.95,
            "bbox": [100.0, 100.0, 200.0, 160.0],
            "model_version": "yolov11n_neudet_defect_v1.0"
        },
        {
            "finding_uid": "F-OCR-01",
            "label": "SERIAL-1234",
            "agent_source": "ocr",
            "detection_confidence": 0.92,
            "bbox": [105.0, 105.0, 195.0, 155.0],  # Direct unresolved severe conflict
            "model_version": "crnn_iiit5k_ocr_v1.0"
        }
    ]
    context_data = {
        "defect_result": {
            "defects_detected": 1,
            "defects": [{"class": "crack", "bbox": [100.0, 100.0, 200.0, 160.0], "confidence": 0.95}]
        },
        "ocr_result": {
            "detected_text": "SERIAL-1234",
            "bounding_boxes": [[105.0, 105.0, 195.0, 155.0]]
        }
    }

    assessment = engine.evaluate(
        query="Inspect cold-rolled steel surface for defects and serial number",
        intent="defect_inspection",
        findings=findings,
        context_data=context_data,
        executed_agents=["defect", "ocr"],
        image_path=valid_test_image
    )

    # Numerical deficit is minimal, but unresolved contradiction veto strictly takes precedence over ACCEPT
    assert assessment.decision != RavenDecision.ACCEPT
    assert assessment.decision == RavenDecision.ABSTAIN
    assert assessment.safety_veto_applied == "CRITICAL_CONTRADICTION_UNRESOLVED"
