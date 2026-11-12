"""
Unit & Integration Tests for RAVEN (Risk-Aware Visual Evidence Navigator)
Tests uncertainty calibration, cross-agent conflict entropy, evidence completeness,
tri-state decision policies (ACCEPT, VERIFY, ABSTAIN), and forensic escalation dossiers.
"""

import pytest
from backend.app.raven.schemas import RavenDecision, RavenAssessment, HumanEscalationDossier
from backend.app.raven.calibration import (
    calibrate_confidence,
    compute_spatial_uncertainty,
    compute_occlusion_uncertainty,
    compute_composite_uncertainty
)
from backend.app.raven.conflict import CrossAgentConflictEvaluator
from backend.app.raven.engine import RavenEngine


def test_confidence_calibration():
    """Verify that temperature scaling moderates overconfident neural predictions."""
    # Extreme overconfidence (0.999) must be moderated to prevent false certainty
    cal_high = calibrate_confidence(0.999, temperature=1.35)
    assert 0.90 <= cal_high <= 0.98

    # Low confidence (0.20)
    cal_low = calibrate_confidence(0.20, temperature=1.35)
    assert cal_low < 0.30

    # Boundary conditions
    assert calibrate_confidence(0.0) == 0.0
    assert calibrate_confidence(1.0) == 0.98


def test_spatial_uncertainty_resolution_bounds():
    """Verify that sub-resolution bounding boxes carry high epistemic uncertainty."""
    # Large object (w=150, h=300): ample pixel resolution
    large_box = [100.0, 100.0, 250.0, 400.0]
    unc_large = compute_spatial_uncertainty(large_box, min_dimension_px=75.0)
    assert unc_large <= 0.25

    # Small object (w=20, h=30): low pixel footprint
    small_box = [100.0, 100.0, 120.0, 130.0]
    unc_small = compute_spatial_uncertainty(small_box, min_dimension_px=75.0)
    assert unc_small >= 0.60

    # Degenerate or missing box
    assert compute_spatial_uncertainty([]) >= 0.50
    assert compute_spatial_uncertainty([10.0, 10.0, 10.0, 10.0]) >= 0.80


def test_occlusion_uncertainty_bounds():
    """Verify that mutual bounding box overlaps inflate occlusion uncertainty."""
    target_box = [100.0, 100.0, 200.0, 300.0]
    non_overlapping = [[300.0, 300.0, 400.0, 500.0]]
    strongly_overlapping = [[120.0, 120.0, 220.0, 320.0]]  # High IoU

    unc_none = compute_occlusion_uncertainty(target_box, non_overlapping)
    assert unc_none <= 0.10

    unc_overlap = compute_occlusion_uncertainty(target_box, strongly_overlapping)
    assert unc_overlap >= 0.50


def test_conflict_evaluator_count_discrepancy():
    """Detect entity count contradictions between object detection and PPE agent."""
    context_data = {
        "object_result": {
            "objects": [
                {"label": "person", "bbox": [10, 10, 50, 100]},
                {"label": "person", "bbox": [60, 10, 100, 100]},
                {"label": "person", "bbox": [110, 10, 150, 100]}
            ]
        },
        "ppe_result": {
            "total_persons_evaluated": 0,  # Contradiction: 3 persons found by detector, 0 evaluated by PPE
            "person_evaluations": []
        }
    }

    report = CrossAgentConflictEvaluator.evaluate(
        findings=[],
        context_data=context_data,
        executed_agents=["object", "ppe"]
    )

    assert report.has_conflict is True
    assert report.conflict_entropy >= 0.40
    assert any(d["type"] == "COUNT_DISCREPANCY_SEVERE" for d in report.discrepancies)


def test_conflict_evaluator_spatial_defect_ocr_intersection():
    """Detect spatial conflict when a defect bbox overlaps an OCR text reading region."""
    context_data = {
        "defect_result": {
            "defects": [
                {"class": "scratches", "bbox": [50.0, 50.0, 150.0, 100.0], "anomaly_score": 0.85}
            ]
        },
        "ocr_result": {
            "detected_text": "SERIAL NO: 49201",
            "bounding_boxes": [
                [55.0, 52.0, 145.0, 98.0]  # Direct overlap with defect
            ]
        }
    }

    report = CrossAgentConflictEvaluator.evaluate(
        findings=[],
        context_data=context_data,
        executed_agents=["defect", "ocr"]
    )

    assert report.has_conflict is True
    assert any(d["type"] == "SPATIAL_INTERSECTION_CONFLICT" for d in report.discrepancies)


def test_raven_decision_accept_on_complete_evidence():
    """When evidence is complete, high confidence, and no conflict, decision must be ACCEPT."""
    engine = RavenEngine(tau_accept=0.30, tau_escalate=0.65)

    findings = [
        {
            "finding_uid": "F-CNT-01",
            "label": "2 Industrial Cranes",
            "agent_source": "object",
            "detection_confidence": 0.94,
            "human_review_required": False
        }
    ]
    context_data = {
        "object_result": {
            "objects": [
                {"label": "crane", "bbox": [10, 10, 100, 200]},
                {"label": "crane", "bbox": [200, 10, 300, 200]}
            ]
        }
    }

    assessment = engine.evaluate(
        query="Count the number of industrial cranes",
        intent="object_counting_detection",
        findings=findings,
        context_data=context_data,
        executed_agents=["object"]
    )

    assert assessment.decision == RavenDecision.ACCEPT
    assert assessment.risk_score < 0.30
    assert assessment.evidence_completeness >= 0.80
    assert assessment.escalation_dossier is None


def test_raven_decision_verify_on_moderate_ambiguity():
    """When localized ambiguity exists, decision must be VERIFY with secondary agent suggestions."""
    engine = RavenEngine(tau_accept=0.30, tau_escalate=0.65)

    # Moderate defect without full ASTM standard lookup
    findings = [
        {
            "finding_uid": "F-DFT-01",
            "label": "Potential inclusion flaw",
            "agent_source": "defect",
            "anomaly_score": 0.65,
            "human_review_required": False
        }
    ]
    context_data = {
        "defect_result": {
            "defects": [
                {"class": "inclusion", "anomaly_score": 0.70, "confidence": 0.82, "is_confirmed": True}
            ]
        }
    }

    assessment = engine.evaluate(
        query="Inspect cold-rolled steel surface for inclusion defects",
        intent="defect_inspection",
        findings=findings,
        context_data=context_data,
        executed_agents=["defect"]
    )

    # Risk score should land in [0.30, 0.65)
    assert assessment.decision == RavenDecision.VERIFY
    assert 0.30 <= assessment.risk_score < 0.65
    assert len(assessment.secondary_agents_suggested) > 0
    assert "rag" in assessment.secondary_agents_suggested or "scene" in assessment.secondary_agents_suggested


def test_raven_decision_abstain_on_critical_occluded_ppe():
    """In safety-critical queries with occlusion, decision must be ABSTAIN with full dossier."""
    engine = RavenEngine(tau_accept=0.30, tau_escalate=0.65)

    findings = [
        {
            "finding_uid": "F-PPE-01",
            "label": "Worker #1: Uncertain PPE Compliance (Occluded)",
            "agent_source": "ppe",
            "human_review_required": True
        }
    ]
    context_data = {
        "ppe_result": {
            "total_persons_evaluated": 1,
            "uncertain_count": 1,
            "person_evaluations": [
                {
                    "person_id": "Worker #1",
                    "bbox": [50, 50, 120, 200],
                    "compliance_status": "UNCERTAIN",
                    "is_occluded": True,
                    "uncertainty": 0.75
                }
            ]
        }
    }

    assessment = engine.evaluate(
        query="Are workers wearing required hard hats and safety gear?",
        intent="ppe_safety_compliance",
        findings=findings,
        context_data=context_data,
        executed_agents=["object", "ppe"]
    )

    assert assessment.decision == RavenDecision.ABSTAIN
    assert assessment.risk_score >= 0.65
    assert assessment.hazard_severity >= 0.85
    assert assessment.escalation_dossier is not None
    assert assessment.escalation_dossier.urgency_level in ["HIGH", "CRITICAL"]
    assert len(assessment.escalation_dossier.primary_concerns) > 0
    assert len(assessment.escalation_dossier.recommended_human_actions) > 0
    assert any("1926" in clause for clause in assessment.escalation_dossier.regulatory_clauses)
    assert "advisory" in assessment.escalation_dossier.bounded_liability_statement.lower()


def test_raven_engine_robustness_on_empty_and_corrupt_inputs():
    """Engine must never crash on empty findings or None context; scores must stay in [0, 1]."""
    engine = RavenEngine()

    assessment = engine.evaluate(
        query="",
        intent="general_scene_inquiry",
        findings=[],
        context_data={},
        executed_agents=[]
    )

    assert assessment.decision in [RavenDecision.ACCEPT, RavenDecision.VERIFY, RavenDecision.ABSTAIN]
    assert 0.0 <= assessment.risk_score <= 1.0
    assert 0.0 <= assessment.hazard_severity <= 1.0
    assert 0.0 <= assessment.evidence_completeness <= 1.0
    assert 0.0 <= assessment.uncertainty_index <= 1.0
    assert 0.0 <= assessment.conflict_entropy <= 1.0
    assert assessment.to_dict()["risk_score"] == assessment.risk_score
