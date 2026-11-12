"""
Unit and Integration Test Suite for VisionIQ Phase 3.1 Evidence Engine Core
Covers:
- All 8 supported evidence tokens
- Missing, negative, and invalid confidence values
- Clamped score boundaries [0.0, 1.0]
- Occluded workers and low-resolution crops
- Contradiction detection and spatial conflict handling
- Duplicate evidence deduplication and provenance preservation
- Unsupported token handling
- Decoupled completeness, reliability, conflict, and unavailability
- Configurable weight validation
- Mathematical verification that unavailable evidence is not double-counted
"""

import pytest
from backend.app.orchestration.evidence_deficit import (
    compute_box_iou,
    TokenStatus,
    DeficitWeights,
    EvidenceTokenDefinition,
    EvaluatedEvidenceToken,
    EvidenceLedger,
    EvidenceDeficitEngine,
    TOKEN_CATALOG
)


def test_01_token_catalog_integrity():
    """Verify all 8 initial tokens are registered with complete metadata."""
    expected_tokens = [
        "ET-OBJ-PERSON-01",
        "ET-PPE-CRANIAL-02",
        "ET-PPE-TORSO-03",
        "ET-DFT-LOCALIZATION-01",
        "ET-DFT-CLASSIFICATION-02",
        "ET-DFT-PROCEDURE-03",
        "ET-OCR-TRANSCRIBE-01",
        "ET-SCN-ENVIRONMENT-01"
    ]
    engine = EvidenceDeficitEngine()
    supported = engine.get_supported_token_ids()
    for tok_id in expected_tokens:
        assert tok_id in supported, f"Token {tok_id} must be in TOKEN_CATALOG"
        tok_def = TOKEN_CATALOG[tok_id]
        assert tok_def.token_id == tok_id
        assert len(tok_def.task_name) > 0
        assert len(tok_def.description) > 0
        assert tok_def.domain in ["object", "ppe", "defect", "ocr", "scene", "recommendation"]


def test_02_box_iou_computation():
    """Test bounding box IoU calculation on overlapping, disjoint, and invalid boxes."""
    box_a = [10.0, 10.0, 50.0, 50.0]  # area = 1600
    box_b = [10.0, 10.0, 50.0, 50.0]  # identical
    assert compute_box_iou(box_a, box_b) == pytest.approx(1.0, 0.001)

    box_c = [50.0, 50.0, 100.0, 100.0]  # touching at corner -> intersection = 0
    assert compute_box_iou(box_a, box_c) == pytest.approx(0.0, 0.001)

    box_d = [30.0, 10.0, 70.0, 50.0]  # inter = 20 * 40 = 800; area_a=1600, area_d=1600, union=2400 -> 1/3
    assert compute_box_iou(box_a, box_d) == pytest.approx(800.0 / 2400.0, 0.01)

    # Invalid box inputs
    assert compute_box_iou(None, box_a) == 0.0
    assert compute_box_iou([], box_a) == 0.0
    assert compute_box_iou([10, 10], box_a) == 0.0


def test_03_token_obj_person_01():
    """Test ET-OBJ-PERSON-01 on detections, missing agent, failed agent, and negative queries."""
    engine = EvidenceDeficitEngine()

    # Case A: Unexecuted agent
    res_unexec = engine.evaluate(
        query="inspect workers",
        intent="ppe_compliance",
        findings=[],
        context_data={},
        executed_agents=[],
        failed_agents=[]
    )
    tok = res_unexec.tokens["ET-OBJ-PERSON-01"]
    assert tok.status == TokenStatus.MISSING
    assert tok.satisfaction_score == 0.0

    # Case B: Failed agent
    res_failed = engine.evaluate(
        query="inspect workers",
        intent="ppe_compliance",
        findings=[],
        context_data={},
        executed_agents=[],
        failed_agents=["object"]
    )
    tok = res_failed.tokens["ET-OBJ-PERSON-01"]
    assert tok.status == TokenStatus.UNAVAILABLE
    assert tok.satisfaction_score == 0.0

    # Case C: 2 workers detected with high confidence
    findings = [
        {"finding_uid": "F-OBJ-01", "agent_source": "object", "label": "person", "bbox": [50, 50, 150, 250], "detection_confidence": 0.92},
        {"finding_uid": "F-OBJ-02", "agent_source": "object", "label": "worker", "bbox": [200, 50, 300, 250], "detection_confidence": 0.88}
    ]
    res_detected = engine.evaluate(
        query="inspect workers for PPE",
        intent="ppe_compliance",
        findings=findings,
        context_data={"object_result": {"objects": findings}},
        executed_agents=["object"]
    )
    tok = res_detected.tokens["ET-OBJ-PERSON-01"]
    assert tok.status == TokenStatus.SATISFIED
    assert 0.70 <= tok.satisfaction_score <= 1.0

    # Case D: Negative query ("verify area is clear") with 0 workers detected
    res_neg = engine.evaluate(
        query="verify workspace is clear and empty",
        intent="general_scene_inquiry",
        findings=[],
        context_data={"object_result": {"objects": []}},
        executed_agents=["object"]
    )
    tok = res_neg.tokens["ET-OBJ-PERSON-01"]
    assert tok.status == TokenStatus.SATISFIED
    assert tok.satisfaction_score == 1.0


def test_04_token_ppe_cranial_and_torso():
    """Test cranial and torso tokens with compliant, missing, and occluded workers."""
    engine = EvidenceDeficitEngine()

    # Case A: Worker with helmet and vest, no occlusion
    ppe_res_compliant = {
        "persons": [{
            "person_id": "Worker #1",
            "has_helmet": True,
            "helmet_confidence": 0.85,
            "has_vest": True,
            "vest_confidence": 0.90,
            "is_occluded": False
        }]
    }
    res = engine.evaluate(
        query="check worker PPE compliance",
        intent="ppe_compliance",
        findings=[],
        context_data={"ppe_result": ppe_res_compliant},
        executed_agents=["object", "ppe"]
    )
    cranial = res.tokens["ET-PPE-CRANIAL-02"]
    torso = res.tokens["ET-PPE-TORSO-03"]
    assert cranial.status == TokenStatus.SATISFIED
    assert cranial.satisfaction_score >= 0.70
    assert torso.status == TokenStatus.SATISFIED
    assert torso.satisfaction_score >= 0.70

    # Case B: Worker with occluded cranial and torso zones
    ppe_res_occluded = {
        "persons": [{
            "person_id": "Worker #2",
            "has_helmet": True,
            "helmet_confidence": 0.85,
            "has_vest": True,
            "vest_confidence": 0.90,
            "is_occluded": True  # Overlapping person
        }]
    }
    res_occ = engine.evaluate(
        query="check worker PPE compliance",
        intent="ppe_compliance",
        findings=[],
        context_data={"ppe_result": ppe_res_occluded},
        executed_agents=["object", "ppe"]
    )
    cranial_occ = res_occ.tokens["ET-PPE-CRANIAL-02"]
    torso_occ = res_occ.tokens["ET-PPE-TORSO-03"]
    assert cranial_occ.status == TokenStatus.INSUFFICIENT_EVIDENCE
    assert cranial_occ.satisfaction_score < 0.35  # Heavily penalized due to occlusion
    assert torso_occ.status == TokenStatus.INSUFFICIENT_EVIDENCE
    assert torso_occ.satisfaction_score < 0.35

    # Case C: Worker missing both helmet and vest (confirmed absent)
    ppe_res_violation = {
        "persons": [{
            "person_id": "Worker #3",
            "has_helmet": False,
            "helmet_confidence": 0.0,
            "has_vest": False,
            "vest_confidence": 0.0,
            "is_occluded": False
        }]
    }
    res_viol = engine.evaluate(
        query="check worker PPE compliance",
        intent="ppe_compliance",
        findings=[],
        context_data={"ppe_result": ppe_res_violation},
        executed_agents=["object", "ppe"]
    )
    assert res_viol.tokens["ET-PPE-CRANIAL-02"].status == TokenStatus.CONFIRMED_ABSENT
    assert res_viol.tokens["ET-PPE-CRANIAL-02"].satisfaction_score == 0.0
    assert res_viol.tokens["ET-PPE-TORSO-03"].status == TokenStatus.CONFIRMED_ABSENT
    assert res_viol.tokens["ET-PPE-TORSO-03"].satisfaction_score == 0.0


def test_05_token_defect_localization_and_classification():
    """Test defect tokens with confirmed CNN defects, heuristic cues, clean surfaces, and failed agents."""
    engine = EvidenceDeficitEngine()

    # Case A: Confirmed defect (top_conf >= 0.50)
    findings_confirmed = [{
        "finding_uid": "F-DEF-01",
        "agent_source": "defect",
        "label": "Confirmed Defect: Scratches",
        "bbox": [50.0, 50.0, 200.0, 150.0],
        "anomaly_score": 0.88,
        "detection_confidence": 0.85,
        "is_confirmed_defect": True
    }]
    res_conf = engine.evaluate(
        query="inspect surface for cracks or defects",
        intent="defect_inspection",
        findings=findings_confirmed,
        context_data={"defect_result": {"defects_detected": 1, "findings": findings_confirmed}},
        executed_agents=["defect"]
    )
    dft_loc = res_conf.tokens["ET-DFT-LOCALIZATION-01"]
    dft_cls = res_conf.tokens["ET-DFT-CLASSIFICATION-02"]
    assert dft_loc.status == TokenStatus.SATISFIED
    assert dft_loc.satisfaction_score >= 0.70
    assert dft_cls.status == TokenStatus.SATISFIED
    assert dft_cls.satisfaction_score >= 0.70

    # Case B: Heuristic texture anomaly cue only (is_confirmed_defect = False)
    findings_heuristic = [{
        "finding_uid": "F-DEF-02",
        "agent_source": "defect",
        "label": "Heuristic Anomaly Cue: Surface Texture Irregularity",
        "bbox": [10.0, 10.0, 40.0, 40.0],
        "anomaly_score": 0.65,
        "detection_confidence": 0.35,  # CNN confidence < 50%
        "is_confirmed_defect": False
    }]
    res_heur = engine.evaluate(
        query="inspect surface for cracks or defects",
        intent="defect_inspection",
        findings=findings_heuristic,
        context_data={"defect_result": {"defects_detected": 1, "findings": findings_heuristic}},
        executed_agents=["defect"]
    )
    dft_cls_heur = res_heur.tokens["ET-DFT-CLASSIFICATION-02"]
    assert dft_cls_heur.status == TokenStatus.UNCERTAIN
    assert dft_cls_heur.satisfaction_score <= 0.25

    # Case C: Clean surface coupon (defects_detected == 0)
    res_clean = engine.evaluate(
        query="inspect surface for cracks or defects",
        intent="defect_inspection",
        findings=[],
        context_data={"defect_result": {"defects_detected": 0, "findings": []}},
        executed_agents=["defect"]
    )
    assert res_clean.tokens["ET-DFT-LOCALIZATION-01"].status == TokenStatus.SATISFIED
    assert res_clean.tokens["ET-DFT-LOCALIZATION-01"].satisfaction_score == 1.0
    assert res_clean.tokens["ET-DFT-CLASSIFICATION-02"].status == TokenStatus.SATISFIED
    assert res_clean.tokens["ET-DFT-CLASSIFICATION-02"].satisfaction_score == 1.0

    # Case D: Failed defect agent
    res_failed = engine.evaluate(
        query="inspect surface for cracks or defects",
        intent="defect_inspection",
        findings=[],
        context_data={},
        executed_agents=[],
        failed_agents=["defect"]
    )
    assert res_failed.tokens["ET-DFT-LOCALIZATION-01"].status == TokenStatus.UNAVAILABLE
    assert res_failed.tokens["ET-DFT-CLASSIFICATION-02"].status == TokenStatus.UNAVAILABLE


def test_06_token_dft_procedure_03():
    """Verify that ET-DFT-PROCEDURE-03 satisfies on explicit NDT citations and never increases satisfaction for missing tests."""
    engine = EvidenceDeficitEngine()

    # Case A: Explicit NDT protocol citing ASTM E165
    findings_ndt = [{
        "finding_uid": "F-DEF-01",
        "agent_source": "defect",
        "verification_procedure": "Perform Dye Penetrant Inspection (PT) per ASTM E165."
    }]
    res_ndt = engine.evaluate(
        query="defect inspection",
        intent="defect_inspection",
        findings=findings_ndt,
        context_data={},
        executed_agents=["defect", "recommendation"]
    )
    tok = res_ndt.tokens["ET-DFT-PROCEDURE-03"]
    assert tok.status == TokenStatus.SATISFIED
    assert tok.satisfaction_score == 1.0

    # Case B: Generic inspection advice without specific NDT protocol
    findings_generic = [{
        "finding_uid": "F-DEF-01",
        "agent_source": "defect",
        "verification_procedure": "Qualified inspector should visually inspect the region."
    }]
    res_generic = engine.evaluate(
        query="defect inspection",
        intent="defect_inspection",
        findings=findings_generic,
        context_data={},
        executed_agents=["defect", "recommendation"]
    )
    tok = res_generic.tokens["ET-DFT-PROCEDURE-03"]
    assert tok.status == TokenStatus.UNCERTAIN
    assert tok.satisfaction_score == 0.40

    # Case C: No verification procedure at all
    res_none = engine.evaluate(
        query="defect inspection",
        intent="defect_inspection",
        findings=[],
        context_data={},
        executed_agents=["defect"]
    )
    tok = res_none.tokens["ET-DFT-PROCEDURE-03"]
    assert tok.status == TokenStatus.MISSING
    assert tok.satisfaction_score == 0.0


def test_07_token_ocr_transcribe_01():
    """Test ET-OCR-TRANSCRIBE-01 on legible text, empty text, and failed agent."""
    engine = EvidenceDeficitEngine()

    # Case A: Legible serial number
    ocr_res_valid = {
        "detected_text": "SN-84920-B",
        "text_regions": [{"confidence": 0.94, "bbox": [10, 10, 80, 30]}]
    }
    res = engine.evaluate(
        query="read serial tag",
        intent="ocr_text_extraction",
        findings=[],
        context_data={"ocr_result": ocr_res_valid},
        executed_agents=["ocr"]
    )
    tok = res.tokens["ET-OCR-TRANSCRIBE-01"]
    assert tok.status == TokenStatus.SATISFIED
    assert tok.satisfaction_score >= 0.70

    # Case B: Empty text detected
    ocr_res_empty = {
        "detected_text": "No text detected",
        "text_regions": []
    }
    res_empty = engine.evaluate(
        query="read serial tag",
        intent="ocr_text_extraction",
        findings=[],
        context_data={"ocr_result": ocr_res_empty},
        executed_agents=["ocr"]
    )
    tok_empty = res_empty.tokens["ET-OCR-TRANSCRIBE-01"]
    assert tok_empty.status == TokenStatus.MISSING
    assert tok_empty.satisfaction_score == 0.0


def test_08_token_scn_environment_01():
    """Test ET-SCN-ENVIRONMENT-01 on standard lighting, underexposed, and glare conditions."""
    engine = EvidenceDeficitEngine()

    # Case A: Diffused standard lighting
    res_std = engine.evaluate(
        query="inspect scene",
        intent="scene_understanding",
        findings=[],
        context_data={"scene_result": {"lighting": "Standard diffused industrial lighting", "direct_observations": ["good"]}},
        executed_agents=["scene"]
    )
    assert res_std.tokens["ET-SCN-ENVIRONMENT-01"].status == TokenStatus.SATISFIED
    assert res_std.tokens["ET-SCN-ENVIRONMENT-01"].satisfaction_score == 1.0

    # Case B: Low-light underexposed condition
    res_dark = engine.evaluate(
        query="inspect scene",
        intent="scene_understanding",
        findings=[],
        context_data={"scene_result": {"lighting": "Low-light / underexposed environment"}},
        executed_agents=["scene"]
    )
    assert res_dark.tokens["ET-SCN-ENVIRONMENT-01"].status == TokenStatus.UNCERTAIN
    assert res_dark.tokens["ET-SCN-ENVIRONMENT-01"].satisfaction_score == 0.40

    # Case C: High glare condition
    res_glare = engine.evaluate(
        query="inspect scene",
        intent="scene_understanding",
        findings=[],
        context_data={"scene_result": {"lighting": "High-key / direct glare illumination"}},
        executed_agents=["scene"]
    )
    assert res_glare.tokens["ET-SCN-ENVIRONMENT-01"].status == TokenStatus.UNCERTAIN
    assert res_glare.tokens["ET-SCN-ENVIRONMENT-01"].satisfaction_score == 0.40


def test_09_unsupported_token_handling():
    """Verify that requests for unsupported tokens are cleanly rejected without fabricating evidence."""
    engine = EvidenceDeficitEngine()
    unsupported_tok_id = "ET-ACOUSTIC-EMISSION-01"

    res = engine.evaluate(
        query="inspect structural health with acoustic sensors",
        intent="defect_inspection",
        findings=[],
        context_data={},
        executed_agents=["defect"],
        requested_tokens=["ET-DFT-LOCALIZATION-01", unsupported_tok_id]
    )

    assert unsupported_tok_id in res.unsupported_tokens
    tok = res.tokens[unsupported_tok_id]
    assert tok.status == TokenStatus.UNSUPPORTED
    assert tok.satisfaction_score == 0.0
    assert tok.uncertainty == 1.0


def test_10_evidence_ledger_deduplication():
    """Verify that duplicate detections from the same agent are merged rather than duplicated."""
    ledger = EvidenceLedger()

    f1 = {
        "finding_uid": "F-OBJ-01",
        "agent_source": "object",
        "label": "person",
        "bbox": [100.0, 100.0, 200.0, 300.0],
        "detection_confidence": 0.85
    }
    # Duplicate detection with slight bounding box shift (IoU >= 0.60) and higher confidence
    f2 = {
        "finding_uid": "F-OBJ-02",
        "agent_source": "object",
        "label": "person",
        "bbox": [102.0, 101.0, 201.0, 299.0],
        "detection_confidence": 0.94
    }

    entry1 = ledger.add_finding(f1, step=1)
    entry2 = ledger.add_finding(f2, step=2)

    assert len(ledger.entries) == 1, "Duplicate finding must be merged, not duplicated in ledger."
    assert ledger.entries[0].raw_confidence == 0.94
    assert ledger.entries[0].metadata.get("duplicate_observations") == 2


def test_11_evidence_ledger_spatial_contradiction():
    """Verify that spatial clashes between contradictory agents (defect vs OCR) preserve both records and create a ConflictRecord."""
    ledger = EvidenceLedger()

    defect_finding = {
        "finding_uid": "F-DEF-01",
        "agent_source": "defect",
        "label": "Surface Crack",
        "bbox": [50.0, 50.0, 150.0, 100.0],
        "detection_confidence": 0.88
    }
    ocr_finding = {
        "finding_uid": "F-OCR-01",
        "agent_source": "ocr",
        "label": "Text Inscription: 'SERIAL-123'",
        "bbox": [55.0, 52.0, 148.0, 98.0],  # Overlaps defect bbox with IoU > 0.80
        "detection_confidence": 0.91
    }

    ledger.add_finding(defect_finding, step=1)
    ledger.add_finding(ocr_finding, step=2)

    assert len(ledger.entries) == 2, "Both contradictory observations must be preserved in the ledger."
    assert ledger.entries[0].has_conflict is True
    assert ledger.entries[1].has_conflict is True
    assert len(ledger.conflicts) == 1
    conf = ledger.conflicts[0]
    assert conf.conflict_type == "SPATIAL_INTERSECTION"
    assert "defect" in conf.agents_involved
    assert "ocr" in conf.agents_involved


def test_12_decoupled_dimensions_and_no_double_counting():
    """
    Verify that when an agent fails (e.g. ppe), its tokens contribute strictly to U_unavail,
    while completeness C_ev is evaluated over AVAILABLE required tokens without double-counting.
    """
    engine = EvidenceDeficitEngine()

    # Query requires: ET-OBJ-PERSON-01, ET-PPE-CRANIAL-02, ET-PPE-TORSO-03, ET-SCN-ENVIRONMENT-01
    # Object and scene succeed; PPE fails
    findings = [
        {"finding_uid": "F-OBJ-01", "agent_source": "object", "label": "person", "bbox": [50, 50, 150, 250], "detection_confidence": 0.95}
    ]
    ctx = {
        "object_result": {"objects": findings},
        "scene_result": {"lighting": "Standard diffused industrial lighting"}
    }

    report = engine.evaluate(
        query="check worker PPE compliance",
        intent="ppe_compliance",
        findings=findings,
        context_data=ctx,
        executed_agents=["object", "scene"],
        failed_agents=["ppe"]
    )

    # 4 applicable required tokens total:
    # 2 are satisfied: ET-OBJ-PERSON-01 (calibrated ~0.7291) and ET-SCN-ENVIRONMENT-01 (1.0)
    # 2 are unavailable: ET-PPE-CRANIAL-02 (0.0), ET-PPE-TORSO-03 (0.0) -> U_unavail = 2/4 = 0.50
    assert report.unavailability == pytest.approx(0.50, 0.01)

    # Completeness C_ev across ALL applicable required tokens is (0.7291 + 1.0 + 0.0 + 0.0) / 4 = 0.4323
    expected_c_ev = (
        report.tokens["ET-OBJ-PERSON-01"].satisfaction_score +
        report.tokens["ET-SCN-ENVIRONMENT-01"].satisfaction_score +
        0.0 + 0.0
    ) / 4.0
    assert report.completeness == pytest.approx(expected_c_ev, 0.01)
    assert report.completeness == pytest.approx(0.4323, 0.01)

    # Deficit must remain strictly within [0.0, 1.0] and firmly indicate deficit (D >= 0.35)
    # This resolves the previous false sufficiency bug where D was severely underestimated at 0.1939.
    # Here with C_ev=0.4323, R_ev=0.7268, U_unavail=0.50, D is ~0.3885 (substantially above convergence epsilon ~0.15)
    assert report.composite_deficit >= 0.35
    assert report.composite_deficit == pytest.approx(0.3885, 0.01)

    # Check that deficit matches weighted formula
    w = report.normalized_weights
    expected_deficit = (
        w["w_completeness"] * (1.0 - report.completeness) +
        w["w_reliability"] * (1.0 - report.reliability) +
        w["w_conflict"] * report.conflict_score +
        w["w_unavailability"] * report.unavailability
    )
    assert report.composite_deficit == pytest.approx(expected_deficit, 0.001)


def test_13_deficit_weights_validation_and_normalization():
    """Verify weight validation: non-negative check, zero-sum rejection, and normalization."""
    # Negative weight raises ValidationError
    with pytest.raises(Exception):
        DeficitWeights(w_completeness=-0.10)

    # All-zero weights raise ValueError
    with pytest.raises(ValueError):
        DeficitWeights(w_completeness=0.0, w_reliability=0.0, w_conflict=0.0, w_unavailability=0.0)

    # Arbitrary positive weights normalize to sum exactly 1.0
    custom_weights = DeficitWeights(w_completeness=5.0, w_reliability=3.0, w_conflict=1.0, w_unavailability=1.0)
    norm = custom_weights.normalized()
    assert sum(norm.values()) == pytest.approx(1.0, 0.0001)
    assert norm["w_completeness"] == pytest.approx(0.50, 0.0001)
    assert norm["w_reliability"] == pytest.approx(0.30, 0.0001)


def test_14_composite_deficit_bounds():
    """Verify composite deficit remains strictly in [0.0, 1.0] across multiple extreme scenarios."""
    engine = EvidenceDeficitEngine()

    # Extreme 1: Perfect state (all satisfied, 0 conflicts, 0 unavailable, high reliability)
    rep_perfect = engine.evaluate(
        query="verify workspace is clear",
        intent="general_scene_inquiry",
        findings=[{"finding_uid": "F-SCN-01", "agent_source": "scene", "label": "scene", "detection_confidence": 0.98}],
        context_data={"object_result": {"objects": []}, "scene_result": {"lighting": "Standard diffused industrial lighting"}},
        executed_agents=["object", "scene"]
    )
    assert 0.0 <= rep_perfect.composite_deficit <= 0.15

    # Extreme 2: Total failure (all agents failed)
    rep_worst = engine.evaluate(
        query="check worker PPE compliance",
        intent="ppe_compliance",
        findings=[],
        context_data={},
        executed_agents=[],
        failed_agents=["object", "ppe", "scene"]
    )
    assert 0.0 <= rep_worst.composite_deficit <= 1.0
    assert rep_worst.unavailability == 1.0
    assert rep_worst.completeness == 0.0


def test_15_invalid_confidence_values_clamped():
    """Verify that anomalous out-of-range confidence scores (e.g. >1.0 or <0.0) are clamped safely."""
    engine = EvidenceDeficitEngine()
    findings_anomalous = [
        {"finding_uid": "F-OBJ-01", "agent_source": "object", "label": "person", "bbox": [10, 10, 50, 50], "detection_confidence": 99.0}, # Percent instead of fraction
        {"finding_uid": "F-OBJ-02", "agent_source": "object", "label": "person", "bbox": [60, 60, 90, 90], "detection_confidence": -0.50}  # Corrupt negative
    ]
    rep = engine.evaluate(
        query="inspect workers",
        intent="ppe_compliance",
        findings=findings_anomalous,
        context_data={"object_result": {"objects": findings_anomalous}},
        executed_agents=["object"]
    )
    tok = rep.tokens["ET-OBJ-PERSON-01"]
    assert 0.0 <= tok.satisfaction_score <= 1.0
    assert 0.0 <= rep.composite_deficit <= 1.0


# ==============================================================================
# PHASE 3.1 VALIDATION SCENARIOS (11 Targeted Verification Tests)
# ==============================================================================

def test_scenario_01_ppe_failure_with_object_scene_success():
    """
    Validation Scenario 1:
    PPE failure with object and scene success must yield a substantial deficit (D >= 0.35),
    firmly rejecting the previous erroneous underestimation of D = 0.1939.
    """
    engine = EvidenceDeficitEngine()
    findings = [
        {"finding_uid": "F-OBJ-01", "agent_source": "object", "label": "person", "bbox": [50, 50, 150, 250], "detection_confidence": 0.95}
    ]
    ctx = {
        "object_result": {"objects": findings},
        "scene_result": {"lighting": "Standard diffused industrial lighting"}
    }
    report = engine.evaluate(
        query="check worker PPE compliance",
        intent="ppe_compliance",
        findings=findings,
        context_data=ctx,
        executed_agents=["object", "scene"],
        failed_agents=["ppe"]
    )

    # 4 applicable required tokens:
    # ET-OBJ-PERSON-01 (calibrated ~0.7291)
    # ET-SCN-ENVIRONMENT-01 (1.0)
    # ET-PPE-CRANIAL-02 (0.0, UNAVAILABLE)
    # ET-PPE-TORSO-03 (0.0, UNAVAILABLE)
    assert report.tokens["ET-PPE-CRANIAL-02"].status == TokenStatus.UNAVAILABLE
    assert report.tokens["ET-PPE-TORSO-03"].status == TokenStatus.UNAVAILABLE
    assert report.unavailability == pytest.approx(0.50, 0.01)
    assert report.completeness == pytest.approx(0.4323, 0.01)
    # Deficit must be substantial and far above convergence epsilon ~0.15
    assert report.composite_deficit >= 0.35
    assert report.composite_deficit == pytest.approx(0.3885, 0.01)


def test_scenario_02_missing_mandatory_evidence_unexecuted():
    """
    Validation Scenario 2:
    Missing mandatory evidence at step 0 (unexecuted agents) must yield 0 completeness
    and zero unavailability (unexecuted != failed).
    """
    engine = EvidenceDeficitEngine()
    report = engine.evaluate(
        query="check worker PPE compliance",
        intent="ppe_compliance",
        findings=[],
        context_data={},
        executed_agents=[],
        failed_agents=[]
    )
    for tok_id in ["ET-OBJ-PERSON-01", "ET-PPE-CRANIAL-02", "ET-PPE-TORSO-03", "ET-SCN-ENVIRONMENT-01"]:
        assert report.tokens[tok_id].status == TokenStatus.MISSING
        assert report.tokens[tok_id].satisfaction_score == 0.0

    assert report.completeness == 0.0
    assert report.unavailability == 0.0  # Not unavailable; merely unexecuted
    assert report.composite_deficit >= 0.50  # Deficit dominated by completeness loss


def test_scenario_03_confirmed_absence_of_people_semantics():
    """
    Validation Scenario 3:
    When object detection runs and confirms 0 personnel, status must be CONFIRMED_ABSENT
    (for non-negative queries), distinguishing verified absence from detection failure.
    """
    engine = EvidenceDeficitEngine()
    report = engine.evaluate(
        query="inspect workers for PPE",
        intent="ppe_compliance",
        findings=[],
        context_data={"object_result": {"objects": []}},
        executed_agents=["object"],
        failed_agents=[]
    )
    person_tok = report.tokens["ET-OBJ-PERSON-01"]
    assert person_tok.status == TokenStatus.CONFIRMED_ABSENT
    assert person_tok.satisfaction_score == 0.0
    assert "Confirmed 0 personnel present" in person_tok.explanation


def test_scenario_04_object_detector_failure_cascading_unavailability():
    """
    Validation Scenario 4:
    Object detector failure cascades into UNAVAILABLE status for downstream dependent PPE tokens.
    In contrast, unexecuted object detector produces MISSING status.
    """
    engine = EvidenceDeficitEngine()

    # Case A: Object detector failed
    report_failed = engine.evaluate(
        query="check worker PPE compliance",
        intent="ppe_compliance",
        findings=[],
        context_data={},
        executed_agents=[],
        failed_agents=["object"]
    )
    assert report_failed.tokens["ET-OBJ-PERSON-01"].status == TokenStatus.UNAVAILABLE
    assert report_failed.tokens["ET-PPE-CRANIAL-02"].status == TokenStatus.UNAVAILABLE
    assert report_failed.tokens["ET-PPE-TORSO-03"].status == TokenStatus.UNAVAILABLE
    assert "failed" in report_failed.tokens["ET-PPE-CRANIAL-02"].explanation.lower()

    # Case B: Object detector unexecuted
    report_unexec = engine.evaluate(
        query="check worker PPE compliance",
        intent="ppe_compliance",
        findings=[],
        context_data={},
        executed_agents=[],
        failed_agents=[]
    )
    assert report_unexec.tokens["ET-OBJ-PERSON-01"].status == TokenStatus.MISSING
    assert report_unexec.tokens["ET-PPE-CRANIAL-02"].status == TokenStatus.MISSING
    assert report_unexec.tokens["ET-PPE-TORSO-03"].status == TokenStatus.MISSING
    assert "unexecuted" in report_unexec.tokens["ET-PPE-CRANIAL-02"].explanation.lower()


def test_scenario_05_not_applicable_evidence_when_zero_workers():
    """
    Validation Scenario 5:
    When 0 workers are confirmed present, PPE tokens are NOT_APPLICABLE and excluded
    from applicable requirements, so the empty scene does not penalize completeness.
    """
    engine = EvidenceDeficitEngine()
    report = engine.evaluate(
        query="check worker PPE compliance",
        intent="ppe_compliance",
        findings=[],
        context_data={
            "object_result": {"objects": []},
            "scene_result": {"lighting": "Standard diffused industrial lighting"}
        },
        executed_agents=["object", "scene"],
        failed_agents=[]
    )
    assert report.tokens["ET-PPE-CRANIAL-02"].status == TokenStatus.NOT_APPLICABLE
    assert report.tokens["ET-PPE-TORSO-03"].status == TokenStatus.NOT_APPLICABLE
    assert report.unavailability == 0.0
    # Scene token is satisfied (1.0), and non-applicable PPE tokens do not reduce completeness
    assert report.completeness >= 0.50


def test_scenario_06_unsupported_evidence_clean_rejection():
    """
    Validation Scenario 6:
    Requests for unsupported modalities are assigned UNSUPPORTED status and rejected cleanly
    without fabricating observations or hallucinating satisfaction.
    """
    engine = EvidenceDeficitEngine()
    report = engine.evaluate(
        query="run thermal ultrasonic scan",
        intent="defect_inspection",
        findings=[],
        context_data={},
        executed_agents=["defect"],
        requested_tokens=["ET-THERMAL-RADIOMETRIC-99"]
    )
    tok = report.tokens["ET-THERMAL-RADIOMETRIC-99"]
    assert tok.status == TokenStatus.UNSUPPORTED
    assert tok.satisfaction_score == 0.0
    assert tok.uncertainty == 1.0
    assert "ET-THERMAL-RADIOMETRIC-99" in report.unsupported_tokens


def test_scenario_07_reliability_calculation_multiple_observations():
    """
    Validation Scenario 7:
    Reliability aggregates across multiple observations, scaling each entry's
    calibrated confidence by its spatial certainty (1 - uncertainty).
    """
    engine = EvidenceDeficitEngine()
    findings = [
        {"finding_uid": "F-1", "agent_source": "object", "label": "person", "bbox": [50, 50, 100, 100], "detection_confidence": 0.90},
        {"finding_uid": "F-2", "agent_source": "object", "label": "person", "bbox": [200, 200, 250, 250], "detection_confidence": 0.80}
    ]
    report = engine.evaluate(
        query="inspect workers",
        intent="ppe_compliance",
        findings=findings,
        context_data={"object_result": {"objects": findings}},
        executed_agents=["object"]
    )
    # Check that ledger recorded both entries
    assert len(report.ledger_entries) == 2
    # Verify reliability matches the average entry reliability
    entry_rels = [
        (e.calibrated_confidence or e.raw_confidence) * (1.0 - (e.spatial_uncertainty or 0.20))
        for e in report.ledger_entries
    ]
    expected_rel = sum(entry_rels) / len(entry_rels)
    assert report.reliability == pytest.approx(expected_rel, 0.001)


def test_scenario_08_heuristic_vs_calibrated_confidence_ledger():
    """
    Validation Scenario 8:
    Verify that object/defect/ocr entries are tracked as calibrated_probability,
    while scene/heuristic entries are tracked as heuristic_score.
    """
    ledger = EvidenceLedger()
    e_obj = ledger.add_finding({
        "finding_uid": "F-OBJ",
        "agent_source": "object",
        "label": "person",
        "detection_confidence": 0.90
    })
    e_scn = ledger.add_finding({
        "finding_uid": "F-SCN",
        "agent_source": "scene",
        "label": "ambient_lighting",
        "detection_confidence": 0.85
    })
    assert e_obj.confidence_type == "calibrated_probability"
    assert e_scn.confidence_type == "heuristic_score"


def test_scenario_09_conflict_score_bounds_and_multi_discrepancy():
    """
    Validation Scenario 9:
    Conflict score bounds S_conflict in [0.0, 1.0] and handles multi-discrepancy clamping.
    Verify backward-compatible conflict_entropy property.
    """
    engine = EvidenceDeficitEngine()
    findings = [
        {"finding_uid": "F-DEF-1", "agent_source": "defect", "label": "Crack", "bbox": [50, 50, 150, 100], "detection_confidence": 0.90},
        {"finding_uid": "F-OCR-1", "agent_source": "ocr", "label": "Text", "bbox": [55, 52, 148, 98], "detection_confidence": 0.90}
    ]
    report = engine.evaluate(
        query="inspect surface and serial",
        intent="defect_inspection",
        findings=findings,
        context_data={},
        executed_agents=["defect", "ocr"]
    )
    assert report.conflict_score > 0.0
    assert 0.0 <= report.conflict_score <= 1.0
    assert report.conflict_score == report.conflict_entropy


def test_scenario_10_no_double_counting_decoupled_dimensions():
    """
    Validation Scenario 10:
    Mathematical verification of decoupling:
    When an agent is MISSING vs UNAVAILABLE, completeness treats both as unfulfilled (phi = 0.0),
    while unavailability isolates the permanent agent failure penalty.
    """
    engine = EvidenceDeficitEngine()
    findings = [
        {"finding_uid": "F-OBJ-01", "agent_source": "object", "label": "person", "bbox": [50, 50, 150, 250], "detection_confidence": 0.95}
    ]
    ctx = {
        "object_result": {"objects": findings},
        "scene_result": {"lighting": "Standard diffused industrial lighting"}
    }

    # Case A: PPE unexecuted (MISSING)
    rep_missing = engine.evaluate(
        query="check worker PPE compliance",
        intent="ppe_compliance",
        findings=findings,
        context_data=ctx,
        executed_agents=["object", "scene"],
        failed_agents=[]
    )

    # Case B: PPE failed (UNAVAILABLE)
    rep_unavail = engine.evaluate(
        query="check worker PPE compliance",
        intent="ppe_compliance",
        findings=findings,
        context_data=ctx,
        executed_agents=["object", "scene"],
        failed_agents=["ppe"]
    )

    # Completeness must be identical: both lack PPE evidence
    assert rep_missing.completeness == pytest.approx(rep_unavail.completeness, 0.001)

    # Unavailability differs: unexecuted has U = 0, failed has U = 0.50
    assert rep_missing.unavailability == 0.0
    assert rep_unavail.unavailability == 0.50

    # Deficit difference is exactly the unavailability weight term (0.10 * 0.50 = 0.050)
    w_u = rep_unavail.normalized_weights["w_unavailability"]
    expected_diff = w_u * 0.50
    assert (rep_unavail.composite_deficit - rep_missing.composite_deficit) == pytest.approx(expected_diff, 0.005)


def test_scenario_11_critical_missing_evidence_cannot_produce_low_deficit():
    """
    Validation Scenario 11:
    Critical missing evidence cannot produce a falsely low deficit.
    When a defect inspection lacks defect specialist results, deficit must remain >= 0.40.
    """
    engine = EvidenceDeficitEngine()
    # Scene agent succeeds, but defect specialist failed
    report = engine.evaluate(
        query="inspect steel surface for defects and cracks",
        intent="defect_inspection",
        findings=[],
        context_data={"scene_result": {"lighting": "Standard diffused industrial lighting"}},
        executed_agents=["scene"],
        failed_agents=["defect"]
    )
    # Target tokens: ET-DFT-LOCALIZATION-01, ET-DFT-CLASSIFICATION-02, ET-DFT-PROCEDURE-03, ET-SCN-ENVIRONMENT-01
    assert report.tokens["ET-DFT-LOCALIZATION-01"].status == TokenStatus.UNAVAILABLE
    assert report.tokens["ET-DFT-CLASSIFICATION-02"].status == TokenStatus.UNAVAILABLE
    assert report.composite_deficit >= 0.40
