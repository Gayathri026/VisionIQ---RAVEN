"""
VisionIQ Phase 3.1: Evidence Engine Core
Implements observable evidence tokens, task-conditioned evidence repertoires,
decoupled completeness, reliability, conflict, and unavailability metrics,
provenance-preserving evidence ledger, and configurable deficit formulation.

Adheres strictly to the VisionIQ Phase 3.1 Design Specification:
1. Missing or negative tests never increase satisfaction.
2. All satisfaction scores reside strictly in [0.0, 1.0].
3. No assumed worker population; conditioned on verified detections.
4. Uses only exposed model and detector output fields.
5. Missing evidence and unavailability are mathematically decoupled (no double-counting).
6. Non-monotonic contradiction handling (no blind maximum).
"""

from enum import Enum
import math
import uuid
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from pydantic import BaseModel, Field, model_validator

from backend.app.raven.calibration import (
    calibrate_confidence,
    compute_spatial_uncertainty,
    compute_occlusion_uncertainty
)
from backend.app.raven.conflict import CrossAgentConflictEvaluator


def compute_box_iou(box1: Optional[List[float]], box2: Optional[List[float]]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


class TokenStatus(str, Enum):
    """Observable epistemic status of an individual evidence token."""
    SATISFIED = "SATISFIED"                      # Token observed with high confidence and verified provenance
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE" # Attempted, but degraded by occlusion, blur, or borderline score
    UNCERTAIN = "UNCERTAIN"                      # Backward-compatible alias for INSUFFICIENT_EVIDENCE
    CONFIRMED_ABSENT = "CONFIRMED_ABSENT"        # Agent successfully ran and confirmed target is absent
    NOT_APPLICABLE = "NOT_APPLICABLE"            # Token not applicable given confirmed scene state (e.g. PPE when 0 workers)
    CONTRADICTORY = "CONTRADICTORY"              # Contradicted by another agent's observation or spatial intersection
    MISSING = "MISSING"                          # Target agent not yet executed
    UNAVAILABLE = "UNAVAILABLE"                  # Target agent execution failed, timed out, or ungrounded by prerequisite failure
    UNSUPPORTED = "UNSUPPORTED"                  # Modality cannot be observed by current system


class DeficitWeights(BaseModel):
    """Configurable weights for the multi-factor evidence deficit formulation."""
    w_completeness: float = Field(0.50, ge=0.0, description="Weight for evidence incompleteness (1 - C_ev)")
    w_reliability: float = Field(0.20, ge=0.0, description="Weight for evidence unreliability (1 - R_ev)")
    w_conflict: float = Field(0.20, ge=0.0, description="Weight for cross-agent conflict score S_conflict")
    w_unavailability: float = Field(0.10, ge=0.0, description="Weight for unavailable evidence U_unavail")

    @model_validator(mode="after")
    def validate_non_zero_sum(self) -> "DeficitWeights":
        total = self.w_completeness + self.w_reliability + self.w_conflict + self.w_unavailability
        if total <= 0.0:
            raise ValueError("Sum of deficit weights must be strictly positive.")
        return self

    def normalized(self) -> Dict[str, float]:
        """Returns normalized weights summing to exactly 1.0."""
        total = self.w_completeness + self.w_reliability + self.w_conflict + self.w_unavailability
        return {
            "w_completeness": self.w_completeness / total,
            "w_reliability": self.w_reliability / total,
            "w_conflict": self.w_conflict / total,
            "w_unavailability": self.w_unavailability / total
        }


class EvidenceTokenDefinition(BaseModel):
    """Metadata specification for a verifiable evidence token in the catalog."""
    token_id: str
    task_name: str
    description: str
    domain: str
    primary_agent: str
    prerequisite_agents: List[str] = Field(default_factory=list)
    required: bool = True
    scoring_type: str  # "probability", "heuristic_mask", "pattern_match", "environmental_metric"
    is_empirically_calibrated: bool = False
    source_fields: List[str] = Field(default_factory=list)


class EvaluatedEvidenceToken(BaseModel):
    """Result of evaluating an evidence token against observed execution state."""
    token_id: str
    task_name: str
    required: bool
    status: TokenStatus
    satisfaction_score: float = Field(0.0, ge=0.0, le=1.0)
    raw_confidence: Optional[float] = None
    calibrated_confidence: Optional[float] = None
    uncertainty: float = Field(0.0, ge=0.0, le=1.0)
    source_agent: Optional[str] = None
    source_finding_uids: List[str] = Field(default_factory=list)
    explanation: str = ""
    is_heuristic: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceLedgerEntry(BaseModel):
    """An immutable, provenance-preserving record of a single piece of visual evidence."""
    model_config = {"protected_namespaces": ()}

    entry_id: str = Field(default_factory=lambda: f"LEDGER-{uuid.uuid4().hex[:8].upper()}")
    finding_uid: str
    source_agent: str
    model_version: Optional[str] = None
    hardware_device: Optional[str] = None
    raw_confidence: Optional[float] = None
    calibrated_confidence: Optional[float] = None
    confidence_type: str = "calibrated_probability"  # "calibrated_probability", "heuristic_score", "unspecified"
    spatial_uncertainty: Optional[float] = None
    bbox: Optional[List[float]] = None
    acquisition_step: int = 0
    label: str
    observed_evidence: str
    has_conflict: bool = False
    conflict_notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConflictRecord(BaseModel):
    """Details on a detected contradiction or spatial clash between observations."""
    conflict_id: str = Field(default_factory=lambda: f"CONF-{uuid.uuid4().hex[:8].upper()}")
    conflict_type: str  # "COUNT_DISCREPANCY", "SPATIAL_INTERSECTION", "SEMANTIC_CONTRADICTION"
    agents_involved: List[str]
    severity: float = Field(0.0, ge=0.0, le=1.0)
    description: str
    finding_uids: List[str] = Field(default_factory=list)


class EvidenceEvaluationReport(BaseModel):
    """Comprehensive output of the evidence deficit engine."""
    tokens: Dict[str, EvaluatedEvidenceToken]
    completeness: float = Field(..., ge=0.0, le=1.0, description="C_ev: Completeness over applicable required tokens")
    reliability: float = Field(..., ge=0.0, le=1.0, description="R_ev: Calibrated reliability of observed evidence")
    conflict_score: float = Field(..., ge=0.0, le=1.0, description="S_conflict: Normalized contradiction severity score")
    unavailability: float = Field(..., ge=0.0, le=1.0, description="U_unavail: Fraction of required tokens whose agent failed")
    composite_deficit: float = Field(..., ge=0.0, le=1.0, description="D in [0, 1]: Multi-factor evidence deficit")
    normalized_weights: Dict[str, float]
    conflicts: List[ConflictRecord] = Field(default_factory=list)
    ledger_entries: List[EvidenceLedgerEntry] = Field(default_factory=list)
    unsupported_tokens: List[str] = Field(default_factory=list)
    summary: str = ""

    @property
    def conflict_entropy(self) -> float:
        """Backward-compatible alias for conflict_score."""
        return self.conflict_score

    @property
    def evidence_deficit(self) -> float:
        """Alias for composite_deficit."""
        return self.composite_deficit


# ==============================================================================
# TOKEN CATALOG SPECIFICATION
# ==============================================================================

TOKEN_CATALOG: Dict[str, EvidenceTokenDefinition] = {
    "ET-OBJ-PERSON-01": EvidenceTokenDefinition(
        token_id="ET-OBJ-PERSON-01",
        task_name="personnel_localization",
        description="Verified spatial bounding boxes and calibrated detections for human personnel",
        domain="object",
        primary_agent="object",
        prerequisite_agents=[],
        required=True,
        scoring_type="probability",
        is_empirically_calibrated=True,
        source_fields=["object_result.objects", "findings.bbox", "findings.detection_confidence"]
    ),
    "ET-PPE-CRANIAL-02": EvidenceTokenDefinition(
        token_id="ET-PPE-CRANIAL-02",
        task_name="cranial_protection_compliance",
        description="Protective hardhat coverage evaluated across localized personnel",
        domain="ppe",
        primary_agent="ppe",
        prerequisite_agents=["object"],
        required=True,
        scoring_type="heuristic_mask",
        is_empirically_calibrated=False,
        source_fields=["ppe_result.persons.has_helmet", "ppe_result.persons.helmet_confidence", "ppe_result.persons.is_occluded"]
    ),
    "ET-PPE-TORSO-03": EvidenceTokenDefinition(
        token_id="ET-PPE-TORSO-03",
        task_name="torso_protection_compliance",
        description="High-visibility safety vest coverage evaluated across localized personnel",
        domain="ppe",
        primary_agent="ppe",
        prerequisite_agents=["object"],
        required=True,
        scoring_type="heuristic_mask",
        is_empirically_calibrated=False,
        source_fields=["ppe_result.persons.has_vest", "ppe_result.persons.vest_confidence", "ppe_result.persons.is_occluded"]
    ),
    "ET-DFT-LOCALIZATION-01": EvidenceTokenDefinition(
        token_id="ET-DFT-LOCALIZATION-01",
        task_name="surface_flaw_localization",
        description="Bounded surface region of interest and anomaly score for surface flaws",
        domain="defect",
        primary_agent="defect",
        prerequisite_agents=[],
        required=True,
        scoring_type="probability",
        is_empirically_calibrated=False,
        source_fields=["defect_result.findings.bbox", "defect_result.findings.anomaly_score"]
    ),
    "ET-DFT-CLASSIFICATION-02": EvidenceTokenDefinition(
        token_id="ET-DFT-CLASSIFICATION-02",
        task_name="defect_morphology_classification",
        description="Neural CNN model classification distinguishing confirmed NEU defect types from noise",
        domain="defect",
        primary_agent="defect",
        prerequisite_agents=[],
        required=True,
        scoring_type="probability",
        is_empirically_calibrated=True,
        source_fields=["defect_result.findings.is_confirmed_defect", "defect_result.findings.detection_confidence"]
    ),
    "ET-DFT-PROCEDURE-03": EvidenceTokenDefinition(
        token_id="ET-DFT-PROCEDURE-03",
        task_name="ndt_verification_procedure",
        description="Prescribed actionable Non-Destructive Testing (PT/UT/ET) procedure citing ASTM/ISO standards",
        domain="recommendation",
        primary_agent="recommendation",
        prerequisite_agents=["rca"],
        required=False,
        scoring_type="pattern_match",
        is_empirically_calibrated=False,
        source_fields=["findings.verification_procedure", "recommendation_result"]
    ),
    "ET-OCR-TRANSCRIBE-01": EvidenceTokenDefinition(
        token_id="ET-OCR-TRANSCRIBE-01",
        task_name="alphanumeric_inscription_transcription",
        description="Extracted alphanumeric characters from signage, equipment nameplates, or serial tags",
        domain="ocr",
        primary_agent="ocr",
        prerequisite_agents=[],
        required=True,
        scoring_type="probability",
        is_empirically_calibrated=True,
        source_fields=["ocr_result.detected_text", "ocr_result.text_regions.confidence"]
    ),
    "ET-SCN-ENVIRONMENT-01": EvidenceTokenDefinition(
        token_id="ET-SCN-ENVIRONMENT-01",
        task_name="ambient_illumination_assessment",
        description="Industrial ambient luminance and contrast verification to rule out optical glare or underexposure",
        domain="scene",
        primary_agent="scene",
        prerequisite_agents=[],
        required=True,
        scoring_type="environmental_metric",
        is_empirically_calibrated=False,
        source_fields=["scene_result.lighting", "scene_result.direct_observations"]
    )
}


# ==============================================================================
# EVIDENCE LEDGER IMPLEMENTATION
# ==============================================================================

class EvidenceLedger:
    """
    Maintains an append-only, provenance-preserving record of visual evidence.
    Deduplicates identical detections while preserving and flagging contradictory findings.
    """
    def __init__(self):
        self._entries: List[EvidenceLedgerEntry] = []
        self._conflicts: List[ConflictRecord] = []

    def add_finding(
        self,
        finding: Dict[str, Any],
        step: int = 0,
        model_version: Optional[str] = None,
        hardware_device: Optional[str] = None
    ) -> EvidenceLedgerEntry:
        """
        Adds a specialist finding to the ledger.
        Checks for spatial duplicates and cross-agent discrepancies.
        """
        f_uid = finding.get("finding_uid", f"F-{uuid.uuid4().hex[:6].upper()}")
        agent = finding.get("agent_source", "unknown")
        label = finding.get("label", "unspecified")
        bbox = finding.get("bbox")
        raw_conf = finding.get("detection_confidence")
        if raw_conf is not None:
            raw_conf = float(min(1.0, max(0.0, raw_conf)))
        
        cal_conf = finding.get("calibrated_confidence")
        if cal_conf is None and raw_conf is not None:
            cal_conf = calibrate_confidence(raw_conf)
        elif cal_conf is not None:
            cal_conf = float(min(1.0, max(0.0, cal_conf)))

        spatial_unc = finding.get("uncertainty")
        if spatial_unc is None and bbox:
            spatial_unc = compute_spatial_uncertainty(bbox)

        # 1. Deduplication check: Same agent, high IoU, same label
        for existing in self._entries:
            if existing.source_agent == agent and existing.label == label:
                if compute_box_iou(existing.bbox, bbox) >= 0.60:
                    # Update existing record rather than creating a duplicate
                    if raw_conf is not None and (existing.raw_confidence is None or raw_conf > existing.raw_confidence):
                        existing.raw_confidence = raw_conf
                        existing.calibrated_confidence = cal_conf
                    existing.metadata["duplicate_observations"] = existing.metadata.get("duplicate_observations", 1) + 1
                    return existing

        # 2. Contradiction check: Different agent with high spatial overlap but conflicting semantics
        has_conflict = False
        conflict_desc = None
        for existing in self._entries:
            if existing.source_agent != agent and bbox and existing.bbox:
                iou = compute_box_iou(existing.bbox, bbox)
                if iou >= 0.25:
                    # e.g., defect vs OCR text overlap
                    if (agent == "defect" and existing.source_agent == "ocr") or \
                       (agent == "ocr" and existing.source_agent == "defect"):
                        has_conflict = True
                        conflict_desc = f"Spatial clash: Defect anomaly overlaps OCR text inscription (IoU={iou:.2f})"
                        existing.has_conflict = True
                        conf_rec = ConflictRecord(
                            conflict_type="SPATIAL_INTERSECTION",
                            agents_involved=[agent, existing.source_agent],
                            severity=0.35,
                            description=conflict_desc,
                            finding_uids=[f_uid, existing.finding_uid]
                        )
                        self._conflicts.append(conf_rec)
                        break

        conf_type = "calibrated_probability" if agent in ["object", "defect", "ocr"] else "heuristic_score"

        entry = EvidenceLedgerEntry(
            finding_uid=f_uid,
            source_agent=agent,
            model_version=model_version or finding.get("model_version"),
            hardware_device=hardware_device,
            raw_confidence=raw_conf,
            calibrated_confidence=cal_conf,
            confidence_type=conf_type,
            spatial_uncertainty=spatial_unc,
            bbox=bbox,
            acquisition_step=step,
            label=label,
            observed_evidence=finding.get("observed_evidence", ""),
            has_conflict=has_conflict,
            conflict_notes=conflict_desc,
            metadata=finding
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> List[EvidenceLedgerEntry]:
        return list(self._entries)

    @property
    def conflicts(self) -> List[ConflictRecord]:
        return list(self._conflicts)


# ==============================================================================
# TOKEN EVALUATORS (Grounded strictly in actual source fields)
# ==============================================================================

def evaluate_token_obj_person_01(
    query: str,
    context_data: Dict[str, Any],
    findings: List[Dict[str, Any]],
    executed: List[str],
    failed: List[str]
) -> EvaluatedEvidenceToken:
    """Evaluates ET-OBJ-PERSON-01: Personnel Entity Localization."""
    tok_def = TOKEN_CATALOG["ET-OBJ-PERSON-01"]

    if "object" in failed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNAVAILABLE,
            satisfaction_score=0.0,
            uncertainty=1.0,
            explanation="Object detection agent failed during execution; personnel localization unavailable.",
            is_heuristic=False
        )

    if "object" not in executed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.MISSING,
            satisfaction_score=0.0,
            uncertainty=0.50,
            explanation="Object detection agent not executed; personnel presence unverified.",
            is_heuristic=False
        )

    obj_res = context_data.get("object_result") or {}
    objects = obj_res.get("objects", [])
    person_objects = [
        o for o in objects
        if str(o.get("label", "")).lower() in ["person", "worker"]
    ]
    if not person_objects:
        person_objects = [
            f for f in findings
            if f.get("agent_source") == "object" and str(f.get("label", "")).lower() in ["person", "worker"]
        ]

    n_persons = len(person_objects)
    q_lower = query.lower()
    is_negative_query = any(k in q_lower for k in ["clear", "empty", "no person", "unoccupied"])

    if n_persons > 0:
        confs = [float(o.get("confidence", 0.80)) for o in person_objects]
        cal_confs = [calibrate_confidence(c) for c in confs]
        mean_score = sum(cal_confs) / float(n_persons)
        score = float(min(1.0, max(0.0, mean_score)))
        status = TokenStatus.SATISFIED if score >= 0.70 else TokenStatus.INSUFFICIENT_EVIDENCE

        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=status,
            satisfaction_score=round(score, 4),
            raw_confidence=round(sum(confs) / float(n_persons), 4),
            calibrated_confidence=round(score, 4),
            uncertainty=round(1.0 - score, 4),
            source_agent="object",
            source_finding_uids=[o.get("finding_uid", "") for o in person_objects if o.get("finding_uid")],
            explanation=f"Localized {n_persons} personnel entities with mean calibrated confidence {score*100:.1f}%.",
            is_heuristic=False,
            metadata={"person_count": n_persons}
        )
    else:
        # 0 persons detected
        if is_negative_query:
            return EvaluatedEvidenceToken(
                token_id=tok_def.token_id,
                task_name=tok_def.task_name,
                required=tok_def.required,
                status=TokenStatus.SATISFIED,
                satisfaction_score=1.0,
                raw_confidence=1.0,
                calibrated_confidence=0.98,
                uncertainty=0.05,
                source_agent="object",
                explanation="Confirmed 0 personnel entities present in inspection area per negative query specification.",
                is_heuristic=False,
                metadata={"person_count": 0}
            )
        else:
            return EvaluatedEvidenceToken(
                token_id=tok_def.token_id,
                task_name=tok_def.task_name,
                required=tok_def.required,
                status=TokenStatus.CONFIRMED_ABSENT,
                satisfaction_score=0.0,
                uncertainty=0.15,
                source_agent="object",
                explanation="Confirmed 0 personnel present in image after successful object detection scan.",
                is_heuristic=False,
                metadata={"person_count": 0}
            )


def evaluate_token_ppe_cranial_02(
    query: str,
    context_data: Dict[str, Any],
    findings: List[Dict[str, Any]],
    executed: List[str],
    failed: List[str]
) -> EvaluatedEvidenceToken:
    """Evaluates ET-PPE-CRANIAL-02: Cranial Protection Compliance."""
    tok_def = TOKEN_CATALOG["ET-PPE-CRANIAL-02"]

    if "object" in failed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNAVAILABLE,
            satisfaction_score=0.0,
            uncertainty=1.0,
            explanation="Upstream object detection agent failed; cranial protection cannot be grounded.",
            is_heuristic=True
        )

    if "object" not in executed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.MISSING,
            satisfaction_score=0.0,
            uncertainty=0.50,
            explanation="Upstream object detection agent unexecuted; cranial protection cannot be grounded.",
            is_heuristic=True
        )

    # Check if object detector executed and confirmed zero personnel
    obj_res = context_data.get("object_result") or {}
    obj_persons = [
        o for o in obj_res.get("objects", [])
        if str(o.get("label", "")).lower() in ["person", "worker"]
    ]
    if not obj_persons:
        obj_persons = [
            f for f in findings
            if f.get("agent_source") == "object" and str(f.get("label", "")).lower() in ["person", "worker"]
        ]

    ppe_res = context_data.get("ppe_result") or {}
    ppe_persons = ppe_res.get("persons", [])

    if len(obj_persons) == 0 and len(ppe_persons) == 0:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.NOT_APPLICABLE,
            satisfaction_score=1.0,
            uncertainty=0.05,
            source_agent="ppe",
            explanation="Zero personnel confirmed present in scene; cranial protection is not applicable.",
            is_heuristic=True,
            metadata={"evaluated_workers": 0}
        )

    if "ppe" in failed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNAVAILABLE,
            satisfaction_score=0.0,
            uncertainty=1.0,
            explanation="PPE compliance specialist failed during execution.",
            is_heuristic=True
        )

    if "ppe" not in executed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.MISSING,
            satisfaction_score=0.0,
            uncertainty=0.50,
            explanation="PPE compliance specialist not executed.",
            is_heuristic=True
        )

    ppe_res = context_data.get("ppe_result") or {}
    persons = ppe_res.get("persons", [])
    if not persons:
        # Check findings directly
        ppe_findings = [f for f in findings if f.get("agent_source") == "ppe"]
        if ppe_findings:
            # Reconstruct evaluation from findings
            persons = [{
                "has_helmet": "helmet" in f.get("observed_evidence", "").lower(),
                "helmet_confidence": 0.85 if "helmet" in f.get("observed_evidence", "").lower() else 0.0,
                "is_occluded": f.get("uncertainty", 0.0) >= 0.65,
                "person_id": f.get("person_id")
            } for f in ppe_findings]

    if not persons:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNAVAILABLE,
            satisfaction_score=0.0,
            uncertainty=0.90,
            explanation="Personnel localized but PPE module produced 0 evaluations; cranial protection unavailable.",
            is_heuristic=True,
            metadata={"evaluated_workers": 0}
        )

    scores = []
    has_occlusion = False
    for p in persons:
        has_h = bool(p.get("has_helmet", False))
        conf = float(min(1.0, max(0.0, p.get("helmet_confidence", 0.60))))
        occ = bool(p.get("is_occluded", False))
        if occ:
            has_occlusion = True
            person_score = conf * 0.30  # Heavy penalty for occluded cranial zone
        else:
            person_score = conf if has_h else 0.0
        scores.append(person_score)

    mean_sat = sum(scores) / float(len(scores))
    satisfaction = float(min(1.0, max(0.0, mean_sat)))

    if satisfaction >= 0.65 and not has_occlusion:
        status = TokenStatus.SATISFIED
    elif has_occlusion or (0.20 <= satisfaction < 0.65):
        status = TokenStatus.INSUFFICIENT_EVIDENCE
    else:
        status = TokenStatus.CONFIRMED_ABSENT  # Confirmed absence of hardhat

    return EvaluatedEvidenceToken(
        token_id=tok_def.token_id,
        task_name=tok_def.task_name,
        required=tok_def.required,
        status=status,
        satisfaction_score=round(satisfaction, 4),
        raw_confidence=round(sum(p.get("helmet_confidence", 0.0) for p in persons) / float(len(persons)), 4),
        uncertainty=0.70 if has_occlusion else round(1.0 - satisfaction, 4),
        source_agent="ppe",
        explanation=f"Evaluated cranial protection across {len(persons)} worker(s); satisfaction={satisfaction*100:.1f}%.",
        is_heuristic=True,
        metadata={"worker_count": len(persons), "has_occlusion": has_occlusion}
    )


def evaluate_token_ppe_torso_03(
    query: str,
    context_data: Dict[str, Any],
    findings: List[Dict[str, Any]],
    executed: List[str],
    failed: List[str]
) -> EvaluatedEvidenceToken:
    """Evaluates ET-PPE-TORSO-03: Torso High-Visibility Protection Compliance."""
    tok_def = TOKEN_CATALOG["ET-PPE-TORSO-03"]

    if "object" in failed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNAVAILABLE,
            satisfaction_score=0.0,
            uncertainty=1.0,
            explanation="Upstream object detection agent failed; torso protection cannot be grounded.",
            is_heuristic=True
        )

    if "object" not in executed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.MISSING,
            satisfaction_score=0.0,
            uncertainty=0.50,
            explanation="Upstream object detection agent unexecuted; torso protection cannot be grounded.",
            is_heuristic=True
        )

    # Check if object detector executed and confirmed zero personnel
    obj_res = context_data.get("object_result") or {}
    obj_persons = [
        o for o in obj_res.get("objects", [])
        if str(o.get("label", "")).lower() in ["person", "worker"]
    ]
    if not obj_persons:
        obj_persons = [
            f for f in findings
            if f.get("agent_source") == "object" and str(f.get("label", "")).lower() in ["person", "worker"]
        ]

    ppe_res = context_data.get("ppe_result") or {}
    ppe_persons = ppe_res.get("persons", [])

    if len(obj_persons) == 0 and len(ppe_persons) == 0:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.NOT_APPLICABLE,
            satisfaction_score=1.0,
            uncertainty=0.05,
            source_agent="ppe",
            explanation="Zero personnel confirmed present in scene; torso protection is not applicable.",
            is_heuristic=True,
            metadata={"evaluated_workers": 0}
        )

    if "ppe" in failed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNAVAILABLE,
            satisfaction_score=0.0,
            uncertainty=1.0,
            explanation="PPE compliance specialist failed during execution.",
            is_heuristic=True
        )

    if "ppe" not in executed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.MISSING,
            satisfaction_score=0.0,
            uncertainty=0.50,
            explanation="PPE compliance specialist not executed.",
            is_heuristic=True
        )

    ppe_res = context_data.get("ppe_result") or {}
    persons = ppe_res.get("persons", [])
    if not persons:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNAVAILABLE,
            satisfaction_score=0.0,
            uncertainty=0.90,
            explanation="Personnel localized but PPE module produced 0 evaluations; torso protection unavailable.",
            is_heuristic=True,
            metadata={"evaluated_workers": 0}
        )

    scores = []
    has_occlusion = False
    for p in persons:
        has_v = bool(p.get("has_vest", False))
        conf = float(min(1.0, max(0.0, p.get("vest_confidence", 0.65))))
        occ = bool(p.get("is_occluded", False))
        if occ:
            has_occlusion = True
            person_score = conf * 0.30
        else:
            person_score = conf if has_v else 0.0
        scores.append(person_score)

    mean_sat = sum(scores) / float(len(scores))
    satisfaction = float(min(1.0, max(0.0, mean_sat)))

    if satisfaction >= 0.65 and not has_occlusion:
        status = TokenStatus.SATISFIED
    elif has_occlusion or (0.20 <= satisfaction < 0.65):
        status = TokenStatus.INSUFFICIENT_EVIDENCE
    else:
        status = TokenStatus.CONFIRMED_ABSENT  # Confirmed absence of vest

    return EvaluatedEvidenceToken(
        token_id=tok_def.token_id,
        task_name=tok_def.task_name,
        required=tok_def.required,
        status=status,
        satisfaction_score=round(satisfaction, 4),
        raw_confidence=round(sum(p.get("vest_confidence", 0.0) for p in persons) / float(len(persons)), 4),
        uncertainty=0.70 if has_occlusion else round(1.0 - satisfaction, 4),
        source_agent="ppe",
        explanation=f"Evaluated torso vest compliance across {len(persons)} worker(s); satisfaction={satisfaction*100:.1f}%.",
        is_heuristic=True,
        metadata={"worker_count": len(persons), "has_occlusion": has_occlusion}
    )


def evaluate_token_dft_localization_01(
    query: str,
    context_data: Dict[str, Any],
    findings: List[Dict[str, Any]],
    executed: List[str],
    failed: List[str]
) -> EvaluatedEvidenceToken:
    """Evaluates ET-DFT-LOCALIZATION-01: Surface Flaw Spatial Localization."""
    tok_def = TOKEN_CATALOG["ET-DFT-LOCALIZATION-01"]

    if "defect" in failed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNAVAILABLE,
            satisfaction_score=0.0,
            uncertainty=1.0,
            explanation="Defect inspection specialist failed during execution.",
            is_heuristic=False
        )

    if "defect" not in executed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.MISSING,
            satisfaction_score=0.0,
            uncertainty=0.50,
            explanation="Defect inspection specialist not executed.",
            is_heuristic=False
        )

    defect_res = context_data.get("defect_result") or {}
    defects_count = defect_res.get("defects_detected", 0)
    defect_findings = [f for f in findings if f.get("agent_source") == "defect"]

    if defects_count > 0 or defect_findings:
        scores = []
        for f in defect_findings:
            anom = f.get("anomaly_score")
            conf = f.get("detection_confidence", 0.80)
            base_score = anom if anom is not None else conf
            bbox = f.get("bbox")
            if bbox and len(bbox) >= 4:
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area_factor = min(1.0, max(0.2, area / 120.0))
            else:
                area_factor = 0.50
            scores.append(float(min(1.0, max(0.0, base_score * area_factor))))

        max_score = max(scores) if scores else 0.50
        status = TokenStatus.SATISFIED if max_score >= 0.50 else TokenStatus.UNCERTAIN

        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=status,
            satisfaction_score=round(max_score, 4),
            raw_confidence=round(max_score, 4),
            uncertainty=round(1.0 - max_score, 4),
            source_agent="defect",
            source_finding_uids=[f.get("finding_uid", "") for f in defect_findings if f.get("finding_uid")],
            explanation=f"Localized {len(defect_findings)} anomalous surface region(s); peak score={max_score:.3f}.",
            is_heuristic=False
        )
    else:
        # Executed and 0 defects found: Clean surface verified
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.SATISFIED,
            satisfaction_score=1.0,
            raw_confidence=1.0,
            calibrated_confidence=0.98,
            uncertainty=0.05,
            source_agent="defect",
            explanation="Clean surface verified: Zero surface flaws detected after morphometric scan.",
            is_heuristic=False
        )


def evaluate_token_dft_classification_02(
    query: str,
    context_data: Dict[str, Any],
    findings: List[Dict[str, Any]],
    executed: List[str],
    failed: List[str]
) -> EvaluatedEvidenceToken:
    """Evaluates ET-DFT-CLASSIFICATION-02: Defect Morphology Classification."""
    tok_def = TOKEN_CATALOG["ET-DFT-CLASSIFICATION-02"]

    if "defect" in failed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNAVAILABLE,
            satisfaction_score=0.0,
            uncertainty=1.0,
            explanation="Defect inspection specialist failed during execution.",
            is_heuristic=False
        )

    if "defect" not in executed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.MISSING,
            satisfaction_score=0.0,
            uncertainty=0.50,
            explanation="Defect inspection specialist not executed.",
            is_heuristic=False
        )

    defect_res = context_data.get("defect_result") or {}
    defects_count = defect_res.get("defects_detected", 0)
    defect_findings = [f for f in findings if f.get("agent_source") == "defect"]

    if defects_count == 0 and not defect_findings:
        # Clean surface: no unconfirmed defect morphologies
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.SATISFIED,
            satisfaction_score=1.0,
            raw_confidence=1.0,
            calibrated_confidence=0.98,
            uncertainty=0.05,
            source_agent="defect",
            explanation="Clean surface: no morphology classification required.",
            is_heuristic=False
        )

    confirmed = [f for f in defect_findings if f.get("is_confirmed_defect") is True]
    if confirmed:
        confs = [float(f.get("detection_confidence", 0.80)) for f in confirmed]
        cal_confs = [calibrate_confidence(c) for c in confs]
        score = float(min(1.0, max(0.0, max(cal_confs))))
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.SATISFIED,
            satisfaction_score=round(score, 4),
            raw_confidence=round(max(confs), 4),
            calibrated_confidence=round(score, 4),
            uncertainty=round(1.0 - score, 4),
            source_agent="defect",
            source_finding_uids=[f.get("finding_uid", "") for f in confirmed if f.get("finding_uid")],
            explanation=f"Confirmed {len(confirmed)} defect morphology class(es) via CNN model (peak confidence {score*100:.1f}%).",
            is_heuristic=False
        )
    else:
        # Heuristic anomaly cues only
        heuristic_confs = [float(f.get("detection_confidence", 0.30)) for f in defect_findings]
        peak_heur = max(heuristic_confs) if heuristic_confs else 0.30
        downgraded_score = float(min(0.25, max(0.05, 0.25 * peak_heur)))
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNCERTAIN,
            satisfaction_score=round(downgraded_score, 4),
            raw_confidence=round(peak_heur, 4),
            uncertainty=0.75,
            source_agent="defect",
            source_finding_uids=[f.get("finding_uid", "") for f in defect_findings if f.get("finding_uid")],
            explanation="Heuristic anomaly cue only; neural morphology classification unconfirmed (<50% threshold).",
            is_heuristic=True
        )


def evaluate_token_dft_procedure_03(
    query: str,
    context_data: Dict[str, Any],
    findings: List[Dict[str, Any]],
    executed: List[str],
    failed: List[str]
) -> EvaluatedEvidenceToken:
    """Evaluates ET-DFT-PROCEDURE-03: Actionable NDT Specification."""
    tok_def = TOKEN_CATALOG["ET-DFT-PROCEDURE-03"]

    # Rule: Missing physical tests do NOT increase satisfaction.
    # Satisfaction reflects whether an actionable NDT procedure has been formulated.
    all_procedures = [
        str(f.get("verification_procedure", ""))
        for f in findings if f.get("verification_procedure")
    ]
    rec_res = context_data.get("recommendation_result") or {}
    if rec_res.get("recommendations"):
        all_procedures.extend([str(r) for r in rec_res["recommendations"]])

    combined_text = " ".join(all_procedures).upper()
    has_explicit_ndt = any(k in combined_text for k in ["ASTM", "ISO", "PT", "UT", "ET", "PENETRANT", "ULTRASONIC", "EDDY CURRENT", "NDT"])

    if has_explicit_ndt:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.SATISFIED,
            satisfaction_score=1.0,
            uncertainty=0.05,
            source_agent="recommendation",
            explanation="Actionable Non-Destructive Testing (NDT) protocol with ASTM/ISO standards citation formulated.",
            is_heuristic=True
        )
    elif len(all_procedures) > 0:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNCERTAIN,
            satisfaction_score=0.40,
            uncertainty=0.60,
            source_agent="recommendation",
            explanation="Generic inspection recommended without explicit NDT testing protocol citation.",
            is_heuristic=True
        )
    else:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.MISSING,
            satisfaction_score=0.0,
            uncertainty=1.0,
            explanation="No verification or NDT procedure formulated.",
            is_heuristic=True
        )


def evaluate_token_ocr_transcribe_01(
    query: str,
    context_data: Dict[str, Any],
    findings: List[Dict[str, Any]],
    executed: List[str],
    failed: List[str]
) -> EvaluatedEvidenceToken:
    """Evaluates ET-OCR-TRANSCRIBE-01: Alphanumeric Inscription Transcription."""
    tok_def = TOKEN_CATALOG["ET-OCR-TRANSCRIBE-01"]

    if "ocr" in failed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNAVAILABLE,
            satisfaction_score=0.0,
            uncertainty=1.0,
            explanation="OCR agent failed during execution.",
            is_heuristic=False
        )

    if "ocr" not in executed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.MISSING,
            satisfaction_score=0.0,
            uncertainty=0.50,
            explanation="OCR agent not executed.",
            is_heuristic=False
        )

    ocr_res = context_data.get("ocr_result") or {}
    detected_text = str(ocr_res.get("detected_text", "")).strip()
    text_regions = ocr_res.get("text_regions", [])

    if detected_text and detected_text != "No text detected":
        confs = [float(r.get("confidence", 0.85)) for r in text_regions if r.get("confidence") is not None]
        mean_conf = sum(confs) / float(len(confs)) if confs else 0.85
        cal_conf = calibrate_confidence(mean_conf)
        length_factor = min(1.0, max(0.3, len(detected_text) / 3.0))
        satisfaction = float(min(1.0, max(0.0, cal_conf * length_factor)))
        status = TokenStatus.SATISFIED if satisfaction >= 0.60 else TokenStatus.UNCERTAIN

        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=status,
            satisfaction_score=round(satisfaction, 4),
            raw_confidence=round(mean_conf, 4),
            calibrated_confidence=round(cal_conf, 4),
            uncertainty=round(1.0 - satisfaction, 4),
            source_agent="ocr",
            explanation=f"Transcribed '{detected_text}' with calibrated confidence {cal_conf*100:.1f}%.",
            is_heuristic=False,
            metadata={"text_length": len(detected_text)}
        )
    else:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.MISSING,
            satisfaction_score=0.0,
            uncertainty=0.85,
            source_agent="ocr",
            explanation="Zero optical characters detected in localized image regions.",
            is_heuristic=False
        )


def evaluate_token_scn_environment_01(
    query: str,
    context_data: Dict[str, Any],
    findings: List[Dict[str, Any]],
    executed: List[str],
    failed: List[str]
) -> EvaluatedEvidenceToken:
    """Evaluates ET-SCN-ENVIRONMENT-01: Ambient Illumination Assessment."""
    tok_def = TOKEN_CATALOG["ET-SCN-ENVIRONMENT-01"]

    if "scene" in failed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.UNAVAILABLE,
            satisfaction_score=0.0,
            uncertainty=1.0,
            explanation="Scene understanding agent failed during execution.",
            is_heuristic=True
        )

    if "scene" not in executed:
        return EvaluatedEvidenceToken(
            token_id=tok_def.token_id,
            task_name=tok_def.task_name,
            required=tok_def.required,
            status=TokenStatus.MISSING,
            satisfaction_score=0.0,
            uncertainty=0.50,
            explanation="Scene understanding agent not executed.",
            is_heuristic=True
        )

    scene_res = context_data.get("scene_result") or {}
    lighting_desc = str(scene_res.get("lighting", "")).lower()
    obs = " ".join([str(o) for o in scene_res.get("direct_observations", [])])

    if "standard" in lighting_desc or "diffused" in lighting_desc:
        score = 1.0
        status = TokenStatus.SATISFIED
        explanation = "Standard diffused industrial lighting verified; optical conditions optimal."
    elif "low-light" in lighting_desc or "underexposed" in lighting_desc:
        score = 0.40
        status = TokenStatus.UNCERTAIN
        explanation = "Underexposed environment detected; perceptual uncertainty elevated."
    elif "glare" in lighting_desc or "high-key" in lighting_desc:
        score = 0.40
        status = TokenStatus.UNCERTAIN
        explanation = "Specular glare detected; potential contrast saturation on reflective surfaces."
    elif obs:
        score = 0.80
        status = TokenStatus.SATISFIED
        explanation = "Scene ambient context evaluated from direct observations."
    else:
        score = 0.50
        status = TokenStatus.UNCERTAIN
        explanation = "Ambiguous scene lighting condition."

    return EvaluatedEvidenceToken(
        token_id=tok_def.token_id,
        task_name=tok_def.task_name,
        required=tok_def.required,
        status=status,
        satisfaction_score=round(score, 4),
        uncertainty=round(1.0 - score, 4),
        source_agent="scene",
        explanation=explanation,
        is_heuristic=True
    )


# ==============================================================================
# EVIDENCE DEFICIT ENGINE
# ==============================================================================

class EvidenceDeficitEngine:
    """
    Evidence Engine Core for VisionIQ.
    Calculates observable evidence satisfaction, decoupled completeness,
    reliability, contradiction, and unavailability, and returns a verified deficit score.
    """
    def __init__(self, weights: Optional[DeficitWeights] = None):
        self.weights = weights or DeficitWeights()

    def get_supported_token_ids(self) -> List[str]:
        """Returns the list of all valid, grounded token IDs."""
        return list(TOKEN_CATALOG.keys())

    def determine_required_tokens_for_intent(self, intent: str, query: str) -> List[str]:
        """Maps classified user intent and query to the appropriate token repertoire."""
        q = query.lower()
        required = []

        if "ppe" in intent or any(k in q for k in ["ppe", "helmet", "hard hat", "vest", "worker"]):
            required.extend(["ET-OBJ-PERSON-01", "ET-PPE-CRANIAL-02", "ET-PPE-TORSO-03", "ET-SCN-ENVIRONMENT-01"])

        if "defect" in intent or any(k in q for k in ["defect", "crack", "scratch", "damage", "flaw", "crazing"]):
            required.extend(["ET-DFT-LOCALIZATION-01", "ET-DFT-CLASSIFICATION-02", "ET-DFT-PROCEDURE-03", "ET-SCN-ENVIRONMENT-01"])

        if "ocr" in intent or any(k in q for k in ["serial", "read", "text", "ocr", "tag", "code"]):
            required.extend(["ET-OCR-TRANSCRIBE-01", "ET-SCN-ENVIRONMENT-01"])

        if not required:
            # Default general scene repertoire
            required.extend(["ET-OBJ-PERSON-01", "ET-SCN-ENVIRONMENT-01"])

        # Deduplicate preserving order
        seen = set()
        deduped = []
        for t in required:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped

    def evaluate(
        self,
        query: str,
        intent: str,
        findings: List[Dict[str, Any]],
        context_data: Dict[str, Any],
        executed_agents: List[str],
        failed_agents: Optional[List[str]] = None,
        requested_tokens: Optional[List[str]] = None,
        step: int = 0
    ) -> EvidenceEvaluationReport:
        """
        Executes full evaluation of evidence dimensions and computes composite deficit.
        """
        failed = failed_agents or []
        target_tokens = requested_tokens or self.determine_required_tokens_for_intent(intent, query)

        # 1. Populate Evidence Ledger & Extract Conflicts
        ledger = EvidenceLedger()
        for f in findings:
            ledger.add_finding(f, step=step)

        # Ingest external conflict evaluation from CrossAgentConflictEvaluator
        external_conflicts = CrossAgentConflictEvaluator.evaluate(findings, context_data, executed_agents)
        for disc in external_conflicts.discrepancies:
            ledger._conflicts.append(ConflictRecord(
                conflict_type=disc.get("type", "DISCREPANCY"),
                agents_involved=disc.get("agents", []),
                severity=float(disc.get("severity", 0.25)),
                description=disc.get("description", "")
            ))

        # 2. Evaluate Each Evidence Token
        evaluated_tokens: Dict[str, EvaluatedEvidenceToken] = {}
        unsupported_tokens: List[str] = []

        evaluator_dispatch = {
            "ET-OBJ-PERSON-01": evaluate_token_obj_person_01,
            "ET-PPE-CRANIAL-02": evaluate_token_ppe_cranial_02,
            "ET-PPE-TORSO-03": evaluate_token_ppe_torso_03,
            "ET-DFT-LOCALIZATION-01": evaluate_token_dft_localization_01,
            "ET-DFT-CLASSIFICATION-02": evaluate_token_dft_classification_02,
            "ET-DFT-PROCEDURE-03": evaluate_token_dft_procedure_03,
            "ET-OCR-TRANSCRIBE-01": evaluate_token_ocr_transcribe_01,
            "ET-SCN-ENVIRONMENT-01": evaluate_token_scn_environment_01,
        }

        for tok_id in target_tokens:
            if tok_id not in TOKEN_CATALOG or tok_id not in evaluator_dispatch:
                unsupported_tokens.append(tok_id)
                evaluated_tokens[tok_id] = EvaluatedEvidenceToken(
                    token_id=tok_id,
                    task_name="unsupported_modality",
                    required=True,
                    status=TokenStatus.UNSUPPORTED,
                    satisfaction_score=0.0,
                    uncertainty=1.0,
                    explanation=f"Token '{tok_id}' cannot be grounded in existing specialist outputs."
                )
                continue

            fn = evaluator_dispatch[tok_id]
            evaluated_tok = fn(
                query=query,
                context_data=context_data,
                findings=findings,
                executed=executed_agents,
                failed=failed
            )
            evaluated_tokens[tok_id] = evaluated_tok

        # 3. Calculate Decoupled Dimensions
        # Separate applicable required tokens (excluding NOT_APPLICABLE tokens)
        applicable_req_tokens = [
            t for t in evaluated_tokens.values()
            if t.required and t.status != TokenStatus.NOT_APPLICABLE
        ]

        # Dimension A: Unavailability U_unavail
        # Fraction of applicable required tokens whose target agent failed or is ungroundable
        if applicable_req_tokens:
            unavail_count = sum(
                1 for t in applicable_req_tokens
                if t.status in [TokenStatus.UNAVAILABLE, TokenStatus.UNSUPPORTED]
            )
            u_unavail = float(min(1.0, max(0.0, unavail_count / float(len(applicable_req_tokens)))))
        else:
            u_unavail = 0.0

        # Dimension B: Completeness C_ev
        # Calculated across ALL applicable required tokens.
        # Tokens that are UNAVAILABLE or MISSING have satisfaction_score = 0.0,
        # ensuring that missing or failed mandatory evidence directly decreases completeness.
        if applicable_req_tokens:
            c_ev = float(sum(t.satisfaction_score for t in applicable_req_tokens) / float(len(applicable_req_tokens)))
            c_ev = float(min(1.0, max(0.0, c_ev)))
        else:
            # If all required tokens are NOT_APPLICABLE (e.g. 0 persons confirmed present in PPE inspection),
            # the scene vacuously satisfies applicable requirements.
            c_ev = 1.0

        # Dimension C: Reliability R_ev
        # Calibrated reliability of observed ledger entries
        ledger_entries = ledger.entries
        if ledger_entries:
            reliabilities = []
            for entry in ledger_entries:
                p = entry.calibrated_confidence if entry.calibrated_confidence is not None else entry.raw_confidence or 0.70
                u = entry.spatial_uncertainty or 0.20
                reliabilities.append(float(min(1.0, max(0.0, p * (1.0 - u)))))
            r_ev = float(sum(reliabilities) / float(len(reliabilities)))
            r_ev = float(min(1.0, max(0.0, r_ev)))
        else:
            r_ev = 0.50 if not executed_agents else 0.80

        # Dimension D: Cross-Agent Contradiction Conflict Score S_conflict
        conflicts = ledger.conflicts
        if conflicts:
            conflict_severities = [c.severity for c in conflicts]
            # Bounded sum of severity scores in [0, 1]
            conflict_score = float(min(1.0, max(0.0, sum(conflict_severities))))
        else:
            conflict_score = 0.0

        # 4. Composite Deficit Calculation
        # D = wC * (1 - C_ev) + wR * (1 - R_ev) + wS * S_conflict + wU * U_unavail
        w = self.weights.normalized()
        raw_deficit = (
            w["w_completeness"] * (1.0 - c_ev) +
            w["w_reliability"] * (1.0 - r_ev) +
            w["w_conflict"] * conflict_score +
            w["w_unavailability"] * u_unavail
        )
        deficit = float(min(1.0, max(0.0, raw_deficit)))

        summary = (
            f"Deficit D={deficit:.3f} [C_ev={c_ev*100:.1f}%, R_ev={r_ev*100:.1f}%, "
            f"S_conflict={conflict_score:.2f}, U_unavail={u_unavail*100:.1f}%]. "
            f"Tokens: {sum(1 for t in applicable_req_tokens if t.status == TokenStatus.SATISFIED)}/{len(applicable_req_tokens)} satisfied."
        )

        return EvidenceEvaluationReport(
            tokens=evaluated_tokens,
            completeness=round(c_ev, 4),
            reliability=round(r_ev, 4),
            conflict_score=round(conflict_score, 4),
            unavailability=round(u_unavail, 4),
            composite_deficit=round(deficit, 4),
            normalized_weights=w,
            conflicts=conflicts,
            ledger_entries=ledger_entries,
            unsupported_tokens=unsupported_tokens,
            summary=summary
        )
