"""
RAVEN: Risk-Aware Visual Evidence Navigator - Data Contracts & Schemas
Defines core data models for evidence repertoires, uncertainty calibration,
cross-agent conflict entropy, finding provenance, site-specific physical-absence
certificates, structured human review records, and tri-state decision policies.
"""

import time
import math
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from backend.app.raven.calibration import recompute_clopper_pearson_upper_bound


class RavenDecision(str, Enum):
    """Tri-state operational decision policy for multi-agent evidence synthesis."""
    ACCEPT = "ACCEPT"    # Low risk & sufficient evidence: findings certified for reporting
    VERIFY = "VERIFY"    # Resolvable contradiction or required additional verification
    ABSTAIN = "ABSTAIN"  # Critical evidence missing, unreliable, unavailable, or unsafe


class FindingProvenance(BaseModel):
    """Auditable provenance record for an individual finding."""
    model_config = {"protected_namespaces": ()}

    finding_uid: str
    specialist: str
    model_name: str
    model_version: str
    checkpoint_hash: Optional[str] = None
    image_ref: str
    region: Optional[List[float]] = None
    confidence_type: str = "calibrated"  # "calibrated", "raw", "heuristic_mask", "pattern_match"
    evidence_token: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    execution_status: str = "SUCCESS"
    dependency_satisfied: bool = True
    is_valid: bool = True
    validation_error: Optional[str] = None


class ConflictRecord(BaseModel):
    """Structured record of a cross-agent contradiction or spatial discrepancy."""
    model_config = {"protected_namespaces": ()}

    conflict_id: str
    involved_specialists: List[str]
    finding_ids: List[str] = Field(default_factory=list)
    relevant_regions: List[List[float]] = Field(default_factory=list)
    conflict_type: str
    severity: float = Field(..., ge=0.0, le=1.0)
    description: str
    resolution_status: str = "UNRESOLVED"  # "UNRESOLVED" or "RESOLVED"
    evidence_required: str
    resolving_specialist: Optional[str] = None
    resolution_rationale: Optional[str] = None


class SiteAbsenceValidationCertificate(BaseModel):
    """
    Validated site-specific physical-absence certification.
    Binds absence clearance to workcell, optics, detector checkpoint hash,
    and underlying statistical trials.
    """
    model_config = {"protected_namespaces": ()}

    certificate_id: str
    workcell_id: str
    camera_config: Dict[str, Any] = Field(default_factory=dict)
    detector_checkpoint_hash: str
    operational_subgroup: str
    validation_manifest_id: str
    validation_trials_n: int = Field(..., ge=1)
    observed_false_negatives_k: int = Field(..., ge=0)
    stored_upper_bound: float
    recomputed_upper_bound: float = 1.0
    expiry_timestamp: float
    is_revoked: bool = False
    image_quality_requirements: Dict[str, float] = Field(default_factory=dict)

    def validate_certificate(
        self,
        current_time: float,
        checkpoint_hash: str,
        workcell_id: str,
        image_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """
        Validates absence certification against current operating parameters.
        Recomputes exact Clopper-Pearson bound directly from k and n.
        """
        if self.is_revoked:
            return False, f"Certificate {self.certificate_id} is revoked."
        if current_time > self.expiry_timestamp:
            return False, f"Certificate {self.certificate_id} has expired."
        if self.workcell_id != workcell_id:
            return False, f"Certificate workcell mismatch: {self.workcell_id} != {workcell_id}."
        if self.detector_checkpoint_hash != checkpoint_hash:
            return False, f"Detector checkpoint hash mismatch: expected {self.detector_checkpoint_hash[:8]}."

        actual_bound = recompute_clopper_pearson_upper_bound(
            self.observed_false_negatives_k, self.validation_trials_n
        )
        self.recomputed_upper_bound = actual_bound
        if actual_bound > 0.05:
            return False, f"Recomputed Clopper-Pearson upper bound {actual_bound:.4f} exceeds safety threshold 0.05."

        if image_metrics and self.image_quality_requirements:
            min_blur = self.image_quality_requirements.get("min_laplacian_variance", 5.0)
            if image_metrics.get("laplacian_variance", 100.0) < min_blur:
                return False, f"Image blur index {image_metrics.get('laplacian_variance', 0.0):.1f} fails minimum {min_blur}."

        return True, "Certificate valid and statistically certified."


class HumanReviewRecord(BaseModel):
    """
    Concise human-review escalation record generated when RAVEN returns VERIFY or ABSTAIN.
    """
    model_config = {"protected_namespaces": ()}

    inspection_id: str
    query: str
    decision: str  # "VERIFY" or "ABSTAIN"
    reason: str
    relevant_regions: List[Dict[str, Any]] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    conflicting_findings: List[Dict[str, Any]] = Field(default_factory=list)
    failed_specialists: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    regulatory_clauses: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    disclaimer: str = Field(
        "Inspection decision support only. RAVEN decisions do not constitute a certified safety guarantee. "
        "Qualified human inspection required."
    )


class EvidenceToken(BaseModel):
    """An individual verifiable unit of visual or contextual evidence."""
    model_config = {"protected_namespaces": ()}

    token_id: str = Field(..., description="Unique token identifier, e.g. ET-PPE-CRANIAL-01")
    description: str = Field(..., description="Semantic meaning of this evidence token")
    required: bool = Field(True, description="Whether this token is mandatory for the query intent")
    present: bool = Field(False, description="Whether this token was successfully observed in perceptual output")
    raw_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Raw model confidence or mask fraction")
    calibrated_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Temperature-scaled / Platt-calibrated confidence")
    uncertainty: float = Field(0.0, ge=0.0, le=1.0, description="Epistemic uncertainty associated with this token")
    source_agent: str = Field(..., description="Originating agent (object, ppe, defect, ocr, scene, rag)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Bounding coordinates, mask size, or token details")


class EvidenceRepertoire(BaseModel):
    """Query-conditioned set of required and optional visual evidence tokens."""
    intent: str = Field(..., description="Classified user query intent")
    tokens: List[EvidenceToken] = Field(default_factory=list, description="List of evidence tokens evaluated")
    total_required: int = Field(0, description="Count of mandatory tokens")
    total_satisfied: int = Field(0, description="Count of satisfied mandatory tokens")
    completeness_score: float = Field(0.0, ge=0.0, le=1.0, description="C_ev in [0, 1]: Fraction of satisfied evidence")


class ConflictReport(BaseModel):
    """Quantitative report on cross-agent discrepancies and spatial/semantic conflict."""
    has_conflict: bool = Field(False, description="Whether any significant contradiction between agents was detected")
    conflict_entropy: float = Field(0.0, ge=0.0, le=1.0, description="Normalized conflict entropy H_conflict in [0, 1]")
    discrepancy_count: int = Field(0, description="Total number of discrete discrepancies detected")
    discrepancies: List[Dict[str, Any]] = Field(default_factory=list, description="Itemized discrepancy entries with agent pairs and details")
    conflict_records: List[ConflictRecord] = Field(default_factory=list, description="Preserved conflict records with resolution status")


class HumanEscalationDossier(BaseModel):
    """Structured forensic audit dossier generated when automated clearance is refused."""
    escalation_uid: str = Field(..., description="Unique escalation tracking identifier, e.g. ESC-RAVEN-9F3A")
    urgency_level: str = Field("HIGH", description="Urgency categorization: LOW, MEDIUM, HIGH, CRITICAL")
    reason: str = Field(..., description="Concise executive justification for refusal of automated certification")
    primary_concerns: List[str] = Field(default_factory=list, description="Specific safety or structural hazards identified")
    missing_evidence: List[str] = Field(default_factory=list, description="Unsatisfied evidence tokens requiring physical inspection")
    recommended_human_actions: List[str] = Field(default_factory=list, description="Actionable NDT, physical check, or supervisor interventions")
    regulatory_clauses: List[str] = Field(default_factory=list, description="Applicable OSHA / ASTM / ISO regulatory standards")
    bounded_liability_statement: str = Field(
        "VisionIQ/RAVEN is an advisory decision-support system, not a certified safety certifier. "
        "Automated clearance is explicitly withheld. Qualified on-site personnel must conduct manual verification.",
        description="Legal and operational liability disclaimers"
    )


class RavenAssessment(BaseModel):
    """Complete risk-aware verification assessment produced by RAVEN."""
    model_config = {"protected_namespaces": ()}

    decision: RavenDecision = Field(..., description="Operational decision: ACCEPT, VERIFY, or ABSTAIN")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Composite operational risk R in [0, 1]")
    hazard_severity: float = Field(..., ge=0.0, le=1.0, description="Domain hazard severity S_hazard in [0, 1]")
    evidence_completeness: float = Field(..., ge=0.0, le=1.0, description="Evidence completeness C_ev in [0, 1]")
    uncertainty_index: float = Field(..., ge=0.0, le=1.0, description="Epistemic uncertainty index U(E) in [0, 1]")
    conflict_entropy: float = Field(..., ge=0.0, le=1.0, description="Cross-agent conflict entropy H_conflict in [0, 1]")
    decision_rationale: str = Field(..., description="Transparent, auditable justification for the tri-state decision")
    evidence_repertoire: EvidenceRepertoire = Field(..., description="Query-conditioned token evaluation")
    conflict_report: ConflictReport = Field(..., description="Cross-agent discrepancy evaluation")
    secondary_agents_suggested: List[str] = Field(default_factory=list, description="Targeted agents to invoke if decision is VERIFY")
    escalation_dossier: Optional[HumanEscalationDossier] = Field(None, description="Escalation dossier if decision is ABSTAIN or human review required")
    human_review_record: Optional[HumanReviewRecord] = Field(None, description="Concise human review record for VERIFY and ABSTAIN")
    validated_findings: List[Dict[str, Any]] = Field(default_factory=list, description="Findings that passed full provenance and consistency checks")
    invalid_findings: List[Dict[str, Any]] = Field(default_factory=list, description="Findings rejected due to invalid provenance or dependencies")
    failed_critical_specialists: List[str] = Field(default_factory=list, description="Specialists essential to query that failed")
    failed_irrelevant_specialists: List[str] = Field(default_factory=list, description="Failed specialists that are non-critical to query")
    conflict_records: List[ConflictRecord] = Field(default_factory=list, description="Preserved conflict records with resolution status")
    absence_certificate: Optional[SiteAbsenceValidationCertificate] = Field(None, description="Absence certificate if provided")
    provenance_certified: bool = Field(True, description="Whether all evidence provenance checks passed")
    safety_veto_applied: Optional[str] = Field(None, description="Name of safety veto rule applied if overridden")
    verification_timestamp: float = Field(..., description="UNIX epoch timestamp of verification execution")

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to serializable dictionary for JSON and DB persistence."""
        return self.model_dump()
