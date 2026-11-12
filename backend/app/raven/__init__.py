"""
RAVEN: Risk-Aware Visual Evidence Navigator
Provides evidence verification, cross-agent conflict detection,
and tri-state operational decision policies (ACCEPT, VERIFY, ABSTAIN).
"""

from backend.app.raven.schemas import (
    RavenDecision,
    EvidenceToken,
    EvidenceRepertoire,
    ConflictReport,
    HumanEscalationDossier,
    RavenAssessment
)
from backend.app.raven.calibration import (
    calibrate_confidence,
    compute_spatial_uncertainty,
    compute_occlusion_uncertainty,
    compute_composite_uncertainty
)
from backend.app.raven.conflict import CrossAgentConflictEvaluator
from backend.app.raven.engine import RavenEngine

__all__ = [
    "RavenDecision",
    "EvidenceToken",
    "EvidenceRepertoire",
    "ConflictReport",
    "HumanEscalationDossier",
    "RavenAssessment",
    "calibrate_confidence",
    "compute_spatial_uncertainty",
    "compute_occlusion_uncertainty",
    "compute_composite_uncertainty",
    "CrossAgentConflictEvaluator",
    "RavenEngine",
]
