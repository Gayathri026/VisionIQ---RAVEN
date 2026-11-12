"""
RAVEN Core Mathematical Engine & Safety Decision Layer (Phase 4)
Implements query-conditioned evidence requirement extraction, multi-dimensional
provenance verification, site-specific physical-absence statistical certification,
cross-agent contradiction resolution, and strict safety-gate precedence over tri-state decisions:
D in { ACCEPT, VERIFY, ABSTAIN }.
"""

import os
import time
import uuid
import math
from typing import Dict, Any, List, Optional, Tuple

from backend.app.core.config import settings
from backend.app.raven.schemas import (
    RavenDecision,
    EvidenceToken,
    EvidenceRepertoire,
    ConflictReport,
    ConflictRecord,
    HumanEscalationDossier,
    HumanReviewRecord,
    SiteAbsenceValidationCertificate,
    FindingProvenance,
    RavenAssessment
)
from backend.app.raven.calibration import (
    calibrate_confidence,
    compute_spatial_uncertainty,
    compute_composite_uncertainty,
    recompute_clopper_pearson_upper_bound,
    measure_image_quality
)
from backend.app.raven.conflict import CrossAgentConflictEvaluator
from backend.app.orchestration.evidence_deficit import TOKEN_CATALOG, TokenStatus


class RavenEngine:
    """
    Risk-Aware Visual Evidence Navigator (RAVEN) Engine.
    Acts as the safety and evidence-validation layer for VisionIQ.
    Determines whether findings support ACCEPT, VERIFY, or ABSTAIN with
    strict safety-gate precedence, auditable conflict records, and human escalation.
    """

    def __init__(
        self,
        tau_accept: float = 0.30,
        tau_escalate: float = 0.65,
        lambda_conflict: float = 0.35
    ):
        self.tau_accept = tau_accept
        self.tau_escalate = tau_escalate
        self.lambda_conflict = lambda_conflict

    @staticmethod
    def is_specialist_critical(specialist: str, intent: str, query: str) -> bool:
        """
        Determines whether a specialist is critical to the current query intent.
        Section 5: An irrelevant specialist failure must NOT automatically cause ABSTAIN.
        """
        q = query.lower()
        spec = specialist.lower()

        # PPE intent: object and ppe are critical
        if "ppe" in intent or any(k in q for k in ["ppe", "worker", "helmet", "vest"]):
            return spec in ("object", "ppe")

        # Defect intent: defect is critical
        if "defect" in intent or any(k in q for k in ["defect", "crack", "scratch", "damage", "flaw"]):
            return spec in ("defect",)

        # OCR intent: ocr is critical
        if "ocr" in intent or any(k in q for k in ["ocr", "serial", "read", "text"]):
            return spec in ("ocr",)

        # Scene intent: scene is critical
        if "scene" in intent or any(k in q for k in ["scene", "lighting", "overview"]):
            return spec in ("scene", "object")

        # General scene inquiry
        return spec in ("object", "scene")

    def validate_findings_provenance(
        self,
        findings: List[Dict[str, Any]],
        image_path: Optional[str],
        executed_agents: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool, Optional[str]]:
        """
        Section 2: Validates finding provenance, source specialist, model version,
        image reference, confidence type, evidence token, execution and dependency status.
        Rejects unsupported findings.
        """
        validated: List[Dict[str, Any]] = []
        invalid: List[Dict[str, Any]] = []

        from backend.app.orchestration.router import AVAILABLE_AGENTS, MANDATORY_PREREQUISITES

        for f in findings:
            source = f.get("agent_source") or f.get("specialist")
            if not source or source not in AVAILABLE_AGENTS:
                f["validation_error"] = f"Unsupported or unknown source specialist: {source}"
                invalid.append(f)
                continue

            # Check model version provenance
            model_ver = f.get("model_version") or settings.AGENT_MODEL_VERSIONS.get(source)
            if not model_ver or "untrusted" in str(model_ver).lower() or "tampered" in str(model_ver).lower():
                f["validation_error"] = f"Invalid or untrusted model version provenance: {model_ver}"
                invalid.append(f)
                continue

            # Check image reference
            img_ref = f.get("image_ref") or f.get("image_path")
            if image_path and img_ref:
                # Normalise base names for comparison
                if os.path.basename(img_ref) != os.path.basename(image_path) and not os.path.exists(img_ref):
                    f["validation_error"] = f"Image reference mismatch: {img_ref} vs {image_path}"
                    invalid.append(f)
                    continue

            # Check execution status
            if source not in executed_agents:
                f["validation_error"] = f"Specialist {source} not present in executed agents list"
                invalid.append(f)
                continue

            # Check dependency status
            prereqs = MANDATORY_PREREQUISITES.get(source, [])
            if prereqs:
                if source == "rca":
                    if not any(p in executed_agents for p in prereqs):
                        f["validation_error"] = f"Prerequisite failure: RCA requires at least one perceptual specialist"
                        invalid.append(f)
                        continue
                else:
                    if not all(p in executed_agents for p in prereqs):
                        f["validation_error"] = f"Prerequisite failure: {source} requires {prereqs}"
                        invalid.append(f)
                        continue

            # Check token grounding if token_id is provided
            tok_id = f.get("evidence_token")
            if tok_id and tok_id not in TOKEN_CATALOG:
                f["validation_error"] = f"Unregistered evidence token: {tok_id}"
                invalid.append(f)
                continue

            # Finding passed full provenance verification
            f["provenance_verified"] = True
            validated.append(f)

        all_valid = (len(invalid) == 0)
        err_msg = invalid[0].get("validation_error") if invalid else None
        return validated, invalid, all_valid, err_msg

    def _determine_hazard_severity(self, intent: str, query: str, findings: List[Dict[str, Any]]) -> float:
        """
        Calculates operational domain hazard severity S_hazard in [0, 1] based on regulatory context.
        """
        q = query.lower()

        # Critical life-safety: overhead hazards, missing fall/head protection
        if any(k in q for k in ["helmet", "hard hat", "fall", "crane", "overhead", "high voltage", "scaffold"]):
            return 0.95

        # Standard PPE compliance
        if "ppe" in intent or any(k in q for k in ["ppe", "vest", "protective", "violation"]):
            for f in findings:
                if "CRITICAL" in str(f.get("label", "")).upper():
                    return 0.95
            return 0.85

        # Surface flaws with lower structural severity (scratches, patches, inclusion)
        if any(k in q for k in ["scratch", "pitted", "patch", "inclusion", "blemish"]):
            return 0.60

        # Structural / crack defect in load-bearing steel
        if "defect" in intent or any(k in q for k in ["crack", "crazing", "fracture", "fatigue", "corrosion"]):
            return 0.85

        # Root Cause Analysis
        if "rca" in intent or any(k in q for k in ["root cause", "failure", "halted", "breakdown"]):
            return 0.75

        # Industrial OCR / rating plate reading
        if "ocr" in intent or any(k in q for k in ["serial", "voltage", "rating", "ocr", "read"]):
            return 0.40

        # Object counting and spatial location
        if "count" in intent or any(k in q for k in ["count", "how many"]):
            return 0.25

        # General scene context
        return 0.20

    def _build_evidence_repertoire(
        self,
        intent: str,
        query: str,
        findings: List[Dict[str, Any]],
        context_data: Dict[str, Any]
    ) -> EvidenceRepertoire:
        """
        Extracts query-conditioned evidence tokens and evaluates whether each is satisfied.
        """
        tokens: List[EvidenceToken] = []
        object_res = context_data.get("object_result") or {}
        ppe_res = context_data.get("ppe_result") or {}
        defect_res = context_data.get("defect_result") or {}
        ocr_res = context_data.get("ocr_result") or {}
        rag_refs = context_data.get("rag_references") or []

        q = query.lower()

        # 1. PPE Compliance Repertoire
        if "ppe" in intent or any(k in q for k in ["ppe", "worker", "helmet", "vest"]):
            detected_persons = [
                o for o in object_res.get("objects", [])
                if o.get("label", "").lower() in ["person", "worker"]
            ] or [
                f for f in findings
                if f.get("agent_source") == "object" and f.get("label", "").lower() in ["person", "worker"]
            ]
            
            # If person evaluation from PPE specialist is available, check occlusion and uncertainty
            ppe_evals = ppe_res.get("person_evaluations", [])
            has_severe_occlusion = any(p.get("is_occluded") and p.get("uncertainty", 0.0) >= 0.60 for p in ppe_evals if isinstance(p, dict))
            
            # Person presence is valid only if detected without severe occlusion
            has_valid_person = (len(detected_persons) > 0 or len(ppe_evals) > 0) and not has_severe_occlusion
            
            person_conf = 0.92
            if detected_persons:
                person_conf = max([d.get("confidence", 0.92) for d in detected_persons] or [0.92])
            elif ppe_evals and not has_severe_occlusion:
                person_conf = max([1.0 - p.get("uncertainty", 0.08) for p in ppe_evals if isinstance(p, dict)] or [0.92])

            tokens.append(EvidenceToken(
                token_id="ET-PPE-PERSON-01",
                description="Personnel body bounding box localization with valid spatial coordinates",
                required=True,
                present=has_valid_person,
                raw_confidence=person_conf if has_valid_person else 0.0,
                calibrated_confidence=calibrate_confidence(person_conf) if has_valid_person else 0.0,
                uncertainty=round(1.0 - person_conf, 3) if has_valid_person else 0.85,
                source_agent="object",
                metadata={"detected_count": len(detected_persons) or len(ppe_evals)}
            ))

            has_helmet_eval = (
                (ppe_res.get("compliant_count", 0) > 0 or ppe_res.get("non_compliant_count", 0) > 0) and not has_severe_occlusion or
                any("helmet" in f.get("label", "").lower() or "hard hat" in f.get("label", "").lower() for f in findings if not f.get("human_review_required"))
            )
            tokens.append(EvidenceToken(
                token_id="ET-PPE-CRANIAL-02",
                description="Head zone protective equipment (hard hat / helmet) visual verification",
                required=True,
                present=has_helmet_eval,
                raw_confidence=0.88 if has_helmet_eval else 0.0,
                calibrated_confidence=calibrate_confidence(0.88) if has_helmet_eval else 0.0,
                uncertainty=0.12 if has_helmet_eval else 0.90,
                source_agent="ppe",
                metadata={"compliance_status": ppe_res.get("compliance_status", "UNKNOWN")}
            ))

            has_vest_eval = (
                ppe_res.get("vest_count", 0) > 0 or
                any("vest" in f.get("label", "").lower() for f in findings)
            )
            tokens.append(EvidenceToken(
                token_id="ET-PPE-TORSO-03",
                description="Torso zone high-visibility safety garment inspection",
                required=False,
                present=has_vest_eval,
                raw_confidence=0.85 if has_vest_eval else 0.0,
                calibrated_confidence=calibrate_confidence(0.85) if has_vest_eval else 0.0,
                uncertainty=0.15 if has_vest_eval else 0.50,
                source_agent="ppe"
            ))

        # 2. Defect Inspection Repertoire
        if "defect" in intent or any(k in q for k in ["defect", "crack", "scratch", "damage"]):
            has_defect_scan = defect_res.get("defect_scan_completed", True) and (
                defect_res.get("defects_detected", 0) > 0 or
                any(f.get("agent_source") == "defect" for f in findings)
            )
            tokens.append(EvidenceToken(
                token_id="ET-DFT-LOCALIZATION-01",
                description="Surface defect spatial bounding box / segmentation contour localization",
                required=True,
                present=has_defect_scan,
                raw_confidence=0.90 if has_defect_scan else 0.0,
                calibrated_confidence=calibrate_confidence(0.90) if has_defect_scan else 0.0,
                uncertainty=0.10 if has_defect_scan else 0.85,
                source_agent="defect"
            ))

            has_class = any(f.get("is_confirmed_defect") or f.get("detection_confidence", 0.0) >= 0.50 for f in findings if f.get("agent_source") == "defect")
            tokens.append(EvidenceToken(
                token_id="ET-DFT-CLASSIFICATION-02",
                description="Neural defect morphology classification (e.g. crazing, scratch, inclusion)",
                required=True,
                present=has_class,
                raw_confidence=0.88 if has_class else 0.0,
                calibrated_confidence=calibrate_confidence(0.88) if has_class else 0.0,
                uncertainty=0.12 if has_class else 0.90,
                source_agent="defect"
            ))

        # 3. OCR Text Extraction Repertoire
        if "ocr" in intent or any(k in q for k in ["serial", "read", "text", "ocr", "code"]):
            has_ocr = (
                ocr_res.get("ocr_scan_completed", True) and
                str(ocr_res.get("detected_text", "")).strip() not in ("", "No text detected") or
                any(f.get("agent_source") == "ocr" for f in findings)
            )
            tokens.append(EvidenceToken(
                token_id="ET-OCR-TRANSCRIBE-01",
                description="Alphanumeric serial number optical character transcription",
                required=True,
                present=has_ocr,
                raw_confidence=0.86 if has_ocr else 0.0,
                calibrated_confidence=calibrate_confidence(0.86) if has_ocr else 0.0,
                uncertainty=0.14 if has_ocr else 0.85,
                source_agent="ocr"
            ))

        # Default fallback token if repertoire is otherwise empty
        if not tokens:
            has_obj = len(findings) > 0 or len(object_res.get("objects", [])) > 0
            conf = 0.85
            if findings:
                conf = max([f.get("detection_confidence", f.get("confidence", 0.85)) for f in findings] or [0.85])
            elif object_res.get("objects"):
                conf = max([o.get("confidence", 0.85) for o in object_res["objects"]] or [0.85])
            tokens.append(EvidenceToken(
                token_id="ET-GEN-PRESENCE-01",
                description="General scene perceptual grounding verification",
                required=True,
                present=has_obj,
                raw_confidence=conf if has_obj else 0.0,
                calibrated_confidence=calibrate_confidence(conf) if has_obj else 0.0,
                uncertainty=round(1.0 - conf, 3) if has_obj else 0.80,
                source_agent="object"
            ))

        # Calculate completeness across REQUIRED tokens
        req_tokens = [t for t in tokens if t.required]
        total_req = len(req_tokens)
        satisfied_req = sum(1 for t in req_tokens if t.present)

        if total_req > 0:
            weighted_sum = sum(t.calibrated_confidence for t in req_tokens if t.present)
            completeness = round(weighted_sum / float(total_req), 3)
        else:
            completeness = 1.0

        return EvidenceRepertoire(
            intent=intent,
            tokens=tokens,
            total_required=total_req,
            total_satisfied=satisfied_req,
            completeness_score=min(1.0, max(0.0, completeness))
        )

    def _generate_escalation_dossier(
        self,
        query: str,
        hazard_severity: float,
        repertoire: EvidenceRepertoire,
        conflict_report: ConflictReport,
        risk_score: float,
        reason_override: Optional[str] = None
    ) -> HumanEscalationDossier:
        """
        Synthesizes a structured forensic audit dossier for human safety officers.
        """
        urgency = "CRITICAL" if hazard_severity >= 0.90 or risk_score >= 0.75 else "HIGH"
        primary_concerns = []
        missing_evidence = []
        recommended_actions = []
        regulatory_clauses = []

        for t in repertoire.tokens:
            if t.required and not t.present:
                missing_evidence.append(f"Unsatisfied Evidence: {t.description} ({t.token_id})")

        for d in conflict_report.discrepancies:
            if d.get("resolution_status") != "RESOLVED":
                primary_concerns.append(f"Cross-Agent Conflict [{d.get('type')}]: {d.get('description')}")

        if "ppe" in repertoire.intent or "worker" in query.lower():
            regulatory_clauses.append("OSHA 29 CFR 1926.100 (Head Protection in Construction)")
            regulatory_clauses.append("OSHA 29 CFR 1926.95 (Personal Protective Equipment Criteria)")
            recommended_actions.append("Dispatch floor safety supervisor to perform immediate visual verification of active personnel.")
            recommended_actions.append("Audit gate turnstile logs and inspect secondary vantage camera streams.")
            primary_concerns.append("Worker compliance could not be certified autonomously due to spatial occlusion or low crop resolution.")

        elif "defect" in repertoire.intent or "crack" in query.lower():
            regulatory_clauses.append("ASTM A568/A568M (Standard Specification for Steel Sheet Products)")
            regulatory_clauses.append("ISO 9712 (Non-Destructive Testing — Qualification and Certification)")
            recommended_actions.append("Halt processing of suspect coil/batch until certified Dye Penetrant (PT) inspection is completed.")
            recommended_actions.append("Measure surface roughness profilometry and compare against contract acceptance limits.")
            primary_concerns.append("Automated vision detected anomalous surface morphology requiring physical depth verification.")

        else:
            recommended_actions.append("Qualified inspector must visually verify region of interest.")

        reason = reason_override or (
            f"Automated clearance withheld due to elevated operational risk (R={risk_score:.3f}). "
            f"Evidence completeness is {repertoire.completeness_score*100:.1f}%, with {len(conflict_report.discrepancies)} active discrepancy(ies)."
        )

        return HumanEscalationDossier(
            escalation_uid=f"ESC-RAVEN-{uuid.uuid4().hex[:8].upper()}",
            urgency_level=urgency,
            reason=reason,
            primary_concerns=primary_concerns,
            missing_evidence=missing_evidence,
            recommended_human_actions=recommended_actions,
            regulatory_clauses=regulatory_clauses
        )

    def _generate_human_review_record(
        self,
        inspection_id: str,
        query: str,
        decision: RavenDecision,
        reason: str,
        repertoire: EvidenceRepertoire,
        conflict_report: ConflictReport,
        failed_specialists: List[str],
        findings: List[Dict[str, Any]]
    ) -> HumanReviewRecord:
        """
        Section 6: Synthesizes a concise human-review record when RAVEN returns VERIFY or ABSTAIN.
        Rule: Do NOT automatically convert a human-review recommendation into ACCEPT.
        """
        relevant_regions = []
        for f in findings:
            if f.get("bbox"):
                relevant_regions.append({
                    "finding_uid": f.get("finding_uid", ""),
                    "specialist": f.get("agent_source", ""),
                    "bbox": f.get("bbox")
                })

        missing_evidence = [
            f"{t.token_id}: {t.description}"
            for t in repertoire.tokens if t.required and not t.present
        ]

        conflicting_findings = [
            {
                "conflict_id": rec.conflict_id,
                "type": rec.conflict_type,
                "specialists": rec.involved_specialists,
                "description": rec.description,
                "regions": rec.relevant_regions,
                "resolution_status": rec.resolution_status
            }
            for rec in conflict_report.conflict_records
            if rec.resolution_status == "UNRESOLVED"
        ]

        recommended_actions = []
        regulatory_clauses = []

        if decision == RavenDecision.ABSTAIN:
            recommended_actions.append("Mandatory physical audit by certified on-site quality inspector.")
            recommended_actions.append("Halt automated batch release until manual sign-off is logged in facility ERP.")
        else:
            recommended_actions.append("Secondary targeted optical verification pass over flagged regions.")
            recommended_actions.append("Human supervisor visual check on terminal interface.")

        if "ppe" in repertoire.intent or "worker" in query.lower():
            regulatory_clauses.append("OSHA 29 CFR 1926.100")
        elif "defect" in repertoire.intent or "crack" in query.lower():
            regulatory_clauses.append("ASTM A568/A568M")
            regulatory_clauses.append("ISO 9712")

        return HumanReviewRecord(
            inspection_id=inspection_id,
            query=query,
            decision=decision.value,
            reason=reason,
            relevant_regions=relevant_regions,
            missing_evidence=missing_evidence,
            conflicting_findings=conflicting_findings,
            failed_specialists=failed_specialists,
            recommended_actions=recommended_actions,
            regulatory_clauses=regulatory_clauses
        )

    def evaluate(
        self,
        query: str,
        intent: str,
        findings: List[Dict[str, Any]],
        context_data: Dict[str, Any],
        executed_agents: List[str],
        failed_agents: Optional[List[str]] = None,
        advisory_absence_flags: Optional[Dict[str, bool]] = None,
        evidence_report: Optional[Any] = None,
        image_path: Optional[str] = None,
        absence_certificate: Optional[SiteAbsenceValidationCertificate] = None,
        worker_cleanup_failed: bool = False,
        budget_exhausted: bool = False,
        workcell_id: str = "WORKCELL_01"
    ) -> RavenAssessment:
        """
        Executes full RAVEN evidence validation and safety decision synthesis.
        Enforces strict safety-gate precedence (Section 5):
        Safety gates apply before ordinary acceptance. Numerical stopping thresholds
        must NEVER override a safety veto.
        """
        failed = failed_agents or []
        inspection_id = context_data.get("session_id") or f"INSP-{uuid.uuid4().hex[:8].upper()}"

        # ----------------------------------------------------------------------
        # 1. Image Provenance & Optical Quality Verification (Section 2 & 5)
        # ----------------------------------------------------------------------
        img_metrics: Optional[Dict[str, float]] = None
        if image_path:
            img_quality = measure_image_quality(image_path)
            if not img_quality["is_valid"]:
                # Safety Veto 1: Image provenance failure or poor optical quality
                veto_reason = f"SAFETY GATE VETO (Image Provenance): {img_quality['error']}"
                rep = self._build_evidence_repertoire(intent, query, findings, context_data)
                conf = CrossAgentConflictEvaluator.evaluate(findings, context_data, executed_agents)
                dossier = self._generate_escalation_dossier(query, 0.90, rep, conf, 0.90, reason_override=veto_reason)
                h_rec = self._generate_human_review_record(inspection_id, query, RavenDecision.ABSTAIN, veto_reason, rep, conf, failed, findings)
                return RavenAssessment(
                    decision=RavenDecision.ABSTAIN,
                    risk_score=0.90,
                    hazard_severity=0.90,
                    evidence_completeness=0.0,
                    uncertainty_index=1.0,
                    conflict_entropy=0.0,
                    decision_rationale=veto_reason,
                    evidence_repertoire=rep,
                    conflict_report=conf,
                    escalation_dossier=dossier,
                    human_review_record=h_rec,
                    provenance_certified=False,
                    safety_veto_applied="INVALID_IMAGE_PROVENANCE_OR_QUALITY",
                    verification_timestamp=time.time()
                )
            img_metrics = {
                "laplacian_variance": img_quality.get("laplacian_variance", 100.0),
                "mean_luminance": img_quality.get("mean_luminance", 128.0)
            }

        # ----------------------------------------------------------------------
        # 2. Worker Containment & Execution Isolation Check (Section 5 & 8)
        # ----------------------------------------------------------------------
        if worker_cleanup_failed:
            veto_reason = "SAFETY GATE VETO (Worker Containment): Isolated worker process failed cleanup or remained uncontained."
            rep = self._build_evidence_repertoire(intent, query, findings, context_data)
            conf = CrossAgentConflictEvaluator.evaluate(findings, context_data, executed_agents)
            dossier = self._generate_escalation_dossier(query, 0.95, rep, conf, 0.95, reason_override=veto_reason)
            h_rec = self._generate_human_review_record(inspection_id, query, RavenDecision.ABSTAIN, veto_reason, rep, conf, failed, findings)
            return RavenAssessment(
                decision=RavenDecision.ABSTAIN,
                risk_score=0.95,
                hazard_severity=0.95,
                evidence_completeness=rep.completeness_score,
                uncertainty_index=1.0,
                conflict_entropy=0.0,
                decision_rationale=veto_reason,
                evidence_repertoire=rep,
                conflict_report=conf,
                escalation_dossier=dossier,
                human_review_record=h_rec,
                provenance_certified=False,
                safety_veto_applied="WORKER_CONTAINMENT_FAILURE",
                verification_timestamp=time.time()
            )

        # ----------------------------------------------------------------------
        # 3. Evidence Validation & Finding Provenance (Section 2)
        # ----------------------------------------------------------------------
        val_findings, inval_findings, prov_ok, prov_err = self.validate_findings_provenance(
            findings, image_path, executed_agents
        )
        if not prov_ok:
            veto_reason = f"SAFETY GATE VETO (Evidence Provenance): {prov_err}"
            rep = self._build_evidence_repertoire(intent, query, val_findings, context_data)
            conf = CrossAgentConflictEvaluator.evaluate(val_findings, context_data, executed_agents)
            dossier = self._generate_escalation_dossier(query, 0.90, rep, conf, 0.90, reason_override=veto_reason)
            h_rec = self._generate_human_review_record(inspection_id, query, RavenDecision.ABSTAIN, veto_reason, rep, conf, failed, val_findings)
            return RavenAssessment(
                decision=RavenDecision.ABSTAIN,
                risk_score=0.90,
                hazard_severity=0.90,
                evidence_completeness=rep.completeness_score,
                uncertainty_index=0.85,
                conflict_entropy=conf.conflict_entropy,
                decision_rationale=veto_reason,
                evidence_repertoire=rep,
                conflict_report=conf,
                validated_findings=val_findings,
                invalid_findings=inval_findings,
                escalation_dossier=dossier,
                human_review_record=h_rec,
                provenance_certified=False,
                safety_veto_applied="INVALID_EVIDENCE_PROVENANCE",
                verification_timestamp=time.time()
            )

        # ----------------------------------------------------------------------
        # 4. Critical vs. Irrelevant Specialist Failure (Section 5)
        # ----------------------------------------------------------------------
        failed_critical: List[str] = []
        failed_irrelevant: List[str] = []

        for f_agent in failed:
            if self.is_specialist_critical(f_agent, intent, query):
                failed_critical.append(f_agent)
            else:
                failed_irrelevant.append(f_agent)

        if failed_critical:
            veto_reason = f"SAFETY GATE VETO (Specialist Failure): Critical specialist(s) {failed_critical} failed during execution."
            rep = self._build_evidence_repertoire(intent, query, val_findings, context_data)
            conf = CrossAgentConflictEvaluator.evaluate(val_findings, context_data, executed_agents)
            dossier = self._generate_escalation_dossier(query, 0.90, rep, conf, 0.90, reason_override=veto_reason)
            h_rec = self._generate_human_review_record(inspection_id, query, RavenDecision.ABSTAIN, veto_reason, rep, conf, failed_critical, val_findings)
            return RavenAssessment(
                decision=RavenDecision.ABSTAIN,
                risk_score=0.90,
                hazard_severity=0.90,
                evidence_completeness=rep.completeness_score,
                uncertainty_index=0.90,
                conflict_entropy=conf.conflict_entropy,
                decision_rationale=veto_reason,
                evidence_repertoire=rep,
                conflict_report=conf,
                validated_findings=val_findings,
                invalid_findings=inval_findings,
                failed_critical_specialists=failed_critical,
                failed_irrelevant_specialists=failed_irrelevant,
                escalation_dossier=dossier,
                human_review_record=h_rec,
                provenance_certified=True,
                safety_veto_applied="CRITICAL_SPECIALIST_FAILURE",
                verification_timestamp=time.time()
            )

        # ----------------------------------------------------------------------
        # 5. Domain Hazard & Evidence Repertoire Evaluation
        # ----------------------------------------------------------------------
        hazard_severity = self._determine_hazard_severity(intent, query, val_findings)
        repertoire = self._build_evidence_repertoire(intent, query, val_findings, context_data)
        completeness = repertoire.completeness_score

        # ----------------------------------------------------------------------
        # 6. Budget Exhaustion with Incomplete Evidence Check (Section 5)
        # ----------------------------------------------------------------------
        req_missing = [t for t in repertoire.tokens if t.required and not t.present]
        if budget_exhausted and len(req_missing) > 0:
            veto_reason = f"SAFETY GATE VETO (Budget Exhaustion): Execution budget exhausted with {len(req_missing)} mandatory evidence token(s) unsatisfied."
            conf = CrossAgentConflictEvaluator.evaluate(val_findings, context_data, executed_agents)
            dossier = self._generate_escalation_dossier(query, hazard_severity, repertoire, conf, 0.85, reason_override=veto_reason)
            h_rec = self._generate_human_review_record(inspection_id, query, RavenDecision.ABSTAIN, veto_reason, repertoire, conf, failed, val_findings)
            return RavenAssessment(
                decision=RavenDecision.ABSTAIN,
                risk_score=0.85,
                hazard_severity=hazard_severity,
                evidence_completeness=completeness,
                uncertainty_index=0.80,
                conflict_entropy=conf.conflict_entropy,
                decision_rationale=veto_reason,
                evidence_repertoire=repertoire,
                conflict_report=conf,
                validated_findings=val_findings,
                invalid_findings=inval_findings,
                failed_critical_specialists=failed_critical,
                failed_irrelevant_specialists=failed_irrelevant,
                escalation_dossier=dossier,
                human_review_record=h_rec,
                provenance_certified=True,
                safety_veto_applied="BUDGET_EXHAUSTION_INCOMPLETE_EVIDENCE",
                verification_timestamp=time.time()
            )

        # ----------------------------------------------------------------------
        # 7. Physical Absence Statistical Certification (Section 4)
        # ----------------------------------------------------------------------
        has_advisory_absence = bool(advisory_absence_flags and any(advisory_absence_flags.values()))
        absence_certified = False

        if absence_certificate is not None:
            # Check certificate validity and recompute exact Clopper-Pearson bound
            ckpt_hash = settings.AGENT_MODEL_VERSIONS.get("object", "chk_yolo_v11_baseline_hash")
            cert_valid, cert_msg = absence_certificate.validate_certificate(
                current_time=time.time(),
                checkpoint_hash=ckpt_hash,
                workcell_id=workcell_id,
                image_metrics=img_metrics
            )
            if not cert_valid:
                # Expired, revoked, or statistically invalid certificate
                veto_reason = f"SAFETY GATE VETO (Absence Certification): {cert_msg}"
                conf = CrossAgentConflictEvaluator.evaluate(val_findings, context_data, executed_agents)
                dossier = self._generate_escalation_dossier(query, hazard_severity, repertoire, conf, 0.85, reason_override=veto_reason)
                h_rec = self._generate_human_review_record(inspection_id, query, RavenDecision.ABSTAIN, veto_reason, repertoire, conf, failed, val_findings)
                return RavenAssessment(
                    decision=RavenDecision.ABSTAIN,
                    risk_score=0.85,
                    hazard_severity=hazard_severity,
                    evidence_completeness=completeness,
                    uncertainty_index=0.80,
                    conflict_entropy=conf.conflict_entropy,
                    decision_rationale=veto_reason,
                    evidence_repertoire=repertoire,
                    conflict_report=conf,
                    absence_certificate=absence_certificate,
                    escalation_dossier=dossier,
                    human_review_record=h_rec,
                    provenance_certified=True,
                    safety_veto_applied="INVALID_OR_EXPIRED_ABSENCE_CERTIFICATE",
                    verification_timestamp=time.time()
                )
            else:
                absence_certified = True
                has_advisory_absence = False  # Valid certificate discharges advisory flag

        # ----------------------------------------------------------------------
        # 8. Cross-Agent Contradiction & Conflict Resolution (Section 3)
        # ----------------------------------------------------------------------
        conflict_report = CrossAgentConflictEvaluator.evaluate(val_findings, context_data, executed_agents)
        conflict_entropy = conflict_report.conflict_entropy

        unresolved_critical_conflicts = [
            r for r in conflict_report.conflict_records
            if r.resolution_status == "UNRESOLVED" and r.severity >= 0.50
        ]
        unresolved_moderate_conflicts = [
            r for r in conflict_report.conflict_records
            if r.resolution_status == "UNRESOLVED" and 0.20 <= r.severity < 0.50
        ]

        # ----------------------------------------------------------------------
        # 9. Composite Risk Index Formulation
        # R = S_hazard * (1 - C_ev) + lambda_conflict * H_conflict
        # ----------------------------------------------------------------------
        token_uncertainties = [t.uncertainty for t in repertoire.tokens]
        uncertainty_index = round(
            float(sum(token_uncertainties) / max(1, len(token_uncertainties))),
            3
        ) if token_uncertainties else 0.50

        risk_score = (hazard_severity * (1.0 - completeness)) + (self.lambda_conflict * conflict_entropy)
        risk_score = round(float(min(1.0, max(0.0, risk_score))), 3)

        # ----------------------------------------------------------------------
        # 10. Tri-State Decision Synthesis with Safety Gate Precedence (Section 5)
        # ----------------------------------------------------------------------
        decision: RavenDecision
        rationale: str
        safety_veto_applied: Optional[str] = None
        secondary_agents: List[str] = []

        # Safety Gate A: Unresolved critical conflict forces ABSTAIN
        if unresolved_critical_conflicts:
            decision = RavenDecision.ABSTAIN
            safety_veto_applied = "CRITICAL_CONTRADICTION_UNRESOLVED"
            c_desc = unresolved_critical_conflicts[0].description
            rationale = f"SAFETY GATE VETO: Unresolved critical cross-agent contradiction (severity >= 0.50). {c_desc}"

        # Safety Gate B: Unresolved moderate conflict forces VERIFY
        elif unresolved_moderate_conflicts:
            decision = RavenDecision.VERIFY
            safety_veto_applied = "MODERATE_CONTRADICTION_UNRESOLVED"
            c_desc = unresolved_moderate_conflicts[0].description
            rationale = f"SAFETY GATE: Resolvable contradiction requires secondary verification pass. {c_desc}"
            if "evidence_validation" not in executed_agents:
                secondary_agents.append("evidence_validation")
            if "scene" not in executed_agents:
                secondary_agents.append("scene")

        # Safety Gate C: Unvalidated physical absence forces VERIFY
        elif has_advisory_absence:
            decision = RavenDecision.VERIFY
            safety_veto_applied = "UNVALIDATED_PHYSICAL_ABSENCE"
            rationale = (
                "SAFETY GATE OVERRIDE (Unvalidated Absence): Advisory physical absence asserted without site statistical validation. "
                "Detector non-detection does not certify physical absence; human verification required."
            )
            if "scene" not in executed_agents:
                secondary_agents.append("scene")
            secondary_agents.append("evidence_re-scan")

        # Safety Gate D: Missing critical evidence in high-hazard inspection
        elif hazard_severity >= 0.85 and any(t.source_agent in executed_agents for t in req_missing):
            decision = RavenDecision.ABSTAIN
            safety_veto_applied = "CRITICAL_EVIDENCE_UNAVAILABLE"
            missing_tokens = [t.token_id for t in req_missing if t.source_agent in executed_agents]
            rationale = (
                f"SAFETY GATE VETO (Critical Evidence Unavailable): Mandatory evidence token(s) {missing_tokens} "
                f"remain unsatisfied in high-hazard domain (S={hazard_severity:.2f}) despite specialist execution. "
                f"Safe autonomous clearance cannot be granted."
            )

        # Ordinary Numerical Risk Thresholding (only when no safety veto applied)
        elif risk_score < self.tau_accept and len(req_missing) == 0:
            decision = RavenDecision.ACCEPT
            rationale = (
                f"Operational risk R={risk_score:.3f} < {self.tau_accept:.2f}. "
                f"Evidence completeness is high ({completeness*100:.1f}%), all mandatory tokens are substantiated, "
                f"provenance is verified, and zero unresolved conflicts exist. Certified for reporting."
            )
        elif risk_score < self.tau_escalate:
            decision = RavenDecision.VERIFY
            if "ppe" in intent and "ocr" not in executed_agents:
                secondary_agents.append("ocr")
            if "defect" in intent and "rag" not in executed_agents:
                secondary_agents.append("rag")
            if "scene" not in executed_agents:
                secondary_agents.append("scene")

            rationale = (
                f"Operational risk R={risk_score:.3f} falls in verification band [{self.tau_accept:.2f}, {self.tau_escalate:.2f}). "
                f"Evidence completeness is {completeness*100:.1f}%. Targeted secondary verification recommended."
            )
        else:
            decision = RavenDecision.ABSTAIN
            rationale = (
                f"Operational risk R={risk_score:.3f} >= {self.tau_escalate:.2f}. "
                f"Automated clearance withheld due to high domain hazard ({hazard_severity:.2f}) "
                f"or incomplete visual evidence ({completeness*100:.1f}%). Mandatory human safety audit initiated."
            )

        # ----------------------------------------------------------------------
        # 11. Human Escalation Dossier & Concise Review Record Synthesis (Section 6)
        # ----------------------------------------------------------------------
        dossier: Optional[HumanEscalationDossier] = None
        human_review_rec: Optional[HumanReviewRecord] = None

        if decision in (RavenDecision.VERIFY, RavenDecision.ABSTAIN) or any(f.get("human_review_required") for f in val_findings):
            dossier = self._generate_escalation_dossier(
                query=query,
                hazard_severity=hazard_severity,
                repertoire=repertoire,
                conflict_report=conflict_report,
                risk_score=risk_score,
                reason_override=rationale
            )
            human_review_rec = self._generate_human_review_record(
                inspection_id=inspection_id,
                query=query,
                decision=decision,
                reason=rationale,
                repertoire=repertoire,
                conflict_report=conflict_report,
                failed_specialists=failed_critical,
                findings=val_findings
            )

        return RavenAssessment(
            decision=decision,
            risk_score=risk_score,
            hazard_severity=hazard_severity,
            evidence_completeness=completeness,
            uncertainty_index=uncertainty_index,
            conflict_entropy=conflict_entropy,
            decision_rationale=rationale,
            evidence_repertoire=repertoire,
            conflict_report=conflict_report,
            secondary_agents_suggested=secondary_agents,
            escalation_dossier=dossier,
            human_review_record=human_review_rec,
            validated_findings=val_findings,
            invalid_findings=inval_findings,
            failed_critical_specialists=failed_critical,
            failed_irrelevant_specialists=failed_irrelevant,
            conflict_records=conflict_report.conflict_records,
            absence_certificate=absence_certificate,
            provenance_certified=True,
            safety_veto_applied=safety_veto_applied,
            verification_timestamp=time.time()
        )
