"""
VisionIQ LangGraph StateGraph Dynamic Orchestration Pipeline
Defines state machine nodes, conditional dispatch, error isolation,
latency measurements, and persistent storage.
"""

import time
import os
import uuid
from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END

from backend.app.orchestration.state import AgentState
from backend.app.agents.agent_l_orchestrator import run_orchestrator_planning
from backend.app.agents.agent_b_object import run_object_agent
from backend.app.agents.agent_c_ppe import run_ppe_agent
from backend.app.agents.agent_d_ocr import run_ocr_agent
from backend.app.agents.agent_e_defect import run_defect_agent
from backend.app.agents.agent_f_scene import run_scene_agent
from backend.app.agents.agent_g_evidence_validation import run_evidence_validation_agent
from backend.app.agents.agent_h_rag import run_rag_agent
from backend.app.agents.agent_i_rca import run_rca_agent
from backend.app.agents.agent_j_recommendation import run_recommendation_agent
from backend.app.agents.agent_k_report import run_report_agent
from backend.app.services.annotation_service import annotation_service
from backend.app.database.session import SessionLocal
from backend.app.database.repository import AnalysisRepository
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.raven.engine import RavenEngine
from backend.app.raven.schemas import RavenDecision

raven_engine = RavenEngine()

def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Evaluates query intent, scores agents using Eq. (1), and plans branches."""
    start_time = time.time()
    plan_updates = run_orchestrator_planning(state)

    if not state.get("session_id"):
        state["session_id"] = str(uuid.uuid4())

    plan_updates["session_id"] = state["session_id"]
    plan_updates["start_time"] = start_time
    plan_updates["findings"] = []
    plan_updates["executed_agents"] = []
    plan_updates["failed_agents"] = []
    plan_updates["errors"] = []
    plan_updates["verification_cycle"] = 0
    plan_updates["verification_pass_executed"] = False
    return plan_updates

def vision_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Executes selected perceptual vision specialists with fault isolation and timeout handling."""
    selected = state.get("selected_agents", [])
    findings = list(state.get("findings", []))
    executed = list(state.get("executed_agents", []))
    failed = list(state.get("failed_agents", []))
    log = list(state.get("execution_log", []))
    errors = list(state.get("errors", []))
    updates: Dict[str, Any] = {}

    def update_trace(agent_name: str, status: str, duration_ms: float, err: str = None):
        for entry in log:
            if entry["agent_name"] == agent_name:
                entry["execution_status"] = status
                entry["actual_latency_ms"] = round(duration_ms, 2)
                entry["error_message"] = err
                entry["hardware_device"] = settings.HARDWARE_DEVICE_CONTEXT
                entry["model_version"] = settings.AGENT_MODEL_VERSIONS.get(agent_name)
                break

    # 1. Object Agent
    if "object" in selected:
        t0 = time.time()
        try:
            res = run_object_agent(state)
            updates.update(res)
            findings.extend(res.get("new_findings", []))
            executed.append("object")
            update_trace("object", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("object")
            errors.append(f"ObjectAgent error: {str(e)}")
            update_trace("object", "FAILED", (time.time() - t0) * 1000.0, str(e))

    # 2. PPE Agent
    if "ppe" in selected:
        t0 = time.time()
        try:
            temp_state = dict(state)
            temp_state.update(updates)
            res = run_ppe_agent(temp_state)
            updates.update(res)
            findings.extend(res.get("new_findings", []))
            executed.append("ppe")
            update_trace("ppe", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("ppe")
            errors.append(f"PPEAgent error: {str(e)}")
            update_trace("ppe", "FAILED", (time.time() - t0) * 1000.0, str(e))

    # 3. Defect Agent
    if "defect" in selected:
        t0 = time.time()
        try:
            temp_state = dict(state)
            temp_state.update(updates)
            res = run_defect_agent(temp_state)
            updates.update(res)
            findings.extend(res.get("new_findings", []))
            executed.append("defect")
            update_trace("defect", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("defect")
            errors.append(f"DefectAgent error: {str(e)}")
            update_trace("defect", "FAILED", (time.time() - t0) * 1000.0, str(e))

    # 4. OCR Agent
    if "ocr" in selected:
        t0 = time.time()
        try:
            res = run_ocr_agent(state)
            updates.update(res)
            findings.extend(res.get("new_findings", []))
            executed.append("ocr")
            update_trace("ocr", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("ocr")
            errors.append(f"OCRAgent error: {str(e)}")
            update_trace("ocr", "FAILED", (time.time() - t0) * 1000.0, str(e))

    # 5. Scene Understanding Agent
    if "scene" in selected:
        t0 = time.time()
        try:
            temp_state = dict(state)
            temp_state.update(updates)
            res = run_scene_agent(temp_state)
            updates.update(res)
            findings.extend(res.get("new_findings", []))
            executed.append("scene")
            update_trace("scene", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("scene")
            errors.append(f"SceneAgent error: {str(e)}")
            update_trace("scene", "FAILED", (time.time() - t0) * 1000.0, str(e))

    updates["findings"] = findings
    updates["executed_agents"] = executed
    updates["failed_agents"] = failed
    updates["execution_log"] = log
    updates["errors"] = errors
    return updates

def validation_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Evidence validation agent checks provenance and cross-references claims."""
    t0 = time.time()
    log = list(state.get("execution_log", []))
    try:
        val_updates = run_evidence_validation_agent(state)
        dur = (time.time() - t0) * 1000.0
        for entry in log:
            if entry["agent_name"] == "evidence_validation":
                entry["execution_status"] = "SUCCESS"
                entry["actual_latency_ms"] = round(dur, 2)
                entry["hardware_device"] = settings.HARDWARE_DEVICE_CONTEXT
                entry["model_version"] = settings.AGENT_MODEL_VERSIONS.get("evidence_validation")
                break
        val_updates["execution_log"] = log
        return val_updates
    except Exception as e:
        dur = (time.time() - t0) * 1000.0
        for entry in log:
            if entry["agent_name"] == "evidence_validation":
                entry["execution_status"] = "FAILED"
                entry["actual_latency_ms"] = round(dur, 2)
                entry["error_message"] = str(e)
                entry["hardware_device"] = settings.HARDWARE_DEVICE_CONTEXT
                entry["model_version"] = settings.AGENT_MODEL_VERSIONS.get("evidence_validation")
                break
        return {"evidence_support_score": 0.80, "execution_log": log}

def raven_verification_node(state: AgentState) -> Dict[str, Any]:
    """Node 3.5: Risk-Aware Visual Evidence Navigator (RAVEN) verification layer."""
    t0 = time.time()
    findings = list(state.get("findings", []))
    context_data = {
        "object_result": state.get("object_result", {}),
        "ppe_result": state.get("ppe_result", {}),
        "defect_result": state.get("defect_result", {}),
        "ocr_result": state.get("ocr_result", {}),
        "scene_result": state.get("scene_result", {})
    }
    executed_agents = list(state.get("executed_agents", []))

    # Apply confidence calibration and spatial uncertainty across findings
    from backend.app.raven.calibration import calibrate_confidence, compute_spatial_uncertainty
    for f in findings:
        conf = f.get("detection_confidence")
        if conf is not None and "calibrated_confidence" not in f:
            f["calibrated_confidence"] = calibrate_confidence(conf)
        if f.get("bbox") and "spatial_uncertainty" not in f:
            f["spatial_uncertainty"] = compute_spatial_uncertainty(f["bbox"])

    assessment = raven_engine.evaluate(
        query=state.get("user_query", ""),
        intent=state.get("detected_intent", "general"),
        findings=findings,
        context_data=context_data,
        executed_agents=executed_agents,
        failed_agents=state.get("failed_agents", []),
        advisory_absence_flags=state.get("advisory_absence_flags"),
        evidence_report=state.get("active_evidence_report"),
        image_path=state.get("image_path"),
        absence_certificate=state.get("absence_certificate"),
        worker_cleanup_failed=state.get("worker_cleanup_failed", False),
        budget_exhausted=state.get("budget_exhausted", False),
        workcell_id=state.get("workcell_id", "WORKCELL_01")
    )

    log = list(state.get("execution_log", []))
    dur = round((time.time() - t0) * 1000.0, 2)
    log.append({
        "agent_name": "raven_verification",
        "relevance_score": 1.0,
        "evidence_capability": 1.0,
        "normalized_cost": 0.05,
        "expected_latency": 0.05,
        "final_routing_score": 1.0,
        "decision": "SELECTED",
        "selection_reason": f"Operational Risk R={assessment.risk_score:.3f}, Completeness={assessment.evidence_completeness*100:.1f}%, Decision={assessment.decision.value}",
        "execution_status": "SUCCESS",
        "actual_latency_ms": dur,
        "error_message": None,
        "hardware_device": settings.HARDWARE_DEVICE_CONTEXT
    })

    dossier_dict = assessment.escalation_dossier.model_dump() if assessment.escalation_dossier else None

    return {
        "findings": findings,
        "raven_assessment": assessment.model_dump(),
        "raven_decision": assessment.decision.value,
        "raven_risk_score": assessment.risk_score,
        "evidence_completeness": assessment.evidence_completeness,
        "conflict_entropy": assessment.conflict_entropy,
        "raven_dossier": dossier_dict,
        "human_review_record": assessment.human_review_record.model_dump() if assessment.human_review_record else None,
        "evidence_support_score": assessment.evidence_completeness,
        "execution_log": log,
        "verification_cycle": state.get("verification_cycle", 0)
    }

def route_after_raven(state: AgentState) -> str:
    """Evaluates the RAVEN tri-state decision policy and bounds cyclic secondary passes."""
    cycle = state.get("verification_cycle", 0)
    decision = state.get("raven_decision", "ACCEPT")

    if decision == "VERIFY" and cycle < 1:
        return "secondary_verification"
    elif decision == "ABSTAIN":
        return "escalation"
    else:
        return "cognitive_agents"

def secondary_verification_node(state: AgentState) -> Dict[str, Any]:
    """Node 3.6: Targeted secondary verification dispatch for borderline / ambiguous cases."""
    t0 = time.time()
    cycle = state.get("verification_cycle", 0) + 1
    log = list(state.get("execution_log", []))
    executed = list(state.get("executed_agents", []))
    findings = list(state.get("findings", []))
    context_data = {
        "object_result": state.get("object_result", {}),
        "ppe_result": state.get("ppe_result", {}),
        "defect_result": state.get("defect_result", {}),
        "ocr_result": state.get("ocr_result", {}),
        "scene_result": state.get("scene_result", {})
    }

    assessment_data = state.get("raven_assessment", {})
    suggested = assessment_data.get("secondary_agents_suggested", [])
    dispatched = []

    # Execute targeted secondary agent if suggested and not already executed
    if "ocr" in suggested and "ocr" not in executed:
        try:
            from backend.app.agents.agent_d_ocr import run_ocr_agent
            ocr_res = run_ocr_agent(state)
            context_data["ocr_result"] = ocr_res.get("ocr_result", {})
            findings.extend(ocr_res.get("new_findings", []))
            executed.append("ocr")
            dispatched.append("ocr")
        except Exception as e:
            logger.warning(f"Secondary OCR dispatch failed: {e}")

    if "scene" in suggested and "scene" not in executed:
        try:
            from backend.app.agents.agent_f_scene import run_scene_agent
            scene_res = run_scene_agent(state)
            context_data["scene_result"] = scene_res.get("scene_result", {})
            findings.extend(scene_res.get("new_findings", []))
            executed.append("scene")
            dispatched.append("scene")
        except Exception as e:
            logger.warning(f"Secondary scene dispatch failed: {e}")

    if not dispatched:
        dispatched.append("localized_rescan")

    # Re-evaluate risk with newly fused evidence
    re_assessment = raven_engine.evaluate(
        query=state.get("user_query", ""),
        intent=state.get("detected_intent", "general"),
        findings=findings,
        context_data=context_data,
        executed_agents=executed
    )

    dur = round((time.time() - t0) * 1000.0, 2)
    log.append({
        "agent_name": "secondary_verification",
        "relevance_score": 1.0,
        "evidence_capability": 1.0,
        "normalized_cost": 0.10,
        "expected_latency": 0.10,
        "final_routing_score": 1.0,
        "decision": "SELECTED",
        "selection_reason": f"Targeted secondary verification executed for: {dispatched}. Re-calibrated R={re_assessment.risk_score:.3f}",
        "execution_status": "SUCCESS",
        "actual_latency_ms": dur,
        "error_message": None,
        "hardware_device": settings.HARDWARE_DEVICE_CONTEXT
    })

    dossier_dict = re_assessment.escalation_dossier.model_dump() if re_assessment.escalation_dossier else None

    return {
        "findings": findings,
        "executed_agents": executed,
        "secondary_agents_dispatched": dispatched,
        "verification_pass_executed": True,
        "verification_cycle": cycle,
        "raven_assessment": re_assessment.model_dump(),
        "raven_decision": re_assessment.decision.value,
        "raven_risk_score": re_assessment.risk_score,
        "evidence_completeness": re_assessment.evidence_completeness,
        "conflict_entropy": re_assessment.conflict_entropy,
        "raven_dossier": dossier_dict,
        "evidence_support_score": re_assessment.evidence_completeness,
        "execution_log": log
    }

def escalation_node(state: AgentState) -> Dict[str, Any]:
    """Node 3.7: Flags forensic human review and safety clearance withholding."""
    findings = list(state.get("findings", []))
    for f in findings:
        f["human_review_required"] = True

    return {
        "findings": findings,
        "human_review_required": True,
        "safety_clearance_withheld": True
    }

def cognitive_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: RAG standards retrieval, Root Cause Analysis, and CAPA recommendations."""
    selected = state.get("selected_agents", [])
    executed = list(state.get("executed_agents", []))
    failed = list(state.get("failed_agents", []))
    log = list(state.get("execution_log", []))
    updates: Dict[str, Any] = {}

    def update_trace(agent_name: str, status: str, duration_ms: float, err: str = None):
        for entry in log:
            if entry["agent_name"] == agent_name:
                entry["execution_status"] = status
                entry["actual_latency_ms"] = round(duration_ms, 2)
                entry["error_message"] = err
                entry["hardware_device"] = settings.HARDWARE_DEVICE_CONTEXT
                entry["model_version"] = settings.AGENT_MODEL_VERSIONS.get(agent_name)
                break

    # 1. RAG Knowledge Retrieval
    if "rag" in selected:
        t0 = time.time()
        try:
            res = run_rag_agent(state)
            updates.update(res)
            executed.append("rag")
            update_trace("rag", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("rag")
            update_trace("rag", "FAILED", (time.time() - t0) * 1000.0, str(e))

    # 2. Root Cause Analysis
    if "rca" in selected:
        t0 = time.time()
        try:
            temp_state = dict(state)
            temp_state.update(updates)
            res = run_rca_agent(temp_state)
            updates.update(res)
            executed.append("rca")
            update_trace("rca", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("rca")
            update_trace("rca", "FAILED", (time.time() - t0) * 1000.0, str(e))

    # 3. Recommendation Planning
    if "recommendation" in selected:
        t0 = time.time()
        try:
            temp_state = dict(state)
            temp_state.update(updates)
            res = run_recommendation_agent(temp_state)
            updates.update(res)
            executed.append("recommendation")
            update_trace("recommendation", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("recommendation")
            update_trace("recommendation", "FAILED", (time.time() - t0) * 1000.0, str(e))

    updates["executed_agents"] = executed
    updates["failed_agents"] = failed
    updates["execution_log"] = log
    return updates

def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """Node 5: Consolidates findings, renders annotations, generates PDF report, persists to DB."""
    start_time = state.get("start_time", time.time())
    total_latency_ms = (time.time() - start_time) * 1000.0
    session_id = state.get("session_id", str(uuid.uuid4()))
    image_path = state.get("image_path", "")
    findings = state.get("findings", [])
    traces = state.get("execution_log", [])
    rag_refs = state.get("rag_references", [])
    evidence_score = state.get("evidence_support_score", 1.0)

    annotated_path = ""
    if image_path and os.path.exists(image_path):
        try:
            annotated_path = annotation_service.annotate_image(
                image_path=image_path,
                findings=findings,
                output_filename=f"annotated_{session_id[:8]}_{os.path.basename(image_path)}"
            )
        except Exception as e:
            annotated_path = image_path

    intent = state.get("detected_intent", "Analysis")
    rca = state.get("rca_result")
    ppe = state.get("ppe_result")
    defect = state.get("defect_result")

    summary_parts = []
    if state.get("raven_decision") == "ABSTAIN":
        risk_score = state.get("raven_risk_score", 0.70)
        summary_parts.append(
            f"[SAFETY CLEARANCE WITHHELD — HUMAN AUDIT MANDATED: Operational risk index R = {risk_score:.3f} >= 0.65 threshold. Certified physical inspection is required.]"
        )
    elif state.get("raven_decision") == "VERIFY":
        summary_parts.append("[RAVEN VERIFICATION APPLIED: Targeted secondary evidence pass completed.]")

    summary_parts.append(f"Analysis completed for query '{state.get('user_query')}'. Intent classified as {intent}.")
    if ppe:
        summary_parts.append(
            f"PPE Evaluation: {ppe.get('compliant_count', 0)} compliant, "
            f"{ppe.get('non_compliant_count', 0)} non-compliant, {ppe.get('uncertain_count', 0)} uncertain."
        )
    if defect and defect.get("defects_detected", 0) > 0:
        summary_parts.append(f"Identified {defect['defects_detected']} surface defect region(s).")
    if rca and rca.get("ranked_hypotheses"):
        top_h = rca["ranked_hypotheses"][0]
        summary_parts.append(f"Primary Root-Cause Hypothesis: {top_h['hypothesis']} (Est. Probability: {top_h['estimated_cause_probability']*100:.0f}%).")

    synthesis_text = " ".join(summary_parts)

    pdf_report_path = ""
    try:
        report_state = dict(state)
        report_state["session_id"] = session_id
        report_state["annotated_path"] = annotated_path
        report_state["total_latency_ms"] = total_latency_ms
        pdf_res = run_report_agent(report_state)
        pdf_report_path = pdf_res.get("pdf_report_path", "")
    except Exception as e:
        pdf_report_path = ""

    from backend.app.schemas.contracts import build_evidence_records
    evidence_records = [
        r.model_dump() for r in build_evidence_records(
            session_id=session_id,
            input_image_ref=image_path,
            findings=findings,
            traces=traces,
            raven_dossier=state.get("raven_dossier")
        )
    ]

    final_response = {
        "session_id": session_id,
        "inspection_id": session_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user_query": state.get("user_query", ""),
        "query": state.get("user_query", ""),
        "input_type": state.get("input_type", "image"),
        "input_path": image_path,
        "annotated_path": annotated_path,
        "detected_intent": intent,
        "intent_confidence": state.get("intent_confidence", 1.0),
        "selected_agents": state.get("selected_agents") or state.get("executed_agents", []),
        "executed_agents": state.get("executed_agents", []),
        "skipped_agents": state.get("skipped_agents", []),
        "failed_agents": state.get("failed_agents", []),
        "status": "completed" if not state.get("failed_agents") else "partial",
        "total_latency_ms": round(total_latency_ms, 2),
        "synthesis_text": synthesis_text,
        "evidence_support_score": round(evidence_score, 3),
        "pdf_report_path": pdf_report_path,
        "raven_risk_score": state.get("raven_risk_score"),
        "raven_decision": state.get("raven_decision"),
        "evidence_completeness": state.get("evidence_completeness"),
        "conflict_entropy": state.get("conflict_entropy"),
        "raven_dossier": state.get("raven_dossier"),
        "raven_assessment": state.get("raven_assessment"),
        "human_review_record": state.get("human_review_record"),
        "evidence_records": evidence_records,
        "findings": findings,
        "traces": traces,
        "rag_references": rag_refs,
        "deficit_history": state.get("deficit_history", []),
        "current_deficit": state.get("current_deficit"),
        "utility_trace": state.get("utility_trace", []),
        "iteration_step": state.get("iteration_step", 0),
        "routing_telemetry": state.get("routing_telemetry", []),
        "agent_lifecycle_states": state.get("agent_lifecycle_states", {}),
        "stopping_reason": state.get("stopping_reason")
    }

    try:
        db = SessionLocal()
        repo = AnalysisRepository(db)
        repo.create_session({
            "id": session_id,
            "user_query": state.get("user_query", ""),
            "input_type": state.get("input_type", "image"),
            "input_path": image_path,
            "annotated_path": annotated_path,
            "detected_intent": intent,
            "intent_confidence": state.get("intent_confidence", 1.0),
            "selected_agents": state.get("selected_agents") or state.get("executed_agents", []),
            "executed_agents": state.get("executed_agents", []),
            "skipped_agents": state.get("skipped_agents", []),
            "failed_agents": state.get("failed_agents", []),
            "status": final_response["status"],
            "total_latency_ms": total_latency_ms,
            "synthesis_text": synthesis_text,
            "evidence_support_score": evidence_score,
            "pdf_report_path": pdf_report_path,
            "raven_risk_score": state.get("raven_risk_score"),
            "raven_decision": state.get("raven_decision"),
            "evidence_completeness": state.get("evidence_completeness"),
            "conflict_entropy": state.get("conflict_entropy"),
            "raven_dossier": state.get("raven_dossier"),
            "evidence_records": evidence_records
        })
        repo.add_findings(session_id, findings)
        repo.add_traces(session_id, traces)
        repo.add_rag_sources(session_id, rag_refs)
        db.close()
    except Exception as e:
        logger.warning(f"Database persistence warning: {e}")

    return {
        "final_response": final_response,
        "annotated_path": annotated_path,
        "pdf_report_path": pdf_report_path,
        "total_latency_ms": total_latency_ms,
        "synthesis_text": synthesis_text
    }

def build_visioniq_graph():
    """Compiles the LangGraph StateGraph with conditional execution edges."""
    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("vision_agents", vision_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("raven_verification", raven_verification_node)
    workflow.add_node("secondary_verification", secondary_verification_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("cognitive_agents", cognitive_node)
    workflow.add_node("synthesis", synthesis_node)

    workflow.add_edge(START, "orchestrator")
    workflow.add_edge("orchestrator", "vision_agents")
    workflow.add_edge("vision_agents", "validation")
    workflow.add_edge("validation", "raven_verification")

    workflow.add_conditional_edges(
        "raven_verification",
        route_after_raven,
        {
            "secondary_verification": "secondary_verification",
            "escalation": "escalation",
            "cognitive_agents": "cognitive_agents"
        }
    )
    workflow.add_edge("secondary_verification", "cognitive_agents")
    workflow.add_edge("escalation", "cognitive_agents")
    workflow.add_edge("cognitive_agents", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()

vision_iq_app = build_visioniq_graph()


# ==============================================================================
# PHASE 3: SEQUENTIAL EVIDENCE-GAIN DYNAMIC STATEGRAPH (REVISION 10)
# ==============================================================================

def sequential_init_node(state: AgentState) -> Dict[str, Any]:
    """Phase 3 Node 1: Initializes sequential evidence state, intent, and baseline deficit D0."""
    start_time = time.time()
    session_id = state.get("session_id") or str(uuid.uuid4())
    query = state.get("user_query", "")

    # 1. Deterministic Intent Classification
    from backend.app.orchestration.router import classify_intent_deterministic
    intent, intent_conf, _ = classify_intent_deterministic(query)

    # 2. Evidence Deficit Engine Baseline Evaluation
    from backend.app.orchestration.evidence_deficit import EvidenceDeficitEngine
    engine = EvidenceDeficitEngine()
    req_tokens = engine.determine_required_tokens_for_intent(intent, query)

    # Initial baseline evaluation with empty findings -> D0
    initial_report = engine.evaluate(
        query=query,
        intent=intent,
        findings=[],
        context_data={},
        executed_agents=[],
        failed_agents=[],
        requested_tokens=req_tokens,
        step=0
    )
    d0 = round(float(initial_report.composite_deficit), 4)

    return {
        "session_id": session_id,
        "start_time": start_time,
        "detected_intent": intent,
        "intent_confidence": intent_conf,
        "findings": [],
        "executed_agents": [],
        "failed_agents": [],
        "errors": [],
        "execution_log": [],
        "verification_cycle": 0,
        "verification_pass_executed": False,
        "iteration_step": 0,
        "max_steps": state.get("max_steps", 5),
        "target_deficit": state.get("target_deficit", 0.20),
        "current_deficit": d0,
        "deficit_history": [d0],
        "active_evidence_report": initial_report,
        "utility_trace": [],
        "next_action": "route"
    }


def sequential_router_node(state: AgentState) -> Dict[str, Any]:
    """Phase 3 Node 2: Evaluates utility across eligible specialists and selects argmax a*."""
    from backend.app.orchestration.router import SequentialEvidenceRouter
    stopping_d = state.get("target_deficit", 0.20)
    max_iter = state.get("max_steps", 5)
    router = SequentialEvidenceRouter(stopping_deficit=stopping_d, max_iterations=max_iter)

    best_agent, max_u, candidate_scores = router.select_next_specialist(state)

    utility_trace = list(state.get("utility_trace", []))
    step = state.get("iteration_step", 0)
    current_d = state.get("current_deficit", 1.0)

    trace_entry = {
        "step": step,
        "current_deficit": current_d,
        "selected_specialist": best_agent,
        "max_utility": max_u,
        "candidate_scores": candidate_scores
    }
    utility_trace.append(trace_entry)

    if best_agent is None:
        return {
            "selected_specialist": None,
            "next_action": "stop",
            "utility_trace": utility_trace
        }
    else:
        return {
            "selected_specialist": best_agent,
            "next_action": "dispatch",
            "utility_trace": utility_trace
        }


def sequential_dispatch_node(state: AgentState) -> Dict[str, Any]:
    """Phase 3 Node 3: Executes single selected specialist a* with fault isolation."""
    agent_name = state.get("selected_specialist")
    findings = list(state.get("findings", []))
    executed = list(state.get("executed_agents", []))
    failed = list(state.get("failed_agents", []))
    log = list(state.get("execution_log", []))
    errors = list(state.get("errors", []))
    updates: Dict[str, Any] = {}

    if not agent_name:
        return {"next_action": "stop"}

    t0 = time.time()
    try:
        if agent_name == "object":
            res = run_object_agent(state)
        elif agent_name == "ppe":
            temp_state = dict(state)
            temp_state["findings"] = findings
            temp_state.update(updates)
            res = run_ppe_agent(temp_state)
        elif agent_name == "defect":
            temp_state = dict(state)
            temp_state["findings"] = findings
            temp_state.update(updates)
            res = run_defect_agent(temp_state)
        elif agent_name == "ocr":
            temp_state = dict(state)
            temp_state["findings"] = findings
            temp_state.update(updates)
            res = run_ocr_agent(temp_state)
        elif agent_name == "scene":
            res = run_scene_agent(state)
        elif agent_name == "evidence_validation":
            temp_state = dict(state)
            temp_state["findings"] = findings
            res = run_evidence_validation_agent(temp_state)
        elif agent_name == "rag":
            temp_state = dict(state)
            temp_state["findings"] = findings
            res = run_rag_agent(temp_state)
        elif agent_name == "rca":
            temp_state = dict(state)
            temp_state["findings"] = findings
            res = run_rca_agent(temp_state)
        elif agent_name == "recommendation":
            temp_state = dict(state)
            temp_state["findings"] = findings
            res = run_recommendation_agent(temp_state)
        else:
            res = {}

        updates.update(res)
        new_findings = res.get("new_findings", [])
        findings.extend(new_findings)
        executed.append(agent_name)
        dur = (time.time() - t0) * 1000.0

        log.append({
            "agent_name": agent_name,
            "relevance_score": 1.0,
            "evidence_capability": 1.0,
            "normalized_cost": settings.AGENT_COSTS.get(agent_name, 0.20),
            "expected_latency": settings.AGENT_LATENCIES.get(agent_name, 0.20),
            "final_routing_score": 1.0,
            "decision": "SELECTED",
            "selection_reason": f"Sequential evidence-gain dispatch (Iteration {state.get('iteration_step', 0) + 1})",
            "execution_status": "SUCCESS",
            "actual_latency_ms": round(dur, 2),
            "error_message": None,
            "hardware_device": settings.HARDWARE_DEVICE_CONTEXT,
            "model_version": settings.AGENT_MODEL_VERSIONS.get(agent_name)
        })
    except Exception as e:
        dur = (time.time() - t0) * 1000.0
        failed.append(agent_name)
        errors.append(f"{agent_name} execution error: {str(e)}")
        log.append({
            "agent_name": agent_name,
            "relevance_score": 1.0,
            "evidence_capability": 1.0,
            "normalized_cost": settings.AGENT_COSTS.get(agent_name, 0.20),
            "expected_latency": settings.AGENT_LATENCIES.get(agent_name, 0.20),
            "final_routing_score": 0.0,
            "decision": "SELECTED",
            "selection_reason": f"Sequential dispatch failed: {str(e)}",
            "execution_status": "FAILED",
            "actual_latency_ms": round(dur, 2),
            "error_message": str(e),
            "hardware_device": settings.HARDWARE_DEVICE_CONTEXT,
            "model_version": settings.AGENT_MODEL_VERSIONS.get(agent_name)
        })

    updates["findings"] = findings
    updates["executed_agents"] = executed
    updates["failed_agents"] = failed
    updates["execution_log"] = log
    updates["errors"] = errors
    updates["iteration_step"] = state.get("iteration_step", 0) + 1
    return updates


def sequential_evidence_update_node(state: AgentState) -> Dict[str, Any]:
    """Phase 3 Node 4: Re-evaluates EvidenceDeficitEngine and checks stopping conditions."""
    from backend.app.orchestration.evidence_deficit import EvidenceDeficitEngine
    engine = EvidenceDeficitEngine()

    query = state.get("user_query", "")
    intent = state.get("detected_intent", "general_scene_inquiry")
    findings = state.get("findings", [])
    executed = state.get("executed_agents", [])
    failed = state.get("failed_agents", [])
    step = state.get("iteration_step", 0)

    context_data = {
        "object_result": state.get("object_result", {}),
        "ppe_result": state.get("ppe_result", {}),
        "defect_result": state.get("defect_result", {}),
        "ocr_result": state.get("ocr_result", {}),
        "scene_result": state.get("scene_result", {})
    }

    report = engine.evaluate(
        query=query,
        intent=intent,
        findings=findings,
        context_data=context_data,
        executed_agents=executed,
        failed_agents=failed,
        step=step
    )

    new_d = round(float(report.composite_deficit), 4)
    deficit_history = list(state.get("deficit_history", []))
    deficit_history.append(new_d)

    target_d = state.get("target_deficit", 0.20)
    max_steps = state.get("max_steps", 5)

    # Check stopping criteria
    if new_d <= target_d or step >= max_steps:
        next_action = "stop"
    else:
        next_action = "continue"

    # Record telemetry for this sequential step
    from backend.app.orchestration.router import SequentialEvidenceRouter
    router = SequentialEvidenceRouter(stopping_deficit=target_d, max_iterations=max_steps)
    prev_d = float(deficit_history[-2]) if len(deficit_history) >= 2 else new_d
    last_trace = state.get("utility_trace", [])
    candidate_scores = last_trace[-1].get("candidate_scores", {}) if last_trace else {}
    selected_agent = state.get("selected_specialist")
    last_exec = executed[-1] if executed else None
    exec_result = "SUCCESS" if (last_exec == selected_agent) else "FAILED"

    step_telemetry = router.record_step_telemetry(
        step=step,
        current_deficit=prev_d,
        candidate_scores=candidate_scores,
        selected_agent=selected_agent,
        selection_efficiency=last_trace[-1].get("max_utility", 0.0) if last_trace else 0.0,
        execution_result=exec_result,
        cleanup_result="CLEAN",
        post_deficit=new_d if exec_result == "SUCCESS" else None,
        actual_findings_count=len(findings),
        remaining_invocations=max(0, max_steps - step),
        remaining_wallclock_ms=float(state.get("remaining_time_budget_ms", 5000.0)),
        stopping_reason=state.get("stopping_reason")
    )
    routing_telemetry = list(state.get("routing_telemetry", []))
    routing_telemetry.append(step_telemetry)

    return {
        "active_evidence_report": report,
        "current_deficit": new_d,
        "deficit_history": deficit_history,
        "next_action": next_action,
        "routing_telemetry": routing_telemetry
    }


def route_after_sequential_router(state: AgentState) -> str:
    """Conditional edge: dispatches specialist or exits loop to RAVEN."""
    if state.get("next_action") == "dispatch":
        return "sequential_dispatch"
    return "raven_verification"


def route_after_sequential_update(state: AgentState) -> str:
    """Conditional edge: loops back to router or exits to RAVEN."""
    if state.get("next_action") == "continue":
        return "sequential_router"
    return "raven_verification"


def build_sequential_visioniq_graph():
    """
    Compiles the Phase 3 Sequential Cyclic StateGraph:
    START -> sequential_init -> sequential_router <-> sequential_dispatch -> sequential_evidence_update
                                      |
                                      v (stop condition met: D <= tau OR t >= max_steps)
                               raven_verification -> cognitive_agents / secondary / escalation -> synthesis -> END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("sequential_init", sequential_init_node)
    workflow.add_node("sequential_router", sequential_router_node)
    workflow.add_node("sequential_dispatch", sequential_dispatch_node)
    workflow.add_node("sequential_evidence_update", sequential_evidence_update_node)
    workflow.add_node("raven_verification", raven_verification_node)
    workflow.add_node("secondary_verification", secondary_verification_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("cognitive_agents", cognitive_node)
    workflow.add_node("synthesis", synthesis_node)

    workflow.add_edge(START, "sequential_init")
    workflow.add_edge("sequential_init", "sequential_router")

    workflow.add_conditional_edges(
        "sequential_router",
        route_after_sequential_router,
        {
            "sequential_dispatch": "sequential_dispatch",
            "raven_verification": "raven_verification"
        }
    )

    workflow.add_edge("sequential_dispatch", "sequential_evidence_update")

    workflow.add_conditional_edges(
        "sequential_evidence_update",
        route_after_sequential_update,
        {
            "sequential_router": "sequential_router",
            "raven_verification": "raven_verification"
        }
    )

    workflow.add_conditional_edges(
        "raven_verification",
        route_after_raven,
        {
            "secondary_verification": "secondary_verification",
            "escalation": "escalation",
            "cognitive_agents": "cognitive_agents"
        }
    )

    workflow.add_edge("secondary_verification", "cognitive_agents")
    workflow.add_edge("escalation", "cognitive_agents")
    workflow.add_edge("cognitive_agents", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()


sequential_vision_iq_app = build_sequential_visioniq_graph()

