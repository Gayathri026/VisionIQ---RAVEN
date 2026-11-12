from enum import Enum
import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

class AgentStatus(str, Enum):
    SELECTED = "SELECTED"
    EXECUTED = "EXECUTED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"

class EvidenceRecord(BaseModel):
    model_config = {"protected_namespaces": ()}

    evidence_id: str = Field(..., description="Unique evidence identifier, e.g. EV-OBJ-01")
    inspection_id: str = Field(..., description="Parent inspection or session ID")
    query_id: Optional[str] = Field(None, description="Associated query ID")
    agent_name: str = Field(..., description="Agent producing the evidence (object, ppe, defect, ocr, scene)")
    agent_status: str = Field(..., description="Execution status: EXECUTED, PARTIAL, FAILED, SKIPPED")
    model_id: Optional[str] = Field(None, description="Underlying model identifier (e.g. YOLO11n, NEU-DefectNet)")
    model_version: Optional[str] = Field(None, description="Model checkpoint version (e.g. v1.0.0)")
    input_image_ref: Optional[str] = Field(None, description="Path or URL of input image")
    predicted_class: Optional[str] = Field(None, description="Predicted class label or defect categorization")
    object_id: Optional[str] = Field(None, description="Persistent tracking or person identifier")
    bbox: Optional[List[float]] = Field(None, description="Bounding box [x1, y1, x2, y2]")
    mask_ref: Optional[str] = Field(None, description="Segmentation mask reference or contour polygon")
    raw_confidence: Optional[float] = Field(None, description="Perception model raw confidence in [0, 1]")
    calibrated_confidence: Optional[float] = Field(None, description="Calibrated confidence score in [0, 1]")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    processing_time_ms: float = Field(0.0, description="Agent execution time in milliseconds")
    evidence_source: str = Field("vision_model", description="Evidence source type (vision_model, ocr_engine, heuristic, rag)")
    error_info: Optional[str] = Field(None, description="Error diagnostics if agent status is FAILED or PARTIAL")
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Metadata on spatial coordinates, crop bounds, etc.")

class EvidenceRetrievalResponse(BaseModel):
    inspection_id: str
    session_id: str
    query: str
    total_evidence_records: int
    evidence_records: List[EvidenceRecord]
    raven_assessment: Optional[Dict[str, Any]] = None
    completeness_score: float = 1.0
    conflict_reports: List[Dict[str, Any]] = Field(default_factory=list)

class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "online"})
    project: str = Field(..., json_schema_extra={"example": "VisionIQ: Adaptive Multi-Agent Vision Intelligence"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    database_connected: bool = True
    models: Dict[str, bool] = Field(..., description="Availability status of registered vision models")
    routing_parameters: Dict[str, float] = Field(..., description="Active routing weights and threshold")

class FindingContract(BaseModel):
    model_config = {"protected_namespaces": ()}

    finding_uid: str = Field(..., json_schema_extra={"example": "F-OBJ-01"})
    agent_source: str = Field(..., json_schema_extra={"example": "object"})
    label: str = Field(..., json_schema_extra={"example": "Worker"})
    bbox: Optional[List[float]] = Field(None, json_schema_extra={"example": [120.0, 85.0, 310.0, 480.0]})
    detection_confidence: Optional[float] = Field(None, json_schema_extra={"example": 0.92})
    anomaly_score: Optional[float] = None
    routing_score: Optional[float] = Field(None, json_schema_extra={"example": 0.552})
    observed_evidence: str = Field(..., json_schema_extra={"example": "Worker detected with high-visibility vest."})
    root_cause_hypothesis: Optional[str] = None
    alternative_explanations: Optional[str] = None
    recommendation: Optional[str] = None
    verification_procedure: Optional[str] = None
    human_review_required: bool = False
    frame_timestamp: Optional[float] = None
    track_id: Optional[int] = None
    model_name: Optional[str] = Field(None, json_schema_extra={"example": "Ultralytics YOLO11n"})
    model_version: Optional[str] = Field(None, json_schema_extra={"example": "v1.0.0"})
    score_interpretation: Optional[str] = Field(None, json_schema_extra={"example": "High confidence visual match (92.0%)"})
    person_id: Optional[str] = Field(None, json_schema_extra={"example": "Worker #1 (W-01)"})
    is_confirmed_defect: Optional[bool] = Field(None, description="True for model-classified defects; False for heuristic anomaly cues")

class AgentTraceContract(BaseModel):
    model_config = {"protected_namespaces": ()}

    agent_name: str = Field(..., json_schema_extra={"example": "ppe"})
    relevance_score: float = Field(..., json_schema_extra={"example": 0.98}, description="R(a, q) - Query relevance")
    evidence_capability: float = Field(..., json_schema_extra={"example": 0.90}, description="E(a) - Evidence capability")
    normalized_cost: float = Field(..., json_schema_extra={"example": 0.20}, description="C(a) - Normalized compute cost")
    expected_latency: float = Field(..., json_schema_extra={"example": 0.22}, description="L(a) - Normalized latency")
    final_routing_score: float = Field(..., json_schema_extra={"example": 0.552}, description="S(a, q) - Formula 1 result")
    decision: str = Field(..., json_schema_extra={"example": "SELECTED"})
    selection_reason: Optional[str] = Field(None, json_schema_extra={"example": "Score 0.552 >= threshold 0.45"})
    execution_status: str = Field(..., json_schema_extra={"example": "SUCCESS"})
    actual_latency_ms: float = Field(..., json_schema_extra={"example": 42.5})
    error_message: Optional[str] = None
    
    # Phase 2 Explainability fields
    static_latency_prior: Optional[float] = Field(None, description="Static configured latency prior L_prior(a)")
    historical_ema_latency_ms: Optional[float] = Field(None, description="Historical EMA measured latency in milliseconds")
    normalized_historical_latency: Optional[float] = Field(None, description="Normalized historical latency in [0, 1]")
    latency_observations_count: Optional[int] = Field(0, description="Number of valid historical observations in rolling window")
    latency_fallback_used: Optional[bool] = Field(True, description="True if fallback to static prior was used")
    selected_latency: Optional[float] = Field(None, description="Selected L(a) used in Equation 1")
    hardware_device: Optional[str] = Field(None, description="Hardware device execution context (e.g., cpu, cuda:0)")
    model_version: Optional[str] = Field(None, description="Underlying model checkpoint version")

class RAGSourceContract(BaseModel):
    document_title: str = Field(..., json_schema_extra={"example": "OSHA Standard 1926.100 - Head Protection"})
    section_reference: str = Field(..., json_schema_extra={"example": "29 CFR 1926.100(a)"})
    excerpt: str = Field(..., json_schema_extra={"example": "Employees working in areas where there is a possible danger..."})
    similarity_score: float = Field(..., json_schema_extra={"example": 0.88})
    source_uri: Optional[str] = None

class AnalysisResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    # Primary identifiers (supporting both inspection_id and session_id)
    session_id: str
    inspection_id: Optional[str] = None
    created_at: str
    user_query: str
    query: Optional[str] = None
    input_type: str = "image"
    input_path: str
    input_url: Optional[str] = None
    annotated_path: Optional[str] = None
    annotated_url: Optional[str] = None
    detected_intent: Optional[str] = None
    intent_confidence: float = 1.0

    # Agent orchestration categorization
    selected_agents: List[str]
    executed_agents: List[str]
    skipped_agents: List[str]
    failed_agents: List[str]
    partial_agents: List[str] = Field(default_factory=list)

    status: str = "completed"
    total_latency_ms: float
    synthesis_text: Optional[str] = None
    evidence_support_score: float = 1.0
    pdf_report_path: Optional[str] = None
    pdf_url: Optional[str] = None

    # RAVEN verification & risk assessment
    raven_risk_score: Optional[float] = None
    raven_decision: Optional[str] = None
    evidence_completeness: Optional[float] = None
    conflict_entropy: Optional[float] = None
    raven_dossier: Optional[Dict[str, Any]] = None
    raven_assessment: Optional[Dict[str, Any]] = None
    human_review_record: Optional[Dict[str, Any]] = None

    # Phase 3 Sequential Routing & Telemetry
    routing_telemetry: Optional[List[Dict[str, Any]]] = None
    deficit_history: Optional[List[float]] = None
    current_deficit: Optional[float] = None
    stopping_reason: Optional[str] = None

    # Common standardized evidence & decision models
    detections: List[Dict[str, Any]] = Field(default_factory=list)
    findings: List[FindingContract]
    evidence_records: List[EvidenceRecord] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    human_review_status: bool = False
    execution_timing: Dict[str, float] = Field(default_factory=dict)

    traces: List[AgentTraceContract]
    rag_sources: List[RAGSourceContract] = []

    @model_validator(mode="before")
    @classmethod
    def populate_compatibility_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # inspection_id <-> session_id
            if not data.get("inspection_id") and data.get("session_id"):
                data["inspection_id"] = data["session_id"]
            elif not data.get("session_id") and data.get("inspection_id"):
                data["session_id"] = data["inspection_id"]

            # query <-> user_query
            if not data.get("query") and data.get("user_query"):
                data["query"] = data["user_query"]
            elif not data.get("user_query") and data.get("query"):
                data["user_query"] = data["query"]

            # recommendations extraction
            if not data.get("recommendations"):
                recs = []
                for f in data.get("findings", []):
                    rec = f.get("recommendation") if isinstance(f, dict) else getattr(f, "recommendation", None)
                    if rec and rec not in recs:
                        recs.append(rec)
                data["recommendations"] = recs

            # detections extraction
            if not data.get("detections"):
                dets = []
                for f in data.get("findings", []):
                    if isinstance(f, dict):
                        dets.append({
                            "label": f.get("label", "Unknown"),
                            "confidence": f.get("detection_confidence"),
                            "bbox": f.get("bbox"),
                            "agent": f.get("agent_source", "unknown"),
                            "object_id": f.get("person_id") or f.get("track_id") or f.get("finding_uid")
                        })
                    elif hasattr(f, "label"):
                        dets.append({
                            "label": f.label,
                            "confidence": f.detection_confidence,
                            "bbox": f.bbox,
                            "agent": f.agent_source,
                            "object_id": f.person_id or f.track_id or f.finding_uid
                        })
                data["detections"] = dets

            # human_review_status
            if "human_review_status" not in data or data.get("human_review_status") is False:
                hr = False
                for f in data.get("findings", []):
                    val = f.get("human_review_required") if isinstance(f, dict) else getattr(f, "human_review_required", False)
                    if val:
                        hr = True
                        break
                if data.get("raven_decision") == "ABSTAIN":
                    hr = True
                data["human_review_status"] = hr

            # execution_timing
            if not data.get("execution_timing"):
                timing = {"total_latency_ms": data.get("total_latency_ms", 0.0)}
                for t in data.get("traces", []):
                    if isinstance(t, dict):
                        name = t.get("agent_name", "agent")
                        timing[f"{name}_ms"] = t.get("actual_latency_ms", 0.0)
                    elif hasattr(t, "agent_name"):
                        timing[f"{t.agent_name}_ms"] = t.actual_latency_ms
                data["execution_timing"] = timing

            # raven_assessment
            if not data.get("raven_assessment"):
                if data.get("raven_risk_score") is not None or data.get("raven_decision"):
                    data["raven_assessment"] = {
                        "risk_score": data.get("raven_risk_score"),
                        "decision": data.get("raven_decision"),
                        "evidence_completeness": data.get("evidence_completeness"),
                        "conflict_entropy": data.get("conflict_entropy"),
                        "dossier": data.get("raven_dossier")
                    }
        return data

def build_evidence_records(
    session_id: str,
    input_image_ref: Optional[str],
    findings: List[Any],
    traces: List[Any],
    raven_dossier: Optional[Dict[str, Any]] = None
) -> List[EvidenceRecord]:
    """Constructs standardized EvidenceRecord representations across all agent findings and traces."""
    records: List[EvidenceRecord] = []
    trace_timing = {}
    for t in traces:
        name = t.get("agent_name") if isinstance(t, dict) else getattr(t, "agent_name", "")
        latency = t.get("actual_latency_ms", 0.0) if isinstance(t, dict) else getattr(t, "actual_latency_ms", 0.0)
        trace_timing[name] = latency

    for idx, f in enumerate(findings):
        f_dict = f if isinstance(f, dict) else f.model_dump()
        source = f_dict.get("agent_source", "vision_agent")
        uid = f_dict.get("finding_uid") or f"EV-{source.upper()}-{idx+1:02d}"
        
        status = "EXECUTED"
        if f_dict.get("is_confirmed_defect") is False or (f_dict.get("human_review_required") and (f_dict.get("detection_confidence") or 0.0) < 0.5):
            status = "PARTIAL"

        rec = EvidenceRecord(
            evidence_id=uid,
            inspection_id=session_id,
            query_id=session_id,
            agent_name=source,
            agent_status=status,
            model_id=f_dict.get("model_name") or f"Specialist-{source}",
            model_version=f_dict.get("model_version") or "v1.0.0",
            input_image_ref=input_image_ref,
            predicted_class=f_dict.get("label"),
            object_id=f_dict.get("person_id") or str(f_dict.get("track_id")) if f_dict.get("track_id") is not None else uid,
            bbox=f_dict.get("bbox"),
            mask_ref=None,
            raw_confidence=f_dict.get("detection_confidence"),
            calibrated_confidence=round(f_dict.get("detection_confidence") / 1.35, 4) if f_dict.get("detection_confidence") else None,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            processing_time_ms=trace_timing.get(source, 0.0),
            evidence_source=f"{source}_specialist",
            error_info=None,
            provenance={
                "score_interpretation": f_dict.get("score_interpretation"),
                "is_confirmed_defect": f_dict.get("is_confirmed_defect"),
                "anomaly_score": f_dict.get("anomaly_score")
            }
        )
        records.append(rec)

    # For failed agents in traces, add explicit error evidence records
    for t in traces:
        t_dict = t if isinstance(t, dict) else t.model_dump()
        status = t_dict.get("execution_status", "SUCCESS")
        if status in ("FAILED", "SKIPPED"):
            name = t_dict.get("agent_name", "agent")
            records.append(
                EvidenceRecord(
                    evidence_id=f"EV-{name.upper()}-TRACE",
                    inspection_id=session_id,
                    query_id=session_id,
                    agent_name=name,
                    agent_status=status,
                    input_image_ref=input_image_ref,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    processing_time_ms=t_dict.get("actual_latency_ms", 0.0),
                    evidence_source=f"{name}_specialist",
                    error_info=t_dict.get("error_message") or t_dict.get("selection_reason")
                )
            )

    return records

class VideoProcessingRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Track personnel and verify PPE compliance"})
    sample_fps: float = Field(1.0, ge=0.2, le=5.0)
    max_frames: int = Field(25, ge=1, le=120)

class VideoProcessingResponse(BaseModel):
    session_id: Optional[str] = None
    video_path: str
    duration_sec: float
    total_video_frames: int
    sampled_frames_processed: int
    sample_fps: float
    unique_tracked_entities: Dict[str, int]
    total_unique_entities: int
    frame_results: List[Dict[str, Any]]
    temporal_violations: Optional[List[Dict[str, Any]]] = None
    detected_intent: Optional[str] = None
    selected_agents: Optional[List[str]] = None
    synthesis_text: Optional[str] = None
    total_latency_ms: float

class CameraFrameRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Continuous PPE monitoring"})
    frame_base64: str = Field(..., description="Base64 encoded JPEG/PNG frame payload")
    timestamp: float = Field(0.0)

class CameraFrameResponse(BaseModel):
    status: str
    session_id: Optional[str] = None
    timestamp: float
    latency_ms: float
    detected_intent: Optional[str] = None
    selected_agents: List[str] = []
    executed_agents: List[str] = []
    findings: List[FindingContract] = []
    annotated_frame_base64: Optional[str] = None
    synthesis: Optional[str] = None

class HistorySessionContract(BaseModel):
    id: str
    created_at: str
    user_query: str
    input_type: str
    detected_intent: Optional[str]
    status: str
    total_latency_ms: float
    evidence_support_score: float
    pdf_report_path: Optional[str] = None
    raven_risk_score: Optional[float] = None
    raven_decision: Optional[str] = None

class HistoryResponse(BaseModel):
    status: str = "success"
    total: int
    sessions: List[HistorySessionContract]

class OrchestrationPlanRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Count all visible people."})
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0)
    beta: Optional[float] = Field(None, ge=0.0, le=1.0)
    gamma: Optional[float] = Field(None, ge=0.0, le=1.0)
    delta: Optional[float] = Field(None, ge=0.0, le=1.0)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

class OrchestrationPlanResponse(BaseModel):
    query: str
    detected_intent: str
    intent_confidence: float
    selected_agents: List[str]
    skipped_agents: List[str]
    considered_agents: List[str]
    threshold: float
    traces: Dict[str, AgentTraceContract]
    latency_ms: float

class SessionTraceResponse(BaseModel):
    session_id: str
    user_query: str
    detected_intent: Optional[str]
    selected_agents: List[str]
    executed_agents: List[str]
    skipped_agents: List[str]
    failed_agents: List[str]
    total_latency_ms: float
    traces: List[AgentTraceContract]
