from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import or_
from backend.app.database.models import (
    AnalysisSession, Finding, AgentExecutionTrace, RAGSource, ReportRecord, ExperimentRun
)

class AnalysisRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_session(self, session_data: Dict[str, Any]) -> AnalysisSession:
        session = AnalysisSession(
            id=session_data.get("id"),
            user_query=session_data["user_query"],
            input_type=session_data.get("input_type", "image"),
            input_path=session_data["input_path"],
            annotated_path=session_data.get("annotated_path"),
            detected_intent=session_data.get("detected_intent"),
            intent_confidence=session_data.get("intent_confidence", 1.0),
            selected_agents=session_data.get("selected_agents", []),
            executed_agents=session_data.get("executed_agents", []),
            skipped_agents=session_data.get("skipped_agents", []),
            failed_agents=session_data.get("failed_agents", []),
            status=session_data.get("status", "completed"),
            total_latency_ms=session_data.get("total_latency_ms", 0.0),
            synthesis_text=session_data.get("synthesis_text"),
            evidence_support_score=session_data.get("evidence_support_score", 1.0),
            pdf_report_path=session_data.get("pdf_report_path"),
            raven_risk_score=session_data.get("raven_risk_score"),
            raven_decision=session_data.get("raven_decision"),
            evidence_completeness=session_data.get("evidence_completeness"),
            conflict_entropy=session_data.get("conflict_entropy"),
            raven_dossier=session_data.get("raven_dossier"),
            evidence_records=session_data.get("evidence_records", [])
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def add_findings(self, session_id: str, findings: List[Dict[str, Any]]):
        for f in findings:
            finding = Finding(
                session_id=session_id,
                finding_uid=f.get("finding_uid", "F-00"),
                agent_source=f.get("agent_source", "unknown"),
                label=f.get("label", "Unknown"),
                bbox=f.get("bbox"),
                detection_confidence=f.get("detection_confidence"),
                anomaly_score=f.get("anomaly_score"),
                routing_score=f.get("routing_score"),
                observed_evidence=f.get("observed_evidence", ""),
                root_cause_hypothesis=f.get("root_cause_hypothesis"),
                alternative_explanations=f.get("alternative_explanations"),
                recommendation=f.get("recommendation"),
                verification_procedure=f.get("verification_procedure"),
                human_review_required=f.get("human_review_required", False),
                frame_timestamp=f.get("frame_timestamp"),
                track_id=f.get("track_id"),
                model_name=f.get("model_name"),
                model_version=f.get("model_version"),
                score_interpretation=f.get("score_interpretation"),
                person_id=f.get("person_id"),
                is_confirmed_defect=f.get("is_confirmed_defect")
            )
            self.db.add(finding)
        self.db.commit()

    def add_traces(self, session_id: str, traces: List[Dict[str, Any]]):
        for t in traces:
            trace = AgentExecutionTrace(
                session_id=session_id,
                agent_name=t.get("agent_name", "unknown"),
                relevance_score=t.get("relevance_score", 0.0),
                evidence_capability=t.get("evidence_capability", 0.0),
                normalized_cost=t.get("normalized_cost", 0.0),
                expected_latency=t.get("expected_latency", 0.0),
                final_routing_score=t.get("final_routing_score", 0.0),
                decision=t.get("decision", "SELECTED"),
                selection_reason=t.get("selection_reason", ""),
                execution_status=t.get("execution_status", "SUCCESS"),
                actual_latency_ms=t.get("actual_latency_ms", 0.0),
                error_message=t.get("error_message"),
                static_latency_prior=t.get("static_latency_prior"),
                historical_ema_latency_ms=t.get("historical_ema_latency_ms"),
                normalized_historical_latency=t.get("normalized_historical_latency"),
                latency_observations_count=t.get("latency_observations_count", 0),
                latency_fallback_used=t.get("latency_fallback_used", True),
                hardware_device=t.get("hardware_device"),
                model_version=t.get("model_version")
            )
            self.db.add(trace)
        self.db.commit()

    def get_historical_agent_latencies(
        self,
        agent_name: str,
        window_size: int = 50,
        hardware_device: Optional[str] = None,
        model_version: Optional[str] = None,
        allow_legacy_null: bool = True
    ) -> List[float]:
        """
        Retrieves recent valid successful execution latency measurements (in ms) for an agent.
        Filtered by:
          - execution_status == 'SUCCESS'
          - actual_latency_ms > 0 and not null
          - hardware_device matches if specified (or records without device specified)
          - model_version matches:
              * Exact model_version matches are strictly preferred.
              * If no exact match records exist and allow_legacy_null is True,
                legacy records where model_version IS NULL are admitted as fallback.
              * Records with conflicting known model versions are strictly excluded.
        Ordered chronologically [oldest, ..., newest] within the rolling window.
        """
        q = self.db.query(AgentExecutionTrace.actual_latency_ms).filter(
            AgentExecutionTrace.agent_name == agent_name,
            AgentExecutionTrace.execution_status == "SUCCESS",
            AgentExecutionTrace.actual_latency_ms > 0.0
        )
        if hardware_device:
            q = q.filter(
                or_(
                    AgentExecutionTrace.hardware_device == hardware_device,
                    AgentExecutionTrace.hardware_device.is_(None)
                )
            )

        if model_version:
            # Model-version isolation policy:
            # 1. Prefer exact model_version matches
            exact_q = q.filter(AgentExecutionTrace.model_version == model_version)
            if exact_q.count() > 0:
                q = exact_q
            elif allow_legacy_null:
                # 2. Compatibility policy for legacy records:
                # If no records exist for the explicit model version, admit legacy records where model_version IS NULL.
                # Incompatible known model versions are strictly excluded.
                q = q.filter(AgentExecutionTrace.model_version.is_(None))
            else:
                # Strict mode: no legacy fallback allowed, only exact matches
                q = exact_q

        recent_records = q.order_by(AgentExecutionTrace.id.desc()).limit(window_size).all()
        return [r[0] for r in reversed(recent_records) if r[0] is not None and r[0] > 0.0]

    def get_all_historical_latencies(
        self,
        window_size: int = 50,
        hardware_device: Optional[str] = None,
        agent_model_versions: Optional[Dict[str, str]] = None,
        allow_legacy_null: bool = True
    ) -> Dict[str, List[float]]:
        """
        Retrieves chronological successful execution latencies grouped by candidate agent.
        """
        from backend.app.orchestration.router import AVAILABLE_AGENTS
        versions = agent_model_versions or {}
        results: Dict[str, List[float]] = {}
        for agent in AVAILABLE_AGENTS:
            results[agent] = self.get_historical_agent_latencies(
                agent_name=agent,
                window_size=window_size,
                hardware_device=hardware_device,
                model_version=versions.get(agent),
                allow_legacy_null=allow_legacy_null
            )
        return results

    def add_rag_sources(self, session_id: str, sources: List[Dict[str, Any]]):
        for s in sources:
            src = RAGSource(
                session_id=session_id,
                document_title=s.get("document_title", "Technical Documentation"),
                section_reference=s.get("section_reference", "General"),
                excerpt=s.get("excerpt", ""),
                similarity_score=s.get("similarity_score", 0.0),
                source_uri=s.get("source_uri")
            )
            self.db.add(src)
        self.db.commit()

    def update_pdf_path(self, session_id: str, pdf_path: str):
        session = self.db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
        if session:
            session.pdf_report_path = pdf_path
            self.db.commit()

    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        return self.db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()

    def get_inspection(self, inspection_id: str) -> Optional[AnalysisSession]:
        """Alias for get_session adhering to industrial inspection terminology."""
        return self.get_session(inspection_id)

    def get_evidence_records(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves persisted or derived evidence records for an inspection session."""
        session = self.get_session(session_id)
        if not session:
            return []
        if session.evidence_records:
            return session.evidence_records
        
        # Derive dynamically if session was created prior to evidence persistence
        from backend.app.schemas.contracts import build_evidence_records
        derived = build_evidence_records(
            session_id=session.id,
            input_image_ref=session.input_path,
            findings=[f.to_dict() for f in session.findings],
            traces=[t.to_dict() for t in session.traces],
            raven_dossier=session.raven_dossier
        )
        return [r.model_dump() for r in derived]

    def list_sessions(self, limit: int = 50) -> List[AnalysisSession]:
        return self.db.query(AnalysisSession).order_by(AnalysisSession.created_at.desc()).limit(limit).all()

    def search_sessions(
        self,
        query: Optional[str] = None,
        input_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        order_desc: bool = True
    ) -> List[AnalysisSession]:
        q = self.db.query(AnalysisSession)
        if query:
            search_pattern = f"%{query.strip()}%"
            q = q.filter(
                or_(
                    AnalysisSession.user_query.ilike(search_pattern),
                    AnalysisSession.detected_intent.ilike(search_pattern),
                    AnalysisSession.id.ilike(search_pattern)
                )
            )
        if input_type and input_type.lower() != "all":
            q = q.filter(AnalysisSession.input_type == input_type.lower())
        if status and status.lower() != "all":
            q = q.filter(AnalysisSession.status == status.lower())

        if order_desc:
            q = q.order_by(AnalysisSession.created_at.desc())
        else:
            q = q.order_by(AnalysisSession.created_at.asc())

        return q.limit(limit).all()

    def create_report_record(
        self,
        session_id: str,
        pdf_path: str,
        file_size_bytes: int = 0,
        report_format: str = "PDF/A-1b"
    ) -> ReportRecord:
        record = ReportRecord(
            session_id=session_id,
            pdf_path=pdf_path,
            file_size_bytes=file_size_bytes,
            format=report_format
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_report_records(self, session_id: str) -> List[ReportRecord]:
        return self.db.query(ReportRecord).filter(ReportRecord.session_id == session_id).order_by(ReportRecord.created_at.desc()).all()

    def create_experiment_run(self, data: Dict[str, Any]) -> ExperimentRun:
        run = ExperimentRun(**data)
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def list_experiment_runs(self, limit: int = 20) -> List[ExperimentRun]:
        return self.db.query(ExperimentRun).order_by(ExperimentRun.created_at.desc()).limit(limit).all()
