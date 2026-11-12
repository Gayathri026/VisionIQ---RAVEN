from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.core.config import settings
from backend.app.database.models import Base

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=settings.DEBUG
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes all database tables defined in SQLAlchemy Base and applies schema migrations."""
    Base.metadata.create_all(bind=engine)
    
    # Auto-migrate SQLite columns if added to models after initial creation
    if settings.DATABASE_URL.startswith("sqlite"):
        from sqlalchemy import text
        with engine.connect() as conn:
            cursor = conn.execute(text("PRAGMA table_info(findings);"))
            existing_cols = {row[1] for row in cursor.fetchall()}
            
            new_columns = [
                ("model_name", "VARCHAR(120)"),
                ("model_version", "VARCHAR(64)"),
                ("score_interpretation", "TEXT"),
                ("person_id", "VARCHAR(64)"),
                ("is_confirmed_defect", "BOOLEAN")
            ]
            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE findings ADD COLUMN {col_name} {col_type};"))

            # Auto-migrate analysis_sessions columns for RAVEN
            cursor_sessions = conn.execute(text("PRAGMA table_info(analysis_sessions);"))
            session_cols = {row[1] for row in cursor_sessions.fetchall()}
            raven_columns = [
                ("raven_risk_score", "FLOAT"),
                ("raven_decision", "VARCHAR(32)"),
                ("evidence_completeness", "FLOAT"),
                ("conflict_entropy", "FLOAT"),
                ("raven_dossier", "JSON"),
                ("evidence_records", "JSON")
            ]
            for col_name, col_type in raven_columns:
                if col_name not in session_cols:
                    conn.execute(text(f"ALTER TABLE analysis_sessions ADD COLUMN {col_name} {col_type};"))

            # Auto-migrate agent_execution_traces columns for Phase 2 dynamic latency feedback
            cursor_traces = conn.execute(text("PRAGMA table_info(agent_execution_traces);"))
            trace_cols = {row[1] for row in cursor_traces.fetchall()}
            trace_columns = [
                ("static_latency_prior", "FLOAT"),
                ("historical_ema_latency_ms", "FLOAT"),
                ("normalized_historical_latency", "FLOAT"),
                ("latency_observations_count", "INTEGER"),
                ("latency_fallback_used", "BOOLEAN"),
                ("hardware_device", "VARCHAR(64)"),
                ("model_version", "VARCHAR(64)"),
                ("created_at", "DATETIME")
            ]
            for col_name, col_type in trace_columns:
                if col_name not in trace_cols:
                    conn.execute(text(f"ALTER TABLE agent_execution_traces ADD COLUMN {col_name} {col_type};"))

            conn.commit()

def get_db():
    """FastAPI request dependency yielding an active database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
