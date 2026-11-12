"""
VisionIQ Phase 2: Dynamic Cost and Latency Feedback Automated Test Suite
Verifies:
1. No historical measurements: static fallback.
2. One historical measurement.
3. Multiple measurements: correct EMA calculation.
4. Invalid latency records (0, negative, None, NaN, inf).
5. Failed/skipped executions excluded from historical measurements.
6. Insufficient history (< min_observations) triggers fallback.
7. Dynamic feedback enabled in router Equation (1).
8. Dynamic feedback disabled in router Equation (1).
9. Existing mandatory dependency behavior preserved under dynamic feedback.
10. Existing routing score compatibility (exact match to static baseline).
"""

import math
import uuid
import pytest
from backend.app.core.config import settings
from backend.app.orchestration.latency_estimator import (
    compute_ema,
    normalize_latency,
    HistoricalLatencyEstimator,
    LatencyEstimate
)
from backend.app.orchestration.router import (
    route_query,
    compute_routing_scores,
    AVAILABLE_AGENTS
)
from backend.app.database.session import SessionLocal
from backend.app.database.models import AnalysisSession, AgentExecutionTrace
from backend.app.database.repository import AnalysisRepository


# --------------------------------------------------------------------------
# Test 1: No historical measurements: static fallback
# --------------------------------------------------------------------------
def test_01_no_historical_measurements_static_fallback():
    estimator = HistoricalLatencyEstimator(min_observations=3)
    res = estimator.estimate_agent_latency("object", historical_measurements=[])

    assert res.fallback_used is True
    assert res.historical_ema_ms is None
    assert res.observation_count == 0
    assert res.selected_latency == settings.AGENT_LATENCIES["object"]
    assert res.static_prior == settings.AGENT_LATENCIES["object"]


# --------------------------------------------------------------------------
# Test 2: One historical measurement
# --------------------------------------------------------------------------
def test_02_one_historical_measurement():
    # When min_observations=1, a single measurement is sufficient
    estimator = HistoricalLatencyEstimator(min_observations=1, ema_alpha=0.30)
    single_measurement = [150.0]
    res = estimator.estimate_agent_latency("object", historical_measurements=single_measurement)

    assert res.fallback_used is False
    assert res.observation_count == 1
    assert res.historical_ema_ms == 150.0
    expected_norm = normalize_latency(
        ema_ms=150.0,
        agent_name="object",
        prior=settings.AGENT_LATENCIES["object"],
        method=estimator.normalization_method
    )
    assert abs(res.selected_latency - expected_norm) < 0.0001


# --------------------------------------------------------------------------
# Test 3: Multiple measurements: correct EMA calculation
# --------------------------------------------------------------------------
def test_03_multiple_measurements_correct_ema_calculation():
    # Sequence: [200.0, 250.0, 300.0] with alpha=0.30
    # S1 = 200.0
    # S2 = 0.30 * 250.0 + 0.70 * 200.0 = 75.0 + 140.0 = 215.0
    # S3 = 0.30 * 300.0 + 0.70 * 215.0 = 90.0 + 150.5 = 240.50
    measurements = [200.0, 250.0, 300.0]
    alpha = 0.30
    ema = compute_ema(measurements, alpha=alpha)
    assert ema == 240.50

    estimator = HistoricalLatencyEstimator(ema_alpha=alpha, min_observations=3)
    res = estimator.estimate_agent_latency("object", historical_measurements=measurements)

    assert res.fallback_used is False
    assert res.observation_count == 3
    assert res.historical_ema_ms == 240.50
    expected_norm = normalize_latency(240.50, "object", settings.AGENT_LATENCIES["object"], method="anchored")
    assert abs(res.selected_latency - expected_norm) < 0.0001


# --------------------------------------------------------------------------
# Test 4: Invalid latency records
# --------------------------------------------------------------------------
def test_04_invalid_latency_records():
    # Feed corrupt/invalid values: 0, negative, None, NaN, inf
    dirty_measurements = [0.0, -15.0, None, float("nan"), float("inf"), 200.0, 250.0, 300.0]
    ema = compute_ema(dirty_measurements, alpha=0.30)
    # Only [200.0, 250.0, 300.0] should be ingested
    assert ema == 240.50

    # If all measurements are invalid, compute_ema should return None safely
    all_invalid = [0.0, -100.0, float("nan"), None]
    assert compute_ema(all_invalid, alpha=0.30) is None


# --------------------------------------------------------------------------
# Test 5: Failed executions excluded from historical measurements
# --------------------------------------------------------------------------
def test_05_failed_executions_excluded_from_db():
    db = SessionLocal()
    test_session_id = f"test-trace-filter-{uuid.uuid4().hex[:8]}"
    test_agent = f"filter_test_{uuid.uuid4().hex[:4]}"

    try:
        session_rec = AnalysisSession(
            id=test_session_id,
            user_query="Filter test query",
            input_path="storage/test.png",
            status="completed"
        )
        db.add(session_rec)
        db.commit()

        # Insert 1 successful trace, 1 failed trace, and 1 zero-latency trace
        traces_to_add = [
            AgentExecutionTrace(
                session_id=test_session_id,
                agent_name=test_agent,
                decision="SELECTED",
                execution_status="SUCCESS",
                actual_latency_ms=120.0
            ),
            AgentExecutionTrace(
                session_id=test_session_id,
                agent_name=test_agent,
                decision="SELECTED",
                execution_status="FAILED",
                actual_latency_ms=9999.0  # Should NOT be ingested!
            ),
            AgentExecutionTrace(
                session_id=test_session_id,
                agent_name=test_agent,
                decision="SELECTED",
                execution_status="SUCCESS",
                actual_latency_ms=0.0     # Should NOT be ingested!
            ),
            AgentExecutionTrace(
                session_id=test_session_id,
                agent_name=test_agent,
                decision="SELECTED",
                execution_status="SUCCESS",
                actual_latency_ms=140.0
            )
        ]
        db.add_all(traces_to_add)
        db.commit()

        repo = AnalysisRepository(db)
        latencies = repo.get_historical_agent_latencies(test_agent, window_size=50)

        # Only the 2 positive SUCCESS records (120.0, 140.0) should be returned
        assert latencies == [120.0, 140.0]

    finally:
        db.query(AgentExecutionTrace).filter(AgentExecutionTrace.session_id == test_session_id).delete()
        db.query(AnalysisSession).filter(AnalysisSession.id == test_session_id).delete()
        db.commit()
        db.close()


# --------------------------------------------------------------------------
# Test 6: Insufficient history triggers fallback
# --------------------------------------------------------------------------
def test_06_insufficient_history():
    # min_observations = 3, but only 2 measurements available
    estimator = HistoricalLatencyEstimator(min_observations=3)
    res = estimator.estimate_agent_latency("ppe", historical_measurements=[90.0, 95.0])

    assert res.fallback_used is True
    assert res.observation_count == 2
    assert res.selected_latency == settings.AGENT_LATENCIES["ppe"]
    assert res.static_prior == settings.AGENT_LATENCIES["ppe"]


# --------------------------------------------------------------------------
# Test 7: Dynamic feedback enabled
# --------------------------------------------------------------------------
def test_07_dynamic_feedback_enabled():
    query = "Check whether workers are wearing helmets."
    static_ppe_prior = settings.AGENT_LATENCIES["ppe"]

    # Mock high latency for PPE (e.g. throttled hardware)
    mock_estimates = {
        "ppe": LatencyEstimate(
            agent_name="ppe",
            static_prior=static_ppe_prior,
            historical_ema_ms=300.0,
            normalized_latency=0.66,
            observation_count=10,
            fallback_used=False,
            selected_latency=0.66,
            hardware_device="cpu",
            raw_measurements=[300.0] * 10
        )
    }

    plan = route_query(
        query=query,
        enable_dynamic_latency=True,
        latency_estimates=mock_estimates
    )

    assert plan["dynamic_latency_enabled"] is True
    ppe_trace = plan["traces"]["ppe"]
    assert ppe_trace["expected_latency"] == 0.66
    assert ppe_trace["selected_latency"] == 0.66
    assert ppe_trace["latency_fallback_used"] is False
    assert ppe_trace["historical_ema_latency_ms"] == 300.0
    assert ppe_trace["latency_observations_count"] == 10

    # Equation 1 fidelity check: S(a,q) must use 0.66
    expected_s = (
        settings.ROUTING_ALPHA * ppe_trace["relevance_score"] +
        settings.ROUTING_DELTA * ppe_trace["evidence_capability"] -
        settings.ROUTING_BETA * ppe_trace["normalized_cost"] -
        settings.ROUTING_GAMMA * 0.66
    )
    assert abs(ppe_trace["final_routing_score"] - expected_s) < 0.001


# --------------------------------------------------------------------------
# Test 8: Dynamic feedback disabled
# --------------------------------------------------------------------------
def test_08_dynamic_feedback_disabled():
    query = "Check whether workers are wearing helmets."
    mock_estimates = {
        "ppe": LatencyEstimate(
            agent_name="ppe",
            static_prior=settings.AGENT_LATENCIES["ppe"],
            historical_ema_ms=300.0,
            normalized_latency=0.66,
            observation_count=10,
            fallback_used=False,
            selected_latency=0.66,
            hardware_device="cpu",
            raw_measurements=[300.0] * 10
        )
    }

    # Explicitly disabled
    plan = route_query(
        query=query,
        enable_dynamic_latency=False,
        latency_estimates=mock_estimates
    )

    assert plan["dynamic_latency_enabled"] is False
    ppe_trace = plan["traces"]["ppe"]
    # Must use original static prior, ignoring the dynamic estimate
    assert ppe_trace["expected_latency"] == settings.AGENT_LATENCIES["ppe"]
    assert ppe_trace["latency_fallback_used"] is True


# --------------------------------------------------------------------------
# Test 9: Existing mandatory dependency behavior
# --------------------------------------------------------------------------
def test_09_existing_mandatory_dependency_behavior():
    query = "Check whether workers are wearing helmets."
    # Suppress object detection with huge latency penalty so its raw score is below threshold
    mock_estimates = {
        "object": LatencyEstimate(
            agent_name="object",
            static_prior=settings.AGENT_LATENCIES["object"],
            historical_ema_ms=5000.0,
            normalized_latency=1.0,
            observation_count=20,
            fallback_used=False,
            selected_latency=1.0,
            hardware_device="cpu",
            raw_measurements=[5000.0] * 20
        )
    }

    plan = route_query(
        query=query,
        threshold=0.45,
        enable_dynamic_latency=True,
        latency_estimates=mock_estimates
    )

    # Even though object detection score was penalized by L=1.0,
    # mandatory dependency resolution must still forcibly include object for PPE!
    assert "ppe" in plan["selected_agents"]
    assert "object" in plan["selected_agents"]
    assert plan["traces"]["object"]["decision"] in ["SELECTED", "MANDATORY_DEPENDENCY"]


# --------------------------------------------------------------------------
# Test 10: Existing routing score compatibility
# --------------------------------------------------------------------------
def test_10_existing_routing_score_compatibility():
    # Calling route_query without extra arguments must produce exact baseline scores
    query = "Locate cracks or scratches on this metal surface."
    plan = route_query(query)

    for agent_name, trace in plan["traces"].items():
        static_l = settings.AGENT_LATENCIES.get(agent_name, 0.20)
        expected_s = (
            settings.ROUTING_ALPHA * trace["relevance_score"] +
            settings.ROUTING_DELTA * trace["evidence_capability"] -
            settings.ROUTING_BETA * trace["normalized_cost"] -
            settings.ROUTING_GAMMA * static_l
        )
        assert abs(trace["final_routing_score"] - expected_s) < 0.0001
        assert trace["expected_latency"] == static_l


# --------------------------------------------------------------------------
# Test 11: Matching model versions
# --------------------------------------------------------------------------
def test_11_matching_model_versions():
    db = SessionLocal()
    sess_id = f"test-mv-match-{uuid.uuid4().hex[:8]}"
    agent_name = f"mv_match_{uuid.uuid4().hex[:4]}"
    target_version = "YOLOv11n (Ultralytics 8.3.0)"

    try:
        session_rec = AnalysisSession(id=sess_id, user_query="test", input_path="dummy.png")
        db.add(session_rec)
        db.commit()

        traces = [
            AgentExecutionTrace(
                session_id=sess_id,
                agent_name=agent_name,
                decision="SELECTED",
                execution_status="SUCCESS",
                actual_latency_ms=105.0,
                model_version=target_version
            ),
            AgentExecutionTrace(
                session_id=sess_id,
                agent_name=agent_name,
                decision="SELECTED",
                execution_status="SUCCESS",
                actual_latency_ms=115.0,
                model_version=target_version
            )
        ]
        db.add_all(traces)
        db.commit()

        repo = AnalysisRepository(db)
        res = repo.get_historical_agent_latencies(agent_name=agent_name, model_version=target_version)
        assert res == [105.0, 115.0]

    finally:
        db.query(AgentExecutionTrace).filter(AgentExecutionTrace.session_id == sess_id).delete()
        db.query(AnalysisSession).filter(AnalysisSession.id == sess_id).delete()
        db.commit()
        db.close()


# --------------------------------------------------------------------------
# Test 12: Different model versions being excluded
# --------------------------------------------------------------------------
def test_12_different_model_versions_excluded():
    db = SessionLocal()
    sess_id = f"test-mv-diff-{uuid.uuid4().hex[:8]}"
    agent_name = f"mv_diff_{uuid.uuid4().hex[:4]}"

    try:
        session_rec = AnalysisSession(id=sess_id, user_query="test", input_path="dummy.png")
        db.add(session_rec)
        db.commit()

        traces = [
            AgentExecutionTrace(
                session_id=sess_id,
                agent_name=agent_name,
                decision="SELECTED",
                execution_status="SUCCESS",
                actual_latency_ms=100.0,
                model_version="v1.0.0"
            ),
            AgentExecutionTrace(
                session_id=sess_id,
                agent_name=agent_name,
                decision="SELECTED",
                execution_status="SUCCESS",
                actual_latency_ms=500.0,
                model_version="v2.0.0"
            )
        ]
        db.add_all(traces)
        db.commit()

        repo = AnalysisRepository(db)
        # Querying v2.0.0 must strictly exclude v1.0.0
        res_v2 = repo.get_historical_agent_latencies(agent_name=agent_name, model_version="v2.0.0")
        assert res_v2 == [500.0]

        # Querying v1.0.0 must strictly exclude v2.0.0
        res_v1 = repo.get_historical_agent_latencies(agent_name=agent_name, model_version="v1.0.0")
        assert res_v1 == [100.0]

    finally:
        db.query(AgentExecutionTrace).filter(AgentExecutionTrace.session_id == sess_id).delete()
        db.query(AnalysisSession).filter(AnalysisSession.id == sess_id).delete()
        db.commit()
        db.close()


# --------------------------------------------------------------------------
# Test 13: Legacy NULL model-version records
# --------------------------------------------------------------------------
def test_13_legacy_null_model_version_fallback_policy():
    db = SessionLocal()
    sess_id = f"test-mv-null-{uuid.uuid4().hex[:8]}"
    agent_name = f"mv_null_{uuid.uuid4().hex[:4]}"

    try:
        session_rec = AnalysisSession(id=sess_id, user_query="test", input_path="dummy.png")
        db.add(session_rec)
        db.commit()

        # Step 1: Insert only legacy record with model_version=None
        legacy_trace = AgentExecutionTrace(
            session_id=sess_id,
            agent_name=agent_name,
            decision="SELECTED",
            execution_status="SUCCESS",
            actual_latency_ms=150.0,
            model_version=None
        )
        db.add(legacy_trace)
        db.commit()

        repo = AnalysisRepository(db)
        # Case A: When no exact match exists, legacy NULL is admitted under compatibility policy
        res_legacy = repo.get_historical_agent_latencies(
            agent_name=agent_name,
            model_version="v1.0.0",
            allow_legacy_null=True
        )
        assert res_legacy == [150.0]

        # Step 2: Now insert an exact match record
        exact_trace = AgentExecutionTrace(
            session_id=sess_id,
            agent_name=agent_name,
            decision="SELECTED",
            execution_status="SUCCESS",
            actual_latency_ms=220.0,
            model_version="v1.0.0"
        )
        db.add(exact_trace)
        db.commit()

        # Case B: Once exact matches exist, they are strictly preferred over legacy NULL
        res_exact = repo.get_historical_agent_latencies(
            agent_name=agent_name,
            model_version="v1.0.0",
            allow_legacy_null=True
        )
        assert res_exact == [220.0]

    finally:
        db.query(AgentExecutionTrace).filter(AgentExecutionTrace.session_id == sess_id).delete()
        db.query(AnalysisSession).filter(AnalysisSession.id == sess_id).delete()
        db.commit()
        db.close()


# --------------------------------------------------------------------------
# Test 14: Hardware and model-version filtering together
# --------------------------------------------------------------------------
def test_14_hardware_and_model_version_combined_filtering():
    db = SessionLocal()
    sess_id = f"test-hw-mv-{uuid.uuid4().hex[:8]}"
    agent_name = f"hw_mv_{uuid.uuid4().hex[:4]}"

    try:
        session_rec = AnalysisSession(id=sess_id, user_query="test", input_path="dummy.png")
        db.add(session_rec)
        db.commit()

        traces = [
            AgentExecutionTrace(
                session_id=sess_id,
                agent_name=agent_name,
                decision="SELECTED",
                execution_status="SUCCESS",
                actual_latency_ms=100.0,
                hardware_device="cpu",
                model_version="v1.0.0"
            ),
            AgentExecutionTrace(
                session_id=sess_id,
                agent_name=agent_name,
                decision="SELECTED",
                execution_status="SUCCESS",
                actual_latency_ms=25.0,
                hardware_device="cuda:0",
                model_version="v1.0.0"
            ),
            AgentExecutionTrace(
                session_id=sess_id,
                agent_name=agent_name,
                decision="SELECTED",
                execution_status="SUCCESS",
                actual_latency_ms=350.0,
                hardware_device="cpu",
                model_version="v2.0.0"
            ),
            AgentExecutionTrace(
                session_id=sess_id,
                agent_name=agent_name,
                decision="SELECTED",
                execution_status="SUCCESS",
                actual_latency_ms=70.0,
                hardware_device="cuda:0",
                model_version="v2.0.0"
            )
        ]
        db.add_all(traces)
        db.commit()

        repo = AnalysisRepository(db)
        # Query (cpu, v1.0.0) -> must return only [100.0]
        res_cpu_v1 = repo.get_historical_agent_latencies(
            agent_name=agent_name,
            hardware_device="cpu",
            model_version="v1.0.0"
        )
        assert res_cpu_v1 == [100.0]

        # Query (cuda:0, v2.0.0) -> must return only [70.0]
        res_cuda_v2 = repo.get_historical_agent_latencies(
            agent_name=agent_name,
            hardware_device="cuda:0",
            model_version="v2.0.0"
        )
        assert res_cuda_v2 == [70.0]

    finally:
        db.query(AgentExecutionTrace).filter(AgentExecutionTrace.session_id == sess_id).delete()
        db.query(AnalysisSession).filter(AnalysisSession.id == sess_id).delete()
        db.commit()
        db.close()


# --------------------------------------------------------------------------
# Test 15: No history for a new model version
# --------------------------------------------------------------------------
def test_15_no_history_for_new_model_version():
    db = SessionLocal()
    sess_id = f"test-new-mv-{uuid.uuid4().hex[:8]}"
    agent_name = f"new_mv_{uuid.uuid4().hex[:4]}"

    try:
        session_rec = AnalysisSession(id=sess_id, user_query="test", input_path="dummy.png")
        db.add(session_rec)
        db.commit()

        trace = AgentExecutionTrace(
            session_id=sess_id,
            agent_name=agent_name,
            decision="SELECTED",
            execution_status="SUCCESS",
            actual_latency_ms=100.0,
            model_version="v1.0.0"
        )
        db.add(trace)
        db.commit()

        repo = AnalysisRepository(db)
        # Querying an unrecorded version with allow_legacy_null=False must return empty list
        res = repo.get_historical_agent_latencies(
            agent_name=agent_name,
            model_version="v3.0.0-experimental",
            allow_legacy_null=False
        )
        assert res == []

    finally:
        db.query(AgentExecutionTrace).filter(AgentExecutionTrace.session_id == sess_id).delete()
        db.query(AnalysisSession).filter(AnalysisSession.id == sess_id).delete()
        db.commit()
        db.close()


# --------------------------------------------------------------------------
# Test 16: Static latency fallback when compatible history is insufficient
# --------------------------------------------------------------------------
def test_16_static_fallback_when_compatible_history_insufficient():
    db = SessionLocal()
    sess_id = f"test-mv-fallback-{uuid.uuid4().hex[:8]}"
    agent_name = f"mv_fb_{uuid.uuid4().hex[:4]}"

    try:
        session_rec = AnalysisSession(id=sess_id, user_query="test", input_path="dummy.png")
        db.add(session_rec)
        db.commit()

        # Only 1 observation exists for v2.0.0
        trace = AgentExecutionTrace(
            session_id=sess_id,
            agent_name=agent_name,
            decision="SELECTED",
            execution_status="SUCCESS",
            actual_latency_ms=250.0,
            model_version="v2.0.0"
        )
        db.add(trace)
        db.commit()

        estimator = HistoricalLatencyEstimator(min_observations=3)
        res = estimator.estimate_agent_latency(
            agent_name=agent_name,
            db_session=db,
            model_version="v2.0.0",
            allow_legacy_null=False
        )

        # 1 observation < min_observations=3 -> must fall back to static prior
        assert res.fallback_used is True
        assert res.observation_count == 1
        assert res.selected_latency == settings.AGENT_LATENCIES.get(agent_name, 0.20)
        assert res.model_version == "v2.0.0"

    finally:
        db.query(AgentExecutionTrace).filter(AgentExecutionTrace.session_id == sess_id).delete()
        db.query(AnalysisSession).filter(AnalysisSession.id == sess_id).delete()
        db.commit()
        db.close()


# --------------------------------------------------------------------------
# Test 17: Existing routing behavior unchanged when dynamic feedback disabled
# --------------------------------------------------------------------------
def test_17_existing_routing_behavior_unchanged_when_disabled_with_model_versions():
    query = "Check whether workers are wearing helmets."
    plan = route_query(
        query=query,
        enable_dynamic_latency=False,
        agent_model_versions={"object": "YOLOv99-Hypothetical"}
    )

    assert plan["dynamic_latency_enabled"] is False
    for agent_name, trace in plan["traces"].items():
        assert trace["latency_fallback_used"] is True
        assert trace["expected_latency"] == settings.AGENT_LATENCIES.get(agent_name, 0.20)

