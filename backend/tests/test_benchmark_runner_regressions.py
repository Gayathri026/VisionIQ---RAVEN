import csv
import json
import os
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from experiments.runner import benchmark_runner


def test_empty_test_set_has_clear_error(tmp_path):
    testset_path = tmp_path / "empty.json"
    testset_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="Benchmark test set is empty"):
        benchmark_runner.BenchmarkRunner(str(testset_path))


def test_missing_image_does_not_use_fallback(tmp_path, monkeypatch):
    runner = benchmark_runner.BenchmarkRunner()
    monkeypatch.setattr(
        benchmark_runner,
        "vision_iq_app",
        SimpleNamespace(
            invoke=lambda state: pytest.fail(
                "Graph should not run when the item's image is missing"
            )
        ),
    )
    missing_path = tmp_path / "missing.png"
    item = {
        "id": "missing-image",
        "session_id": "session-1",
        "query": "Inspect the image",
        "image_path": str(missing_path),
        "required_agents": ["object"],
    }

    with pytest.raises(FileNotFoundError, match="Benchmark image not found for item missing-image"):
        runner.run_pipeline_on_item(item, "adaptive")


def test_agent_count_uses_graph_executed_agents(monkeypatch):
    runner = benchmark_runner.BenchmarkRunner()
    monkeypatch.setattr(
        benchmark_runner,
        "vision_iq_app",
        SimpleNamespace(
            invoke=lambda state: {"executed_agents": ["ocr"], "findings": []}
        ),
    )
    item = {
        "id": "executed-count",
        "session_id": "session-1",
        "query": "Read the warning sign",
        "image_path": "experiments/benchmark/fixtures/ocr_safety_sign.png",
        "required_agents": ["ocr"],
    }

    result = runner.run_pipeline_on_item(item, "fixed")

    assert result["selected_agent_count"] == 8
    assert result["executed_agents"] == ["ocr"]
    assert result["executed_agent_count"] == result["num_agents"] == 1


def test_sqlite_persistence_uses_computed_metrics_only(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    benchmark_runner.ExperimentRun.__table__.create(engine)
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(benchmark_runner, "SessionLocal", session_factory)
    metrics = {
        "pipeline_type": "adaptive",
        "total_queries": 1,
        "routing_precision": 0.8,
        "routing_recall": 0.7,
        "routing_f1": 0.75,
        "median_latency_ms": 120.0,
        "p95_latency_ms": 150.0,
        "latency_reduction_pct": 20.0,
        "mean_agents_invoked": 2.0,
        "required_agent_coverage": 0.5,
        "completion_rate": 1.0,
    }

    runner = benchmark_runner.BenchmarkRunner()
    runner.persist_to_sqlite({"adaptive": metrics})

    with session_factory() as database:
        persisted = database.query(benchmark_runner.ExperimentRun).one()

        assert persisted.raw_metrics_json == metrics
        assert persisted.routing_f1 == metrics["routing_f1"]
        assert persisted.detection_map50 is None
        assert persisted.ocr_wer is None
        assert persisted.ocr_cer is None
        assert persisted.failure_recovery_rate is None

    engine.dispose()


def test_csv_exports_only_available_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(benchmark_runner, "PLOTS_DIR", str(tmp_path / "plots"))
    monkeypatch.setattr(benchmark_runner, "PAPER_FIGS_DIR", str(tmp_path / "paper"))
    for path in (
        benchmark_runner.RESULTS_DIR,
        benchmark_runner.PLOTS_DIR,
        benchmark_runner.PAPER_FIGS_DIR,
    ):
        os.makedirs(path, exist_ok=True)

    pipeline_types = [
        "fixed",
        "rule_based",
        "adaptive",
        "ablation_no_cost",
        "ablation_no_rag",
        "ablation_no_evidence_aggregation",
        "ablation_no_fallback",
    ]
    summary = {
        pipeline: {
            "routing_f1": 0.75,
            "routing_precision": 0.8,
            "routing_recall": 0.7,
            "median_latency_ms": 120.0,
            "p95_latency_ms": 150.0,
            "latency_reduction_pct": 20.0,
            "mean_agents_invoked": 2.0,
            "required_agent_coverage": 0.5,
            "completion_rate": 1.0,
        }
        for pipeline in pipeline_types
    }

    benchmark_runner.BenchmarkRunner().generate_publication_assets(summary, {})

    with open(
        tmp_path / "results" / "pipeline_comparison.csv",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        pipeline_csv = list(csv.DictReader(csv_file))
    with open(
        tmp_path / "results" / "ablation_study.csv",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        ablation_csv = list(csv.DictReader(csv_file))

    assert len(pipeline_csv) == 3
    assert "Required Agent Coverage" in pipeline_csv[0]
    assert "Completion Rate" in pipeline_csv[0]
    assert "Required Agent Coverage" in ablation_csv[0]
    assert "Completion Rate" in ablation_csv[0]
    assert "Evidence Support" not in pipeline_csv[0]
    assert "Task Correctness" not in pipeline_csv[0]
