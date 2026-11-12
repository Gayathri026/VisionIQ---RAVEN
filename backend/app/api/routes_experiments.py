import os
import csv
import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.database.models import ExperimentRun, ModelVersion

router = APIRouter()

ROUTING_COMPARISON_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "experiments", "results", "routing_comparison.csv"
)

def load_routing_comparison_data() -> List[Dict[str, Any]]:
    """Loads measured empirical benchmarks for the 4 routing strategies."""
    if not os.path.exists(ROUTING_COMPARISON_CSV):
        return []
    results = []
    with open(ROUTING_COMPARISON_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "strategy": row.get("strategy"),
                "sample_count": int(row.get("sample_count", 0)),
                "mean_invocations": float(row.get("mean_invocations", 0.0)),
                "median_latency_ms": float(row.get("median_latency_ms", 0.0)),
                "p95_latency_ms": float(row.get("p95_latency_ms", 0.0)),
                "mean_task_f1": float(row.get("mean_task_f1", 0.0)),
                "mean_task_prec": float(row.get("mean_task_prec", 0.0)),
                "mean_task_rec": float(row.get("mean_task_rec", 0.0)),
                "mean_deficit": float(row.get("mean_deficit", 0.0)),
                "mean_completeness": float(row.get("mean_completeness", 0.0)),
                "mean_reliability": float(row.get("mean_reliability", 0.0)),
                "abstention_rate": float(row.get("abstention_rate", 0.0))
            })
    return results

@router.get("/experiments/routing-strategies", summary="Retrieve 4-Strategy Routing Comparison Benchmark")
async def get_routing_strategies_benchmark():
    """
    Returns empirical benchmarks comparing the 4 routing strategies:
    Fixed Pipeline, Phase 2 Weighted Router, Relevance/Dependency Router,
    and Sequential Evidence-Gain Router on real held-out industrial test sets.
    """
    strategies = load_routing_comparison_data()
    return {
        "status": "success",
        "total_strategies": len(strategies),
        "strategies": strategies
    }

@router.get("/experiments", summary="Retrieve Research Experiments and Pipeline Benchmarks")
async def get_research_experiments(db: Session = Depends(get_db)):
    """
    Returns genuine empirical evaluation experiments, dataset versions,
    model versions, latency distributions, and comparisons among Fixed,
    Rule-Based, and Adaptive VisionIQ pipelines. Missing experiments are
    explicitly labeled as 'not_yet_run'.
    """
    db_runs = db.query(ExperimentRun).order_by(ExperimentRun.created_at.desc()).all()
    models = db.query(ModelVersion).all()

    # Identify executed pipeline types
    executed_types = {r.pipeline_type for r in db_runs}

    # Canonical comparison baseline schema
    comparison_matrix = []

    # 1. Fixed Pipeline (Executes all 7 agents indiscriminately)
    fixed_run = next((r.to_dict() for r in db_runs if r.pipeline_type == "fixed"), None)
    if fixed_run:
        comparison_matrix.append({
            "pipeline_type": "fixed",
            "name": "Fixed Multi-Agent Pipeline",
            "description": "Executes all registered specialist agents unconditionally on every query.",
            "status": "completed",
            "metrics": fixed_run
        })
    else:
        comparison_matrix.append({
            "pipeline_type": "fixed",
            "name": "Fixed Multi-Agent Pipeline",
            "description": "Executes all registered specialist agents unconditionally on every query.",
            "status": "not_yet_run",
            "note": "Baseline experiment pending batch test execution on test splits.",
            "theoretical_mean_agents": 7.0,
            "theoretical_latency_ms": 380.0
        })

    # 2. Rule-Based Pipeline (Rigid keyword matching heuristics)
    rule_run = next((r.to_dict() for r in db_runs if r.pipeline_type == "rule_based"), None)
    if rule_run:
        comparison_matrix.append({
            "pipeline_type": "rule_based",
            "name": "Heuristic Rule-Based Pipeline",
            "description": "Regex and keyword routing dictionary mapping tokens to fixed agent sub-graphs.",
            "status": "completed",
            "metrics": rule_run
        })
    else:
        comparison_matrix.append({
            "pipeline_type": "rule_based",
            "name": "Heuristic Rule-Based Pipeline",
            "description": "Regex and keyword routing dictionary mapping tokens to fixed agent sub-graphs.",
            "status": "not_yet_run",
            "note": "Rule baseline ablation experiment not yet run (pending test batch execution).",
            "theoretical_mean_agents": 3.2,
            "theoretical_latency_ms": 210.0
        })

    # 3. VisionIQ Adaptive Pipeline (Equation 1 Cost-Aware Router)
    adaptive_run = next((r.to_dict() for r in db_runs if r.pipeline_type == "adaptive"), None)
    if adaptive_run:
        comparison_matrix.append({
            "pipeline_type": "adaptive",
            "name": "VisionIQ Adaptive Cost-Aware Pipeline (Eq. 1)",
            "description": "Dynamic multi-criteria utility routing balancing relevance, evidence, cost, and latency.",
            "status": "completed",
            "metrics": adaptive_run
        })
    else:
        comparison_matrix.append({
            "pipeline_type": "adaptive",
            "name": "VisionIQ Adaptive Cost-Aware Pipeline (Eq. 1)",
            "description": "Dynamic multi-criteria utility routing balancing relevance, evidence, cost, and latency.",
            "status": "not_yet_run",
            "note": "Adaptive benchmark run not yet initiated."
        })

    return {
        "status": "success",
        "total_experiments_recorded": len(db_runs),
        "executed_runs": [r.to_dict() for r in db_runs],
        "pipeline_comparisons": comparison_matrix,
        "routing_strategies": load_routing_comparison_data(),
        "active_models": [m.to_dict() for m in models],
        "datasets": [
            {
                "name": "Construction Site Safety (Roboflow PPE)",
                "task": "PPE Bounding Box Localization",
                "version": "v1.0.0-YOLO",
                "license": "CC BY 4.0",
                "classes": ["person", "hard_hat", "safety_vest"],
                "status": "verified"
            },
            {
                "name": "NEU Surface Defect Database",
                "task": "Steel Surface Defect Classification",
                "version": "v1.0.0-NEU",
                "license": "Research Use Only",
                "classes": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
                "status": "verified"
            },
            {
                "name": "MVTec Anomaly Detection (MVTec-AD)",
                "task": "Pixel-Level Continuous Anomaly Localization",
                "version": "v1.0.0-Benchmark",
                "license": "CC BY-NC-SA 4.0",
                "classes": ["nominal", "anomaly"],
                "status": "verified"
            }
        ]
    }
