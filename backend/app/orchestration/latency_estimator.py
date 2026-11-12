"""
VisionIQ Historical Latency Estimator (Phase 2)
Computes Exponential Moving Average (EMA) of historical successful execution runtimes,
applies normalization relative to benchmark priors or linear scale,
and provides transparent fallback to static priors when history is unavailable or insufficient.
"""

import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import logger


@dataclass
class LatencyEstimate:
    """Detailed explainable record for an agent's latency estimation."""
    agent_name: str
    static_prior: float
    historical_ema_ms: Optional[float]
    normalized_latency: float
    observation_count: int
    fallback_used: bool
    selected_latency: float
    hardware_device: Optional[str]
    raw_measurements: List[float]
    model_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "static_prior": self.static_prior,
            "historical_ema_ms": self.historical_ema_ms,
            "normalized_latency": self.normalized_latency,
            "observation_count": self.observation_count,
            "fallback_used": self.fallback_used,
            "selected_latency": self.selected_latency,
            "hardware_device": self.hardware_device,
            "model_version": self.model_version,
            "raw_measurements": self.raw_measurements
        }


def compute_ema(measurements: List[float], alpha: float = 0.30) -> Optional[float]:
    """
    Computes standard recursive Exponential Moving Average (EMA) over chronological observations.
    
    Formula:
        S_1 = x_1
        S_k = alpha * x_k + (1 - alpha) * S_{k-1}, for k = 2, ..., N
        
    Args:
        measurements: Chronological list of positive execution latencies in ms [oldest, ..., newest]
        alpha: Smoothing factor in (0.0, 1.0]
        
    Returns:
        Final EMA estimate S_N in milliseconds, or None if no valid measurements exist.
    """
    if not measurements:
        return None

    # Filter out invalid, non-positive, NaN, or infinite measurements
    valid = [
        float(x) for x in measurements
        if x is not None and not math.isnan(x) and not math.isinf(x) and x > 0.0
    ]
    if not valid:
        return None

    s = valid[0]
    for x in valid[1:]:
        s = (alpha * x) + ((1.0 - alpha) * s)

    return round(float(s), 2)


def normalize_latency(
    ema_ms: float,
    agent_name: str,
    prior: float,
    method: str = "anchored",
    max_scale_ms: float = 5000.0,
    bench_latencies: Optional[Dict[str, float]] = None
) -> float:
    """
    Normalizes historical EMA latency in milliseconds into L(a) in [0.01, 1.0] for Equation (1).
    
    Methods:
        - "anchored": Scale proportional to calibrated empirical benchmark reference:
          L(a) = clip(prior * (ema_ms / tau_bench(a)), 0.01, 1.0)
        - "linear": Scale relative to a reference maximum latency:
          L(a) = clip(ema_ms / max_scale_ms, 0.01, 1.0)
    """
    if ema_ms <= 0.0:
        return prior

    if method == "linear":
        norm = ema_ms / max(1.0, max_scale_ms)
    else:  # anchored (default)
        ref_dict = bench_latencies or settings.ROUTING_BENCHMARK_LATENCIES_MS
        tau_bench = ref_dict.get(agent_name, 250.0)
        norm = prior * (ema_ms / max(1.0, tau_bench))

    return round(float(min(1.0, max(0.01, norm))), 4)


class HistoricalLatencyEstimator:
    """
    Manages historical execution latency retrieval from the persistence layer,
    calculates rolling EMA estimates, and provides safe fallback to static priors.
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        ema_alpha: Optional[float] = None,
        min_observations: Optional[int] = None,
        normalization_method: Optional[str] = None,
        max_scale_ms: Optional[float] = None,
        bench_latencies: Optional[Dict[str, float]] = None,
        hardware_device: Optional[str] = None,
        agent_model_versions: Optional[Dict[str, str]] = None
    ):
        self.window_size = window_size or settings.ROUTING_LATENCY_WINDOW_SIZE
        self.ema_alpha = ema_alpha if ema_alpha is not None else settings.ROUTING_LATENCY_EMA_ALPHA
        self.min_observations = min_observations if min_observations is not None else settings.ROUTING_LATENCY_MIN_OBSERVATIONS
        self.normalization_method = normalization_method or settings.ROUTING_LATENCY_NORMALIZATION_METHOD
        self.max_scale_ms = max_scale_ms or settings.ROUTING_LATENCY_MAX_SCALE_MS
        self.bench_latencies = bench_latencies or settings.ROUTING_BENCHMARK_LATENCIES_MS
        self.hardware_device = hardware_device or settings.HARDWARE_DEVICE_CONTEXT
        self.agent_model_versions = agent_model_versions if agent_model_versions is not None else getattr(settings, "AGENT_MODEL_VERSIONS", {})

    def estimate_agent_latency(
        self,
        agent_name: str,
        historical_measurements: Optional[List[float]] = None,
        db_session: Optional[Session] = None,
        hardware_device: Optional[str] = None,
        model_version: Optional[str] = None,
        allow_legacy_null: bool = True
    ) -> LatencyEstimate:
        """
        Estimates the latency for a single agent.
        
        If historical_measurements is provided, uses that sequence.
        Otherwise, if db_session is provided, queries the database with hardware and model-version isolation.
        Falls back to static prior if history is missing or below min_observations.
        """
        static_prior = settings.AGENT_LATENCIES.get(agent_name, 0.20)
        device = hardware_device or self.hardware_device
        version = model_version or self.agent_model_versions.get(agent_name)

        measurements: List[float] = []
        if historical_measurements is not None:
            measurements = [
                m for m in historical_measurements
                if m is not None and not math.isnan(m) and not math.isinf(m) and m > 0.0
            ]
        elif db_session is not None:
            try:
                from backend.app.database.repository import AnalysisRepository
                repo = AnalysisRepository(db_session)
                measurements = repo.get_historical_agent_latencies(
                    agent_name=agent_name,
                    window_size=self.window_size,
                    hardware_device=device,
                    model_version=version,
                    allow_legacy_null=allow_legacy_null
                )
            except Exception as e:
                logger.warning(f"Error querying historical latency for agent {agent_name}: {e}")
                measurements = []

        # Enforce rolling window slice
        if len(measurements) > self.window_size:
            measurements = measurements[-self.window_size:]

        obs_count = len(measurements)

        # Fallback condition: insufficient observations
        if obs_count < self.min_observations:
            return LatencyEstimate(
                agent_name=agent_name,
                static_prior=round(static_prior, 4),
                historical_ema_ms=None,
                normalized_latency=round(static_prior, 4),
                observation_count=obs_count,
                fallback_used=True,
                selected_latency=round(static_prior, 4),
                hardware_device=device,
                raw_measurements=measurements,
                model_version=version
            )

        # Compute EMA
        ema = compute_ema(measurements, self.ema_alpha)
        if ema is None:
            return LatencyEstimate(
                agent_name=agent_name,
                static_prior=round(static_prior, 4),
                historical_ema_ms=None,
                normalized_latency=round(static_prior, 4),
                observation_count=0,
                fallback_used=True,
                selected_latency=round(static_prior, 4),
                hardware_device=device,
                raw_measurements=[],
                model_version=version
            )

        norm_lat = normalize_latency(
            ema_ms=ema,
            agent_name=agent_name,
            prior=static_prior,
            method=self.normalization_method,
            max_scale_ms=self.max_scale_ms,
            bench_latencies=self.bench_latencies
        )

        return LatencyEstimate(
            agent_name=agent_name,
            static_prior=round(static_prior, 4),
            historical_ema_ms=ema,
            normalized_latency=norm_lat,
            observation_count=obs_count,
            fallback_used=False,
            selected_latency=norm_lat,
            hardware_device=device,
            raw_measurements=measurements,
            model_version=version
        )

    def get_all_agent_estimates(
        self,
        db_session: Optional[Session] = None,
        hardware_device: Optional[str] = None,
        agent_model_versions: Optional[Dict[str, str]] = None,
        allow_legacy_null: bool = True
    ) -> Dict[str, LatencyEstimate]:
        """
        Computes latency estimates for all candidate agents under Equation (1).
        """
        from backend.app.orchestration.router import AVAILABLE_AGENTS
        versions = agent_model_versions or self.agent_model_versions or {}
        estimates: Dict[str, LatencyEstimate] = {}
        for agent in AVAILABLE_AGENTS:
            estimates[agent] = self.estimate_agent_latency(
                agent_name=agent,
                db_session=db_session,
                hardware_device=hardware_device,
                model_version=versions.get(agent),
                allow_legacy_null=allow_legacy_null
            )
        return estimates
