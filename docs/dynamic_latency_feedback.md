# VisionIQ Phase 2: Dynamic Cost and Latency Feedback

## 1. Executive Summary & Research Motivation

In smart manufacturing inspection systems, static routing priors for computational latency $L(a)$ fail to reflect real runtime conditions. Hardware degradation, batch queue contention, varying image resolutions, and model execution profiles cause significant latency drift. 

Phase 2 replaces static latency priors in the VisionIQ dynamic routing mechanism with an empirical, historical **Exponential Moving Average (EMA)** of measured agent execution latencies retrieved directly from the persistence layer (`AgentExecutionTrace`).

---

## 2. Mathematical Formulation

### 2.1 Chronological Filtering, Hardware Isolation, and Model-Version Compatibility Gate
Let $\mathcal{T}(a) = \{t_1, t_2, \dots, t_M\}$ be the set of historical execution traces recorded for candidate agent $a \in \mathcal{A}$. A trace $t_i$ is admitted to the estimation sequence if and only if it satisfies the strict validity and compatibility conditions:

$$\text{Valid}(t_i) \iff \left( \text{status}(t_i) = \text{SUCCESS} \right) \land \left( \tau_i > 0 \right) \land \left( \tau_i \in \mathbb{R}_{< \infty} \right) \land \left( \text{device}(t_i) \equiv \text{device}_{\text{target}} \right) \land \text{CompatibleModel}(t_i, v_{\text{target}})$$

where $\tau_i$ denotes `actual_latency_ms`. Traces resulting from failed executions, timeouts, heuristic skips, or corrupted non-positive measurements are discarded.

#### Model-Version Isolation Policy
Historical execution times can vary drastically between model checkpoints and architectures (e.g., `YOLOv11n` vs `YOLOv11x` or `EasyOCR-1.7.2` vs `PaddleOCR`). To prevent cross-checkpoint bias, latency filtering adheres to a tiered compatibility hierarchy:
1. **Tier 1 (Exact Match Preference):** If historical records matching `model_version == target_version` exist, only exact matches are returned.
2. **Tier 2 (Legacy NULL Fallback Policy):** If zero exact matches exist (e.g., existing records created prior to database schema migration where `model_version IS NULL`), records with `model_version IS NULL` are admitted if `allow_legacy_null=True`.
3. **Tier 3 (Conflicting Version Exclusion):** Traces associated with another known model version ($v \neq v_{\text{target}}$ and $v \neq \text{NULL}$) are strictly excluded. They are never combined with target version estimates.
4. **Hardware & Version Conjunction:** Device filtering (`cpu` vs `cuda:0`) and model-version filtering are applied simultaneously in the SQL predicate.

For a rolling window of size $W \in \mathbb{Z}^+$ (configured via `ROUTING_LATENCY_WINDOW_SIZE`, default $W = 50$), the $N \le W$ most recent valid measurements are sorted chronologically from oldest to newest:

$$\mathbf{x}(a) = [x_1, x_2, \dots, x_N], \quad x_k > 0$$

### 2.2 Recursive Exponential Moving Average (EMA)
Given a smoothing coefficient $\alpha_{\text{ema}} \in (0.0, 1.0]$ (`ROUTING_LATENCY_EMA_ALPHA`, default $\alpha_{\text{ema}} = 0.30$):

$$S_1 = x_1$$
$$S_k = \alpha_{\text{ema}} \cdot x_k + (1 - \alpha_{\text{ema}}) \cdot S_{k-1}, \quad \forall k \in \{2, 3, \dots, N\}$$

The historical execution latency estimate in milliseconds is:

$$\hat{\tau}_{\text{hist}}(a) = S_N$$

### 2.3 Latency Normalization to $L(a) \in [0.01, 1.0]$
To integrate $\hat{\tau}_{\text{hist}}(a)$ into Equation (1), milliseconds must be transformed into a normalized dimensionless penalty term $L(a) \in [0.01, 1.0]$. Two normalization methods are supported:

#### Method A: Benchmark-Anchored Proportional Normalization (`"anchored"`, Default)
Maintains equilibrium with calibrated static priors $L_{\text{prior}}(a)$ while responding proportionally to hardware acceleration or degradation:

$$L_{\text{norm}}(a) = \text{clip}\left( L_{\text{prior}}(a) \cdot \frac{\hat{\tau}_{\text{hist}}(a)}{\tau_{\text{bench}}(a)}, \ 0.01, \ 1.0 \right)$$

where $\tau_{\text{bench}}(a)$ represents the empirical baseline reference runtime (e.g. YOLOv11n = 250 ms, EasyOCR = 3000 ms, Defect = 50 ms).

#### Method B: Global Linear Normalization (`"linear"`)
Direct scaling relative to a global upper-bound latency ceiling $\tau_{\text{max}}$ (`ROUTING_LATENCY_MAX_SCALE_MS`, default 5000 ms):

$$L_{\text{norm}}(a) = \text{clip}\left( \frac{\hat{\tau}_{\text{hist}}(a)}{\tau_{\text{max}}}, \ 0.01, \ 1.0 \right)$$

---

## 3. Integration with Router Equation (1)

The VisionIQ routing score is evaluated as:

$$S(a, q) = \alpha R(a, q) + \delta E(a) - \beta C(a) - \gamma L(a)$$

where:
- $R(a, q) \in [0, 1]$: Query relevance score.
- $E(a) \in [0, 1]$: Evidence capability prior.
- $C(a) \in [0, 1]$: Computational cost prior.
- $L(a) \in [0, 1]$: Selected latency penalty:

$$L(a) = \begin{cases} 
L_{\text{norm}}(a) & \text{if } \text{DynamicFeedbackEnabled} \land (N \ge N_{\min}) \\
L_{\text{prior}}(a) & \text{otherwise (Static Fallback)}
\end{cases}$$

---

## 4. Fallback Policy

The system deterministically reverts to the static latency prior $L_{\text{prior}}(a)$ whenever:
1. `ENABLE_DYNAMIC_LATENCY_FEEDBACK` is set to `False` (Static baseline mode).
2. The agent has never been executed ($N = 0$).
3. The number of valid, compatible historical observations matching the hardware device and model version is strictly less than the minimum threshold ($N_{\text{compat}} < N_{\min}$, where $N_{\min} = 3$).
4. A new model/checkpoint version has been deployed with zero matching historical records (cold start).
5. The persistence layer encounters a transient database disconnection.

When fallback occurs, `latency_fallback_used = True` is stamped in the execution trace, ensuring total traceability.

---

## 5. Configuration Reference

| Parameter Key | Type | Default | Description |
|:---|:---:|:---:|:---|
| `ENABLE_DYNAMIC_LATENCY_FEEDBACK` | `bool` | `False` | Global switch enabling historical EMA latency in Eq. (1) |
| `ROUTING_LATENCY_WINDOW_SIZE` | `int` | `50` | Maximum rolling window size $W$ of historical measurements |
| `ROUTING_LATENCY_EMA_ALPHA` | `float` | `0.30` | Smoothing factor $\alpha_{\text{ema}} \in (0.0, 1.0]$ |
| `ROUTING_LATENCY_MIN_OBSERVATIONS`| `int` | `3` | Minimum valid observations required before activating EMA |
| `ROUTING_LATENCY_NORMALIZATION_METHOD`| `str` | `"anchored"` | Normalization method: `"anchored"` or `"linear"` |
| `ROUTING_LATENCY_MAX_SCALE_MS` | `float` | `5000.0` | Max latency ceiling for `"linear"` mode |
| `HARDWARE_DEVICE_CONTEXT` | `str` | `"cpu"` | Active hardware execution platform (`"cpu"`, `"cuda:0"`) |
| `AGENT_MODEL_VERSIONS` | `dict` | `dict(...)` | Canonical model/checkpoint versions for all 12 specialist agents |

---

## 6. Explainable Routing Trace Schema

Every routing trace produced by `compute_routing_scores()` and persisted to `AgentExecutionTrace` exposes:

```json
{
  "agent_name": "object",
  "relevance_score": 0.8500,
  "evidence_capability": 0.8500,
  "normalized_cost": 0.1500,
  "expected_latency": 0.1732,
  "final_routing_score": 0.4965,
  "threshold": 0.45,
  "decision": "SELECTED",
  "selection_reason": "Routing score 0.496 >= threshold 0.45 (dynamic EMA latency 240.5ms -> L=0.173 from 3 obs)",
  "execution_status": "PENDING",
  "actual_latency_ms": 0.0,
  "error_message": null,
  "static_latency_prior": 0.1800,
  "historical_ema_latency_ms": 240.50,
  "normalized_historical_latency": 0.1732,
  "latency_observations_count": 3,
  "latency_fallback_used": false,
  "selected_latency": 0.1732,
  "hardware_device": "cpu",
  "model_version": "YOLOv11n (Ultralytics 8.3.0)"
}
```

---

## 7. Limitations & Transition to Phase 3

1. **One-Shot Planning**: Phase 2 updates routing scores prior to agent execution based on historical distributions. It does not re-plan iteratively after each agent runs.
2. **Prior Evidence Independence**: Phase 2 does not calculate conditional marginal evidence gain $\Delta E(a \mid \mathcal{E}_{\text{current}})$.
3. **Phase 3 Scope**: The iterative sequential evidence accumulation loop and marginal gain optimization are strictly deferred to **Phase 3**.
