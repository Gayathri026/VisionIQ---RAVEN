# VisionIQ Phase 3.1: Evidence Engine Core Documentation

## 1. Executive Summary & Purpose

Phase 3.1 implements the **Evidence Engine Core** for VisionIQ, establishing an empirical, mathematically grounded evidence ledger and multi-factor evidence deficit formulation. 

It addresses key limitations of static heuristic routing:
1. Replaces generic capability scalars with **grounded, verifiable evidence tokens** tied to real model outputs.
2. Decouples **completeness**, **epistemic reliability**, **cross-agent contradiction**, and **unavailability**, preventing double-counting.
3. Implements an **immutable provenance ledger** that preserves contradictory observations instead of blindly overwriting them.
4. Enforces calibration discipline by separating raw model confidences, Platt/temperature-calibrated probabilities, and rule-based heuristic satisfaction scores.

---

## 2. Evidence Token Catalog

The core engine registers 8 initial tokens spanning perceptual and recommendation specialists:

| Token ID | Task Name | Domain | Primary Agent | Prerequisites | Calibration Type | Actual Source Fields |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `ET-OBJ-PERSON-01` | Personnel Localization | Perception | `object` | None | Platt/Temp Calibrated | `object_result.objects`, `findings.bbox`, `findings.detection_confidence` |
| `ET-PPE-CRANIAL-02` | Cranial Protection Compliance | Safety | `ppe` | `object` | Rule-based Heuristic | `ppe_result.persons.has_helmet`, `helmet_confidence`, `is_occluded` |
| `ET-PPE-TORSO-03` | Torso Protection Compliance | Safety | `ppe` | `object` | Rule-based Heuristic | `ppe_result.persons.has_vest`, `vest_confidence`, `is_occluded` |
| `ET-DFT-LOCALIZATION-01`| Surface Flaw Localization | Quality | `defect` | None | Morphometric Heuristic | `defect_result.findings.bbox`, `anomaly_score` |
| `ET-DFT-CLASSIFICATION-02`| Defect Morphology Class | Quality | `defect` | None | Temperature Calibrated | `defect_result.findings.is_confirmed_defect`, `detection_confidence` |
| `ET-DFT-PROCEDURE-03` | NDT Testing Specification | Cognition | `recommendation` | `rca` | String Pattern Heuristic | `findings.verification_procedure`, `recommendation_result` |
| `ET-OCR-TRANSCRIBE-01` | Alphanumeric Transcription | Perception | `ocr` | None | Temperature Calibrated | `ocr_result.detected_text`, `ocr_result.text_regions.confidence` |
| `ET-SCN-ENVIRONMENT-01`| Ambient Illumination Check | Context | `scene` | None | Photometric Heuristic | `scene_result.lighting`, `scene_result.direct_observations` |

---

## 3. Mathematical Formulation of Decoupled Evidence Dimensions

### 3.1 Observable Token Satisfaction $\phi(k) \in [0.0, 1.0]$
Every token evaluation produces a bounded score $\phi(k) \in [0.0, 1.0]$. Missing evidence strictly receives $\phi(k) = 0.0$. Negative queries (e.g., verifying an automated hazard cell is unoccupied) evaluate to $\phi(k) = 1.0$ when zero persons are detected.

### 3.2 Decoupled Dimensions
Let $\mathcal{K}_{\text{req}}$ be the set of mandatory tokens for the classified query intent.
Let $\mathcal{K}_{\text{req, app}}$ be the applicable required tokens, excluding tokens whose precondition is demonstrably not applicable to the scene (e.g., PPE cranial and torso tokens when zero workers are confirmed present):
$$\mathcal{K}_{\text{req, app}} = \{k \in \mathcal{K}_{\text{req}} \mid \text{status}(k) \neq \mathbf{NOT\_APPLICABLE}\}$$

1. **Completeness ($C_{\text{ev}} \in [0.0, 1.0]$)**:
   Evaluated across **all applicable** required tokens. Tokens that are `UNAVAILABLE` (agent failed) or `MISSING` (agent unexecuted) have $\phi(k) = 0.0$:
   $$C_{\text{ev}} = \frac{1}{|\mathcal{K}_{\text{req, app}}|} \sum_{k \in \mathcal{K}_{\text{req, app}}} \phi(k) \quad (\text{if } |\mathcal{K}_{\text{req, app}}| > 0 \text{ else } 1.0)$$
   *Critical Correction*: Earlier formulations computed $C_{\text{ev}}$ only over available tokens, which caused a severe paradox: when the PPE specialist failed, removing its tokens from the completeness denominator artificially inflated completeness to $86.5\%$ and yielded a falsely low deficit ($D = 0.1939$). In the revised formulation, missing or unavailable mandatory evidence directly drives $C_{\text{ev}}$ downward ($C_{\text{ev}} = 0.4323, D \approx 0.3885 \sim 0.41$).

2. **Unavailability ($U_{\text{unavail}} \in [0.0, 1.0]$)**:
   Proportion of applicable required tokens whose providing agent failed (exception, crash, timeout) or requested an ungroundable modality:
   $$U_{\text{unavail}} = \frac{\sum_{k \in \mathcal{K}_{\text{req, app}}} \mathbb{I}(\text{status}(k) \in \{\mathbf{UNAVAILABLE}, \mathbf{UNSUPPORTED}\})}{|\mathcal{K}_{\text{req, app}}|} \quad (\text{if } |\mathcal{K}_{\text{req, app}}| > 0 \text{ else } 0.0)$$

   *Decoupling Semantics*:
   - A `MISSING` token (agent not yet executed in sequential routing) contributes to the completeness deficit $(1 - C_{\text{ev}})$, but has $U_{\text{unavail}} = 0.0$ because the evidence can still be acquired by scheduling the agent in the next step.
   - An `UNAVAILABLE` token (agent failed) contributes to $(1 - C_{\text{ev}})$ *and* incurs the unrecoverability penalty $U_{\text{unavail}}$, reflecting the severe risk that required evidence cannot be verified at all.

3. **Epistemic Reliability ($R_{\text{ev}} \in [0.0, 1.0]$)**:
   Calibrated confidence penalized by spatial bounding box uncertainty across observed ledger entries $E$:
   $$R_{\text{ev}} = \frac{1}{|E|} \sum_{e \in E} \left[ p_{\text{cal}}(e) \cdot (1.0 - u_{\text{spatial}}(e)) \right]$$
   where $p_{\text{cal}}(e)$ uses calibrated probabilities for neural detectors (`confidence_type = "calibrated_probability"`) and rule scores for heuristics (`confidence_type = "heuristic_score"`).

4. **Cross-Agent Contradiction Conflict Score ($S_{\text{conflict}} \in [0.0, 1.0]$)**:
   Bounded sum of severity scores across spatial intersections (e.g. defect crack vs. OCR text inscription with IoU $\ge 0.25$) and entity count discrepancies between independent specialists:
   $$S_{\text{conflict}} = \min\left(1.0, \sum_{c \in \text{Conflicts}} \text{severity}(c)\right)$$
   *(Renamed from $H_{\text{conflict}}$ to explicitly clarify that it is a bounded discrepancy severity metric rather than an information-theoretic Shannon entropy).*

### 3.3 Composite Deficit Function $D \in [0.0, 1.0]$
$$D = \text{clip}\left( w_C \cdot (1.0 - C_{\text{ev}}) + w_R \cdot (1.0 - R_{\text{ev}}) + w_S \cdot S_{\text{conflict}} + w_U \cdot U_{\text{unavail}}, \ 0.0, \ 1.0 \right)$$
where default initial weights are $w_C = 0.50$, $w_R = 0.20$, $w_S = 0.20$, $w_U = 0.10$. Weights are normalized prior to evaluation.

---

## 4. Provenance-Preserving Evidence Ledger & Deduplication

1. **Spatial Deduplication**:
   When an agent detects an entity matching an existing ledger record (same agent, same semantic label, $\text{IoU} \ge 0.60$), the ledger merges the record, updates confidence to the peak observation, and increments `duplicate_observations`. This prevents repetitive token inflation.
2. **Contradiction Preservation**:
   When two distinct agents report spatially overlapping findings with semantic clash ($\text{IoU} \ge 0.25$), both findings are retained in the ledger with `has_conflict = True`, and a `ConflictRecord` is created.
3. **Semantic Status Categorization**:
   The engine enforces explicit, non-overlapping `TokenStatus` values:
   - `SATISFIED`: Evidence requirement verified with high confidence.
   - `INSUFFICIENT_EVIDENCE`: Occluded, low-resolution crop, or borderline confidence (formerly `UNCERTAIN`).
   - `CONFIRMED_ABSENT`: Verified absence of entity (e.g. 0 workers detected, or bare head confirmed).
   - `NOT_APPLICABLE`: Requirement not applicable to scene (e.g. PPE tokens when 0 workers are present).
   - `CONTRADICTORY`: Spatial or semantic contradiction between specialist observations.
   - `MISSING`: Unexecuted agent; evidence can still be acquired in a subsequent step.
   - `UNAVAILABLE`: Providing agent failed during execution (unrecoverable).
   - `UNSUPPORTED`: Modality cannot be grounded in existing specialist repertoires.

---

## 5. Calibration Discipline & Disclaimers

1. **Calibrated Confidences**:
   Applied to neural detectors (`YOLOv11n`, `IndustrialDefectNet`, `EasyOCR`) using Platt/temperature sigmoid transformation:
   $$\sigma_{\text{cal}}(s) = \frac{1}{1 + \exp\left(-\frac{\text{logit}(s) - 0.05}{1.35}\right)}$$
2. **Heuristic Metrics**:
   Mask color fractions (hardhat yellow/white, vest neon orange), NDT regex text matching, and scene luminance thresholds are explicitly marked `is_heuristic = True`. They are **not** claimed to be empirically calibrated probabilities.
3. **Zero Assumption on Worker Population**:
   The engine does not assume an a priori known worker population. Cranial and torso coverage is normalized strictly by the verified number of localized personnel $N_p$. If $N_p = 0$, tokens are marked ungrounded.

---

## 6. Unsupported Modalities & Limitations

The following modalities cannot be grounded in existing specialist outputs and are rejected as `UNSUPPORTED`:
- Acoustic emission sensor telemetry
- Micro-CT internal volumetric scans
- Real-time RFID badge turnstile pings

Phase 3.1 does not modify routing decisions or LangGraph execution. It serves as an isolated, independently testable evidence engine for downstream sequential planning.
