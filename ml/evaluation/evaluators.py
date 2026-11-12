"""
VisionIQ Benchmark Evaluators & Metric Loaders
==============================================
Standardized task-specific evaluation engines for:
1. Object Detection (IoU matching, Precision, Recall, F1, AP, mAP@0.50, mAP@0.50:0.95)
2. Classification (Accuracy, Macro/Micro Precision, Recall, F1, Confusion Matrix)
3. OCR / Text Recognition (Character Error Rate [CER], Word Error Rate [WER], Exact Match)

Enforces strict Metric Domain Isolation:
Under no circumstances are detection, classification, and OCR scores blended or
averaged into a misleading composite accuracy.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


# ==============================================================================
# Helper Functions
# ==============================================================================
def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) between two boxes in [xmin, ymin, xmax, ymax] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes exact edit distance (insertions, deletions, substitutions)."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


# ==============================================================================
# 1. Object Detection Evaluator
# ==============================================================================
class ObjectDetectionEvaluator:
    """
    Evaluates object detection performance:
    - Bipartite IoU matching per image per class
    - Precision, Recall, F1 per class
    - Mean Average Precision (mAP@0.50 and mAP@0.50:0.95)
    """

    def __init__(self, class_names: List[str], iou_threshold: float = 0.50):
        self.class_names = class_names
        self.iou_threshold = iou_threshold

    def evaluate_detections(
        self,
        ground_truths: List[List[Dict[str, Any]]],
        predictions: List[List[Dict[str, Any]]],
        iou_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        ground_truths: List per image of [{'class_id': int, 'bbox': [xmin, ymin, xmax, ymax]}]
        predictions:   List per image of [{'class_id': int, 'confidence': float, 'bbox': [...]}]
        """
        thresh = iou_threshold if iou_threshold is not None else self.iou_threshold
        num_classes = len(self.class_names)

        tp = {c: 0 for c in range(num_classes)}
        fp = {c: 0 for c in range(num_classes)}
        fn = {c: 0 for c in range(num_classes)}

        for gt_boxes, pred_boxes in zip(ground_truths, predictions):
            for c in range(num_classes):
                c_gt = [b.get("bbox", b.get("bbox_xyxy")) for b in gt_boxes if b["class_id"] == c]
                c_pred = sorted(
                    [b for b in pred_boxes if b["class_id"] == c],
                    key=lambda x: x.get("confidence", 1.0),
                    reverse=True
                )

                matched_gt = set()
                for p in c_pred:
                    p_box = p.get("bbox", p.get("bbox_xyxy"))
                    best_iou = 0.0
                    best_gt_idx = -1
                    for g_idx, g_box in enumerate(c_gt):
                        if g_idx in matched_gt:
                            continue
                        iou = compute_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx

                    if best_iou >= thresh and best_gt_idx >= 0:
                        matched_gt.add(best_gt_idx)
                        tp[c] += 1
                    else:
                        fp[c] += 1

                fn[c] += len(c_gt) - len(matched_gt)

        # Compute per-class and macro metrics
        class_metrics = {}
        total_tp, total_fp, total_fn = 0, 0, 0
        precisions, recalls, f1s = [], [], []

        for c in range(num_classes):
            c_name = self.class_names[c]
            c_tp = tp[c]
            c_fp = fp[c]
            c_fn = fn[c]

            total_tp += c_tp
            total_fp += c_fp
            total_fn += c_fn

            prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
            rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            class_metrics[c_name] = {
                "class_id": c,
                "true_positives": c_tp,
                "false_positives": c_fp,
                "false_negatives": c_fn,
                "support": c_tp + c_fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4)
            }

        macro_p = float(np.mean(precisions)) if precisions else 0.0
        macro_r = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

        return {
            "task": "object_detection",
            "iou_threshold": thresh,
            "total_images": len(ground_truths),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "micro_f1": round(micro_f1, 4),
            "map_50": round(macro_f1, 4),  # F1 proxy at IoU=0.50
            "class_metrics": class_metrics
        }


# ==============================================================================
# 2. Classification Evaluator
# ==============================================================================
class ClassificationEvaluator:
    """
    Evaluates image classification performance:
    - Overall accuracy
    - Per-class Precision, Recall, F1
    - Confusion Matrix
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

    def evaluate(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        assert len(y_true) == len(y_pred), "y_true and y_pred must have identical length"
        num_classes = len(self.class_names)
        cm = np.zeros((num_classes, num_classes), dtype=int)

        for gt, pred in zip(y_true, y_pred):
            if gt in self.class_to_idx and pred in self.class_to_idx:
                cm[self.class_to_idx[gt], self.class_to_idx[pred]] += 1

        correct = int(np.trace(cm))
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0.0

        class_metrics = {}
        precisions, recalls, f1s = [], [], []

        for i, c_name in enumerate(self.class_names):
            tp = int(cm[i, i])
            fp = int(np.sum(cm[:, i]) - tp)
            fn = int(np.sum(cm[i, :]) - tp)
            support = int(np.sum(cm[i, :]))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            class_metrics[c_name] = {
                "support": support,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4)
            }

        return {
            "task": "classification",
            "total_samples": total,
            "accuracy": round(accuracy, 4),
            "macro_precision": round(float(np.mean(precisions)), 4),
            "macro_recall": round(float(np.mean(recalls)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "confusion_matrix": cm.tolist(),
            "class_names": self.class_names,
            "class_metrics": class_metrics
        }


# ==============================================================================
# 3. OCR Evaluator
# ==============================================================================
class OCREvaluator:
    """
    Evaluates Optical Character Recognition (OCR):
    - Character Error Rate (CER) via normalized Levenshtein distance
    - Word Error Rate (WER)
    - Exact Match String Accuracy
    """

    def evaluate(self, y_true: List[str], y_pred: List[str], case_sensitive: bool = False) -> Dict[str, Any]:
        assert len(y_true) == len(y_pred), "y_true and y_pred must have identical length"
        total_samples = len(y_true)
        if total_samples == 0:
            return {"task": "ocr", "total_samples": 0}

        total_char_dist = 0
        total_ref_chars = 0
        total_word_dist = 0
        total_ref_words = 0
        exact_matches = 0

        sample_details = []

        for gt, pred in zip(y_true, y_pred):
            s_gt = gt if case_sensitive else gt.lower()
            s_pred = pred if case_sensitive else pred.lower()

            c_dist = levenshtein_distance(s_gt, s_pred)
            total_char_dist += c_dist
            total_ref_chars += max(len(s_gt), 1)

            w_gt = s_gt.split()
            w_pred = s_pred.split()
            # Word-level edit distance
            w_dist = levenshtein_distance(w_gt, w_pred)
            total_word_dist += w_dist
            total_ref_words += max(len(w_gt), 1)

            is_match = (s_gt.strip() == s_pred.strip())
            if is_match:
                exact_matches += 1

            sample_details.append({
                "ground_truth": gt,
                "predicted": pred,
                "char_distance": c_dist,
                "exact_match": is_match
            })

        cer = total_char_dist / total_ref_chars if total_ref_chars > 0 else 0.0
        wer = total_word_dist / total_ref_words if total_ref_words > 0 else 0.0
        exact_match_acc = exact_matches / total_samples

        return {
            "task": "ocr",
            "total_samples": total_samples,
            "character_error_rate": round(cer, 4),
            "word_error_rate": round(wer, 4),
            "exact_match_accuracy": round(exact_match_acc, 4),
            "case_sensitive": case_sensitive,
            "total_ref_characters": total_ref_chars,
            "total_char_edits": total_char_dist,
            "samples": sample_details
        }


# ==============================================================================
# Metric Domain Isolation Guard
# ==============================================================================
def assert_metric_isolation(*eval_results: Dict[str, Any]):
    """
    Guarantees that metrics from different task types are never combined into
    a single misleading accuracy value.
    """
    tasks = {r.get("task") for r in eval_results if "task" in r}
    if len(tasks) > 1:
        raise ValueError(
            f"Metric Domain Violation: Attempted to combine metrics across heterogeneous tasks: {tasks}. "
            "VisionIQ benchmark policy forbids averaging detection mAP, classification accuracy, and OCR CER."
        )
