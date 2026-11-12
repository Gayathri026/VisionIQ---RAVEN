"""
VisionIQ Automated Test Suite: Phase 1 Dataset Benchmarks & Integrity
======================================================================
Tests:
1. Manifest parsing & schema validation
2. Image loading & dimension verification
3. Annotation parsing (Pascal VOC XML and YOLO format)
4. Bounding box validity (coordinates, normalized ranges, non-degeneracy)
5. Duplicate detection (cryptographic & perceptual)
6. Split reproducibility (deterministic seed invariance)
7. Train/val/test leakage prevention (zero identity or hash intersections)
8. Missing file & corruption error handling
9. Dataset checksum verification (SHA-256 registry validation)
10. Task-specific benchmark evaluators (Detection, Classification, OCR)
11. Metric domain isolation enforcement
"""

import os
import sys
import json
import hashlib
import tempfile
import pytest
from pathlib import Path
import cv2
import numpy as np

# Locate repository root by checking for 'data' directory
current = Path(__file__).resolve().parent
while current.parent != current and not (current / "data").exists():
    current = current.parent
PROJECT_ROOT = current
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data"

from visioniq.ml.evaluation.evaluators import (
    ObjectDetectionEvaluator,
    ClassificationEvaluator,
    OCREvaluator,
    compute_iou,
    levenshtein_distance,
    assert_metric_isolation
)
from scripts.generate_splits import (
    generate_neu_det_splits,
    generate_shwd_splits,
    generate_ocr_splits,
    verify_leakage
)
from scripts.verify_dataset_integrity import (
    audit_neu_det,
    audit_shwd,
    audit_ocr,
    compute_dhash,
    hamming_distance
)

# ------------------------------------------------------------------------------
# 1. Manifest Parsing & Schema Validation
# ------------------------------------------------------------------------------
def test_manifest_parsing_and_schema():
    master_manifest_path = DATA_ROOT / "manifests" / "master_manifest.json"
    assert master_manifest_path.exists(), "master_manifest.json must exist"

    with open(master_manifest_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    assert "manifest_version" in master
    assert "summary" in master
    assert master["summary"]["total_images"] == 700

    datasets = master["datasets"]
    assert "neu_det" in datasets
    assert "shwd" in datasets
    assert "ocr_iiit5k" in datasets

    assert len(datasets["neu_det"]) == 300
    assert len(datasets["shwd"]) == 300
    assert len(datasets["ocr_iiit5k"]) == 100

    # Validate schema fields on sample items
    for ds_key, ds_items in datasets.items():
        sample_entry = next(iter(ds_items.values()))
        assert "image_id" in sample_entry
        assert "dataset_name" in sample_entry
        assert "file_path" in sample_entry
        assert "label_path" in sample_entry
        assert "image_dimensions" in sample_entry
        assert "split" in sample_entry
        assert sample_entry["split"] in ["train", "val", "test"]
        assert "sha256" in sample_entry
        assert len(sample_entry["sha256"]) == 64
        assert "annotation_availability" in sample_entry


# ------------------------------------------------------------------------------
# 2. Image Loading & Dimensions Check
# ------------------------------------------------------------------------------
def test_image_loading_and_dimensions():
    with open(DATA_ROOT / "manifests" / "master_manifest.json", "r", encoding="utf-8") as f:
        master = json.load(f)

    # Check 10 images from each dataset
    for ds_name in ["neu_det", "shwd", "ocr_iiit5k"]:
        entries = list(master["datasets"][ds_name].values())[:10]
        for entry in entries:
            img_path = PROJECT_ROOT / entry["file_path"]
            assert img_path.exists(), f"Image file not found: {img_path}"
            assert img_path.stat().st_size > 0, f"Zero-byte image file: {img_path}"

            img = cv2.imread(str(img_path))
            assert img is not None, f"OpenCV failed to decode image: {img_path}"
            h, w = img.shape[:2]
            assert h == entry["image_dimensions"]["height"]
            assert w == entry["image_dimensions"]["width"]


# ------------------------------------------------------------------------------
# 3. Annotation Parsing (Pascal VOC XML and YOLO Format)
# ------------------------------------------------------------------------------
def test_annotation_parsing_voc_and_yolo():
    with open(DATA_ROOT / "manifests" / "neu_det_manifest.json", "r", encoding="utf-8") as f:
        neu_manifest = json.load(f)

    # Test NEU-DET (VOC XML and YOLO format)
    sample_neu = next(iter(neu_manifest.values()))
    raw_xml_path = PROJECT_ROOT / sample_neu["raw_annotation_path"]
    proc_yolo_path = PROJECT_ROOT / sample_neu["label_path"]

    assert raw_xml_path.exists()
    assert proc_yolo_path.exists()

    # Verify YOLO line format
    with open(proc_yolo_path, "r", encoding="utf-8") as yf:
        lines = yf.readlines()
    if lines:
        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 5
            cid = int(parts[0])
            assert 0 <= cid < 6

    # Test SHWD YOLO format
    with open(DATA_ROOT / "manifests" / "shwd_manifest.json", "r", encoding="utf-8") as f:
        shwd_manifest = json.load(f)
    sample_shwd = next(iter(shwd_manifest.values()))
    shwd_lbl_path = PROJECT_ROOT / sample_shwd["label_path"]
    assert shwd_lbl_path.exists()
    with open(shwd_lbl_path, "r", encoding="utf-8") as sf:
        for line in sf:
            parts = line.strip().split()
            if parts:
                assert len(parts) == 5
                assert int(parts[0]) in [0, 1]


# ------------------------------------------------------------------------------
# 4. Bounding Box Validity & Non-degeneracy
# ------------------------------------------------------------------------------
def test_bounding_box_validity():
    with open(DATA_ROOT / "manifests" / "master_manifest.json", "r", encoding="utf-8") as f:
        master = json.load(f)

    for ds_name in ["neu_det", "shwd"]:
        for item in master["datasets"][ds_name].values():
            w = item["image_dimensions"]["width"]
            h = item["image_dimensions"]["height"]
            for obj in item.get("objects", []):
                # Check xyxy
                xmin, ymin, xmax, ymax = obj["bbox_xyxy"]
                assert xmin < xmax, f"Degenerate width in {item['image_id']}: {xmin} >= {xmax}"
                assert ymin < ymax, f"Degenerate height in {item['image_id']}: {ymin} >= {ymax}"
                assert xmin >= 0.0, f"Negative xmin in {item['image_id']}"
                assert ymin >= 0.0, f"Negative ymin in {item['image_id']}"

                # Check normalized YOLO bounds
                xc, yc, bw, bh = obj["bbox_yolo_norm"]
                assert 0.0 <= xc <= 1.0
                assert 0.0 <= yc <= 1.0
                assert 0.0 < bw <= 1.0
                assert 0.0 < bh <= 1.0


# ------------------------------------------------------------------------------
# 5. Duplicate Detection
# ------------------------------------------------------------------------------
def test_duplicate_detection():
    with open(DATA_ROOT / "manifests" / "master_manifest.json", "r", encoding="utf-8") as f:
        master = json.load(f)

    # Check zero duplicates in master dataset
    seen_hashes = {}
    for ds_name, ds_items in master["datasets"].items():
        for item in ds_items.values():
            sha = item["sha256"]
            assert sha not in seen_hashes, f"Duplicate image hash found in {ds_name}: {item['image_id']} == {seen_hashes[sha]}"
            seen_hashes[sha] = item["image_id"]

    # Verify synthetic duplicate detector
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img1 = tmp_path / "img1.png"
        img2 = tmp_path / "img2.png"
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img1), blank)
        cv2.imwrite(str(img2), blank)

        h1 = hashlib.sha256(img1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(img2.read_bytes()).hexdigest()
        assert h1 == h2, "Synthetic duplicate must be detected"


# ------------------------------------------------------------------------------
# 6. Split Reproducibility (Deterministic Seed Check)
# ------------------------------------------------------------------------------
def test_split_reproducibility():
    with open(DATA_ROOT / "splits" / "master_splits.json", "r", encoding="utf-8") as f:
        saved_splits = json.load(f)

    # Re-run split generation
    neu_rerun = generate_neu_det_splits(DATA_ROOT)
    shwd_rerun = generate_shwd_splits(DATA_ROOT)
    ocr_rerun = generate_ocr_splits(DATA_ROOT)

    assert neu_rerun["splits"]["train"] == saved_splits["neu_det"]["splits"]["train"]
    assert neu_rerun["splits"]["val"] == saved_splits["neu_det"]["splits"]["val"]
    assert neu_rerun["splits"]["test"] == saved_splits["neu_det"]["splits"]["test"]

    assert shwd_rerun["splits"]["train"] == saved_splits["shwd"]["splits"]["train"]
    assert shwd_rerun["splits"]["test"] == saved_splits["shwd"]["splits"]["test"]

    assert ocr_rerun["splits"]["train"] == saved_splits["ocr_iiit5k"]["splits"]["train"]
    assert ocr_rerun["splits"]["test"] == saved_splits["ocr_iiit5k"]["splits"]["test"]


# ------------------------------------------------------------------------------
# 7. Train / Val / Test Leakage Checks
# ------------------------------------------------------------------------------
def test_train_test_leakage_prevention():
    with open(DATA_ROOT / "splits" / "master_splits.json", "r", encoding="utf-8") as f:
        master_splits = json.load(f)

    for ds_name, ext, subpath in [
        ("neu_det", ".jpg", "raw/neu_det/images"),
        ("shwd", ".jpg", "raw/shwd/images"),
        ("ocr_iiit5k", ".png", "raw/ocr/images")
    ]:
        split_data = master_splits[ds_name]
        leak_res = verify_leakage(split_data, DATA_ROOT / subpath, ext)
        assert leak_res["leak_free"] is True
        assert len(leak_res["id_overlaps"]["train_test"]) == 0
        assert len(leak_res["hash_overlaps"]["train_test"]) == 0
        assert len(leak_res["id_overlaps"]["train_val"]) == 0


# ------------------------------------------------------------------------------
# 8. Missing File & Corruption Handling
# ------------------------------------------------------------------------------
def test_missing_file_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_dir = Path(tmpdir)
        # Empty directory audit must not crash
        neu_audit = audit_neu_det(dummy_dir)
        assert neu_audit["total_images"] == 0
        assert neu_audit["valid_images"] == 0

        shwd_audit = audit_shwd(dummy_dir)
        assert shwd_audit["total_images"] == 0

        ocr_audit = audit_ocr(dummy_dir)
        assert ocr_audit["total_images"] == 0


# ------------------------------------------------------------------------------
# 9. Dataset Checksum Verification
# ------------------------------------------------------------------------------
def test_dataset_checksum_verification():
    with open(DATA_ROOT / "manifests" / "master_manifest.json", "r", encoding="utf-8") as f:
        master = json.load(f)

    # Verify SHA-256 for 5 random images from each dataset against disk
    for ds_name in ["neu_det", "shwd", "ocr_iiit5k"]:
        sample_entries = list(master["datasets"][ds_name].values())[:5]
        for entry in sample_entries:
            file_on_disk = PROJECT_ROOT / entry["file_path"]
            assert file_on_disk.exists()
            computed_sha = hashlib.sha256(file_on_disk.read_bytes()).hexdigest()
            assert computed_sha == entry["sha256"], f"Checksum mismatch for {entry['image_id']}"


# ------------------------------------------------------------------------------
# 10. Benchmark Evaluators (Detection, Classification, OCR)
# ------------------------------------------------------------------------------
def test_evaluators_detection_precision_recall():
    classes = ["hat", "person"]
    evaluator = ObjectDetectionEvaluator(classes, iou_threshold=0.50)

    gt = [[
        {"class_id": 0, "bbox": [10, 10, 50, 50]},
        {"class_id": 1, "bbox": [60, 60, 100, 100]}
    ]]
    pred = [[
        {"class_id": 0, "confidence": 0.95, "bbox": [12, 12, 52, 52]},  # IoU > 0.50 -> TP
        {"class_id": 1, "confidence": 0.80, "bbox": [150, 150, 180, 180]}  # FP (FN for GT 1)
    ]]

    res = evaluator.evaluate_detections(gt, pred)
    assert res["class_metrics"]["hat"]["true_positives"] == 1
    assert res["class_metrics"]["hat"]["false_positives"] == 0
    assert res["class_metrics"]["person"]["true_positives"] == 0
    assert res["class_metrics"]["person"]["false_positives"] == 1
    assert res["class_metrics"]["person"]["false_negatives"] == 1


def test_evaluators_classification():
    classes = ["crazing", "inclusion", "patches"]
    evaluator = ClassificationEvaluator(classes)

    y_true = ["crazing", "inclusion", "patches", "crazing"]
    y_pred = ["crazing", "inclusion", "crazing", "crazing"]

    res = evaluator.evaluate(y_true, y_pred)
    assert res["accuracy"] == 0.75
    assert res["class_metrics"]["crazing"]["recall"] == 1.0


def test_evaluators_ocr():
    evaluator = OCREvaluator()
    y_true = ["VISIONIQ", "SAFETY"]
    y_pred = ["VISIONIQ", "SAFTY"]  # 1 deletion in second word

    res = evaluator.evaluate(y_true, y_pred)
    assert res["total_samples"] == 2
    assert res["exact_match_accuracy"] == 0.5
    assert res["character_error_rate"] == round(1.0 / (8 + 6), 4)


# ------------------------------------------------------------------------------
# 11. Metric Domain Isolation Enforcement
# ------------------------------------------------------------------------------
def test_metric_domain_isolation():
    det_eval = {"task": "object_detection", "map_50": 0.85}
    cls_eval = {"task": "classification", "accuracy": 0.92}
    ocr_eval = {"task": "ocr", "character_error_rate": 0.05}

    # Same domain is allowed
    assert_metric_isolation(det_eval, det_eval)
    assert_metric_isolation(cls_eval, cls_eval)

    # Mixing domains must raise ValueError
    with pytest.raises(ValueError, match="Metric Domain Violation"):
        assert_metric_isolation(det_eval, ocr_eval)

    with pytest.raises(ValueError, match="Metric Domain Violation"):
        assert_metric_isolation(det_eval, cls_eval)
