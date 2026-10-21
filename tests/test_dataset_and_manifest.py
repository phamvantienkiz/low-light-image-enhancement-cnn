"""
Unit tests for split manifest integrity and PairedLowLightDataset.
"""

import json
from pathlib import Path
import pytest
import torch
from src.datasets import PairedLowLightDataset, load_dataset_from_manifest


def test_manifest_partition_integrity():
    manifest_path = Path("docs/ai/split_manifest.json")
    assert manifest_path.exists(), "Manifest file missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    val = manifest["validation_set"]
    train_initial = manifest["train_initial_250"]
    quarantine = manifest["quarantine_slices"]

    assert len(val) == 200, f"Expected 200 validation pairs, got {len(val)}"
    assert len(train_initial) == 250, f"Expected 250 initial train pairs, got {len(train_initial)}"

    # Check 6 slices
    assert len(quarantine) == 6, "Expected 6 quarantine slices"
    all_quarantine_ids = set()
    for slice_name, slice_items in quarantine.items():
        assert len(slice_items) == 50, f"Slice {slice_name} should have 50 items"
        for item in slice_items:
            assert item["id"] not in all_quarantine_ids, f"Duplicate in quarantine: {item['id']}"
            all_quarantine_ids.add(item["id"])

    assert len(all_quarantine_ids) == 300, "Total quarantined should be 300"

    # Verify zero data leakage (disjoint sets)
    val_ids = {item["id"] for item in val}
    train_ids = {item["id"] for item in train_initial}

    assert val_ids.isdisjoint(train_ids), "Data leakage between val and train!"
    assert val_ids.isdisjoint(all_quarantine_ids), "Data leakage between val and quarantine!"
    assert train_ids.isdisjoint(all_quarantine_ids), "Data leakage between train and quarantine!"

    # Total unique pairs in active pool
    total_unique = len(val_ids | train_ids | all_quarantine_ids)
    assert total_unique == 750, f"Expected 750 unique pairs, got {total_unique}"


def test_dataset_loading_sample():
    manifest_path = Path("docs/ai/split_manifest.json")
    data_root = Path(".")

    train_ds, val_ds = load_dataset_from_manifest(
        manifest_path=manifest_path,
        data_root=data_root,
        mode="initial_250",
        patch_size=128,
    )

    assert len(train_ds) == 250
    assert len(val_ds) == 200

    low_t, high_t, item_id = train_ds[0]
    assert isinstance(low_t, torch.Tensor)
    assert isinstance(high_t, torch.Tensor)
    assert low_t.shape == (3, 128, 128)
    assert high_t.shape == (3, 128, 128)
    assert low_t.min() >= 0.0 and low_t.max() <= 1.0
