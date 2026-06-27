"""Build a manifest over the tokenized MEG data.

Reads a ``manifest.csv`` from the fm_datasets (on BMRC) corpus (one row per
session) selects the training split for one dataset/task, and writes the
pipeline's ``data/manifest.pt`` (path + token length + index per session)
alongside a frozen held-out split ``data/holdout_sessions.json``.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

# fm_datasets release corpus. The v1 release is Schaefer100 Cam-CAN, tokenized by
# a tokenizer trained on Cam-CAN *all* (rest + passive/smt task). Its ``train``
# split now holds camcan rest (621) alongside the passive/smt task sessions; the
# defaults below select rest only. ``manifest.csv`` columns include: session,
# split, dataset, task, subject, age, sex, n_samples, n_tokens, n_channels,
# tokens_file (root-relative path to the token .h5).
DATA_ROOT = "/well/woolrich/projects/fm_datasets"
MANIFEST_CSV = f"{DATA_ROOT}/tokenized/v1/manifest.csv"
SFREQ = 250
# Token ids are 1-indexed: the v1 (camcan-all) tokenizer emits compacted ids
# 1..91 on the camcan/rest corpus (id 0 is the label_map's catch-all for raw
# codes that never occur here, so it's absent from train/held-out but still
# needs an embedding row). The embedding needs max_id + 1 = 92 rows; label_map
# caps all ids at 91, so vocab 92 also bounds the OOD test sets.
VOCAB_SIZE = 92


def select_sessions(manifest_csv, split, dataset, task):
    """Rows of manifest.csv matching split/dataset/task, sorted by session id."""
    with open(manifest_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    sel = [
        r
        for r in rows
        if (split is None or r["split"] == split)
        and (dataset is None or r["dataset"] == dataset)
        and (task is None or r["task"] == task)
    ]
    sel.sort(key=lambda r: r["session"])
    return sel


def scan_manifest(rows, data_root):
    """Per session, read the h5 token shape (header only) + demographics.

    The token length is taken from the actual ``.h5`` (not the CSV) so the
    downstream windowing always matches the bytes on disk.
    """

    entries, n_parcels = [], None
    for idx, r in enumerate(rows):
        path = os.path.join(data_root, r["tokens_file"])
        with h5py.File(path, "r") as h:
            shp = h["tokens"].shape  # (T, C)

        if n_parcels is None:
            n_parcels = int(shp[1])
        elif int(shp[1]) != n_parcels:
            raise ValueError(f"{path}: {shp[1]} channels, expected {n_parcels}")

        entries.append({
            "index": idx,  # session_id
            "session": r["session"],
            "path": path,
            "n_samples": int(shp[0]),
            "n_parcels": n_parcels,
            "age": float(r["age"]) if r["age"] else float("nan"),
            "sex": r["sex"],
        })

    return entries, n_parcels


def write_holdout(output_dir, n_total, frac, seed):
    """Freeze the held-out session set (deterministic) to JSON for the pipeline."""
    n_holdout = round(n_total * frac)
    sessions = sorted(np.random.RandomState(seed).choice(n_total, size=n_holdout, replace=False).tolist())
    p = Path(output_dir) / "holdout_sessions.json"
    p.write_text(json.dumps({"seed": seed, "frac": frac, "n_total": n_total, "sessions": sessions}, indent=2))
    print(f"Saved holdout -> {p} ({n_holdout}/{n_total} held out, seed={seed})")


def prepare(output_dir, manifest_csv, data_root, split, dataset, task, holdout_frac, holdout_seed):
    rows = select_sessions(manifest_csv, split, dataset, task)
    if not rows:
        raise SystemExit(f"No sessions match split={split!r} dataset={dataset!r} task={task!r}")

    print(f"Scanning {len(rows)} {dataset}/{task} [{split}] sessions under {data_root} …")
    entries, n_parcels = scan_manifest(rows, data_root)

    lens = np.array([e["n_samples"] for e in entries])
    ages = np.array([e["age"] for e in entries])
    print(f"  sessions: {len(entries)} | n_parcels: {n_parcels} | vocab: {VOCAB_SIZE}")
    print(f"  token length: min {lens.min()} median {int(np.median(lens))} "
          f"max {lens.max()}  (@{SFREQ}Hz median {np.median(lens)/SFREQ:.0f}s)")
    print(f"  age: {np.nanmin(ages):.0f}–{np.nanmax(ages):.0f} (mean {np.nanmean(ages):.0f})")

    manifest = {
        "entries": entries,
        "n_parcels": n_parcels,
        "sfreq": SFREQ,
        "vocab_size": VOCAB_SIZE,
        "dataset": f"{dataset}/{task}",
        "data_root": data_root,
        "manifest_csv": manifest_csv,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "manifest.pt"
    torch.save(manifest, str(p))
    print(f"Saved manifest -> {p} ({os.path.getsize(p)/1024:.1f} KB, {len(entries)} entries)")

    write_holdout(output_dir, len(entries), holdout_frac, holdout_seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Manifest over tokenized .h5 sessions")
    ap.add_argument("--manifest-csv", default=MANIFEST_CSV)
    ap.add_argument("--data-root", default=DATA_ROOT, help="root the manifest's tokens_file paths are relative to")
    ap.add_argument("--split", default="train")
    ap.add_argument("--dataset", default="camcan")
    ap.add_argument("--task", default="rest")
    ap.add_argument("--output-dir", default="./data")
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--holdout-seed", type=int, default=42)
    a = ap.parse_args()
    prepare(a.output_dir, a.manifest_csv, a.data_root, a.split, a.dataset, a.task, a.holdout_frac, a.holdout_seed)
