"""Download only the LIBERO demo files this project trains on.

Why not LIBERO's own benchmark_scripts/download_libero_datasets.py:
  - it pulls a whole suite (libero_spatial alone is ~5.9GB),
  - it passes force_download=True, so it re-downloads on every call,
  - it prompts on stdin, which EOFs on a headless training box.

This fetches one task's HDF5 (~486MB for libero_spatial task 0), skips it if
already on disk, and is safe to re-run -- a setup that hits Beekeeper's 300s
timeout mid-download finishes on the next launch.

Deliberately imports nothing from `libero`: setup.sh runs BEFORE
`pip install -r requirements.txt`, and libero.libero.benchmark imports torch.
The dataset root is read straight out of LIBERO's config.yaml instead.

Usage:
    python scripts/fetch_datasets.py
    python scripts/fetch_datasets.py --pattern 'libero_spatial/*'
"""
import argparse
import os
import sys

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"   # download_utils.py:108

# Must match the task Agent trains on (agent.py: benchmark_name + task_id).
# Duplicated here because resolving it through the benchmark needs torch, which
# is not installed yet when this runs. A mismatch fails loudly in DataLoader
# with a missing-file error, not silently.
DEFAULT_PATTERN = (
    "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin"
    "_and_place_it_on_the_plate_demo.hdf5"
)

CONFIG_FILE = os.path.join(
    os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")),
    "config.yaml",
)


def dataset_root():
    """LIBERO's configured dataset dir, per libero/libero/__init__.py."""
    import yaml
    with open(CONFIG_FILE) as f:
        return dict(yaml.safe_load(f))["datasets"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default=DEFAULT_PATTERN,
                   help="HF allow_patterns glob, relative to the dataset root")
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    download_dir = dataset_root()
    os.makedirs(download_dir, exist_ok=True)

    target = os.path.join(download_dir, args.pattern)
    if "*" not in args.pattern and os.path.exists(target):
        print(f"[fetch_datasets] already present: {target}")
        return 0

    print(f"[fetch_datasets] {HF_REPO_ID} :: {args.pattern} -> {download_dir}")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=args.pattern,
    )
    print("[fetch_datasets] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
