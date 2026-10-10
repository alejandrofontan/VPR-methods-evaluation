"""
VSLAM-LAB capability: VPR distance matrix (entry point of the VPR-methods-evaluation fork used by VSLAM-LAB).

For every given VSLAM-LAB sequence folder, extracts global descriptors of its rgb_0 frames (rgb.csv) with a VPR
method (--method, default megaloc) at 0/90/180/270 degree rotations and writes <sequence>/vpr-lab/:
D_<angle>.npy (faiss squared L2 between every pair of frames, database = queries = the sequence) and D.npy, their
element-wise minimum. The extraction is vpr-lab.py's (multi-GPU, batch size from free memory), run unchanged as a
subprocess. A sequence whose D.npy exists is skipped unless --overwrite.

VSLAM-LAB (Capabilities/vpr.py) resolves dataset/sequence targets into --sequence-path folders and calls this script
through the `inference` pixi task; D.npy feeds rgb_vpr (sample_vpr.py) and rgb_placecell. Model weights are fetched
by torch.hub into TORCH_HOME (the pixi tasks point it at this repo's weights/); --prefetch downloads them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_FOLDER = "vpr-lab"
DEFAULT_METHOD = "megaloc"

SCRIPT_LABEL = "\033[95m[vslamlab_vpr.py]\033[0m "


def print_info(msg: str) -> None:
    print(f"{SCRIPT_LABEL}{msg}", flush=True)


def print_warning(msg: str) -> None:
    print(f"{SCRIPT_LABEL}\033[93m[warning]\033[0m {msg}", flush=True)


def prefetch(method: str) -> None:
    """Load the method's model once, so torch.hub downloads its code and weights into TORCH_HOME."""
    sys.path.insert(0, str(REPO_ROOT))
    import parser as vpr_parser  # this repo's parser.py (per-method backbone / descriptor size)
    import vpr_models

    argv, sys.argv = sys.argv, [sys.argv[0], "--method", method]
    try:
        args = vpr_parser.parse_arguments()
    finally:
        sys.argv = argv
    vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    print_info(f"{method} weights ready ({args.descriptors_dimension}-d descriptors)")


def run_sequence(seq_path: Path, method: str, overwrite: bool) -> None:
    label = f"{seq_path.parent.name}/{seq_path.name}"
    rgb_csv = seq_path / "rgb.csv"
    if not rgb_csv.exists():
        print_warning(f"Skipping {label} - missing {rgb_csv}")
        return
    out_dir = seq_path / OUTPUT_FOLDER
    if (out_dir / "D.npy").exists() and not overwrite:
        print_info(f"Skipping {label} - {OUTPUT_FOLDER}/D.npy already exists (use --overwrite to recompute)")
        return

    print_info(f"{label} - {method} descriptors at 4 rotations -> {out_dir}")
    cmd = [sys.executable, str(REPO_ROOT / "vpr-lab.py"), f"--method={method}", "--rot_360", "--exp_yaml=None",
           f"--rgb_csv_db={rgb_csv}", f"--rgb_csv_q={rgb_csv}", f"--log_dir={out_dir}"]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="VPR distance matrix <sequence>/vpr-lab/D.npy of VSLAM-LAB sequences.")
    parser.add_argument("--sequence-path", nargs="+", type=Path, dest="sequence_paths", help="VSLAM-LAB sequence folders")
    parser.add_argument("--method", default=DEFAULT_METHOD, help=f"VPR method (default: {DEFAULT_METHOD})")
    parser.add_argument("--overwrite", action="store_true", help="Recompute D.npy even if it exists")
    parser.add_argument("--prefetch", action="store_true", help="Download the method's weights and exit")
    args = parser.parse_args()

    if args.prefetch:
        prefetch(args.method)
        return
    if not args.sequence_paths:
        parser.error("--sequence-path is required")
    for seq_path in args.sequence_paths:
        run_sequence(seq_path.resolve(), args.method, args.overwrite)


if __name__ == "__main__":
    main()
