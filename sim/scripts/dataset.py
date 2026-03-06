"""MinIO dataset management for sim-generated images.

Splits local sim images into train/eval subsets and syncs them with the team
MinIO server under datasets/v1/sim/{train,eval}.

Automatically configures the mc alias on first use.

Usage:
    # Upload sim-images7 split 80/20 train/eval (defaults)
    python sim/scripts/dataset.py push

    # Upload with a custom source directory or eval ratio
    python sim/scripts/dataset.py push --source sim-images7 --eval-ratio 0.15

    # Pull the train split to a local directory
    python sim/scripts/dataset.py pull --split train --path data/sim/train

    # Pull the eval split
    python sim/scripts/dataset.py pull --split eval --path data/sim/eval

    # Pull both splits (each goes to its default path)
    python sim/scripts/dataset.py pull --split all

    # List available sim dataset contents
    python sim/scripts/dataset.py list
"""

from __future__ import annotations

import argparse
import getpass
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_SOURCE = "sim-images13"
TRAIN_VERSION = "v1/sim/train"
EVAL_VERSION = "v1/sim/eval"
DEFAULT_EVAL_RATIO = 0.2
RANDOM_SEED = 42

ALIAS = "drtech"
ENDPOINT = "http://tech.doosanrobotics.com:9000"
ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
BUCKET = "datasets"


def _require_mc() -> str:
    """Return the path to the mc binary, or exit if not found."""
    mc = shutil.which("mc")
    if mc is None:
        print(
            "Error: mc (MinIO client) not found on PATH.\n"
            "Install it:\n"
            "  curl -O https://dl.min.io/client/mc/release/linux-amd64/mc\n"
            "  chmod +x mc && sudo mv mc /usr/local/bin/",
            file=sys.stderr,
        )
        sys.exit(1)
    return mc


def _ensure_alias(mc: str) -> None:
    """Configure the drtech alias if it does not already exist.

    Credentials are read from MINIO_ACCESS_KEY / MINIO_SECRET_KEY env vars.
    If not set, the user is prompted interactively (first-time setup).
    """
    result = subprocess.run(
        [mc, "alias", "ls", ALIAS],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or ALIAS not in result.stdout:
        access_key = ACCESS_KEY or input("MinIO access key: ")
        secret_key = SECRET_KEY or getpass.getpass("MinIO secret key: ")
        print(f"Configuring mc alias '{ALIAS}' -> {ENDPOINT}")
        subprocess.run(
            [mc, "alias", "set", ALIAS, ENDPOINT, access_key, secret_key],
            check=True,
        )


def _remote_path(version: str) -> str:
    return f"{ALIAS}/{BUCKET}/{version}"


def _split_files(
    source: Path,
    eval_ratio: float,
) -> tuple[list[Path], list[Path]]:
    """Return (train_files, eval_files) sorted and shuffled with a fixed seed."""
    files = sorted(source.glob("*.png"))
    if not files:
        print(f"Error: no .png files found in '{source}'", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(files)

    n_eval = max(1, round(len(files) * eval_ratio))
    return files[n_eval:], files[:n_eval]


def cmd_push(args: argparse.Namespace) -> None:
    mc = _require_mc()
    _ensure_alias(mc)

    source = Path(args.source)
    if not source.is_dir():
        print(f"Error: source directory '{source}' does not exist.", file=sys.stderr)
        sys.exit(1)

    train_files, eval_files = _split_files(source, args.eval_ratio)
    print(
        f"Split: {len(train_files)} train / {len(eval_files)} eval "
        f"(eval_ratio={args.eval_ratio}, seed={RANDOM_SEED})"
    )

    with tempfile.TemporaryDirectory() as tmp:
        train_dir = Path(tmp) / "train"
        eval_dir = Path(tmp) / "eval"
        train_dir.mkdir()
        eval_dir.mkdir()

        for f in train_files:
            shutil.copy2(f, train_dir / f.name)
        for f in eval_files:
            shutil.copy2(f, eval_dir / f.name)

        train_dest = _remote_path(TRAIN_VERSION)
        eval_dest = _remote_path(EVAL_VERSION)

        mirror_base = [mc, "mirror"]
        if args.overwrite:
            mirror_base.append("--overwrite")

        print(f"Uploading train -> {train_dest}")
        subprocess.run([*mirror_base, str(train_dir), train_dest], check=True)

        print(f"Uploading eval  -> {eval_dest}")
        subprocess.run([*mirror_base, str(eval_dir), eval_dest], check=True)

    print("Upload complete.")


def cmd_pull(args: argparse.Namespace) -> None:
    mc = _require_mc()
    _ensure_alias(mc)

    splits: dict[str, str] = {}
    if args.split in ("train", "all"):
        splits["train"] = args.path if args.split == "train" and args.path else "data/sim/train"
    if args.split in ("eval", "all"):
        splits["eval"] = args.path if args.split == "eval" and args.path else "data/sim/eval"

    for split, local_path in splits.items():
        version = TRAIN_VERSION if split == "train" else EVAL_VERSION
        source = _remote_path(version)
        print(f"Pulling {source} -> {local_path}")
        cmd = [mc, "mirror"]
        if args.overwrite:
            cmd.append("--overwrite")
        cmd += [source, local_path]
        subprocess.run(cmd, check=True)


def cmd_list(args: argparse.Namespace) -> None:
    mc = _require_mc()
    _ensure_alias(mc)
    result = subprocess.run(
        [mc, "du", "--depth", "4", f"{ALIAS}/{BUCKET}/v1/sim/"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in reversed(result.stdout.strip().splitlines()):
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync sim image datasets with the team MinIO server.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # push
    push_parser = subparsers.add_parser(
        "push",
        help="Upload sim images split into train/eval",
    )
    push_parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Local image directory to upload (default: {DEFAULT_SOURCE})",
    )
    push_parser.add_argument(
        "--eval-ratio",
        type=float,
        default=DEFAULT_EVAL_RATIO,
        help=f"Fraction of images reserved for eval (default: {DEFAULT_EVAL_RATIO})",
    )
    push_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files on the destination",
    )
    push_parser.set_defaults(func=cmd_push)

    # pull
    pull_parser = subparsers.add_parser(
        "pull",
        help="Download sim images from MinIO",
    )
    pull_parser.add_argument(
        "--split",
        choices=["train", "eval", "all"],
        default="all",
        help="Which split to download (default: all)",
    )
    pull_parser.add_argument(
        "--path",
        default=None,
        help="Local destination directory (only for single-split pulls)",
    )
    pull_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files on the destination",
    )
    pull_parser.set_defaults(func=cmd_pull)

    # list
    list_parser = subparsers.add_parser(
        "list",
        help="List sim dataset contents on MinIO",
    )
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
