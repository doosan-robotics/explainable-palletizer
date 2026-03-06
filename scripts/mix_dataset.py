"""Generate a mixed sim+gemini dataset for palletizer SFT training.

Combines box images from two sources:
  - Gemini: cropped box images from conveyor-view photos (via crop_boxes.py)
  - Isaac Sim: synthetic box renders

For each generated scenario, each box image slot is randomly assigned
to either source (weighted by --sim-ratio). Pallet images get gray
placeholders until Isaac Sim renders are available.

Usage:
    python scripts/mix_dataset.py --num-train 100 --num-eval 50

    # Custom sim ratio and seed:
    python scripts/mix_dataset.py \
        --num-train 100 --num-eval 50 \
        --sim-ratio 0.6 --seed 123
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from drp_training.data.generator import (
    format_sample_image,
    generate_scenario,
    solve_scenario,
)

# -- Defaults ----------------------------------------------------------------

DEFAULT_OUTPUT = Path("training/dataset/v1/mixed")
DEFAULT_GEMINI_MANIFEST = Path("training/dataset/v1/gemini/box_library/manifest.json")
DEFAULT_SIM_TRAIN = Path("training/dataset/v1/sim/images/train")
DEFAULT_SIM_EVAL = Path("training/dataset/v1/sim/images/eval")

# -- Image helpers -----------------------------------------------------------


def load_gemini_manifest(manifest_path: Path) -> list[dict]:
    """Load the gemini box library manifest."""
    with open(manifest_path) as f:
        return json.load(f)["boxes"]


def load_sim_pool(sim_dir: Path) -> list[Path]:
    """Glob all PNG images from a sim directory."""
    return sorted(sim_dir.glob("*.png"))


def assign_image(
    profile: str,
    sim_pool: list[Path],
    gemini_manifest: list[dict],
    rng: random.Random,
    sim_ratio: float,
) -> tuple[Path, str]:
    """Pick a source image for a box, returns (src_path, source_label)."""
    if rng.random() < sim_ratio and sim_pool:
        return rng.choice(sim_pool), "sim"
    # Fall back to gemini crop matching profile
    matches = [e for e in gemini_manifest if e["profile"] == profile]
    if matches:
        entry = rng.choice(matches)
        return Path(entry["path"]), "gemini"
    # Last resort: any gemini crop
    entry = rng.choice(gemini_manifest)
    return Path(entry["path"]), "gemini"


# -- Split generation --------------------------------------------------------


def generate_split(n: int, seed: int) -> tuple[list[dict], dict[str, list[str]]]:
    """Generate n SFT image samples and track per-box profile names.

    Returns:
        (samples, box_profiles) where box_profiles maps sid -> [profile_name, ...]
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)

    samples: list[dict] = []
    box_profiles: dict[str, list[str]] = {}

    for i in range(n):
        sc = generate_scenario(i + 1, rng=rng)
        action, reasoning = solve_scenario(sc, image_reasoning=True)
        sid = f"rand_{i:05d}"
        sample = format_sample_image(sid, sc, action, reasoning)
        samples.append(sample)
        box_profiles[sid] = [box.profile.name for box in sc.boxes]

    # Shuffle with deterministic seed
    indices = list(range(len(samples)))
    random.shuffle(indices)
    samples = [samples[i] for i in indices]

    return samples, box_profiles


def process_split(
    samples: list[dict],
    box_profiles: dict[str, list[str]],
    sim_pool: list[Path],
    gemini_manifest: list[dict],
    output_images_dir: Path,
    rng: random.Random,
    sim_ratio: float,
) -> dict[str, str]:
    """Assign real images to each sample's box slots and copy files.

    Rewrites each sample's ``images`` field to contain just filenames.

    Returns:
        Provenance dict mapping filename -> "sim" or "gemini".
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, str] = {}

    for sample in samples:
        sid = sample["id"]
        profiles = box_profiles[sid]
        new_images: list[str] = []

        for img_ref in sample.get("images", []):
            name = Path(img_ref).name

            if "_box" in name:
                idx = int(name.split("_box")[1].split(".")[0])
                profile = profiles[idx]
                src, label = assign_image(profile, sim_pool, gemini_manifest, rng, sim_ratio)
                shutil.copy2(src, output_images_dir / name)
                provenance[name] = label
                new_images.append(name)

            # Skip pallet images -- no real renders available yet

        sample["images"] = new_images

    return provenance


# -- CLI ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mixed sim+gemini SFT dataset",
    )
    parser.add_argument("--num-train", type=int, default=100)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--sim-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gemini-manifest",
        type=Path,
        default=DEFAULT_GEMINI_MANIFEST,
    )
    parser.add_argument(
        "--sim-train-dir",
        type=Path,
        default=DEFAULT_SIM_TRAIN,
    )
    parser.add_argument(
        "--sim-eval-dir",
        type=Path,
        default=DEFAULT_SIM_EVAL,
    )
    args = parser.parse_args()

    # Load image pools
    gemini_manifest = load_gemini_manifest(args.gemini_manifest)
    sim_train = load_sim_pool(args.sim_train_dir)
    sim_eval = load_sim_pool(args.sim_eval_dir)

    print(f"Gemini crops: {len(gemini_manifest)}")
    print(f"Sim train images: {len(sim_train)}")
    print(f"Sim eval images: {len(sim_eval)}")

    # Generate scenarios
    print(f"\nGenerating {args.num_train} train samples (seed={args.seed})...")
    train_samples, train_profiles = generate_split(args.num_train, seed=args.seed)

    eval_seed = args.seed + 10_000
    print(f"Generating {args.num_eval} eval samples (seed={eval_seed})...")
    eval_samples, eval_profiles = generate_split(args.num_eval, seed=eval_seed)

    # Assign images
    rng = random.Random(args.seed)

    print("\nAssigning images to train split...")
    train_prov = process_split(
        train_samples,
        train_profiles,
        sim_train,
        gemini_manifest,
        args.output / "images" / "train",
        rng,
        args.sim_ratio,
    )

    print("Assigning images to eval split...")
    eval_prov = process_split(
        eval_samples,
        eval_profiles,
        sim_eval,
        gemini_manifest,
        args.output / "images" / "eval",
        rng,
        args.sim_ratio,
    )

    # Write raw JSON
    raw_dir = args.output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "seed": args.seed,
        "sim_ratio": args.sim_ratio,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }

    for split_name, samples, count in [
        ("train", train_samples, args.num_train),
        ("eval", eval_samples, args.num_eval),
    ]:
        payload = {
            "metadata": {**metadata, "num_samples": count},
            "samples": samples,
        }
        with open(raw_dir / f"{split_name}.json", "w") as f:
            json.dump(payload, f, indent=2)

    # Write provenance
    provenance = {
        **{f"train/{k}": v for k, v in train_prov.items()},
        **{f"eval/{k}": v for k, v in eval_prov.items()},
    }
    with open(args.output / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    # Summary
    sim_count = sum(1 for v in provenance.values() if v == "sim")
    gemini_count = sum(1 for v in provenance.values() if v == "gemini")
    print("\nDone!")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Eval samples: {len(eval_samples)}")
    print(f"  Box images: {sim_count} sim, {gemini_count} gemini")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
