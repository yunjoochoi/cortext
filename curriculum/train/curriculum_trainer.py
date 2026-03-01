"""Multi-stage curriculum orchestrator for Z-Image LoRA training."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# fmt: off
PHASES = [
    {
        "name":                 "easy",
        "contrastive_coeff":    0.01,
        "d1_severity_min":      0.0,
        "d2_difficulty_max":    0.4,
    },
    {
        "name":                 "medium",
        "contrastive_coeff":    0.05,
        "d1_severity_min":      0.5,
        "d2_difficulty_max":    0.7,
    },
    {
        "name":                 "hard",
        "contrastive_coeff":    0.10,
        "d1_severity_min":      0.8,
        "d2_difficulty_max":    1.0,
    },
]
# fmt: on

TRAIN_SCRIPT = Path(__file__).parent / "train_lora_z_image.py"


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_hard_negatives(
    src_jsonl: Path,
    out_jsonl: Path,
    severity_min: float,
) -> int:
    """Write neg pairs whose d1 severity >= severity_min. Returns count."""
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(src_jsonl) as fin, open(out_jsonl, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec.get("severity", 0.0) >= severity_min:
                fout.write(line)
                count += 1
    return count


def filter_manifest(
    src_jsonl: Path,
    out_jsonl: Path,
    difficulty_max: float,
) -> int:
    """Write records whose difficulty.combined <= difficulty_max. Returns count."""
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(src_jsonl) as fin, open(out_jsonl, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            d = rec.get("difficulty", {}).get("combined", 1.0)
            if d <= difficulty_max:
                fout.write(line)
                count += 1
    return count


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def run_phase(phase: dict, args: argparse.Namespace, resume_from: str | None) -> str:
    """Launch train_lora_z_image.py for one curriculum phase. Returns output_dir."""
    phase_out = Path(args.output_dir) / phase["name"]
    phase_out.mkdir(parents=True, exist_ok=True)

    # Filter hard negatives for this phase
    filtered_neg = phase_out / "hard_negatives_filtered.jsonl"
    neg_count = filter_hard_negatives(
        Path(args.hard_negatives_jsonl),
        filtered_neg,
        phase["d1_severity_min"],
    )
    print(f"  [{phase['name']}] neg pairs after severity filter: {neg_count}")

    # Filter training manifest for this phase
    filtered_manifest = phase_out / "manifest_filtered.jsonl"
    manifest_count = filter_manifest(
        Path(args.difficulty_scored_jsonl),
        filtered_manifest,
        phase["d2_difficulty_max"],
    )
    print(f"  [{phase['name']}] training samples after difficulty filter: {manifest_count}")

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--pretrained_model_name_or_path", args.pretrained_model_name_or_path,
        "--instance_data_dir",             args.instance_data_dir,
        "--instance_prompt",               args.instance_prompt,
        "--output_dir",                    str(phase_out),
        "--train_batch_size",              str(args.train_batch_size),
        "--gradient_accumulation_steps",   str(args.gradient_accumulation_steps),
        "--max_train_steps",               str(args.max_train_steps_per_phase),
        "--learning_rate",                 str(args.learning_rate),
        "--rank",                          str(args.rank),
        "--mixed_precision",               args.mixed_precision,
        "--contrastive_loss_coeff",        str(phase["contrastive_coeff"]),
        "--hard_negatives_jsonl",          str(filtered_neg),
        "--contrastive_proj_dim",          str(args.contrastive_proj_dim),
    ]

    if resume_from:
        cmd += ["--resume_from_checkpoint", resume_from]

    print(f"\n[Phase: {phase['name']}] launching training...")
    result = subprocess.run(cmd, check=True)
    return str(phase_out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--instance_data_dir",             required=True)
    p.add_argument("--instance_prompt",               required=True)
    p.add_argument("--output_dir",                    required=True)
    p.add_argument("--hard_negatives_jsonl",          required=True,
                   help="Output of hard_negative_gen.py")
    p.add_argument("--difficulty_scored_jsonl",       required=True,
                   help="Output of difficulty_scorer.py (score_manifest)")
    p.add_argument("--max_train_steps_per_phase",     type=int,   default=500)
    p.add_argument("--train_batch_size",              type=int,   default=1)
    p.add_argument("--gradient_accumulation_steps",   type=int,   default=4)
    p.add_argument("--learning_rate",                 type=float, default=1e-4)
    p.add_argument("--rank",                          type=int,   default=16)
    p.add_argument("--mixed_precision",               default="bf16")
    p.add_argument("--contrastive_proj_dim",          type=int,   default=256)
    p.add_argument("--phases",                        nargs="+",
                   choices=["easy", "medium", "hard"],
                   default=["easy", "medium", "hard"],
                   help="Which phases to run, in order.")
    return p.parse_args()


def main():
    args = parse_args()
    phase_map = {p["name"]: p for p in PHASES}

    resume_from = None
    for phase_name in args.phases:
        phase = phase_map[phase_name]
        prev_ckpt = resume_from  # carry checkpoint from prior phase
        resume_from = run_phase(phase, args, prev_ckpt)
        print(f"[Phase: {phase_name}] done → {resume_from}")

    print("\nCurriculum training complete.")


if __name__ == "__main__":
    main()
