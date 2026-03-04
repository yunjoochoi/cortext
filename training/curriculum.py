"""3-phase curriculum orchestrator: easy -> medium -> hard with increasing contrastive weight."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PHASES = [
    {"name": "easy",   "contrastive_coeff": 0.01},
    {"name": "medium", "contrastive_coeff": 0.05},
    {"name": "hard",   "contrastive_coeff": 0.10},
]

TRAIN_SCRIPT = Path(__file__).parent / "train_lora.py"


def filter_by_tier(scored_jsonl: Path, out_jsonl: Path, tier: str) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(scored_jsonl) as fin, open(out_jsonl, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec.get("difficulty", {}).get("tier") == tier:
                fout.write(line)
                count += 1
    return count


def filter_hard_negatives_by_tier(neg_jsonl: Path, out_jsonl: Path, tier: str) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(neg_jsonl) as fin, open(out_jsonl, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec.get("tier") == tier:
                fout.write(line)
                count += 1
    return count


def run_phase(phase: dict, args: argparse.Namespace, resume_from: str | None) -> str:
    phase_out = Path(args.output_dir) / phase["name"]
    phase_out.mkdir(parents=True, exist_ok=True)

    filtered_manifest = phase_out / "manifest_filtered.jsonl"
    manifest_count = filter_by_tier(Path(args.scored_manifest), filtered_manifest, phase["name"])
    print(f"  [{phase['name']}] training samples: {manifest_count:,}")

    filtered_neg = phase_out / "hard_negatives_filtered.jsonl"
    neg_count = filter_hard_negatives_by_tier(Path(args.hard_negatives_jsonl), filtered_neg, phase["name"])
    print(f"  [{phase['name']}] contrastive pairs: {neg_count:,}")

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--pretrained_model_name_or_path", args.pretrained_model_name_or_path,
        "--manifest_jsonl",                str(filtered_manifest),
        "--hard_negatives_jsonl",          str(filtered_neg),
        "--output_dir",                    str(phase_out),
        "--train_batch_size",              str(args.train_batch_size),
        "--gradient_accumulation_steps",   str(args.gradient_accumulation_steps),
        "--max_train_steps",               str(args.max_train_steps_per_phase),
        "--learning_rate",                 str(args.learning_rate),
        "--rank",                          str(args.rank),
        "--mixed_precision",               args.mixed_precision,
        "--contrastive_loss_coeff",        str(phase["contrastive_coeff"]),
        "--contrastive_proj_dim",          str(args.contrastive_proj_dim),
    ]

    if resume_from:
        cmd += ["--resume_from_checkpoint", resume_from]

    print(f"\n[Phase: {phase['name']}] launching training...")
    subprocess.run(cmd, check=True)
    return str(phase_out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--scored_manifest", required=True)
    p.add_argument("--hard_negatives_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_train_steps_per_phase", type=int, default=500)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--mixed_precision", default="bf16")
    p.add_argument("--contrastive_proj_dim", type=int, default=1024)
    p.add_argument("--phases", nargs="+", choices=["easy", "medium", "hard"],
                   default=["easy", "medium", "hard"])
    return p.parse_args()


def main():
    args = parse_args()
    phase_map = {p["name"]: p for p in PHASES}

    resume_from = None
    for phase_name in args.phases:
        phase = phase_map[phase_name]
        prev_ckpt = resume_from
        resume_from = run_phase(phase, args, prev_ckpt)
        print(f"[Phase: {phase_name}] done -> {resume_from}")

    print("\nCurriculum training complete.")


if __name__ == "__main__":
    main()
