"""Generate clearly marked synthetic annotator sessions for PoD testing.

These files emulate the browser study schema for software validation and
sensitivity analysis. They are not observations from human participants.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "PoD_code" / "labeling" / "tasks.json"
OUT_DIR = ROOT / "data_synthetic_annotators"

FIELDS = [
    "participant_id",
    "session_id",
    "csv_version",
    "block",
    "induced_condition",
    "trial_idx",
    "item_id",
    "c_star",
    "complexity_bin",
    "y_true",
    "y_user",
    "correct",
    "delib_ms",
    "t_shown_perf",
    "t_clicked_perf",
    "t_shown_iso",
    "age_bracket",
    "experience",
    "started_at_iso",
]


@dataclass(frozen=True)
class Profile:
    baseline_accuracy: float
    gaming_accuracy: float
    fatigue_accuracy: float
    baseline_scale: float
    gaming_mean_ms: float
    fatigue_scale: float
    fatigue_variability: float
    age_bracket: str
    experience: str


PROFILES = [
    Profile(0.90, 0.64, 0.58, 1.00, 245, 2.35, 0.55, "25-34", "experienced"),
    Profile(0.86, 0.60, 0.54, 1.08, 270, 2.15, 0.62, "35-44", "experienced"),
    Profile(0.78, 0.55, 0.50, 0.96, 230, 2.55, 0.72, "18-24", "novice"),
    Profile(0.88, 0.67, 0.60, 1.16, 285, 2.30, 0.58, "45-54", "experienced"),
    Profile(0.82, 0.50, 0.52, 0.91, 205, 2.60, 0.80, "25-34", "intermediate"),
    Profile(0.84, 0.59, 0.43, 1.04, 255, 2.85, 0.92, "35-44", "intermediate"),
    Profile(0.92, 0.72, 0.68, 1.11, 295, 2.05, 0.48, "55-64", "experienced"),
    Profile(0.80, 0.56, 0.51, 0.98, 240, 2.45, 0.70, "18-24", "novice"),
    Profile(0.76, 0.53, 0.46, 1.02, 220, 2.75, 0.88, "25-34", "intermediate"),
    Profile(0.89, 0.65, 0.57, 1.13, 275, 2.25, 0.64, "45-54", "experienced"),
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def accuracy_probability(base: float, complexity: float) -> float:
    """Reduce correctness smoothly as task complexity increases."""
    return clamp(base + 0.12 * (0.5 - complexity), 0.05, 0.98)


def deliberation_ms(
    profile: Profile,
    condition: str,
    complexity: float,
    block_position: int,
    rng: random.Random,
) -> int:
    expected = 650.0 * complexity + 245.0
    if condition == "baseline":
        noise = rng.lognormvariate(0.0, 0.12)
        return round(clamp(expected * profile.baseline_scale * noise, 180, 2400))

    if condition == "gaming":
        # Fast, near-mechanical clicking with weak difficulty dependence.
        value = profile.gaming_mean_ms + 18.0 * (complexity - 0.5)
        value += rng.gauss(0.0, 9.0)
        return round(clamp(value, 160, 305))

    # Increasingly slow and erratic responses in the final block.
    progression = 1.0 + 0.014 * block_position
    scatter = rng.lognormvariate(-0.5 * profile.fatigue_variability**2,
                                 profile.fatigue_variability)
    value = expected * profile.fatigue_scale * progression * scatter
    if block_position in {9, 21, 33}:
        value *= 1.8
    return round(clamp(value, 260, 7800))


def build_session(index: int, profile: Profile, items: list[dict]) -> list[dict]:
    rng = random.Random(2026061000 + index)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    trials = shuffled[5:120]

    participant_id = f"synthetic-annotator-{index:02d}"
    start = datetime(2026, 6, 10, 8, 0, tzinfo=timezone.utc) + timedelta(minutes=20 * index)
    perf_ms = 10_000.0 + rng.uniform(0, 2_000)
    rows: list[dict] = []

    for trial_idx, item in enumerate(trials):
        if trial_idx < 39:
            block, condition, block_position = "baseline", "baseline", trial_idx
            target_accuracy = profile.baseline_accuracy
        elif trial_idx < 76:
            block, condition, block_position = "speed_bonus", "gaming", trial_idx - 39
            target_accuracy = profile.gaming_accuracy
        else:
            block, condition, block_position = "long_block", "fatigue", trial_idx - 76
            target_accuracy = profile.fatigue_accuracy

        complexity = float(item["c_star"])
        delib = deliberation_ms(profile, condition, complexity, block_position, rng)
        p_correct = accuracy_probability(target_accuracy, complexity)
        correct = int(rng.random() < p_correct)
        y_true = int(item["y_true"])
        y_user = y_true if correct else 1 - y_true

        shown_perf = perf_ms
        clicked_perf = shown_perf + delib
        shown_iso = start + timedelta(milliseconds=shown_perf)
        rows.append(
            {
                "participant_id": participant_id,
                "session_id": participant_id,
                "csv_version": 1,
                "block": block,
                "induced_condition": condition,
                "trial_idx": trial_idx,
                "item_id": int(item["item_id"]),
                "c_star": f"{complexity:.15g}",
                "complexity_bin": item["complexity_bin"],
                "y_true": y_true,
                "y_user": y_user,
                "correct": correct,
                "delib_ms": delib,
                "t_shown_perf": f"{shown_perf:.3f}",
                "t_clicked_perf": f"{clicked_perf:.3f}",
                "t_shown_iso": shown_iso.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                "age_bracket": profile.age_bracket,
                "experience": profile.experience,
                "started_at_iso": start.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            }
        )
        perf_ms = clicked_perf + rng.uniform(180.0, 420.0)

    return rows


def main() -> None:
    with POOL_PATH.open(encoding="utf-8") as handle:
        pool = json.load(handle)
    items = list(pool["items"])
    if len(items) < 120:
        raise ValueError(f"Expected at least 120 pool items, found {len(items)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, profile in enumerate(PROFILES, start=1):
        rows = build_session(index, profile, items)
        path = OUT_DIR / f"synthetic_pod_session_{index:02d}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path.name}: {len(rows)} trials")


if __name__ == "__main__":
    main()
