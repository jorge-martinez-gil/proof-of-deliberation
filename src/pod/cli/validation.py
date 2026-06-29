"""
``pod-validate`` -- the Hick-Hyman validation runner.

Orchestrates the four-stage pipeline reported in Section 6.5 of the
paper: corpus loading, out-of-fold entropy estimation, statistical
testing, and figure rendering.
"""

from __future__ import annotations

import argparse

from pod.validation.corpora import build_pairs, load_all_segments
from pod.validation.entropy import compute_oof_entropies
from pod.validation.figure import render_figure


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``pod-validate`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="pod-validate",
        description=(
            "Reproduce the Hick-Hyman validation of Proof-of-Deliberation "
            "on Air Traffic Control logs."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="hick_hyman_pod_validation_v2.png",
        help="Output figure path (PNG/PDF chosen by extension).",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of stratified K-fold splits for the OOF estimator.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=3,
        help="Number of independent K-fold partitions to ensemble.",
    )
    parser.add_argument(
        "--no_temp_scale",
        action="store_true",
        help="Disable post-hoc temperature scaling on the OOF probabilities.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    all_segs = load_all_segments()
    pairs = build_pairs(all_segs)
    print(f"Pilot->controller pairs: {len(pairs)}")
    if len(pairs) < 30:
        raise RuntimeError("Too few pairs.")

    print(
        f"\nComputing OOF model entropies "
        f"({args.n_splits}-fold x {args.n_seeds} seeds + temp-scaling) ..."
    )
    model_ents, delays = compute_oof_entropies(
        pairs,
        n_splits=int(args.n_splits),
        n_seeds=int(args.n_seeds),
        apply_temp_scale=not args.no_temp_scale,
    )
    print(f"  OOF entropy estimates: {len(model_ents)}  (all pairs, no warmup waste)")

    rho_me, p_me, extras = render_figure(pairs, model_ents, delays, args.out)

    sep = "=" * 70
    print(f"\n{sep}")
    print("PROOF-OF-DELIBERATION -- HICK-HYMAN VALIDATION REPORT  [v2]")
    print(sep)
    print(f"  Total pairs                        : {extras['n_total']}")
    print(f"  Sources                            : {extras['src_counts']}")
    print(
        "  Method                             : OOF "
        f"{args.n_splits}-fold x {args.n_seeds} seeds + "
        "isotonic calibration + temp-scaling"
    )
    print(
        "  Delay filter                       : "
        f"[0.1s, {extras['p99']:.1f}s]  ->  {extras['n_kept']} pairs"
    )
    print()
    print("  PRIMARY TEST -- Ct = H(P(y|x;theta))  [PoD Eq. 2]")
    stars = (
        "***" if p_me < 0.001
        else "**" if p_me < 0.010
        else "*" if p_me < 0.050
        else "ns"
    )
    print(f"    Model entropy vs delay :  rho = {rho_me:+.3f}  {stars}")
    print()
    print("  CONTRAST -- Linguistic proxies")
    for name, (rho, pval) in [
        ("Word count", extras["word_count"]),
        ("Lexical entropy", extras["lexical_entropy"]),
        ("Partial TTR", extras["partial_ttr"]),
    ]:
        s = (
            "***" if pval < 0.001
            else "**" if pval < 0.010
            else "*" if pval < 0.050
            else "ns"
        )
        print(f"    {name:32s}:  rho = {rho:+.3f}  {s}")
    print()
    print(f"  Verdict: {extras['verdict']}")
    print(sep)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
