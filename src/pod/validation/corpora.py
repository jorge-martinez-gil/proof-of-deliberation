"""
HuggingFace dataset loaders for ATCO2 and UWB-ATCC.

Two corpora are joined: the one-hour ``atco2_corpus_1h`` test split and
the train+test split of ``uwb_atcc``. The parser extracts pilot-to-
controller pairs together with the inter-utterance delay, the corpus
of origin, and the controller-side command label.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

try:  # pragma: no cover - import guarded for unit-test environments
    from datasets import load_dataset  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]

from pod.validation.labels import command_label, encode_label

_ATCO2_SPK_RE = re.compile(r"^(.*)-([A-Za-z])$")
_ATCO2_CTRL = {"A"}
_ATCO2_PILOT = {"B", "G"}

_UWB_RE = re.compile(r"^(uwb-atcc_[^_]+)_(\d+)_(\d+)_(AT|PI|PIAT)$")


def _parse_atco2_id(seg_id: str):
    left = seg_id.split("__")[0]
    m = _ATCO2_SPK_RE.match(left)
    if not m:
        return left, None
    spk = m.group(2).upper()
    if spk in _ATCO2_CTRL:
        return m.group(1), "CONTROLLER"
    if spk in _ATCO2_PILOT:
        return m.group(1), "PILOT"
    return m.group(1), None


def load_all_segments() -> List[Dict[str, Any]]:
    """Download (idempotently) and parse all ATC segments.

    Requires the optional ``datasets`` dependency. The function prints
    progress to stdout, matching the reference script.
    """
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' package is required for ATC validation. "
            "Install with `pip install datasets`."
        )

    segs: List[Dict[str, Any]] = []
    print("Loading atco2_corpus_1h ...")
    ds = load_dataset("Jzuluaga/atco2_corpus_1h")["test"].select_columns(
        ["id", "text", "segment_start_time", "segment_end_time"]
    )
    for r in ds:
        conv, role = _parse_atco2_id(r["id"])
        if role is None:
            continue
        segs.append(
            {
                "conv": conv,
                "role": role,
                "t_start": float(r["segment_start_time"]),
                "t_end": float(r["segment_end_time"]),
                "text": (r.get("text") or "").strip(),
                "source": "atco2_1h",
            }
        )
    print(f"  -> {sum(1 for s in segs if s['source']=='atco2_1h')} segments")

    print("Loading uwb_atcc (train + test) ...")
    ds_dict = load_dataset("Jzuluaga/uwb_atcc")
    for split_name in ("train", "test"):
        if split_name not in ds_dict:
            continue
        for r in ds_dict[split_name].select_columns(
            ["id", "text", "segment_start_time", "segment_end_time"]
        ):
            m = _UWB_RE.match(r["id"])
            if not m or m.group(4) == "PIAT":
                continue
            role = "CONTROLLER" if m.group(4) == "AT" else "PILOT"
            segs.append(
                {
                    "conv": m.group(1),
                    "role": role,
                    "t_start": float(r["segment_start_time"]),
                    "t_end": float(r["segment_end_time"]),
                    "text": (r.get("text") or "").strip(),
                    "source": "uwb_atcc",
                }
            )
    print(f"  -> {sum(1 for s in segs if s['source']=='uwb_atcc')} segments")
    print(f"Total segments: {len(segs)}")
    return segs


def build_pairs(segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group segments by conversation and emit pilot-controller pairs."""
    by_conv: Dict[str, List[Dict[str, Any]]] = {}
    for s in segs:
        by_conv.setdefault(s["conv"], []).append(s)

    pairs: List[Dict[str, Any]] = []
    for _, conv_segs in by_conv.items():
        conv_segs.sort(key=lambda x: x["t_start"])
        for i in range(len(conv_segs) - 1):
            s1, s2 = conv_segs[i], conv_segs[i + 1]
            if s1["role"] == "PILOT" and s2["role"] == "CONTROLLER":
                if not s1["text"]:
                    continue
                pairs.append(
                    {
                        "x_text": s1["text"],
                        "ctrl_text": s2["text"],
                        "y": encode_label(command_label(s2["text"])),
                        "y_label": command_label(s2["text"]),
                        "delay": max(0.0, s2["t_start"] - s1["t_end"]),
                        "source": s1["source"],
                    }
                )
    return pairs
