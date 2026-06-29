"""
Coarse-grained ATC command labels used for the Hick-Hyman validation.

The label taxonomy mirrors the one in the reference script
``hick_hyman_pod_validation_v2.py``. It is intentionally rule-based
(regular-expression matchers) so that the calibration step does not
depend on a separately-trained sequence-tagging model.
"""

from __future__ import annotations

import re
from typing import List, Pattern, Tuple

LABEL_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    ("CLIMB",    re.compile(r"\bclimb\b|\bclimbing\b|\bflight level\b|\bfl\b", re.I)),
    ("DESCEND",  re.compile(r"\bdescend\b|\bdescending\b|\blower\b", re.I)),
    ("TURN",     re.compile(r"\bturn\b|\bheading\b|\bdegrees?\b", re.I)),
    ("SPEED",    re.compile(r"\bspeed\b|\bknots?\b|\bkts?\b|\baccelerate\b|\bdecelerate\b", re.I)),
    ("CONTACT",  re.compile(r"\bcontact\b|\btower\b|\bradar\b|\bground\b|\bfrequency\b|\bmhz\b", re.I)),
    ("CLEARED",  re.compile(r"\bcleared\b|\bclearance\b", re.I)),
    ("HOLD",     re.compile(r"\bhold\b|\bstan(d)?by\b|\bwait\b|\borbiting\b", re.I)),
    ("APPROACH", re.compile(r"\bapproach\b|\blocalizer\b|\bils\b|\brunway\b|\bglide\b", re.I)),
    ("SQUAWK",   re.compile(r"\bsquawk\b|\btransponder\b|\bmode\b", re.I)),
    ("IDENT",    re.compile(r"\bident\b|\bidentif\b", re.I)),
    ("REPORT",   re.compile(r"\breport\b|\badvise\b|\bconfirm\b", re.I)),
]

LABELS: List[str] = [name for name, _ in LABEL_PATTERNS] + ["OTHER"]
LABEL2ID: dict = {name: idx for idx, name in enumerate(LABELS)}


def command_label(text: str) -> str:
    """Map a controller utterance to its coarse command label."""
    t = (text or "").strip()
    if not t:
        return "OTHER"
    for name, pattern in LABEL_PATTERNS:
        if pattern.search(t):
            return name
    return "OTHER"


def encode_label(label: str) -> int:
    """Return the integer id for a (possibly unknown) label."""
    return LABEL2ID.get(label, LABEL2ID["OTHER"])
