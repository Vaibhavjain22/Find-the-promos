"""Module 3: temporal NMS de-duplication and global ranking."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from evaluator import Promo
from parser import temporal_iou, to_seconds

IOU_THRESHOLD = 0.6


def _intervals(p: Promo):
    return [(to_seconds(s.start_time), to_seconds(s.end_time)) for s in p.segments]


def nms_per_episode(promos: List[Promo], iou_threshold: float = IOU_THRESHOLD) -> List[Promo]:
    """Within each episode keep the highest-scoring promo, drop any later one with IoU >= threshold against a kept one."""
    by_ep = defaultdict(list)
    for p in promos:
        by_ep[p.episode].append(p)
    kept: List[Promo] = []
    for ps in by_ep.values():
        ps.sort(key=lambda p: -p.evaluation.composite_score)
        sel: List[Promo] = []
        for p in ps:
            if all(temporal_iou(_intervals(p), _intervals(k)) < iou_threshold for k in sel):
                sel.append(p)
        kept += sel
    return kept


import re

# Penalties for flag *concepts* that contradict the brief ("makes sense on its own", a clean cut).
# The LLM words flags freely ("context needed for...", "unclear relationships"), so match by regex, not exact string.
# Found necessary on the real run: stitched promos were nearly all flagged jarring; ~75% of promos needed earlier episodes.
PENALTY_RULES = [
    (re.compile(r"jarring|abrupt (stitch|transition|cut)|disjoint", re.I), 1.5),
    (re.compile(r"earlier episode|previous episode|prior episode|context|backstory|unclear (character |family )?(relationship|dynamic)|relationships? (is|are) unclear", re.I), 0.5),
]


def penalty(p: Promo) -> float:
    """Each rule is applied at most once per promo, however many flags match it."""
    flags = " | ".join(p.evaluation.failure_flags)
    return sum(v for rx, v in PENALTY_RULES if rx.search(flags))


def rank_global(promos: List[Promo]) -> List[Promo]:
    """Rank by composite minus flag penalties. Raw composite stays in the output; rank_score shows the adjusted value.
    Tie-break: higher hook score (the first 3s decide the ad's performance), then shorter."""
    for p in promos:
        p.rank_score = round(p.evaluation.composite_score - penalty(p), 3)
    return sorted(promos, key=lambda p: (-p.rank_score, -p.evaluation.hook_score, p.total_duration))


def justify(rank: int, p: Promo) -> str:
    e = p.evaluation
    ttype = "stitched cut" if p.kind == "stitched" else "continuous cut"
    flag = f" Watch-outs: {'; '.join(e.failure_flags)}." if e.failure_flags else ""
    return (f"#{rank} {p.episode} ({ttype}, {p.total_duration:.0f}s, score {e.composite_score:.1f}, adjusted {p.rank_score:.1f}): "
            f"hook {e.hook_score:.0f}/10, cliffhanger {e.cliffhanger_score:.0f}/10, clarity {e.standalone_clarity_score:.0f}/10. "
            f"{e.why_it_works}{flag}")


def top_n(ranked: List[Promo], n: int = 10, max_per_episode: int = 1) -> List[Dict]:
    """Curated post-first list. One promo per episode so the first 10 ads show 10 different moments of the show."""
    seen: Dict[str, int] = defaultdict(int)
    out = []
    for p in ranked:
        if seen[p.episode] >= max_per_episode:
            continue
        seen[p.episode] += 1
        out.append({"rank": len(out) + 1, "justification": justify(len(out) + 1, p), "promo": p.model_dump()})
        if len(out) == n:
            break
    return out

