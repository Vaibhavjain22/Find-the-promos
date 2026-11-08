"""Module 1: deterministic screenplay parser, window slicer and stitched-candidate generator.

No LLM is used here. Everything is regex + interval arithmetic.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple

MIN_TOTAL, MAX_TOTAL = 20.0, 60.0
HOOK_MIN, HOOK_MAX = 3.0, 8.0
CLIFF_MIN, CLIFF_MAX = 15.0, 40.0
MIN_SEGMENT = 3.0

_HEADER_RE = re.compile(
    r"\*\*(?P<header>[^*\n]+?):\*\*\s*\n\s*Start:\s*\((?P<s>[\d:.]+)\)\s*\|\s*End:\s*\((?P<e>[\d:.]+)\)\s*\n"
    r"(?P<text>.*?)(?=\n\s*\*\*|\n\s*SCENE \d+|\n\s*--- SCRIPT END|\Z)",
    re.S,
)
_SCENE_SPLIT = re.compile(r"\n(?=\s*SCENE \d+\s*\n)")

# words that make a moment "high action" / hook-like. Plain keyword list, deliberately simple and auditable.
ACTION_WORDS = re.compile(
    r"slap|strike|struck|shove|grab|hurl|throw|shatter|crash|collapse|faint|gasp|scream|"
    r"blood|wound|knife|sword|dagger|poison|burst|rip|tear|kiss|embrace|draw[s]? (?:a )?(?:sword|blade)",
    re.I,
)
HOOK_WORDS = re.compile(
    r"betray|lie[sd]?\b|secret|truth|dead|die|kill|pregnan|divorce|wedding|bride|duke|scar|baby|never|"
    r"how dare|get out|mine|marry|arrest|reveal|you knew|forgive|monster|beast|no one|nobody",
    re.I,
)
CLIFF_WORDS = re.compile(
    r"\?|\.\.\.|—|cut to black|freeze|stunned|horror|frozen|ominous|swell|stinger|suddenly|"
    r"burst|footsteps|shadow|door|who are|what did|what have|you .{0,20}\bwho\b",
    re.I,
)


# ------------------------------------------------------------------ data model
def to_seconds(ts: str) -> float:
    """'MM:SS.ss' or 'HH:MM:SS.ss' (optionally in parentheses) -> float seconds."""
    parts = [float(p) for p in ts.strip("() ").split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def to_mmss(sec: float) -> str:
    """float seconds -> 'MM:SS.ss'"""
    m = int(sec // 60)
    return f"{m:02d}:{sec - 60 * m:05.2f}"


@dataclass
class Event:
    type: str  # camera | action | sfx | dialogue
    header: str
    start_sec: float
    end_sec: float
    text: str
    scene: int

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class Episode:
    number: int
    name: str
    logline: str
    continuity: str
    duration: float
    events: List[Event]
    raw_text: str = ""


@dataclass
class Candidate:
    episode: str
    kind: str  # "continuous" | "stitched"
    segments: List[Tuple[float, float]]
    heuristic: float = 0.0
    id: str = ""

    @property
    def duration(self) -> float:
        return round(sum(b - a for a, b in self.segments), 2)


# ------------------------------------------------------------------ parsing
def _fix_mojibake(t: str) -> str:
    for bad, good in (("â€“", "–"), ("â€”", "—"), ("â€™", "'"), ("â€œ", '"'), ("â€\x9d", '"'), ("Ã©", "é")):
        t = t.replace(bad, good)
    return t


def parse_episode(path: str) -> Episode:
    raw = _fix_mojibake(open(path, encoding="utf-8", errors="replace").read())
    num_m = re.search(r"EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", raw)
    number = int(num_m.group(1)) if num_m else 0
    logline = (re.search(r"EPISODE LOGLINE:\s*(.*)", raw) or [None, ""])[1].strip()
    cont = (re.search(r"CONTINUITY NOTES:\s*(.*)", raw) or [None, ""])[1].strip()
    body = raw.split("--- SCRIPT START ---")[-1]
    events: List[Event] = []
    scene_idx = -1
    for block in _SCENE_SPLIT.split(body):
        if not re.match(r"\s*SCENE \d+", block):
            continue
        scene_idx += 1
        for m in _HEADER_RE.finditer(block):
            header = m.group("header").strip()
            kind = header.split(" (")[0].strip().upper()
            typ = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx"}.get(kind, "dialogue")
            events.append(Event(typ, header, to_seconds(m.group("s")), to_seconds(m.group("e")),
                                " ".join(m.group("text").split()), scene_idx))
    events.sort(key=lambda e: (e.start_sec, e.end_sec))
    duration = max([e.end_sec for e in events] or [0.0])
    return Episode(number, f"Episode {number}", logline, cont, duration, events, raw)


def load_episodes(directory: str) -> List[Episode]:
    eps = [parse_episode(p) for p in sorted(glob.glob(os.path.join(directory, "*.txt")))]
    return sorted(eps, key=lambda e: e.number)


# ------------------------------------------------------------------ helpers
def events_in(ep: Episode, a: float, b: float, types=("dialogue", "action", "sfx")) -> List[Event]:
    return [e for e in ep.events if e.type in types and e.end_sec > a and e.start_sec < b]


def temporal_iou(x: List[Tuple[float, float]], y: List[Tuple[float, float]]) -> float:
    """IoU of two unions of intervals."""
    def merge(iv):
        out = []
        for a, b in sorted(iv):
            if out and a <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        return out
    mx, my = merge(x), merge(y)
    inter = sum(max(0.0, min(b1, b2) - max(a1, a2)) for a1, b1 in mx for a2, b2 in my)
    union = sum(b - a for a, b in mx) + sum(b - a for a, b in my) - inter
    return inter / union if union > 0 else 0.0


def _is_strong_start(e: Event) -> bool:
    """Window may start only on dialogue or a high-action beat (never camera notes, SFX or silence)."""
    return e.type == "dialogue" or (e.type == "action" and bool(ACTION_WORDS.search(e.text)))


def heuristic_score(ep: Episode, segments: List[Tuple[float, float]]) -> float:
    """Cheap 0-1 prior used only to pre-filter candidates before the LLM sees them."""
    a0, b_last = segments[0][0], segments[-1][1]
    head = " ".join(e.text for e in events_in(ep, a0, a0 + 3.0))
    first_dialogue = any(e.type == "dialogue" and e.start_sec < a0 + 3.0 for e in events_in(ep, a0, a0 + 3.0))
    hook = 0.35 * first_dialogue + 0.2 * ("!" in head or "?" in head) + min(0.45, 0.15 * (len(HOOK_WORDS.findall(head)) + len(ACTION_WORDS.findall(head))))
    tail = " ".join(e.text for e in events_in(ep, b_last - 6.0, b_last))
    cliff = min(0.5, 0.12 * len(CLIFF_WORDS.findall(tail)))
    if b_last >= ep.duration - 3.0:
        cliff += 0.3  # episodes are written to end on a cliffhanger
    last_dlg = [e for e in events_in(ep, a0, b_last, ("dialogue",))]
    if last_dlg and last_dlg[-1].text.rstrip('" ').endswith(("?", "...", "—")):
        cliff += 0.2
    n_dlg = len([e for e in events_in(ep, a0, b_last, ("dialogue",))])
    dens = min(1.0, n_dlg / max(1.0, sum(b - a for a, b in segments) / 5.0))
    return round(0.4 * min(hook, 1) + 0.4 * min(cliff, 1) + 0.2 * dens, 4)


# ------------------------------------------------------------------ generators
def continuous_windows(ep: Episode) -> List[Candidate]:
    """All single-scene windows with 20 <= len <= 60 that start on dialogue / high action."""
    out = []
    ends_by_scene = {}
    for e in ep.events:
        ends_by_scene.setdefault(e.scene, set()).add(e.end_sec)
    for s in (e for e in ep.events if _is_strong_start(e)):
        for end in sorted(ends_by_scene[s.scene]):
            if MIN_TOTAL <= end - s.start_sec <= MAX_TOTAL:
                out.append(Candidate(ep.name, "continuous", [(s.start_sec, end)]))
    return out


def stitched_candidates(ep: Episode) -> List[Candidate]:
    """Hook beat (3-8s, early dialogue/action) + cliffhanger beat (15-40s, later). No segment < 3s."""
    hooks, cliffs = [], []
    early_limit = ep.duration * 0.5
    ends = sorted({e.end_sec for e in ep.events if e.type in ("dialogue", "action")})
    for s in (e for e in ep.events if _is_strong_start(e) and e.start_sec <= early_limit):
        for end in ends:
            if HOOK_MIN <= end - s.start_sec <= HOOK_MAX:
                hooks.append((s.start_sec, end))
    for s in (e for e in ep.events if _is_strong_start(e) and e.start_sec >= ep.duration * 0.3):
        for end in ends:
            if CLIFF_MIN <= end - s.start_sec <= CLIFF_MAX and end >= ep.duration - 4.0:  # ends at the episode cliffhanger
                cliffs.append((s.start_sec, end))
    # keep only the 6 best hooks to bound the product
    hooks = sorted(set(hooks), key=lambda h: -heuristic_score(ep, [h]))[:6]
    out = []
    for h in hooks:
        for c in sorted(set(cliffs)):
            total = (h[1] - h[0]) + (c[1] - c[0])
            if c[0] - h[1] >= 5.0 and MIN_TOTAL <= total <= MAX_TOTAL and min(h[1] - h[0], c[1] - c[0]) >= MIN_SEGMENT:
                out.append(Candidate(ep.name, "stitched", [h, c]))
    return out


def _select(ep: Episode, cands: List[Candidate], k: int, iou_thr: float = 0.5) -> List[Candidate]:
    for c in cands:
        c.heuristic = heuristic_score(ep, c.segments)
    cands = sorted(cands, key=lambda c: -c.heuristic)
    keep: List[Candidate] = []
    for c in cands:
        if all(temporal_iou(c.segments, k_.segments) < iou_thr for k_ in keep):
            keep.append(c)
        if len(keep) >= k:
            break
    return keep


def generate_candidates(ep: Episode, top_continuous: int = 3, top_stitched: int = 1) -> List[Candidate]:
    """Full deterministic candidate pipeline for one episode: generate -> score -> prune -> id."""
    cands = _select(ep, continuous_windows(ep), top_continuous) + _select(ep, stitched_candidates(ep), top_stitched)
    for i, c in enumerate(cands, 1):
        c.id = f"E{ep.number:03d}-{c.kind[0].upper()}{i}"
    return cands


def candidate_transcript(ep: Episode, cand: Candidate) -> str:
    """Compact, camera-free text of the candidate, with timestamps. This is the only thing the LLM sees."""
    lines = []
    for i, (a, b) in enumerate(cand.segments, 1):
        lines.append(f"--- SEGMENT {i} [{to_mmss(a)} - {to_mmss(b)}] ---")
        for e in events_in(ep, a, b, ("dialogue", "action")):
            tag = e.header.upper() if e.type == "dialogue" else "ACTION"
            lines.append(f"({to_mmss(max(e.start_sec, a))}) {tag}: {e.text}")
    return "\n".join(lines)

