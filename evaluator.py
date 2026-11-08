"""Module 2: LangChain semantic evaluator (LLM used ONLY to score pre-filtered candidate windows)."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from parser import Candidate, Episode, candidate_transcript, to_mmss

W_HOOK, W_CLIFF, W_CLARITY = 0.4, 0.4, 0.2


class Segment(BaseModel):
    segment_index: int
    episode: str
    start_time: str = Field(description="MM:SS.ss")
    end_time: str = Field(description="MM:SS.ss")
    duration: float
    role: str = Field(description='e.g. "Hook", "Cliffhanger", "Escalation"')


class PromoEvaluation(BaseModel):
    hook_description: str = Field(description="What happens in the first 3 seconds")
    cliffhanger_description: str = Field(description="The unresolved cutoff moment at the end")
    why_it_works: str = Field(description="One-line marketing rationale")
    hook_score: float = Field(ge=0.0, le=10.0, description="Would the first 3s stop a scrolling viewer?")
    cliffhanger_score: float = Field(ge=0.0, le=10.0, description="Does it end unresolved, forcing the viewer to want more?")
    standalone_clarity_score: float = Field(ge=0.0, le=10.0, description="Understandable to someone who never saw the show?")
    composite_score: float = Field(default=0.0, description="Recomputed in code; ignore")
    failure_flags: List[str] = Field(default_factory=list, description="Short problems, e.g. 'unclear pronoun reference'")

    @model_validator(mode="after")
    def _composite(self):
        # never trust the model's arithmetic
        self.composite_score = round(W_HOOK * self.hook_score + W_CLIFF * self.cliffhanger_score + W_CLARITY * self.standalone_clarity_score, 3)
        return self


class Promo(BaseModel):
    candidate_id: str
    kind: str
    episode: str
    total_duration: float
    segments: List[Segment]
    evaluation: PromoEvaluation
    heuristic_prior: float = 0.0
    evaluator: str = "gpt-4o-mini"
    rank_score: float = 0.0  # composite minus ranker penalties; set by ranker.py


SYSTEM = (
    "You are a senior short-form video editor choosing promo clips for vertical microdrama "
    "(ReelShort / DramaBox style) to acquire new viewers on TikTok, Reels and Shorts. "
    "Judge ONLY from the provided text, and use the full 1-10 range; most clips are average (4-6). "
    "HOOK: judge only the first 3 seconds of the clip (the first 1-2 lines). 9-10 = a slap, reveal, threat or shocking line; "
    "6-7 = emotional but no new information; 3-4 = exposition, greeting, scene-setting, or a line that needs context. "
    "CLIFFHANGER: look only at the last 1-2 lines. 9-10 = a reveal, threat or question that is explicitly unanswered; "
    "6-7 = tension but nothing new is at stake; 3-4 = emotion simply fades or the beat is resolved. "
    "CLARITY: 9-10 = a stranger understands who, what and why; 5 = understandable but characters/relationships are unexplained; "
    "1-3 = needs earlier episodes. "
    "Every clip that scores below 7 on any dimension MUST include at least one failure_flag explaining why. "
    "Penalise clips that depend on earlier episodes, that start on exposition, or that end on a resolved beat. "
    "For stitched clips (2 segments) add a failure flag 'jarring stitch transition' if segment 2 does not follow plausibly from segment 1. "
    "Use failure_flags for concrete problems only; leave empty if none."
)
HUMAN = (
    "Show logline (episode {ep}): {logline}\n"
    "Candidate type: {kind}, total {dur}s.\n\n{transcript}\n\n"
    "Evaluate this as a standalone promo for a viewer who has never seen the show."
)


def _roles(cand: Candidate) -> List[str]:
    return ["Hook", "Cliffhanger"] if cand.kind == "stitched" else ["Hook+Cliffhanger"]


def build_segments(cand: Candidate) -> List[Segment]:
    return [Segment(segment_index=i, episode=cand.episode, start_time=to_mmss(a), end_time=to_mmss(b),
                    duration=round(b - a, 2), role=r)
            for i, ((a, b), r) in enumerate(zip(cand.segments, _roles(cand)), 1)]


class PromoEvaluator:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1, max_concurrency: int = 8):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI  # reads OPENAI_API_KEY; swap model/base_url here to change provider

        self.model = model
        self.max_concurrency = max_concurrency
        llm = ChatOpenAI(model=model, temperature=temperature, max_retries=3)
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
        self.chain = prompt | llm.with_structured_output(PromoEvaluation)

    def evaluate(self, episodes: dict, cands: List[Candidate], batch_size: int = 20) -> List[Promo]:
        """Evaluate candidates in batches. Failed items are skipped (reported), never silently scored."""
        results: List[Promo] = []
        for i in range(0, len(cands), batch_size):
            chunk = cands[i:i + batch_size]
            inputs = []
            for c in chunk:
                ep: Episode = episodes[c.episode]
                inputs.append(dict(ep=ep.number, logline=ep.logline, kind=c.kind, dur=c.duration,
                                   transcript=candidate_transcript(ep, c)))
            outs = self.chain.batch(inputs, config={"max_concurrency": self.max_concurrency}, return_exceptions=True)
            for c, o in zip(chunk, outs):
                if isinstance(o, Exception):
                    print(f"  ! {c.id} failed: {o}")
                    continue
                results.append(Promo(candidate_id=c.id, kind=c.kind, episode=c.episode, total_duration=c.duration,
                                     segments=build_segments(c), evaluation=o, heuristic_prior=c.heuristic,
                                     evaluator=self.model))
            print(f"  evaluated {min(i + batch_size, len(cands))}/{len(cands)}")
        return results


class HeuristicEvaluator:
    """Offline stand-in (NO LLM) so the pipeline runs end-to-end without an API key.
    Scores come from keyword rules, so they are far less reliable than the LLM's. Output is labelled accordingly."""

    model = "heuristic-dry-run"

    def evaluate(self, episodes: dict, cands: List[Candidate], **_) -> List[Promo]:
        from parser import events_in
        out = []
        for c in cands:
            ep = episodes[c.episode]
            a0, b_last = c.segments[0][0], c.segments[-1][1]
            first = next(iter(events_in(ep, a0, b_last, ("dialogue", "action"))), None)
            last = (events_in(ep, c.segments[-1][0], b_last, ("dialogue", "action")) or [None])[-1]
            s = min(10.0, 10 * c.heuristic + 2.0)
            q = lambda e: (f"{e.header}: {e.text}" if e and e.type == "dialogue" else (e.text if e else ""))[:220]
            flags = ["heuristic-only scoring (no LLM)"]
            if not any(e.type == "dialogue" and e.start_sec < a0 + 3 for e in events_in(ep, a0, a0 + 3)):
                flags.append("no dialogue in first 3s")
            ev = PromoEvaluation(hook_description=q(first), cliffhanger_description=q(last),
                                 why_it_works=f"keyword prior {c.heuristic:.2f}", hook_score=s, cliffhanger_score=s,
                                 standalone_clarity_score=5.0, failure_flags=flags)
            out.append(Promo(candidate_id=c.id, kind=c.kind, episode=c.episode, total_duration=c.duration,
                             segments=build_segments(c), evaluation=ev, heuristic_prior=c.heuristic, evaluator=self.model))
        return out

