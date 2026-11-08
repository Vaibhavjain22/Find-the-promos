"""Module 4: batch pipeline runner.

  python main.py --input input --out outputs            # real run, needs OPENAI_API_KEY
  python main.py --input input --out outputs --dry-run  # no LLM, keyword scores (smoke test only)
"""
import argparse
import json
import os
import time

import pandas as pd

try:  # optional: read OPENAI_API_KEY from a local .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from evaluator import HeuristicEvaluator, PromoEvaluator
from parser import generate_candidates, load_episodes
from ranker import nms_per_episode, rank_global, top_n


def flatten(rank_of, p):
    e = p.evaluation
    return {
        "rank": rank_of.get(p.candidate_id, ""), "candidate_id": p.candidate_id, "episode": p.episode, "type": p.kind,
        "segments": " + ".join(f"{s.start_time}-{s.end_time}" for s in p.segments),
        "total_duration": p.total_duration, "hook": e.hook_description, "cliffhanger": e.cliffhanger_description,
        "why_it_works": e.why_it_works, "hook_score": e.hook_score, "cliffhanger_score": e.cliffhanger_score,
        "clarity_score": e.standalone_clarity_score, "composite_score": e.composite_score, "rank_score": p.rank_score,
        "failure_flags": "; ".join(e.failure_flags), "evaluator": p.evaluator,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--top-continuous", type=int, default=3, help="continuous candidates/episode sent to the LLM")
    ap.add_argument("--top-stitched", type=int, default=1, help="stitched candidates/episode sent to the LLM")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="skip the LLM; keyword scores only")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N episodes (cheap test run)")
    ap.add_argument("--rerank", action="store_true", help="skip parsing/LLM; re-run NMS + ranking on <out>/promos_master.json (free)")
    a = ap.parse_args()

    if a.rerank:
        from evaluator import Promo
        saved = json.load(open(os.path.join(a.out, "promos_master.json"), encoding="utf-8"))
        promos = [Promo(**p) for p in saved["promos"]]
        meta = saved["meta"]
        deduped = nms_per_episode(promos, a.iou)
        ranked = rank_global(deduped)
        curated = top_n(ranked, a.top)
        rank_of = {c["promo"]["candidate_id"]: c["rank"] for c in curated}
        meta.update(after_nms=len(deduped), iou_threshold=a.iou, reranked=True)
        json.dump({"meta": meta, "promos": [p.model_dump() for p in ranked]}, open(os.path.join(a.out, "promos_master.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        pd.DataFrame([flatten(rank_of, p) for p in ranked]).to_csv(os.path.join(a.out, "promos_master.csv"), index=False, encoding="utf-8-sig")
        json.dump({"meta": meta, "top": curated}, open(os.path.join(a.out, "top_ranked_promos.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"reranked {len(promos)} saved promos -> {len(deduped)} after NMS -> top {len(curated)}")
        return

    t0 = time.time()
    eps = load_episodes(a.input)
    if a.limit:
        eps = eps[:a.limit]
    if not eps:
        raise SystemExit(f"no .txt episodes found in {a.input}")
    episodes = {e.name: e for e in eps}
    print(f"parsed {len(eps)} episodes, {sum(len(e.events) for e in eps)} events")

    cands = [c for e in eps for c in generate_candidates(e, a.top_continuous, a.top_stitched)]
    assert all(20.0 <= c.duration <= 60.0 for c in cands), "duration constraint violated"
    print(f"{len(cands)} pre-filtered candidates ({sum(c.kind == 'stitched' for c in cands)} stitched)")

    if a.dry_run:
        evaluator = HeuristicEvaluator()
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set. Set it, or use --dry-run.")
        evaluator = PromoEvaluator(model=a.model)
    promos = evaluator.evaluate(episodes, cands)

    deduped = nms_per_episode(promos, a.iou)
    ranked = rank_global(deduped)
    curated = top_n(ranked, a.top)
    rank_of = {c["promo"]["candidate_id"]: c["rank"] for c in curated}
    print(f"{len(promos)} evaluated -> {len(deduped)} after NMS(IoU>={a.iou}) -> top {len(curated)}")

    os.makedirs(a.out, exist_ok=True)
    master = {
        "meta": {"evaluator": evaluator.model, "episodes": len(eps), "candidates_evaluated": len(promos),
                 "after_nms": len(deduped), "iou_threshold": a.iou, "runtime_s": round(time.time() - t0, 1)},
        "promos": [p.model_dump() for p in ranked],
    }
    json.dump(master, open(os.path.join(a.out, "promos_master.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    pd.DataFrame([flatten(rank_of, p) for p in ranked]).to_csv(os.path.join(a.out, "promos_master.csv"), index=False, encoding="utf-8-sig")
    json.dump({"meta": master["meta"], "top": curated}, open(os.path.join(a.out, "top_ranked_promos.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"done in {time.time() - t0:.1f}s -> {a.out}/")


if __name__ == "__main__":
    main()

