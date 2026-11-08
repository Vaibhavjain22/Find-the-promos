"""Module 5: Streamlit dashboard.   streamlit run app.py"""
import glob
import json
import os

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from parser import load_episodes, to_seconds

st.set_page_config(page_title="Promo Finder", layout="wide")
st.title("Microdrama Promo Finder")

out_dir = st.sidebar.text_input("Outputs folder", "outputs")
in_dir = st.sidebar.text_input("Screenplays folder", "input")
master_path = os.path.join(out_dir, "promos_master.json")
if not os.path.exists(master_path):
    st.error(f"{master_path} not found. Run `python main.py` first.")
    st.stop()

master = json.load(open(master_path, encoding="utf-8"))
promos = master["promos"]
top = json.load(open(os.path.join(out_dir, "top_ranked_promos.json"), encoding="utf-8"))["top"]
if "heuristic" in master["meta"]["evaluator"]:
    st.warning("These results come from the offline keyword scorer (--dry-run), not the LLM. Treat scores as rough.")


@st.cache_data
def episodes(path):
    return {e.name: e for e in load_episodes(path)}


eps = episodes(in_dir) if glob.glob(os.path.join(in_dir, "*.txt")) else {}
df = pd.DataFrame([{
    "episode": p["episode"], "type": p["kind"], "duration": p["total_duration"],
    "score": p["evaluation"]["composite_score"], "id": p["candidate_id"]} for p in promos])

c = st.columns(4)
c[0].metric("Episodes processed", master["meta"]["episodes"])
c[1].metric("Promos generated", len(promos))
c[2].metric("Avg duration (s)", f"{df.duration.mean():.1f}")
c[3].metric("Avg score", f"{df.score.mean():.2f}")


def card(p, rank=None):
    e = p["evaluation"]
    title = f"{'#%d  ' % rank if rank else ''}{p['episode']}  |  {p['kind']}  |  {p['total_duration']:.1f}s  |  score {e['composite_score']:.1f}"
    with st.expander(title, expanded=bool(rank and rank <= 3)):
        for s in p["segments"]:
            st.markdown(f"**Segment {s['segment_index']} ({s['role']})**: `{s['start_time']}` to `{s['end_time']}` ({s['duration']}s)")
        st.markdown(f"**Hook (first 3s):** {e['hook_description']}")
        st.markdown(f"**Cliffhanger cut:** {e['cliffhanger_description']}")
        st.markdown(f"**Why it works:** {e['why_it_works']}")
        st.progress(e["hook_score"] / 10, text=f"Hook {e['hook_score']:.1f}")
        st.progress(e["cliffhanger_score"] / 10, text=f"Cliffhanger {e['cliffhanger_score']:.1f}")
        st.progress(e["standalone_clarity_score"] / 10, text=f"Standalone clarity {e['standalone_clarity_score']:.1f}")
        for f in e["failure_flags"]:
            st.warning(f)


tab1, tab2, tab3, tab4 = st.tabs(["Leaderboard", "Episode viewer", "Downloads", "Analyze a new episode"])

with tab1:
    f1, f2 = st.columns(2)
    kinds = f1.multiselect("Type", ["continuous", "stitched"], default=["continuous", "stitched"])
    min_score = f2.slider("Minimum score", 0.0, 10.0, 0.0, 0.1)
    shown = [p for p in promos if p["kind"] in kinds and p["evaluation"]["composite_score"] >= min_score]
    st.caption(f"{len(shown)} promos (ranked by composite score). Curated top 10 (one per episode) are marked with their curated rank.")
    curated = {t["promo"]["candidate_id"]: t["rank"] for t in top}
    for p in shown[:50]:
        card(p, curated.get(p["candidate_id"]))

with tab2:
    if not eps:
        st.info("Point 'Screenplays folder' at the .txt episodes to enable the viewer.")
    else:
        name = st.selectbox("Episode", list(eps))
        ep = eps[name]
        ep_promos = [p for p in promos if p["episode"] == name]
        spans = [(to_seconds(s["start_time"]), to_seconds(s["end_time"]), p["candidate_id"]) for p in ep_promos for s in p["segments"]]
        st.markdown(f"*{ep.logline}*")
        for p in ep_promos:
            card(p)
        st.subheader("Script (promo segments highlighted)")
        lines = []
        for e in ep.events:
            if e.type == "camera":
                continue
            hit = [pid for a, b, pid in spans if e.start_sec >= a - 0.01 and e.end_sec <= b + 0.01]
            txt = f"`{e.start_sec:6.1f}s` **{e.header}**: {e.text}"
            lines.append(f"<div style='background:#fff3b0;padding:2px 6px;border-radius:4px'>{txt} <small>[{', '.join(hit)}]</small></div>" if hit else f"<div style='padding:2px 6px'>{txt}</div>")
        st.markdown("\n".join(lines), unsafe_allow_html=True)

with tab3:
    for fn in ("promos_master.json", "promos_master.csv", "top_ranked_promos.json"):
        fp = os.path.join(out_dir, fn)
        if os.path.exists(fp):
            st.download_button(f"Download {fn}", open(fp, "rb").read(), file_name=fn)


with tab4:
    st.caption("Upload one screenplay .txt. The same pipeline runs on it: parse, candidate windows, LLM scoring, de-dup, rank.")
    up = st.file_uploader("Episode screenplay (.txt)", type="txt")
    dry = st.checkbox("Offline keyword scoring (no LLM, no cost)", value=not os.environ.get("OPENAI_API_KEY"))
    n_cont = st.slider("Continuous candidates to score", 1, 8, 4)
    n_stit = st.slider("Stitched candidates to score", 0, 3, 1)
    if up and st.button("Find promos"):
        import tempfile
        from parser import generate_candidates, parse_episode
        from evaluator import HeuristicEvaluator, PromoEvaluator
        from ranker import nms_per_episode, rank_global
        with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as tf:
            tf.write(up.getvalue())
        ep = parse_episode(tf.name)
        if not ep.events:
            st.error("No timestamped events found. Is this the expected screenplay format?")
            st.stop()
        cands = generate_candidates(ep, n_cont, n_stit)
        st.write(f"{ep.name}: {len(ep.events)} events, {ep.duration:.0f}s, {len(cands)} candidates pre-filtered")
        if dry:
            ev = HeuristicEvaluator()
        elif not os.environ.get("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not set.")
            st.stop()
        else:
            ev = PromoEvaluator()
        with st.spinner("Scoring..."):
            res = rank_global(nms_per_episode(ev.evaluate({ep.name: ep}, cands)))
        if dry:
            st.warning("Keyword scoring only, not the LLM. Scores are rough.")
        for i, p in enumerate(res, 1):
            card(p.model_dump(), i)

