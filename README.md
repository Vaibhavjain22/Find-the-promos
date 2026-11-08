# Microdrama Promo Finder

Finds 20–60s promo clips (continuous or 2-segment stitched) in timestamped microdrama screenplays, scores them with an LLM, de-duplicates and ranks them across a show.

## Quickstart & How to Run

### 1. Installation & Environment Setup
Clone the repository and set up a virtual environment:
```bash
git clone <your-repo-link>
cd Promos

# Create and activate virtual environment
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY="sk-..."
```
*(Alternatively, set it in your environment: `export OPENAI_API_KEY="sk-..."` or in PowerShell `$env:OPENAI_API_KEY="sk-..."`)*

### 3. Launch the Interactive Web Dashboard (UI)
```bash
streamlit run app.py
```
Open `http://localhost:8501` to:
- Browse top ranked promos across the series.
- View scripts with highlighted promo cuts.
- **Upload and analyze a single new episode on the fly** in the *Analyze a New Episode* tab.

### 4. Run Batch Pipeline via CLI (All 50 Episodes)
```bash
# Full LLM-scored run across all screenplays (gpt-4o-mini)
python main.py --input input --out outputs

# Re-rank saved results instantly without spending API tokens
python main.py --rerank --out outputs

# Offline dry-run for smoke testing without LLM
python main.py --dry-run --out outputs_dryrun
```

Generated outputs will be saved in `outputs/`:
- `promos_master.json`: Full nested schema with all evaluated promos.
- `promos_master.csv`: Flattened tabular view for quick review.
- `top_ranked_promos.json`: Curated top picks to post first with marketing rationales.

## Streamlit Web UI (`app.py`)

The interactive web dashboard is designed for editorial teams and live interview evaluations (`streamlit run app.py`):

1. **Leaderboard (`Tab 1`):** Ranked overview of all deduplicated promos. Includes filtering by cut type (Continuous vs. Stitched) and interactive score thresholds, displaying duration, hook, cliffhanger cutoffs, and warning flags.
2. **Episode Viewer (`Tab 2`):** Episode screenplay inspector that dynamically highlights discovered promo segments directly over the dialogue and character actions.
3. **Downloads (`Tab 3`):** One-click export for raw batch outputs (`promos_master.json`, `promos_master.csv`, and `top_ranked_promos.json`).
4. **Analyze a New Episode (`Tab 4`):** Real-time promo extractor. Allows an interviewer or editor to upload any new screenplay `.txt` file, running the end-to-end pipeline (parsing, candidate slicing, LLM scoring, and ranking) on demand with zero manual configuration.

## Architecture

| Step | Tool | Module |
|---|---|---|
| Regex parse, "MM:SS.ss"/"HH:MM:SS.ss" to float seconds | code | `parser.py` |
| Window generation (single scene, 20–60s, must start on dialogue or a high-action beat) | code | `parser.py` |
| Stitched pairs: 3–8s hook from the first half + 15–40s beat ending at the episode's end, ≥5s apart, each segment ≥3s, total 20–60s | code | `parser.py` |
| Keyword prior + IoU pruning down to 3 continuous + 1 stitched per episode | code | `parser.py` |
| Hook / cliffhanger / clarity scores, descriptions, failure flags | LLM, structured output | `evaluator.py` |
| Composite score `0.4*hook + 0.4*cliff + 0.2*clarity` (recomputed in code, model arithmetic is ignored) | code | `evaluator.py` |
| Temporal NMS (IoU ≥ 0.6, per episode), global rank, top 10 | code | `ranker.py` |

The LLM never sees a full screenplay and never invents timestamps. It sees a compact, camera-free transcript of one pre-built window, and the timestamps come from code. A hard `assert` in `main.py` checks that every candidate is 20–60s. The model is set in one place (`PromoEvaluator`, `--model`). Use `ChatOpenAI(base_url=...)` to point at another OpenAI-compatible endpoint.

## Cost (measured on the 50-episode pack, default settings)

- Default: 181 candidates, about 84k input tokens (prompt + transcript, measured with a 4-chars-per-token estimate), plus roughly 36k output tokens and about 50k tokens of schema overhead (estimated).
- gpt-4o-mini at \$0.15 / 1M input and \$0.60 / 1M output: about **\$0.04–0.05** for the whole show.
- **The brief's \$0.02 figure is only reachable with fewer calls**: `--top-continuous 1 --top-stitched 1` gives about 100 candidates, roughly \$0.02–0.025. Confirm against your OpenAI usage page after the real run.
- Without the pre-filter, scoring every window would be about 100× more calls, which is why pruning is done in code.

## Assumptions

- Timestamps are per episode (each file starts at 00:00). Stitched promos never cross episodes.
- "Continuous" means one scene, so there are no hard cuts inside it.
- Episodes end on a written cliffhanger, so a window ending at the last 3s gets a prior bonus and stitched cliff beats must end there.
- Hook strength can be judged from text. Visual hooks (a slap in the camera notes) are only partly visible, because the LLM sees ACTION lines but not CAMERA lines.
- Human in the loop: an editor reviews the top ~20 (the app shows the segments highlighted in the script) before cutting.

## Failure modes

### Observed on the real run (gpt-4o-mini, 50 episodes, 181 candidates scored, 0 failed calls, ~2 min)

1. **Stitching is the weakest feature.** In the first full run all 23 surviving stitched promos were flagged "jarring stitch transition"; in the final run 22 stitched promos survived NMS and 27 promos carried that flag (I did not break the second count down by type). Pairing an early hook with the episode ending gives no causal link. None reach the top 10.
2. **Clarity score barely discriminates.** 146 of 161 promos scored 5. The model sees one episode's logline, so mid/late-series moments almost always read as "needs earlier episodes" (119 flags).
3. **Top 10 is biased to late episodes (23–48).** These score well on hook/cliffhanger but are the hardest for a new viewer to follow. A promo for new viewers would normally come from early episodes.
4. **The ranker penalty is leaky.** `ranker.py` subtracts 1.5 for "jarring stitch" and 0.5 for "earlier episodes", but the LLM writes free-text flags ("context needed for...", "unclear relationships"), so many clarity problems are not matched. It moved Episode 36 from 8.8 to 8.3 but did not change the late-episode bias. A fixed list of allowed flags would fix this.
5. **Weak descriptions occasionally.** For example the "cliffhanger" text can be a pasted dialogue line instead of a description.
6. **Cliffhanger scores cluster high** (most are 7 or 9) even with the anchored rubric.
7. **Run-to-run variance.** Two runs on the same input gave slightly different rankings (temperature 0.1, not 0).

### Expected from the design (not yet measured)

1. **Prior-driven blind spots.** The keyword prior chooses what the LLM sees. Check recall: hand-pick your best promo for 5 episodes and see if it was a candidate.
2. **Text-only judgement.** Silent visual hooks and music-driven cliffhangers are under-scored.
3. **Show bias.** The prior keywords are tuned to romance/revenge drama.
4. **One-per-episode curation** may drop a second great moment in a strong episode (still in `promos_master`).
5. **Dry run is not representative.** The offline scorer saturates at 10/10 with many ties.

## Scaling

All steps are per-episode and independent, so they parallelize and cost grows linearly (about \$0.001 per episode at the default). A new show needs a new input folder and no code changes. To go further: add a second LLM pass on the top 20 with the previous episodes' loglines for clarity, and calibrate the weights against editor picks.

## Tools used (for the solution doc)

Antigravity (Claude) wrote the first draft of the code, with the design decisions reviewed by the author. Add one concrete place the AI got it wrong, from your own run.

