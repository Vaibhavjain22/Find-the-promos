# Kuku — AI Generalist Assignment: Solution Document
**Problem 1: Find the Promos (Ads)**  
**Candidate Name:** Vaibhav Jain
**Roll Number:** 22164020  
**Links:** [GitHub Repository & Code](#) | [Raw Outputs Folder](#) | [Streamlit Dashboard Demo](#) | [Walkthrough Video (Optional)](#)

---

### 1. Problem Choice, Framing & Assumptions (Lens 1)
* **Why Problem 1:** In AI microdrama (1–2 min vertical episodes), user acquisition (UA) on TikTok, Instagram Reels, and YouTube Shorts is the primary growth driver. Today, human editors manually scrub 50+ episodes weekly for 20–60s ad creative—a severe production bottleneck. Automating promo discovery directly accelerates testing velocity and improves ad ROAS.
* **Scope & Constraints:** Automatically extract continuous cuts and 2-segment stitched cuts strictly between 20.0s and 60.0s from screenplay text with explicit timestamps, then evaluate, rank, and critique them.
* **Explicit Assumptions (where the brief was silent):**
  1. *Per-Episode Coordinates:* Each script resets timestamps at `00:00.00`. Stitched clips join beats within the same episode (cross-episode stitching without shared visual context disorients cold viewers).
  2. *Cold-Open Hook Rule:* A 3-second scroll-stopping hook must open on spoken dialogue or visceral physical action—never on silent camera staging or ambient sound effects.
  3. *Cliffhanger Structure:* Microdrama scripts place dramatic climaxes at episode boundaries; cuts ending near episode boundaries naturally maximize unresolved intrigue.
  4. *Human-in-the-Loop:* The tool serves as an editorial copilot that ranks high-potential candidates and flags failure warnings for human approval, not an unreviewed auto-publisher.

---

### 2. Architecture, Approach & Cost at Scale (Lens 2)
To eliminate LLM timestamp hallucinations and prevent token waste, deterministic logic handles timecode math while the LLM acts solely as a semantic judge:
* **Deterministic Engine (`parser.py` & `ranker.py`):**
  - Regex parses `(MM:SS.ss)` timecodes into exact floating-point seconds.
  - Slices single-scene continuous windows (20–60s) starting strictly on high-impact verbs (*slap, accuse, poison, betray*).
  - Generates stitched candidate pairs: a 3–8s early hook beat + 15–40s climax cliffhanger beat (no micro-segment < 3.0s).
  - Pre-filters windows via keyword priors, sending only the top 4 candidates/episode to the LLM.
* **LangChain Semantic Critic (`evaluator.py`):**
  - Passes camera-free, timestamped dialogue windows to `ChatOpenAI(model="gpt-4o-mini", temperature=0.1)`.
  - Pydantic v2 structured output extracts: `hook_description`, `cliffhanger_description`, `why_it_works`, scores (0–10) for hook, cliffhanger, and standalone clarity, and `failure_flags`.
  - Computes composite score in deterministic Python: $0.4 \times \text{Hook} + 0.4 \times \text{Cliffhanger} + 0.2 \times \text{Clarity}$.
* **1D Temporal NMS & Deduplication (`ranker.py`):**
  - Suppresses overlapping cuts using temporal Intersection-over-Union ($\text{IoU} \ge 0.6$).
  - Deducts penalty points for critical flags ($-1.5$ for jarring stitch cuts, $-0.5$ for severe backstory dependence).
* **Cost & Scalability Analysis:**
  - Evaluated all 50 episodes (181 candidate windows, ~84k input tokens) for **~$0.04 total API cost** ($0.0008/episode) in **70 seconds**.
  - Scales linearly across an entire production slate ($0.40 for 10 seasons / 500 episodes). Swappable to open-source models (Llama 3 / Qwen) via LangChain.

---

### 3. Top Promos to Post First & Why (Problem 1 Requirement)
From 161 deduplicated promos across all 50 episodes, here are the top curated promos to post first:

| Rank | Episode & Segments | Type & Len | Hook (First 3s) | Cliffhanger Cutoff | Why It Works |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **#1** | **Ep 36** `02:26.50–03:20.75` | Continuous (54.3s) | Alexander learns of Rupert & Catherine's wedding and Juliana's fate. | Alexander vows to return from the dead and take revenge before the wedding. | High revenge stakes and iconic one-liner cutoff force immediate viewer retention. |
| **#2** | **Ep 32** `01:56.90–02:32.00` | Continuous (35.1s) | Juliana confronts Catherine: *"You never learned to make anyone love you."* | Catherine raises a red-hot branding rod directly toward Juliana's face. | Visceral, imminent physical danger and intense hostility compel scrollers to watch. |
| **#3** | **Ep 39** `01:15.80–01:38.00` | Continuous (22.2s) | Alexander asserts authority: demands to reclaim his house and rightful wife. | Sudden cut to black as Alexander vows to bring Juliana home. | High emotional conviction with an abrupt cut creates urgent curiosity. |
| **#4** | **Ep 46** `00:42.80–01:02.80` | Continuous (20.0s) | High Priest announces royal wedding is invalid due to lack of royal license. | Juliana defiantly states she didn't need anyone's permission to survive. | Institutional scandal combined with female lead defiance creates instant engagement. |
| **#5** | **Ep 45** `00:54.00–01:14.00` | Continuous (20.0s) | Alexander accuses Catherine of poisoning his mother and torturing his wife. | Alexander shocks the room by declaring Juliana his legitimate wife. | Public confrontation and shocking marital reveal deliver high ad click-through. |

---

### 4. Quality Checking, Failure Modes & Where It Picks Badly (Lens 3)
A rigorous audit across all 50 episodes revealed critical limitations:
1. **The Stitched-Cut Disconnect:** While continuous cuts succeeded, **22 out of 22 surviving stitched cuts** received the `jarring stitch transition` warning. Pairing an early hook with a late cliffhanger lacks narrative causality; without dialogue bridge overlays, automated stitching feels disjointed to cold viewers.
2. **Backstory Bias in Later Episodes:** 119 of 161 candidates were flagged with `needs earlier episodes`. The LLM heavily favored climax scenes in late episodes (Eps 32–48) because confrontation stakes are higher, despite being harder for new viewers to understand without context.
3. **Clarity Score Clustering:** 146 of 161 candidates received a clarity score of exactly 5.0/10, showing that single-episode loglines provide insufficient context for the LLM to differentiate between mild and severe backstory confusion.
4. **Recall Verification (Sanity Check):** In hand-labeled episodes (e.g., Episode 1: Altar Abandonment at 01:30–02:17, and Episode 4: Confrontation at 01:55–02:29), the deterministic window slicer achieved **100% candidate recall**, proving that high-potential emotional moments are never missed.

---

### 5. Prototype & Tools Used: Where AI Got It Wrong (Lens 4 & 5)
* **Streamlit Web Application (`app.py`):**
  - **Leaderboard:** Interactive table of all deduplicated promos ranked by composite score, with filters for Continuous vs. Stitched cuts and minimum score sliders.
  - **Script Highlighter:** Full episode screenplay viewer highlighting exact promo timestamps against dialogue.
  - **Interactive Single-Episode Analyzer:** A dedicated upload tab allowing editors to drop in a single `.txt` episode during live review or interview demonstrations, extracting hooks, cliffhangers, and quality warnings on the fly.
  - **One-Click Exports:** Instant download of raw `promos_master.json`, `promos_master.csv`, and `top_ranked_promos.json`.
* **Tools Utilized:** Claude 3.5 Sonnet / Antigravity for pipeline scaffolding; LangChain + OpenAI `gpt-4o-mini` for structured evaluations; Streamlit for UI dashboard.
* **Where AI Got It Wrong:** In initial runs, the LLM suffered severe grade inflation—assigning 8.0–9.0 across almost all candidates with zero failure flags. It had to be re-engineered with explicit boundary anchors (e.g., *“3-4 = greeting/exposition, 9-10 = explicit slap/threat”*) and mandatory failure flags for any sub-7 score, which forced genuine score variance (3.8 to 8.8). Furthermore, an automated regex penalty was required in Python because the LLM consistently gave high numerical scores to stitched cuts that it simultaneously described as "jarring".
