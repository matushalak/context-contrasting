# Model scatter — target behaviour (desiderata)

This file is the reference for what the model rotated-sector scatterplots
**should** look like, distilled from the Seignette et al. chronic-imaging data
(`data_analysis/transitions>threshold.ipynb`, threshold **0.3**) and from the
tuning conversation. The current best-matching generator is
`run_model_scatter_pruned_mini.py`; the targets below are what to tune it against.

Tune against the **cloud shapes and the naive→expert transitions**, not the exact
percentages. The percentages are a secondary check (the data's own classes are
noisy). Iterate at `--n-samples 250 --n-jobs 10` (~38 s).

---

## 1. Axes, conventions, what a "scatter" is

- Each point is one cell × one stimulus. **x = NO** (non-occluded / full-image
  response), **y = O** (occluded / feedback-only response). Both are **z-scored to
  the cell's own naive spontaneous baseline**, so a value is "SD above baseline".
- Three panels per condition: **naive** (Pre), **expert** (Post), and the
  **shift** `(dNO, dO) = expert − naive` which defines the rotated sector.
- **Rotated sectors** (by the angle of the shift, magnitude `<0.3` = "small Δ"):
  `+NO` (blue, dNO>0 dominates), `+O` (red), `−NO` (orange), `−O` (green),
  `small Δ` (grey).
- Two populations: **familiar** = responses to the two trained images (1 & 2,
  pooled); **novel** = responses to the never-trained image (3).
- Model "expert/familiar" ≈ the data's **post-familiar** population; model "novel"
  ≈ the data's **post-novel** population.

### Magnitudes (important)
- Responses live in **0–3 SD**. **Naive ≈ 0–2 SD**; plasticity then strengthens
  responders up to **0–3 SD** at expert.
- Naive O responders should **not exceed ~2.5** (no cells at O = 2.5–3.5 in naive).
- Exclusive (pure) O or NO responders may reach the strong end (**2–3**); cells
  that respond to **both** O and NO sit at a more moderate **~(1, 1)**.

---

## 2. Ground-truth proportions to match (rotated sectors, threshold 0.3)

| population | +NO | +O | −NO | −O | small Δ | dominant |
|---|---|---|---|---|---|---|
| **familiar** (post) | 8.4 | 13.9 | 32.5 | 7.2 | 38.0 | **−NO** |
| **novel** (post) | 25.3 | 9.6 | 14.5 | 7.2 | 43.4 | **+NO** |

Treat these as soft targets (±a few points). The shape requirements below take
priority when they conflict.

---

## 3. The core transition logic (where cells go / come from)

The two populations differ **only** because training is on the familiar images:

- **On familiar images**: FF drive **adapts down** (anti-Hebbian), while feedback
  **strengthens and generalizes**. So a cell's evoked response moves **down in NO**
  and, if it has feedback, **up in O**. Net: naive NO responders become expert −NO
  (strong FF, no FB) **or** travel **up-and-left into the expert O cloud at NO≈0**
  (FF + FB: FF adapts away, FB takes over).
- **On the novel image**: FF does **not** adapt (never trained) and feedback
  **still strengthens** (generalization). So *every* responding cell should move
  **up (+O)** and/or **right (+NO)** — never sit static. This produces the +NO
  cloud and the mixed +NO+O cloud. There should be **no static extreme-NO
  population** on novel.

So the same cell type produces a familiar −NO/+O transition and a novel +NO/mixed
transition. The +NO familiar-vs-novel asymmetry is an **emergent** consequence of
which images were trained, not hand-assigned.

---

## 4. FAMILIAR scatter — desiderata

### Naive (Pre)
- A **distinct O-responder cloud** at **NO ≈ 0**, O from ~0 up to **1–2** (capped
  < 2.5). These are feedback-driven occluded responders.
- A **weak O&NO cloud** at **NO 0.5–1.2, O 0.3–0.8** — broadly tuned cells with
  moderate FF (NO) and moderate FB (O). These are the cells that will **adapt and
  join the expert O cloud**.
- **Strong NO responders** along the NO axis (NO up to ~2, O ≈ 0) — these will
  adapt to **−NO**.
- General **spread** around both the NO and O responders (plasticity then "cleans
  this up" toward the tighter expert clouds).
- −O / weakly-O cells should sit **within the O-responder cloud**, not at the
  origin.

### Expert (Post)
- The **+O responder cloud** is the signature: **centred at NO ≈ 0** (vertical,
  off the diagonal), sitting **directly above** the −NO / unresponsive populations
  — not offset left or right of them. O spans **0.3 up to ~2–2.5**.
- It must include the **weaker O responders** (O 0.3–1 with NO < 0.5), not just the
  strong ones. These come from the naive weak-O&NO and naive-NO cells moving
  **up / up-and-left**.
- **+NO cells centred at O ≈ 0** (never sitting below the NO axis).
- **−NO** spread to the left; **unresponsive / −NO** populations should have some
  spread (not a tight dot).
- Overall: familiar is **−NO dominant** with a clear vertical +O cloud.

---

## 5. NOVEL scatter — desiderata

### Naive (Pre)
- An **O-responder cloud** (NO ≈ 0, O up to ~1.5–2) — same FB-driven responders.
- **Moderate-NO low-O responders** (NO 0.3–1.5, O ≈ 0) — naive novel NO responders
  of varying strength.
- The **unresponsive→novel** cells start **near the origin** (suppressed at naive),
  i.e. there should be cells at ~0 that later become +NO.
- **Cap the extreme naive NO responders** (no static cloud at NO > 2).

### Expert (Post)
- A **strong, dense +NO cloud** along the NO axis at **O ≈ 0** (NO 0.5–3) — the
  dominant novel feature (~25%). It is fed by **three** sources, all of which must
  be present:
  1. **unresponsive → novel-NO** (cells that start ~0 and become NO responders),
  2. **weakly naive-NO → stronger** novel NO,
  3. **strongly naive-NO → stronger** novel NO.
- A **pure +O cloud** at NO ≈ 0 (feedback responders).
- A **mixed +NO+O cloud** at **NO 0.5–2, O 0.3–1** (responding to both, around
  ~(1, 1)). This is *predicted*: on novel, FF stays intact (→ NO) **and** FB grows
  (→ O), so broadly-tuned FB cells move **up-and-right** into this region.
- Pure O or pure NO responders can be **strong (2–3)**; the mixed responders are
  **moderate (~1, 1)**.

---

## 6. How each feature is produced (model mechanisms)

Two tuning-width classes + a tuning draw. Untuned cells = **weak / already-adapted
broad** cells (broad width class with near-silent FF). Feedback is always
**generalized** (equal across all 3 context channels).

| cloud / feature | produced by |
|---|---|
| small Δ background | `silent_broad_FFonly` (near-silent FF, no FB) |
| naive O-responder cloud (NO≈0, O>0), both panels | `silent_broad_FB_strong` (FB→FB-like: silent FF + strong generalized FB) |
| familiar −NO (strong naive NO → adapts) | `mid_broad_FFonly` (real FF, no FB) — keep **small**; with no FB it can't move on novel (would be a static extreme-NO cloud) |
| **naive-NO → expert-O** transition (the up/up-left movers; fills weak expert O responders + novel mixed) | `mid_broad_FB_weak`, `strong_broad_FB_strong` (broad FF **+** feedback) |
| novel **mixed +NO+O** (~1,1) | broad FF+FB cells with a **moderate FF and tight noise** so naive NO stays < 2 but the intact novel FF reaches NO~1; moderate surround so novel NO survives |
| +NO cloud (general, emergent fam/novel asymmetry) | `narrow_weak`, `narrow_mid` (narrow: weak FF plasticity + high drive threshold → FB gain-modulates; random preferred image) |
| novel +NO density / unresponsive→novel | `narrow_novel` (narrow, novel-tuned, gain<1 at naive so it starts ~0, then FB growth amplifies → +NO) |

Key levers / rules learned:
- **Feedback learning rate is the engine.** Too small → whole population stuck at
  naive (everything small-Δ). `lr_fb` is boosted vs the base.
- **+O cloud at NO≈0**: the familiar FF must fully adapt **and** the growing
  FF→PV surround (`w_lat` plastic, `W_pv` fixed) must cancel the feedback drive on
  the full image. Use surround **timing** (low initial `w_lat`, higher `lr_lat`)
  so naive NO survives but expert NO is suppressed — *not* a high initial surround
  (that kills the naive NO cloud too).
- **+NO cells at O≈0**: keep the **narrow baseline low** (low adaptation current →
  the rising gain barely drags the occluded response negative) and the narrow FF
  strong enough that the NO shift dominates the tiny O drift.
- **Naive NO ≤ 2 / expert ≤ 3**: cap broad FF (and use tight FF noise on the mixed
  class) so no extreme naive NO; the gain/FB strengthening then pushes expert to 3.
- **Most broad cells must carry feedback** so they move on novel; keep the no-FB
  `*_FFonly` populations small.
- **Permutation, not hand-assignment**: narrow cells draw a random preferred image
  (`permuted1`); only `narrow_novel` is a deliberate special case (endorsed) for
  novel +NO density.

---

## 7. Known hard constraints / residuals (don't chase these forever)

- **−O is structurally low** (~1–3% vs GT 7.2%). The occluded (O) response has no
  FF→PV surround to suppress it, so naive O responders rarely *lose* O to land in
  −O. A small −O is acceptable.
- Filling the weak-O-responder region inflates the **familiar +O** fraction above
  GT — that's an accepted trade for the requested cloud shape.
- Magnitude vs density are coupled through the z-score threshold (0.3 is fixed in
  z-units): more amplification → more cells cross threshold **and** larger
  magnitudes. Decouple via FF strength (NO only) vs baseline/adaptation (O drift)
  rather than by globally rescaling the z-floor.

---

## 8. Tuning workflow

- **Green dashed circle** on a `todo_*.png` = the cloud should fill this region
  (add or tune a config). **Black dashed circle** = remove points from here.
  Clouds without a circle are fine and should be left alone.
- Visualize each template's traces with
  `python -m context_contrasting.model_scatter.run_model_scatter_pruned_mini --n-jobs 10 --plot-center-panels`
  → `center_panels/` (one naive→expert trace panel per template).
- Compare fractions to GT with the aggregate rows of
  `outputs_*/summaries/sector_fractions.csv`.
