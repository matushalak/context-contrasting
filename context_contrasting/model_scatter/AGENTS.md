# Model scatter — goals & target behaviour (AGENTS.md)

Reference for what the model rotated-sector scatterplots **should** look like,
distilled from the Seignette et al. chronic-imaging data
(`data_analysis/transitions>threshold.ipynb`, threshold **0.3**) and from the
tuning history. The current best generator is `run_model_scatter_pruned_mini.py`;
its latest results live in `outputs_pruned_mini_finally/`. Concrete next steps to
get publication-ready are in **`FinalTuningPlan.md`**.

Tune against the **cloud shapes and the naive→expert transitions first**, the
percentages second. **Both the final figure and all intermediate experiments use
the `--n-steps-per-phase 300 --training-trials 7 --test-trials 4` regime** (steps
per phase / nTR / nTE) — always pass these flags. Iterate at `--n-samples 250
--n-jobs 10` (~1 min). Visualize each template's traces with `--plot-center-panels`
→ `center_panels/`.

---

## 0. The model in one paragraph

A population of single L2/3 PyC + PV two-compartment cells (`minimal2`). Only the
**PyC synapses** `w_ff` (anti-Hebbian, adapts down), `w_fb` (feedback, strengthens
+ generalizes) and `w_lat` (surround) are plastic; **PV tuning is fixed**. Each
cell is probed naive, trained on the two familiar images (the only plastic phase),
and re-probed expert. Read out per stimulus: **NO** (full image) and **O**
(occluded = feedback/ecRF only), z-scored to the cell's naive baseline. The
naive→expert shift `(dNO, dO)` gives the rotated sector. Familiar = pooled images
1&2; novel = image 3. Cells are drawn from a small set of **principled templates**
keyed by **tuning width** (narrow vs broad) with **systematically assigned FF and
FB strengths** and **generalized feedback**.

---

## 1. Ground-truth proportions (rotated sectors, threshold 0.3)

| population | +NO | +O | −NO | −O | small Δ | dominant |
|---|---|---|---|---|---|---|
| **familiar** (post) | 8.4 | 13.9 | 32.5 | 7.2 | 38.0 | **−NO** |
| **novel** (post) | 25.3 | 9.6 | 14.5 | 7.2 | 43.4 | **+NO** |

Soft targets (±a few points). Shape requirements (§3–4) take priority on conflict.

---

## 2. What the current version already does RIGHT — KEEP THESE

These were hard-won and must be preserved through any further tuning:

0. **Clouds are centred on the axes** — O responders sit at **NO ≈ 0** and NO
   responders at **O ≈ 0**, in *all four* panels (naive/expert × familiar/novel).
   This required a fine balance of inhibition against feedback and is the single
   most important property. **Do not break it.**
1. **Proportions match well** (familiar +NO/−NO, novel +NO especially). We
   **own up to −O being ~0**: the PV cells are not plastic and we do not model
   adaptation in higher visual areas, so a feedback-driven cell that changes lands
   in −NO, not −O. This is a stated, principled limitation, not a bug to chase.
2. **Naive familiar ≈ naive novel**, both with clear O and NO responder
   populations. Plasticity then separates them correctly: **familiar experts show
   increased separation** between O and NO responders, while **novel experts show
   the opposite** (less separation, plus cells that strengthen *both* O and NO).
3. **The center-panel example traces look great.** Every template is constructed
   principledly within a now-robust framework (shared parameters; systematic FF
   and FB strength levels), far less finicky than the legacy 17-template set. This
   principled, example-able structure is a core strength — keep it.

---

## 3. Magnitude & density conventions (z-scored SD)

- Responses live in **0–3 SD**. The clouds should be **continuous from weak (≈0)
  upward**, never detached blobs.
- **Naive**: the **extreme ends (z > 1.5) should be relatively sparse**; most naive
  responders are weak/moderate.
- **Expert**: plasticity strengthens responders so the cloud **still starts at weak
  but now has more points out at z 2, 2.5, even 3**.
- **Expert O-responder cloud should not exceed ~2** (realistically O caps ~2).
- **NO-responder cloud (especially expert) may reach up to ~3.**

---

## 4. FAMILIAR scatter — desiderata

### Naive (Pre)
- A **distinct O-responder cloud** at **NO ≈ 0**, O from ~0 up to ~1.5–2 (sparse at
  the top), **continuous up from the unresponsive cloud** (no gap).
- A **weak O&NO cloud** at **NO 0.3–1.0, O 0.3–0.8** — broadly-tuned cells weakly
  responding to *both* (this subpopulation is currently **missing/undersampled**;
  see FinalTuningPlan). It should resemble a lower-amplitude version of the *novel
  expert* response of the `mid_broad_FB_*` cells.
- **NO responders** spanning the whole **0–1.0+** band (not only the extreme),
  O ≈ 0 — these will either adapt to −NO or (with FB) move into the O cloud.
- General **spread** that plasticity then tightens.

### Expert (Post)
- The **+O responder cloud** centred at **NO ≈ 0**, sitting **directly above and
  continuous with** the unresponsive/−NO cloud — **not** a detached high-O blob.
  Fed by: cells moving straight **UP** from unresponsive/weak responders, and
  **UP-and-LEFT** from weak-O&NO and NO responders (FF adapts, FB takes over).
  O spans 0.3 → ~2 (sparse at top).
- **Increased O↔NO separation** vs naive (the −NO-dominant signature).
- **+NO cells centred at O ≈ 0**, and present **all across** the NO cloud (weak to
  strong), not just at the extreme.

---

## 5. NOVEL scatter — desiderata

### Naive (Pre)
- An **O-responder cloud** (NO ≈ 0, O up to ~1.5).
- **NO responders of varying strength** (NO up to ~2, O ≈ 0) — many of these are
  the cells that will become −NO or +NO at expert.
- **unresponsive→novel** cells start **near the origin** (suppressed at naive).
- Sparse extreme ends (z > 1.5).

### Expert (Post)
- A **strong, dense +NO cloud** at **O ≈ 0** (NO 0.5–3) — the dominant feature
  (~25%). Fed by three sources, all needed: (a) unresponsive→novel-NO, (b) weak
  naive-NO → stronger, (c) strong naive-NO → stronger.
- A **pure +O cloud** at NO ≈ 0.
- A **mixed +NO+O cloud** at NO 0.5–2, O 0.3–1 (~(1,1)) — cells strengthening
  *both*, because on novel FF stays intact (→NO) **and** FB grows (→O).
- A real **−NO population** (~14.5%): naive novel NO responders whose NO *drops*
  because PV/lateral inhibition strengthens with no feedback to counteract it. The
  ideal picture: many **naive novel NO responders (yellow, up to z≈2) are replaced
  by +NO (blue, up to z≈3)** at expert. This is currently **under-represented**
  (model −NO ≈ 6 vs 14.5; see FinalTuningPlan).
- Less O↔NO separation than familiar; reaching/strong end up to z≈3 for NO, ≤2 for O.

---

## 6. The core transition logic (where cells go / come from)

Both populations use the **same cells**; they differ only because training is on
the familiar images:

- **Familiar**: FF **adapts down**, FB **strengthens**. Naive NO responders →
  expert **−NO** (no FB) or travel **up-and-left into the +O cloud at NO≈0** (FF
  adapts away, FB takes over). Narrow FB cells → **+NO** via gain.
- **Novel**: FF does **not** adapt + FB still strengthens (generalization). Every
  responder should **move up (+O) and/or right (+NO)**; cells with FB but
  strengthened surround and no FF support can also go **−NO**. No static cloud.

---

## 7. How each feature is produced (current templates)

Two width classes (`WIDTH_CLASSES`: broad / narrow). FF strength levels
(`FF_STRENGTHS`: silent / very_weak / weak / mid / strong) and FB levels
(`FB_LEVELS`: none / weak / mid / strong / strong_sat) are assigned **per
template**, independent of width. `tuning` ∈ {all, permuted1, novel} sets which FF
channels a cell prefers; `context` ∈ {none, all, random1/2, familiar, novel} masks
which context channels deliver feedback (lets a broad cell receive FB from a subset
of stimuli). FB is otherwise **generalized** (equal across received channels).

| cloud / feature | template(s) |
|---|---|
| small Δ background | `silent_broad_FFonly` |
| naive O-responder cloud (NO≈0, O>0) | `silent_broad_FB_strong` (high FB, limited headroom), with `silent_broad_FB_weak/mid/partial2` filling weaker O |
| familiar −NO (strong naive NO adapts) | `mid_broad_FFonly`, `weak_broad_FFonly` (fills the 0.3–0.5 NO band) |
| naive-NO/O → expert-O movers; novel mixed | `mid_broad_FB_weak`, `mid_broad_FB_partial2`, `strong_broad_FB_strong` |
| +NO (emergent fam/novel asymmetry) | `narrow_weak`, `narrow_mid` (random preferred image) |
| novel +NO / unresponsive→novel | `narrow_novel` (novel-tuned; gain<1 at naive → starts ~0, FB growth amplifies) |

### Key levers / rules
- **FB learning rate is the engine** of every learning-induced change; too small →
  everything stuck at naive (small Δ).
- **+O cloud at NO≈0**: familiar FF must fully adapt **and** the growing FF→PV
  surround cancels the feedback drive on the full image. Use surround *timing*
  (low initial `w_lat`, higher `lr_lat`) — not a high initial surround (which kills
  the naive NO cloud).
- **+NO cells at O≈0**: keep the **narrow baseline low** (low adaptation current →
  the rising gain barely drags the occluded response negative) and the narrow FF
  strong enough that the NO shift dominates.
- **Continuity / no detached blobs**: see FinalTuningPlan §1 (subtractive drive
  threshold; FB↔inhibition balance; fill with weak-O&NO movers).
- **Permutation, not hand-assignment** for the narrow +NO asymmetry; only
  `narrow_novel` is a deliberate, endorsed special case.

---

## 8. Known hard constraints (don't chase)

- **−O ≈ 0** is expected and stated (fixed PV; no modeled higher-area adaptation).
- Filling weak-O responders can push **familiar +O** above GT — accepted trade.
- Magnitude vs density are coupled through the fixed 0.3 z-threshold; decouple via
  FF strength (NO) vs baseline/adaptation (O drift), not by rescaling the z-floor.

---

## 9. Tuning workflow (optional)

**Drive the tuning from `FinalTuningPlan.md`** — its goals, levers and ordering are
the primary plan. The notes below are just the mechanics.

- Run (always in the `300 / 7 / 4` regime):
  `python -m context_contrasting.model_scatter.run_model_scatter_pruned_mini --n-jobs 10 --n-steps-per-phase 300 --training-trials 7 --test-trials 4 --plot-center-panels`
  → scatter + per-template trace panels.
- Compare to GT: aggregate rows of `outputs_*/summaries/sector_fractions.csv`.
- **Green / black dashed-circle screenshots** are *not* the main workflow — they are
  just for ad-hoc back-and-forth follow-ups: a **green dashed circle** on a
  `todo_*.png` means fill that region (add/tune a config), a **black dashed circle**
  means remove points there, and un-circled clouds are fine as-is.
