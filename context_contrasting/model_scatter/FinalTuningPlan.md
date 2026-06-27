# Final tuning plan — to publication-ready

Concrete exploration goals and levers to close the remaining gaps in
`run_model_scatter_pruned_mini.py` (baseline: `outputs_pruned_mini_finally/`).
Read `AGENTS.md` first for the targets and the properties that must be
**preserved** (axis-centred clouds; principled templates; matched proportions).

**Run regime — for the final figure AND every intermediate experiment:**
`--n-steps-per-phase 300 --training-trials 7 --test-trials 4` (steps per phase /
nTR / nTE). Always pass these flags so results are comparable.

Workflow per experiment: edit → `python -m
context_contrasting.model_scatter.run_model_scatter_pruned_mini --n-jobs 10
--n-steps-per-phase 300 --training-trials 7 --test-trials 4 --plot-center-panels
--output-dir outputs_pruned_<tag>` → eval fractions vs GT → view familiar+novel
scatters and the center panels. Change **one lever at a time** and confirm the
§AGENTS "keep" properties survive.

Current fractions (baseline): familiar +NO9.6/+O18.4/−NO28.4/−O0/sm43.6;
novel +NO23.2/+O10/−NO6/−O0/sm60.8. (GT: fam 8/14/33/7/38; nov 25/10/14/7/43.)

Priorities: **G1 (smooth/attach the familiar +O cloud)** and **G3 (novel −NO)** are
the biggest visible gaps; G2 and G4 are continuity/aesthetic but matter for the
figure.

---

## G1 — Familiar expert O cloud: smooth & attached (highest priority)

**Problem.** The +O cloud is too extreme and **detached** from the unresponsive
cloud it should grow out of — there is a vertical gap between O≈0 and the O plume.

### Lever G1a — subtractive (shifted-ReLU) apical drive  ★ try first
The hard apical-drive threshold `f(x) = (x>θ)·x` makes the feedback drive **jump
discontinuously by ~θ** when `y_fb` crosses θ, producing the gap. The subtractive
form `max(0, x − θ)` ramps continuously from 0.

**Already wired (June 2025).** `ThresholdReLU` (`context_contrasting/utils.py`) now
takes `subtractive`, `hasMax`, `maxValue`. `minimal_s.CCNeuron` exposes
**`apical_drive_subtractive`** (default `False` = current hard behaviour;
`hasMax=False`). The cellular activation is `subtractive=False, hasMax=True,
maxValue=1.0`. With `apical_drive_subtractive=False` the output is **bit-identical**
to `outputs_pruned_mini_finally`.

- **Run the experiment** (subtractive=True, hasMax=False, nothing else changed):
  set `apical_drive_subtractive=True` for the pruned cells — easiest is one line in
  `_configure_pruned_variant`: `generic_config["apical_drive_subtractive"] = True`
  (or flip it in the `broad` config in `minimal2/config_s.py`). Re-run in the
  300/7/4 regime and compare.
- **Re-tune after**: the effective drive drops by ~θ, so the FB-driven O responders
  weaken — expect to **lower drive thresholds** and/or **raise FB** a little to
  recover the O cloud, but now as a continuum from 0.
- Success: the +O cloud is continuous from O≈0 upward (no gap), still at NO≈0,
  O capped ≲2.

### Lever G1b — FB ↔ inhibition balance for FB-driven cells
If a residual gap/over-extension remains, the FB strengthening may be too
aggressive or the surround on purely-FB-driven cells mis-set.
- Lower `lr_fb` (global engine) slightly and/or trim the strong FB headroom
  (`FB_LEVELS["strong"/"strong_sat"]` centers/hi).
- `w_pv_lat` (`pvlat`) lets PyC activity recruit PV and **suppress a purely
  FB-driven cell** — tune it on the `silent_broad_FB_*` templates to set how far up
  the O cloud reaches and how tightly it hugs NScalar≈0.
- Success: O magnitudes land in 0.3–2 with the bulk at low–mid O (sparse top).

### Lever G1c — fill the gap with the missing weak-O&NO movers  ★ also do this
The cleanest fill: a broadly-tuned subpopulation that in **naive familiar** weakly
responds to **both** NO and O, then moves **slightly +O and −NO** (up-and-left).
Its naive familiar response should look like the **novel-expert** response of
`mid_broad_FB_weak / _partial2 / strong_broad_FB_strong` but at **lower amplitude**.
- Add / retune a `mid_broad_FB_*` variant with **mid FF + weak–mid FB + partial
  context** (fewer feedback sources — experiment with `context` ∈
  {`random1`,`random2`,`all`}) and **lower baseline/amplitude** so naive sits at
  NO~0.3–0.7, O~0.3–0.5 and the expert lands in the lower +O band (O 0.3–1, NO≈0).
- The number of FB sources (`context` mask) is an allowed degree of freedom here.
- Success: the familiar expert +O cloud is continuously populated from the
  unresponsive cloud up through O≈1 before the stronger plume.

---

## G2 — Familiar +NO spread across the whole NO range

**Problem.** Narrow familiar +NO cells appear only at the extreme of the NO cloud;
blue dots should be **all over** the expert NO responder cloud (weak→strong), with
some also small-Δ (that part is fine).

**Mechanism.** Narrow +NO cells should span **naive NO 0–1.0**, and the increased
FB gain then slides each rightward along that continuum (graded, not all-or-none).

- Widen the narrow FF init range so naive NO covers 0–1.0: increase the FF init
  **rel-noise / hi** on `narrow_weak` and `narrow_mid` (and/or add a third narrow
  FF level) so the preferred-channel weight spans a continuum rather than two
  clusters.
- Make the gain amplification **graded**: the `gain_threshold`/`gain_k` sigmoid
  should move weak and strong naive-NO cells proportionally (check the center
  panels). Avoid a steep sigmoid that only fires the strongest.
- Confirm +NO cells stay at **O≈0** (keep narrow baseline low — an AGENTS "keep").
- Success: continuous blue band across expert NO 0.3–3, fed from a continuous
  naive NO 0–1.0.

---

## G3 — Novel −NO population (yellow→blue replacement)

**Problem.** Novel −NO ≈ 6% vs GT 14.5%. The data shows naive novel NO responders
being **replaced** by +NO responders after learning.

**Mechanism (already present, lean into it).** `mid_broad_FFonly` cells respond to
novel at naive (intact FF), have **no feedback**, and their NO **drops** at expert
because the lateral/PV surround strengthened during familiar training. So a naive
novel **NO responder (yellow, up to z≈2)** becomes **−NO**, while narrow_novel and
narrow cells add **+NO (blue, up to z≈3)** in the same NO band — the replacement.

- Increase the weight of the `*_broad_FFonly` (no-FB) broad cells so more naive
  novel NO responders exist and drop to −NO. Balance against G-keep (don't recreate
  a static extreme-NO naive cloud — cap their FF so naive novel NO ≤ ~2, and ensure
  the surround growth actually pulls them **down** at expert).
- Strengthen the **surround growth** for these cells so the novel NO drop is clear:
  raise `lr_lat` and/or their initial `w_lat`/`W_pv` so expert novel NO < naive
  novel NO by > 0.3 (lands in −NO). Verify on center panels (the novel column
  should show a clear naive→expert NO decrease).
- Optionally add a dedicated **`broad_novelNO_drop`** template (broad mid FF, no FB,
  high surround-growth) to make this transition unmistakable.
- Watch the coupling: more broad-FFonly raises familiar −NO too (already ~28, fine)
  and lowers novel small-Δ (currently 60.8, too high — this helps).
- Success: novel −NO ≈ 12–15%; in the figure, naive novel yellow NO responders are
  visibly replaced by blue +NO at expert; novel small-Δ drops toward ~43.

---

## G4 — Naive sparse at extremes, expert denser & higher; O≤2, NO≤3

**Problem.** Naive clouds should taper (sparse for z > 1.5) and **plasticity**
should push expert clouds denser and out to z 2–3; expert O should not exceed ~2
while expert NO may reach ~3.

- **Lower naive amplitude, let plasticity build it.** Reduce initial FF/FB
  **centers** (so naive responders are mostly weak/moderate) and rely on FB/gain
  strengthening to reach the strong end at expert. This is consistent with G1a
  (subtractive drive lowers naive O too).
- **Cap expert O ≤ 2**: keep limited FB headroom (`strong_sat` already caps the O
  shift); if O responders overshoot, lower their FB `hi` or raise their baseline.
- **Allow expert NO → 3**: ensure narrow +NO clip/gain permits NO up to ~3 (don't
  over-cap).
- Success: naive clouds visibly taper; expert clouds are denser and reach 2–3 for
  NO, ≤2 for O, continuous from weak.

---

## Guardrails (verify every iteration)

1. O responders at **NO≈0**, NO responders at **O≈0** in all four panels (AGENTS §2.0).
2. Naive familiar ≈ naive novel; familiar expert = more O/NO separation, novel
   expert = less (+ some both-strengthened).
3. Center panels still principled and example-able.
4. −O stays ~0 (expected; do not add machinery to force it).
5. Re-check the GT proportions after each change; don't regress familiar +NO/−NO or
   novel +NO while fixing the others.

Suggested order: **G1a (subtractive drive) → G1c (weak-O&NO movers) → G3 (novel
−NO) → G2 (+NO spread) → G4 (amplitude polish)**, re-checking guardrails each step.
