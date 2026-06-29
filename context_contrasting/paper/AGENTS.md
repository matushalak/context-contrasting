# Model scatter — goals & target behaviour (AGENTS.md)

Reference for what the model rotated-sector scatterplots **should** look like,
distilled from the Seignette et al. chronic-imaging data
(`data_analysis/transitions>threshold.ipynb`, threshold **0.3**) and from the
tuning history.

Tune against the **cloud shapes and the naive→expert transitions first**, the
percentages second. **Both the final figure and all intermediate experiments use
the `--n-steps-per-phase 300 --training-trials 7 --test-trials 5` regime** (steps
per phase / nTR / nTE) — always pass these flags. Iterate at `--n-samples 250
--n-jobs 10` (~1 min). Visualize each template's traces with `--plot-center_panels`.
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

**The PV surround is DIVISIVE (the key model decision).** In `minimal_divisive` the
soma is `y = act( (apical_gain·y_ff + apical_drive) / (1 + k·y_lat) + baseline_drive
− a )`: the FF→PV surround `y_lat` *normalizes* (divides) the stimulus drive rather
than *subtracting* it. This matches shunting PV biophysics and Carandini–Heeger
normalization, and it is what makes the **neutral zone** (§2.0) possible. Two rules
that fall out of the form and MUST hold:
- `baseline_drive` stays **outside** the division (additive spontaneous floor) so a
  fully-shunted full-image response floors **at** baseline, never below — that is the
  whole point. (`a`, the subtractive adaptation, is the *only* term that can go below
  baseline → the genuine −NO source.)
- `w_lat` is soft-bounded ≤1 and PV is single, so `y_lat≤1`; the **divisive gain `k`**
  (`divisive_gain`, set ≈**10**) is needed for `1+k·y_lat` to divide hard enough to
  pull O-responder NO onto the floor. Without k the surround can only halve.

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

0. **Clouds are centred on the axes, with a NEUTRAL ZONE** — O responders sit at
   **NO ≈ 0** and NO responders at **O ≈ 0**, in *all four* panels, AND there is a
   clear neutral zone: O-responders hug the O-axis just **above** NO=0 and **never
   spill into NO<0** (the real data shows this; see fig C/E). This is the single most
   important property. **Why the divisive model exists:** with *subtractive*
   inhibition, getting an O-responder to NO≈0 means cancelling the feedback drive on
   the full image by subtraction, which overshoots the positive (rectified-noise)
   spontaneous baseline → O-responders fall to NO<0 (37% of them) — a structural
   knife-edge that can't be tuned away. *Divisive* inhibition floors the full-image
   response **at** baseline (NO→0⁺, never below), so the neutral zone is reachable.
   −NO still happens (the grown surround divides the full-image NO down toward 0,
   and `a` can push slightly below) — that is fine and expected; it is just **shrinkage toward
   0**, not manufactured overshoot. **Do not break the floor** (keep `baseline_drive`
   outside the division).
1. **We own up to −O being ~0**: the PV cells are not plastic and we do not model
   adaptation in higher visual areas, so a feedback-driven cell that changes lands
   in −NO, not −O. This is a stated, principled limitation, not a bug to chase.

---

## 3. Magnitude & density conventions (z-scored SD)

- Responses should ive in **0–3 SD** (plasticity / initial tuning shouldn't be so strong to warrant z-scores > 3). The clouds should be **continuous from weak (≈0)
  upward**, never detached blobs.
- **Extreme ends (z > 2) should be relatively sparse**; most naive
  responders are weak/moderate; and even after plasticity few expert responders are strong, most evolve into moderate.
- **Expert O-responder cloud should not exceed ~2** (realistically O caps ~2).
- **NO-responder cloud (especially expert) may reach up to ~2.5.**

---

## 4. NAIVE scatter — desiderata (both for both familiar and novel naive)

### Naive (Pre)
- A **distinct O-responder cloud** at **NO ≈ 0**, O from ~0 up to ~1.5–2 (sparse at
  the top), **continuous up from the unresponsive cloud** (no gap).
- A **mixed O&NO cloud centred around (NO≈0.25 - 0.5, O≈0-0.5)** (in both familiar and naive) — broadly-tuned cells weakly responding to *both*. **FINAL-GOAL refinement:** these "both" cells should cluster near **(0.35, 0.35)**, NOT spread out to high NO *and* high O. A cell that responds strongly to both (e.g. (1.5, 1.5)) is wrong; the data's naive both-responders are modest. In familiar these cells will move along -NO/+O diagonal (equally likely to fall into either sector), adapting their FF and growing their FB to join the expert O-responder cloud. In novel, these cells will move either along the +NO/+O diagonal (if their responses are initially weak), or most likely somewhere in the +NO transition region. Should be contiguous with / border the 3 other naive clouds (with O responders, silent, and NO responders).
- **NO responders** spanning the whole **0–1.0+** band (not only the extreme),
  O ≈ 0 — these will either adapt to −NO or (with FB) move into the O cloud (if broad) or increase their NO responses (if narrow). Also in novel, many of these are
  the cells that will become −NO or +NO at expert.
- General **spread** that plasticity then tightens.

## 5. EXPERT scatter (familiar)
- The **+O responder cloud is a VERTICAL column squarely at NO≈0** (no diagonal lean
  whatsoever; O-responders must **not** drift toward the zO=zNO diagonal and must
  **not** dip below NO=0). It is fed by **TWO populations that converge onto NO≈0**:
  1. **from below** — unresponsive / weak cells that grow O straight **UP** (NO stays
     ~0, O rises); and
  2. **from the right** — **strong naive NO responders that ADAPT** (FF depresses)
     **and recruit inhibition** (surround grows → divides the residual full-image
     drive), travelling **UP-and-LEFT** to land **exactly at NO≈0** (not at NO≈0.3–0.5).
     This NO→O transition was the main thing missing; it must be clear in the
     naive→expert **vectors** and on the **traces**, and it must terminate ON the
     O-axis like population (1), not lean diagonal.
  O spans 0.3 → ~2 (sparse at top).
- **Increased O↔NO separation** vs naive (the −NO-dominant signature). *No mixed O&NO cloud in familiar expert* - it joins O responders.
- **+NO cells centred at O ≈ 0**, and present **all across** the NO cloud (weak to
  strong), not just at the extreme.

---

## 6. EXPERT scatter (novel)
- A **strong, dense +NO cloud** at **O ≈ 0** (NO 0.3–3) — the dominant feature
  (~25%). Fed by three sources, all needed: (a) unresponsive→novel-NO, (b) weak-mid
  naive-NO → stronger, (c) weak-mid naive-mixed-NO&O → stronger NO outgrows O which increases more slowly.
- A **pure +O cloud** at NO ≈ 0.
- An **expanded mixed +NO+O cloud centred around (NO≈0.5, O≈0.5), move from (NO=0.35, O=0.35 naive center)** — cells strengthening
  *both*, because on novel FF stays intact (→NO) **and** FB grows (→O). **FINAL-GOAL
  refinement:** this both-cloud should sit near **(0.5, 0.5)**, NOT spread to high NO
  with high O. The transition to watch is **NO → NO&O**: a naive NO responder keeps a
  (modest, ~0.5) NO and *gains* a (modest, ~0.5) O — visible on the **traces**, not
  swamped. (In the divisive model this means the intact novel FF must be divided down
  to ~0.5 by the grown surround — strong enough to land at 0.5, not 1+, but not so
  strong the no-LAT ablation column blows up and dwarfs the Expert trace.)
- A real **−NO population** (~14.5%): naive novel NO responders whose NO *drops*
  (shrinks toward 0) because the surround strengthens with no feedback to counteract
  it. The ideal picture: many **naive novel NO responders (yellow, up to z≈2) are
  replaced by +NO (blue, up to z≈2.5)** at expert.
- Less O↔NO separation than in familiar/novel naive; and much less than in familiar expert.

---

## 7. The core transition logic (where cells go / come from)

Both populations use the **same cells**; the transitions differ only because training is on
the familiar images:

- **Familiar**: FF **adapts down**, FB **strengthens**. Naive NO responders →
  expert **−NO** (no FB) or travel **up-and-left into the +O cloud at NO≈0** (FF
  adapts away, FB takes over). Narrow weak-mid NO responders with FB → **+NO** via gain.
- **Novel**: FF does **not** adapt + FB still strengthens (generalization). Every
  responder should **move right (+NO) and/or up (+O, if FB strengthening crosses drive threshold)**; cells without FB and strengthened surround can also go **−NO**. 

---

## 8. Known hard constraints (don't chase)

- **−O ≈ 0** is expected and stated (fixed PV; no modeled higher-area adaptation).
- Filling weak-O responders can push **familiar +O** above GT — accepted trade.
- Magnitude vs density are coupled through the fixed 0.3 z-threshold; decouple via
  FF strength (NO) vs surround/baseline (O drift), not by rescaling the z-floor.
- **Never floor/clip the z-scores at 0** to fake the neutral zone — that was an
  explicitly rejected shortcut. The neutral zone must come from the *mechanism*
  (divisive surround flooring at baseline), not from post-hoc clamping.
- `w_lat` is soft-bounded ≤1 (single PV) → divisive ratio capped at `1+k`; reach for
  `k`, not for `w_lat>1`. Loky workers re-import fresh, so the model is chosen by env
  (`CC_MODEL=divisive`) and `k`/overrides ride along in the config — a parent-process
  monkeypatch of `CCNeuron` is invisible to the workers.

---