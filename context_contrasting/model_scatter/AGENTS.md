# Model scatter — goals & target behaviour (AGENTS.md)

Reference for what the model rotated-sector scatterplots **should** look like,
distilled from the Seignette et al. chronic-imaging data
(`data_analysis/transitions>threshold.ipynb`, threshold **0.3**) and from the
tuning history. Concrete next steps to get publication-ready are in
**`FinalTuningPlan.md`**.

**Two generators, two inhibition models:**
- `run_model_scatter_pruned_mini.py` — the **subtractive** reference (model
  `minimal2/minimal_s.py`). Best subtractive result: `outputs_pruned_mini_centered/`.
- `run_model_scatter_pruned_mini_divisive.py` — the **divisive-normalization**
  variant (model `minimal2/minimal_divisive.py`, selected via `CC_MODEL=divisive`).
  **This is now the preferred direction** — see §0/§7. Preferred-shape reference:
  `outputs_pruned_mini_divisive_ENDGAME/`; latest shared-rate tuned output:
  `outputs_pruned_divisive_shared_lr_v9/`.

Both share the same principled template set; the divisive variant only swaps the
model and re-tunes the surround. The divisive variant fixes a *structural* problem
the subtractive one could not (see §2.0), so tune **on the divisive variant** now.

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
   and `a` can push below) — that is fine and expected; it is just **shrinkage toward
   0**, not manufactured overshoot. **Do not break the floor** (keep `baseline_drive`
   outside the division).
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
- A **mixed O&NO cloud centred around (NO≈0.5, O≈0.5)** — broadly-tuned cells weakly
  responding to *both*. **FINAL-GOAL refinement:** these "both" cells should cluster
  near **(0.5, 0.5)**, NOT spread out to high NO *and* high O. A cell that responds
  strongly to both (e.g. (1.5, 1.5)) is wrong; the data's both-responders are modest.
- **NO responders** spanning the whole **0–1.0+** band (not only the extreme),
  O ≈ 0 — these will either adapt to −NO or (with FB) move into the O cloud.
- General **spread** that plasticity then tightens.

### Expert (Post)  — the refined target
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
- A **mixed +NO+O cloud centred around (NO≈0.5, O≈0.5)** — cells strengthening
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
  replaced by +NO (blue, up to z≈3)** at expert.
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
(`FF_STRENGTHS`: silent / very_weak / diag_weak / weak / mid / strong) and FB levels
(`FB_LEVELS`: none / weak / mid / strong / strong_sat) are assigned **per
template**, independent of width. `tuning` ∈ {all, permuted1, novel} sets which FF
channels a cell prefers; `context` ∈ {none, all, random1/2, familiar, novel} masks
which context channels deliver feedback (lets a broad cell receive FB from a subset
of stimuli). FB is otherwise **generalized** (equal across received channels).

| cloud / feature | template(s) |
|---|---|
| small Δ background | `silent_broad_FFonly` |
| naive O-responder cloud (NO≈0, O>0) — STABLE O cells | `silent_broad_FB_strong` (high FB, limited headroom), with `silent_broad_FB_weak/mid/partial2` filling weaker O |
| familiar −NO (strong naive NO adapts) | `mid_broad_FFonly`, `weak_broad_FFonly` (fills the 0.3–0.5 NO band) |
| familiar NO→O MOVERS + novel NO→NO&O | `mid_broad_FB_weak`, `mid_broad_FB_partial2`, `strong_broad_FB_strong`, `very_weak_broad_FB_partial2`, `weak_broad_FB_mixed_bridge`, `novel_weak_FB_diagonal` |
| +NO (emergent fam/novel asymmetry) | `narrow_weak`, `narrow_mid` (random preferred image) |
| novel +NO / unresponsive→novel | `narrow_novel` (novel-tuned; gain<1 at naive → starts ~0, FB growth amplifies) |

### Key levers / rules (DIVISIVE model)
- **FB learning rate is the engine** of every learning-induced change; too small →
  everything stuck at naive (small Δ). It is a shared learning rate, not a
  per-template tuning knob.
- **Divisive gain `k`≈10** sets how hard the surround can divide; **`baseline_drive`
  stays outside the division** (the floor that gives the neutral zone). `k` is
  global/shared across configs; do not introduce per-template `k`.
- **Differentiate the surround by cell role — this is THE divisive lever:**
  - **Stable pure-O cells** (`silent_broad_FB_*`, no FF): **HIGH initial `w_lat`**
    (~0.85) so NO is divided to ~0 at naive AND expert → vertical O at NO≈0 both
    phases. No transition needed.
  - **Movers** (`*_broad_FB_*` with real FF): **surround TIMING** — the surround must
    be **low enough at naive** that they are clear **NO responders**, then grow (and
    FF adapts) so that **at expert the full-image NO is divided/adapted to land
    EXACTLY at NO≈0** (vertical, no diagonal lean). The current divisive tuning uses
    low initial `w_lat` plus the shared `lr_lat`; retune this tradeoff through
    initial surround/PV, FF/FB strength, drive/gain, or a global shared-rate change,
    not through per-template learning-rate scales.
  - **Mixed bridge** (`weak_broad_FB_mixed_bridge`): a small, weak broad+FB subclass
    tuned specifically to put a few familiar-naive cells around (NO≈0.5, O≈0.5).
    In the divisive variant it has a distinct initial surround/PV setting, so it
    travels up-left into the expert O column while staying above NO=0 under the same
    shared learning rates as every other template.
  - **Novel diagonal bridge** (`novel_weak_FB_diagonal`): a small novel-tuned,
    novel-context-only broad+FB subclass. It uses a tight weak FF level
    (`diag_weak`) and weak generalized FB so familiar training grows the novel FB
    channel through feedback generalization, but the cell has no familiar-context
    test response. It adds a few weak-pre → expert NO&O blue diagonal movers in the
    novel mixed interior while preserving the familiar panel.
- **Both-responders at (0.5,0.5)**: on novel the intact FF must be divided down to
  ~0.5 (not left at 1+) so the mixed cloud clusters at (0.5,0.5); tune the movers'
  expert surround / FF so the both-response is modest.
- **+NO cells at O≈0**: keep the **narrow baseline low** and the narrow FF strong
  enough that the NO shift dominates; keep narrow `w_lat` low (no division → +NO
  survives).
- **−NO from shrinkage**: the grown surround divides the full-image NO toward 0
  (familiar, FF-only) — fine and expected; `a` can push slightly below.
- **Permutation, not hand-assignment** for the narrow +NO asymmetry; only
  `narrow_novel` is a deliberate, endorsed special case.

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

## 9. Tuning workflow (optional)

**Drive the tuning from `FinalTuningPlan.md`** — its goals, levers and ordering are
the primary plan. The notes below are just the mechanics.

- Run (always in the `300 / 7 / 4` regime), **divisive variant**:
  `python -m context_contrasting.model_scatter.run_model_scatter_pruned_mini_divisive --n-jobs 10 --n-steps-per-phase 300 --training-trials 7 --test-trials 4 --plot-center-panels`
  → scatter + per-template trace panels. (k defaults to 10; override with env
  `CC_DIVISIVE_K` only for whole-run sweeps. The tuned defaults keep one shared
  learning-rate row from `SHARED_LEARNING_RATES`; per-template learning-rate scales
  and per-template `k` are explicitly out of bounds. Width-class plasticity and
  initial surround/PV retuning live in the divisive script's shared constants and
  `DIVISIVE_OVERRIDES`.)
- Compare to GT: aggregate rows of `outputs_*/summaries/sector_fractions.csv`.
- **Judge the TRANSITIONS, not just endpoints**: look at the naive→expert *vectors*
  (the "Expert − Naive" and "Expert by pre color" panels) and the per-template
  *traces*. The familiar NO→O and novel NO→NO&O movements must be visible there.
- **Green / black dashed-circle screenshots** are *not* the main workflow — they are
  just for ad-hoc back-and-forth follow-ups: a **green dashed circle** on a
  `todo_*.png` means fill that region (add/tune a config), a **black dashed circle**
  means remove points there, and un-circled clouds are fine as-is.
