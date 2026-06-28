# Final tuning plan — to publication-ready (DIVISIVE era)

Concrete exploration goals and levers to close the **remaining** gaps. Read
`AGENTS.md` first for the targets and the properties that must be **preserved**
(neutral-zone axis-centred clouds; principled templates; matched proportions).

**The model is now DIVISIVE.** Tune on
`run_model_scatter_pruned_mini_divisive.py` (model `minimal2/minimal_divisive.py`,
`y = act((apical_gain·y_ff + apical_drive)/(1 + k·y_lat) + baseline_drive − a)`,
`k≈10`). This already solved the structural problem the subtractive model could not:
the O-responder cloud now **floors at baseline (no NO<0 overshoot)** → a real neutral
zone. The earlier subtractive plan (a "G1a: switch to subtractive shifted-ReLU drive"
step, FB↔inhibition continuity hacks, z-floor tricks) is **superseded** — do not
re-litigate it. Historical preferred-shape reference:
`outputs_pruned_mini_divisive_ENDGAME/`; latest shared-rate tuned output:
`outputs_pruned_divisive_shared_lr_v9/`.

**Run regime — for the final figure AND every intermediate experiment:**
`--n-steps-per-phase 300 --training-trials 7 --test-trials 4`. Always pass these.

Workflow per experiment: edit exposed template parameters / `DIVISIVE_OVERRIDES`
(or `CC_DIVISIVE_K` for a whole-run global sweep) → `python -m
context_contrasting.model_scatter.run_model_scatter_pruned_mini_divisive --n-jobs 10
--n-steps-per-phase 300 --training-trials 7 --test-trials 4 --plot-center-panels
--output-dir outputs_pruned_divisive_<tag>` → eval fractions vs GT → view the
familiar+novel scatters (**including the naive→expert vector panels**) and the
center-panel traces. Change **one lever at a time**; confirm the §AGENTS "keep"
properties (esp. the neutral-zone floor) survive. Current tuned defaults use one
shared learning-rate row from `SHARED_LEARNING_RATES` (`lr_ff=0.0135`,
`lr_fb=0.00050`, `lr_lat=0.0240`, `lr_pv=0`, scaled by 200/300 in this run) and
one global divisive `k=10`. Do **not** add per-template learning-rate scales or
per-template `k`. The exposed retuning is: `CC_NARROW_FFPLAST=0.9`,
`CC_MOVER_LAT=0.14`, `CC_MOVER_PV=0.55`, `CC_MIXED_BRIDGE_LAT=0.20`,
`CC_MIXED_BRIDGE_PV=0.62`; the novel diagonal bridge uses
`CC_NOVEL_DIAGONAL_LAT=0.08`, `CC_NOVEL_DIAGONAL_PV_TUNED=0.20`,
`CC_NOVEL_DIAGONAL_PV_SILENT=0.06`, plus its source-level FF/FB/gain/drive and
mixture-weight settings in `TEMPLATES`.

Current fractions (latest shared-rate tuned default, k=10): familiar
+NO10/+O22/−NO27.6/−O0/sm40.4; novel +NO24/+O11.2/−NO18/−O0/sm46.8.
(GT: fam 8/14/33/7/38; nov 25/10/14/7/43.)

Priorities, in order: **G1 (vertical familiar O cloud fed by a clean strong-NO→O
transition)** and **G2 (both-responders centred at (0.5,0.5))** are now close enough
in the latest shared-rate tuned default for visualization polish. G3/G4/G5 are now
minor fraction/amplitude polish only.

---

## G1 — Familiar expert O cloud: VERTICAL at NO≈0, fed by two converging flows (highest priority)

**Target.** The +O cloud is a vertical column squarely on the O-axis (NO≈0), with
**no diagonal lean** and **no dip below NO=0**, fed by BOTH:
1. **from below** — unresponsive/weak cells growing O straight up (NO~0 → O↑); and
2. **from the right** — **strong naive NO responders that adapt + recruit inhibition**
   and travel up-and-left to land **exactly at NO≈0** (the NO→O transition).

**Status.** Flow (1) works, and the latest divisive tuning makes flow (2) visible
without breaking the O-axis floor. The core tension remains the one to watch during
future polish:
- High initial `w_lat` on the mover cells (ENDGAME) → vertical endpoint, BUT divides
  the naive NO away → no strong NO responder to transition FROM (transition invisible).
- Low initial `w_lat` (current) → strong naive NO + visible transition, BUT the
  endpoint lands at NO≈0.3–0.4 (diagonal lean), because neither the grown surround
  nor FF adaptation is strong enough by expert to finish the division to ~0.

**Levers to get BOTH (strong naive NO AND vertical endpoint).** The mover needs a
*low naive / high expert* surround, i.e. strong **growth**, plus full FF adaptation:
- **Stronger surround at expert on the movers**: use allowed shared or initial
  parameters only — a global shared `lr_lat` change if absolutely necessary, or
  template initial `w_lat`/PV/FF/FB/gain/drive changes. Do not use per-template
  learning-rate scales. The goal is expert `w_lat` high enough that `1+k·y_lat`
  divides residual full-image NO toward ~0 without erasing the naive NO source.
- **FF adaptation does part of the job**: familiar `y_ff→` small, so even modest
  division finishes it to ~0. (Raising broad `ff_plasticity_scale` 8→13 gave ~no
  extra adaptation — it self-limits as `w_ff→0` — so don't rely on it alone.)
- Success: in the familiar vector panel, strong-NO naive cells have **up-and-left
  arrows that terminate on the O-axis at NO≈0** (next to the straight-up arrows from
  below); expert O-responder `diag(NO>0.3)` → small; `neg(NO<−0.1)` stays ~0.

**Latest status (`outputs_pruned_divisive_shared_lr_v9`).** The familiar O floor is
preserved: O-responder target `NO<−0.1` is 0/250. The added bridge and existing
movers travel up-left into the expert O cloud while staying above the negative-NO
floor.

---

## G2 — Both-responders (O&NO) centred at (0.5, 0.5)

**Target (user-emphasised).** Cells that respond to *both* O and NO — the
familiar/novel **naive** weak-both cloud and the **novel expert** mixed cloud —
should cluster around **(NO≈0.5, O≈0.5)**, NOT spread out to high NO *and* high O.

**Mechanism (divisive).** On novel the FF is intact, so without enough division the
mover NO stays at ~0.9–1+ (too high) while O grows → an over-extended both-cloud.
The grown surround must divide the intact novel NO down to ~0.5, and FB must lift O
to ~0.5 — a *modest* both-response.
- Tune the movers' expert surround so novel NO lands ~0.5 (same shared
  surround/initial-weight lever as G1 — they are coupled: the growth that
  verticalises familiar also shrinks novel NO toward 0.5).
- Keep FB moderate so O lands ~0.5, not 1+.
- **Watch the no-LAT trace**: if the expert surround is *very* strong, the
  "Expert no LAT" ablation column un-divides to a huge response and dwarfs the real
  Expert NO&O rise on the shared-axis trace. Land the division where novel NO≈0.5 is
  clearly visible AND the no-LAT column is not wildly larger.
- Success: novel-expert both-cloud tight around (0.5,0.5); naive both-cloud likewise.

**Latest status (`outputs_pruned_divisive_shared_lr_v9`).** Added
`weak_broad_FB_mixed_bridge`, a small weak broad+FB class. In familiar naive, the
bridge contributes the desired weak mixed cells around the NO/O middle band; bridge
cells move `−NO,+O` into the expert O cloud.

Added `novel_weak_FB_diagonal`, a small novel-tuned, novel-context-only class for the
novel expert NO&O interior. It contributes three strict blue `+NO` cells inside the
expert `(NO,O)=0.25..1.0` box and seven total diagonal-template cells in that box,
all with positive `dNO` and `dO`.

---

## G3 — Novel −NO population (yellow→blue replacement)

**Status.** Novel −NO is now slightly high (~18% vs GT 14.5%), but acceptable for
the shared-rate final. **−NO is fine in divisive** — it is the full-image NO being
*shrunk toward 0* by the grown surround (not an artifact). Future polish should avoid
increasing it further.

- The `*_broad_FFonly` (no-FB) cells respond to novel at naive (intact FF) and should
  have their novel NO divided down at expert by the surround that grew during familiar
  training → −NO. Give these cells **enough surround growth** (their `w_lat`/`W_pv`)
  that expert novel NO < naive by > 0.3, without recreating a static extreme-NO naive
  cloud (cap their FF so naive novel NO ≤ ~2).
- Coupling: this also tends to lower novel small-Δ. Watch that it does not eat the
  novel +NO (keep narrow surrounds low).
- Success if revisited: novel −NO ≈ 12–15%; naive novel yellow NO responders visibly
  replaced by blue +NO at expert; novel small-Δ stays near the mid-40s.

---

## G4 — Familiar +NO too high / narrow +NO spread

**Status.** Familiar +NO is close (~10% vs GT 8%). Narrow familiar-tuned cells still
set this fraction, but it is no longer a major tuning gap.

- Trim the narrow familiar +NO: the narrow familiar-tuned cells should mostly **adapt**
  (FF down → small/−NO) on familiar; raise their effective adaptation or trim narrow
  gain/FF so familiar +NO falls toward GT 8 without hurting novel +NO (≈24, fine).
- Keep the narrow +NO **spread** across the whole NO range (weak→strong), graded by
  the gain sigmoid, at O≈0 (low narrow baseline) — the §G2-of-old goal still holds.

---

## G5 — Naive sparse at extremes, expert denser & higher; O≤2, NO≤3

- **Lower naive amplitude, let plasticity build it**: most naive responders
  weak/moderate; expert clouds denser and out to z 2–3 for NO, ≤2 for O.
- Cap expert O ≤ 2 (limited FB headroom; the divisive floor already prevents the
  negative tail). Allow expert NO → 3 for narrow +NO.

---

## Guardrails (verify every iteration)

1. **Neutral zone**: O responders at NO≈0 with **NO points below NO=0** (divisive
   floor), NO responders at O≈0 — all four panels. **Never** restore the negative
   overshoot by dividing `baseline_drive`, and **never** z-floor/clip to fake it.
2. **Vertical, not diagonal**: familiar expert O-responders on the O-axis, fed by the
   strong-NO→O transition terminating at NO≈0 (G1) — judged on the **vectors/traces**.
3. **Both-responders at (0.5,0.5)**, not high/high (G2).
4. Naive familiar ≈ naive novel; familiar expert = more O/NO separation, novel expert
   = less (+ the both-strengthened cloud). Center panels still principled/example-able.
5. −O stays ~0 (expected; do not force it).
6. Re-check GT proportions after each change; don't regress familiar/novel +NO while
   fixing the others.

Suggested order: **G1 (vertical familiar via strong-NO→O growth) → G2 ((0.5,0.5)
both-cloud, coupled to G1) → G3 (novel −NO) → G4 (tame familiar +NO) → G5 (amplitude
polish)**, re-checking guardrails each step.
