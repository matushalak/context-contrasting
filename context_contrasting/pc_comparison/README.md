# Matched PPE/NPE comparison

The canonical comparison uses one circuit-independent parameter row for a
matched PPE/NPE pair:

- PyC excitatory vector and tuning: sensory `w_FF` in PPE, contextual `w_FB`
  in NPE.
- PV excitatory vector and tuning: contextual excitation in PPE, sensory
  excitation in NPE.
- Fixed scalar `w_LAT` from the PV cell to the PyC.
- One learning rate for the direct PyC excitatory vector. No other synapse is
  plastic.

For sensory input `x` and contextual input `c`, the signed errors are

```text
PPE: e = w_PyC dot x - w_LAT * PV(w_PV dot c)
NPE: e = w_PyC dot c - w_LAT * PV(w_PV dot x)
```

Both use the same anti-Hebbian update:

```text
delta w_PyC = -learning_rate * e * presynaptic_input
```

The signed error above is the teaching signal. Reported PyC activity additionally
includes short-term adaptation, `y = EMA(ReLU(e + baseline - adaptation))`.
Adaptation shapes within-trial traces but does not change the learned balance
target. The canonical shared baseline drive is `0.2`, which keeps over-inhibited
PPEs above the rectification floor so weight increases can appear as positive
familiar-response changes. Both circuits also use shared baseline-drive noise
with `sigma = 0.30`; this affects response variability and z-scoring but remains
outside the signed-error teaching signal. PV noise remains disabled.

The learning rate is defined at a 400-step reference horizon. A run with `T`
steps uses `learning_rate * 400 / T`, but a short-run result is never accepted
without a full 400-step verification.

## Convergence

Exact floating-point zero is not reached by this bounded asymptotic update.
The operational criterion is therefore `abs(signed prediction error) <= 0.005`
for both familiar features in every sampled row after 400 steps per phase and
seven training trials per familiar image.

For the default 300-row parameter space and seed 7151, the minimum full-horizon
rate is `0.2820285747289002`. The 100-step scaled search proposes
`0.27477164754327305`, but that rate fails the full check (`max abs PE =
0.0051315692`), so it is not used.

Run the calibration with:

```bash
uv run python -m context_contrasting.pc_comparison.calibrate_pc_learning_rate
```

Run the matched population comparison with:

```bash
uv run python -m context_contrasting.pc_comparison.run_pc_comparison
```

Export separate thesis-ready familiar scatter panels and familiar/novel vector
panels in PNG, SVG, and EPS:

```bash
uv run python -m context_contrasting.pc_comparison.export_thesis_panels
```
