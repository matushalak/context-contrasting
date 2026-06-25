## Simulation details

### Simulated experiment

We simulated the chronic two-photon paradigm as a discrete-time version of the
circuit model above (one integration step is the unit of time; time constants are
reported in steps). Each cell was probed in a **naive** state, exposed to a block
of familiar-image **training**, and re-probed in an **expert** state, and we read
out how its responses shifted across learning.

Three stimuli were used, encoded as $3$-dimensional one-hot vectors: two familiar
images ($\mathbf{e}_1,\mathbf{e}_2$) and one novel image ($\mathbf{e}_3$). A test
probe trial lasted $200$ steps, of which the first three quarters were a
stimulus-free inter-trial interval and the final quarter ($50$ steps) was the
stimulus presentation. Probe repeats were run as continuous repeated trials, as
in the original protocol. At every step the feedforward and contextual inputs
were drawn from
$\mathcal{N}(\text{mean},\,0.05)$ (mean $=\mathbf{0}$ in the inter-trial interval,
$=$ the one-hot image during presentation). Each image was probed in two regimes:
a **non-occluded** ("NO") trial in which both the feedforward input
$\mathbf{x}$ and the contextual input $\mathbf{c}$ carried the image, and an
**occluded** ("O") trial in which $\mathbf{x}=\mathbf{0}$ and only the contextual
input $\mathbf{c}$ was present (isolating the feedback-driven / ecRF response).

The three phases were:

1. **Naive probe** — every image, NO and O, $2$ continuous trials each, with plasticity off.
2. **Training** — the two familiar images interleaved, $5$ trials, with plasticity
   on (the only phase in which the local learning rules update the weights).
3. **Expert probe** — identical to the naive probe, with plasticity off.

For each (cell, image, trace) we took the mean PyC rate $y$ over the stimulus
window and $z$-scored it to the cell's naive spontaneous activity (mean and s.d.
of $y$ over the inter-trial intervals of the naive probe). To keep signal-poor
responses (e.g. the $\sim 0$ NO response of a pure occluded responder) from
exploding when divided by a near-zero spontaneous s.d., the $z$-score denominator
was floored per cell at $\max(0.04,\,0.27\,\sigma_y)$, where $\sigma_y$ is the
cell's baseline-drive amplitude (a proxy for its true spontaneous variability).
Both probes were normalised to the **naive** baseline, so the naive$\to$expert
shift $(\Delta\mathrm{NO},\Delta\mathrm{O})$ reflects only the change in evoked
response and not any drift in spontaneous rate.

Each cell was then classified, exactly as the imaging data, from the angle of its
shift vector: shifts with magnitude $<0.3$ were "small", and the rest were
assigned to the $+\mathrm{NO}/+\mathrm{O}/-\mathrm{NO}/-\mathrm{O}$ rotated
sectors. Responses to the two familiar images were pooled into a **familiar**
population and responses to the novel image formed a **novel** population; these
are the model analogues of the post-learning familiar ($-\mathrm{NO}$ dominant)
and novel ($+\mathrm{NO}$ dominant) populations measured in vivo, classified at
the same threshold ($0.3$).

### Modeled population

A single set of circuit equations cannot, on its own, reproduce a *scatter* of
heterogeneous cells. We therefore drew a population of $N=250$ model cells from a
mixture of $17$ transition **templates**, each template being a point in parameter
space that produces one qualitative response type (e.g. a broadly tuned cell whose
feedforward drive adapts away and is replaced by a feedback-driven occluded
response, or a narrowly tuned cell whose novel-image drive is gain-amplified into a
$+\mathrm{NO}$ response). The templates' mixture proportions were set to match the
measured class proportions. Every cell was an independent noisy realisation of its
template: its initial synaptic weights were drawn from Gaussians centred on the
template values (per-element s.d. $\max(22\text{–}65\%\times\text{center},\,
0.002\text{–}0.04)$, clipped to be non-negative and to the template's bounds), and
its scalar parameters were jittered (Table 2) by a global factor of $1.75$. Two
template properties were tied to a single **tuning-width** label rather than set
per template: broadly tuned cells used a strong feedforward (anti-Hebbian)
plasticity scale and a lower apical drive threshold (so feedback can more easily *drive* the soma),
whereas narrowly tuned cells used a weak — but non-zero — feedforward plasticity
scale and a higher drive threshold (so feedback mostly *gain-modulates* the soma).

The learning rates were fixed and shared across the whole population; only the
initial weights and the per-cell scalar parameters varied (Tables 1–2).

### Table 1 — Fixed parameters (shared across all model cells)

| Symbol | Parameter | Value |
|---|---|---|
| $\tau_y$ | PyC integration time constant | $10$ steps |
| $\tau_p$ | PV integration time constant | $4$ steps |
| $\tau_a$ | adaptation time constant | $50$ steps |
| $\eta_{FF}$ | feedforward (anti-Hebbian) base learning rate | $0.015$ |
| $\rho_{FF}$ | feedforward plasticity scale (broad / narrow tuning) | $8.0$ / $0.05$ |
| $\eta_{FB}$ | feedback learning rate | $0.0005$ |
| $\eta_{LAT}=\eta_{pvLAT}$ | lateral and PV-lateral learning rate | $0.0015$ |
| $\eta_{PV}$ | PV feedforward learning rate | $0.003$ |
| $\mathbf{S}^c$ | feedback specificity (diagonal / off-diagonal) | $0.6$ / $0.2$ |
| $\mathbf{S}^p$ | PV feedforward specificity (diagonal / off-diagonal) | $0.8$ / $0.1$ |
| $r_{\max}$ | maximum firing rate | $1.0$ |
| — | weight decay | $0$ (off) |
| — | steps per test trial / stimulus-window length | $200$ / $50$ (final quarter) |
| — | naive and expert probe trials per image | $2$ |
| — | familiar training trials | $5$ |
| $\sigma_{\text{stim}}$ | stimulus and inter-trial input s.d. | $0.05$ |
| — | rotated-sector small-shift threshold | $0.3$ |
| — | $z$-score denominator floor | $\max(0.04,\,0.27\,\sigma_y)$ |
| $N$ | sampled cells | $250$ |

### Table 2 — Per-cell variable parameters (population ranges)

Only the parameters below are independently perturbed per cell and then clipped
to the range shown. LAT, PV-lateral, PV feedforward, and all other scalar
parameters are fixed at their transition-template values. "Jitter" is the noise
applied to the template value: *log-normal* multiplies by
$\exp(\mathcal{N}(0,\sigma))$ and *additive* adds $\mathcal{N}(0,\sigma)$; scalar
widths $\sigma$ are further scaled by the global factor $1.75$.

| Symbol | Parameter | Jitter | Population range |
|---|---|---|---|
| $\mathbf{w}_{FF},\mathbf{w}_{FB}$ | initial synaptic weights | Independent Gaussian per weight element about template center (s.d. $\max(22\text{–}65\%,\,0.002\text{–}0.04)$, transition-specific) | $\ge 0$, template-bounded |
| $g$ | apical maximum gain | log-normal, $\sigma=0.18$ | $2.5\text{–}11$ |
| $\theta$ | apical drive threshold | additive, $\sigma=0.12$ | $0.12\text{–}0.5$ (FB-driving) vs $\ge 1.05$ (gain-only), cap $\le 3$ |
| $\sigma_y$ | baseline-drive amplitude ($I_y$ s.d.) | log-normal, $\sigma=0.20$ | $0.085\text{–}0.52$ |
