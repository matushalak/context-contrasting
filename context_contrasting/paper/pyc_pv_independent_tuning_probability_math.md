# Independent PyC–PV Tuning Overlap Probabilities

## Setup

There are three feedforward inputs:

- inputs 1 and 2 are familiar;
- input 3 is novel.

The PV cell is always broadly tuned to exactly two of the three inputs:

\[
T_{\mathrm{PV}}
\in
\left\{
\{1,2\},
\{1,3\},
\{2,3\}
\right\}.
\]

The PyC can be either:

### Broadly tuned PyC

\[
T_{\mathrm{PyC}}
\in
\left\{
\{1,2\},
\{1,3\},
\{2,3\}
\right\}.
\]

### Narrowly tuned PyC

\[
T_{\mathrm{PyC}}
\in
\left\{
\{1\},
\{2\},
\{3\}
\right\}.
\]

Within each tuning-width class, all allowed tuning sets are assumed equally likely. PyC and PV tuning sets are sampled independently.

The probabilities below concern only whether an input is classified as strongly tuned or untuned. The actual synaptic strengths are defined elsewhere by the neuron templates.

---

# 1. Broadly tuned PyC versus broadly tuned PV

There are

\[
3\times 3=9
\]

equally likely ordered PyC–PV tuning combinations.

## A. Share at least one tuned input

Two subsets of size 2 drawn from a set of size 3 must overlap.

\[
P(\text{at least one shared tuned input}\mid\text{broad PyC})
=
1.
\]

\[
\boxed{P_A^{\mathrm{broad}}=1=100\%}
\]

## B. Share both tuned inputs

This occurs only when the PyC and PV choose the same two-input tuning set.

There are 3 matching combinations among the 9 possible combinations:

\[
P(\text{share both tuned inputs}\mid\text{broad PyC})
=
\frac{3}{9}
=
\frac{1}{3}.
\]

\[
\boxed{P_B^{\mathrm{broad}}=\frac13\approx33.33\%}
\]

## C. Both are tuned to the novel input 3

For either cell, 2 of the 3 possible broad tuning sets contain input 3:

\[
P(3\in T)=\frac23.
\]

By independence:

\[
P(3\in T_{\mathrm{PyC}}\land 3\in T_{\mathrm{PV}})
=
\frac23\frac23
=
\frac49.
\]

\[
\boxed{P_C^{\mathrm{broad}}=\frac49\approx44.44\%}
\]

## D. Share tuning for at least one familiar input

The only combinations with no shared familiar tuned input are:

\[
(\{1,3\},\{2,3\})
\]

and

\[
(\{2,3\},\{1,3\}).
\]

Therefore:

\[
P(\text{no shared familiar tuned input})
=
\frac29,
\]

so:

\[
P(\text{at least one shared familiar tuned input})
=
1-\frac29
=
\frac79.
\]

\[
\boxed{P_D^{\mathrm{broad}}=\frac79\approx77.78\%}
\]

## E. Same tuning status for both familiar inputs

The familiar-input tuning patterns are:

\[
\{1,2\}\rightarrow(1,1),
\qquad
\{1,3\}\rightarrow(1,0),
\qquad
\{2,3\}\rightarrow(0,1).
\]

The PyC and PV match on both familiar inputs only when they choose the same broad tuning set.

\[
\boxed{P_E^{\mathrm{broad}}=\frac13\approx33.33\%}
\]

---

# 2. Narrowly tuned PyC versus broadly tuned PV

Again, there are

\[
3\times 3=9
\]

equally likely ordered PyC–PV tuning combinations.

## A. Share at least one tuned input

For any single PyC-preferred input, 2 of the 3 possible PV tuning sets contain that input.

\[
P(\text{at least one shared tuned input}\mid\text{narrow PyC})
=
\frac23.
\]

\[
\boxed{P_A^{\mathrm{narrow}}=\frac23\approx66.67\%}
\]

## B. Share both tuned inputs

A narrow PyC has only one tuned input, so it cannot share two tuned inputs with the PV.

\[
\boxed{P_B^{\mathrm{narrow}}=0}
\]

## C. Both are tuned to the novel input 3

The narrow PyC is tuned to input 3 with probability:

\[
\frac13.
\]

The broad PV is tuned to input 3 with probability:

\[
\frac23.
\]

By independence:

\[
P(3\in T_{\mathrm{PyC}}\land 3\in T_{\mathrm{PV}})
=
\frac13\frac23
=
\frac29.
\]

\[
\boxed{P_C^{\mathrm{narrow}}=\frac29\approx22.22\%}
\]

## D. Share tuning for at least one familiar input

The narrow PyC is tuned to one of the familiar inputs with probability:

\[
\frac23.
\]

Conditional on which familiar input it prefers, the PV contains that input with probability:

\[
\frac23.
\]

Therefore:

\[
P(\text{at least one shared familiar tuned input})
=
\frac23\frac23
=
\frac49.
\]

\[
\boxed{P_D^{\mathrm{narrow}}=\frac49\approx44.44\%}
\]

## E. Same tuning status for both familiar inputs

The narrow-PyC familiar tuning patterns are:

\[
\{1\}\rightarrow(1,0),
\qquad
\{2\}\rightarrow(0,1),
\qquad
\{3\}\rightarrow(0,0).
\]

The PV familiar tuning patterns are:

\[
\{1,2\}\rightarrow(1,1),
\qquad
\{1,3\}\rightarrow(1,0),
\qquad
\{2,3\}\rightarrow(0,1).
\]

Matching occurs only for:

\[
(\{1\},\{1,3\})
\]

and

\[
(\{2\},\{2,3\}).
\]

These are 2 of the 9 equally likely combinations:

\[
\boxed{P_E^{\mathrm{narrow}}=\frac29\approx22.22\%}
\]

---

# 3. Summary

| Event | Broad PyC | Narrow PyC |
|---|---:|---:|
| A. Share at least one tuned input | \(1\) | \(2/3\) |
| B. Share both tuned inputs | \(1/3\) | \(0\) |
| C. Both tuned to novel input 3 | \(4/9\) | \(2/9\) |
| D. Share at least one familiar tuned input | \(7/9\) | \(4/9\) |
| E. Same tuning status for both familiar inputs | \(1/3\) | \(2/9\) |

In decimal form:

| Event | Broad PyC | Narrow PyC |
|---|---:|---:|
| A | 1.0000 | 0.6667 |
| B | 0.3333 | 0.0000 |
| C | 0.4444 | 0.2222 |
| D | 0.7778 | 0.4444 |
| E | 0.3333 | 0.2222 |

---

# 4. Optional population mixture

If a fraction \(q\) of PyC templates are broad and a fraction \(1-q\) are narrow, the total population-level probability of any event \(X\) is:

\[
P_X
=
qP_X^{\mathrm{broad}}
+
(1-q)P_X^{\mathrm{narrow}}.
\]

Therefore:

\[
P_A=\frac{2+q}{3},
\]

\[
P_B=\frac{q}{3},
\]

\[
P_C=\frac{2+2q}{9},
\]

\[
P_D=\frac{4+3q}{9},
\]

\[
P_E=\frac{2+q}{9}.
\]

These formulas only combine the broad- and narrow-PyC cases according to their relative frequency. They do not prescribe how templates or their synaptic parameters are implemented.
