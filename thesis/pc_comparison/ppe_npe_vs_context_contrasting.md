# Canonical PPE/NPE predictive coding versus context-contrasting in the occlusion experiment

## Purpose

This note summarizes which findings of Seignette et al. can be explained by a classical hierarchical predictive-coding circuit with dedicated positive prediction-error (PPE) and negative prediction-error (NPE) neurons, which findings strain that account, and a minimal simulation that can expose the distinction.

The central claim is deliberately narrow:

> The data do not rule out predictive processing in the broad sense. They challenge the **canonical cell-level interpretation** in which individual L2/3 pyramidal cells are dedicated PPE or NPE units with fixed opposing signs of feedforward and predictive input.

A sufficiently augmented predictive-coding model can absorb most response patterns by adding representation neurons, precision gain, mixed feature pooling, stimulus-specific error identities, or additional plasticity. However, once these additions do the explanatory work, the simple PPE/NPE connectivity motif is no longer the explanation being tested.

---

## 1. Experimental observations to explain

The most important L2/3 observations are:

1. **Most familiar non-occluded (NO) responses weaken after familiarization.**
2. **Occluded (O) responses strengthen after familiarization and are delayed relative to NO responses**, consistent with contextual/recurrent input.
3. **NO and O response populations become increasingly separated with experience and task engagement.** The reported separation index increased from approximately `0.040` in naive mice to `0.426` in expert mice and `0.874` during task performance.
4. **A significant minority of neurons increase their responses to familiar NO images.** Many of these cells were initially weak or unresponsive and became highly selective familiar-image responders. The emerging population had a reported lifetime sparseness of approximately `0.59`.
5. **Initial NO selectivity predicts plasticity.** Broadly responsive neurons are preferentially suppressed, whereas more selective neurons are relatively preserved or enhanced (`R = -0.299`, `P = 0.001`, using the paper's sign convention for response change).
6. **Many cells show mixed positive responses to both O and NO in naive animals.** Separation is therefore not an obvious pre-existing property of two fixed cell classes.
7. **Mixed O/NO responsiveness persists for novel images in expert animals.** Novel feedforward responses remain strong while strengthened contextual inputs can generalize to novel scenes.
8. **Strong dual responsiveness becomes rare for familiar images after learning.** The separation therefore appears to be produced by experience-dependent plasticity rather than simply revealing fixed PPE and NPE populations.

These observations jointly matter more than any single population-average suppression or enhancement.

---

## 2. Strict canonical PPE/NPE model

Let

$$
s_i = \text{feedforward sensory evidence for feature } i,
$$

$$
p_i = \text{top-down prediction of feature } i.
$$

A rectified signed-error implementation is

$$
r_i^{\mathrm{PPE}} = [s_i-p_i]_+,
$$

$$
r_i^{\mathrm{NPE}} = [p_i-s_i]_+,
$$

where $[z]_+ = \max(0,z)$.

The corresponding effective connectivity is:

| Cell class | Feedforward sensory input | Predictive/contextual input |
|---|---:|---:|
| PPE | excitatory | inhibitory |
| NPE | inhibitory | excitatory |

For the occlusion experiment, use the first-order approximation

$$
s^{\mathrm{NO}}>0, \qquad s^{\mathrm O}\approx 0,
$$

and assume that the visible surround generates a comparable prediction in the O and NO versions of the same scene:

$$
p^{\mathrm{NO}}\approx p^{\mathrm O}=p.
$$

Then

$$
r_{\mathrm O}^{\mathrm{PPE}}=0,
$$

and

$$
r_{\mathrm O}^{\mathrm{NPE}}=p,
\qquad
r_{\mathrm{NO}}^{\mathrm{NPE}}=[p-s]_+\leq p.
$$

This gives a strong geometric constraint in the response plane where the horizontal axis is NO activity and the vertical axis is O activity:

- PPE cells lie on the **NO axis**, because they cannot respond positively to O.
- NPE cells may respond to both, but must lie on or above the diagonal:

$$
r_{\mathrm O}\geq r_{\mathrm{NO}}.
$$

Thus, a strict PPE/NPE population cannot occupy the entire positive NO–O plane.


## 2.1 Which synapses are actually plastic in classical HPC?

A crucial distinction is that the formal predictive-coding objective specifies **which effective weights must learn**, but does not uniquely specify which biological synapse class implements that learning.

At hierarchy level $l$, let a higher-level representation $r_{l+1}$ predict the activity at the level below through a generative matrix $U_l$:

$$
\hat r_l = U_l r_{l+1},
$$

with signed prediction error

$$
\varepsilon_l = r_l-U_l r_{l+1}.
$$

The canonical generative-weight update is

$$
\Delta U_l \propto \varepsilon_l r_{l+1}^{\top}.
$$

For a single synapse from higher-level unit $j$ to lower-level unit $i$,

$$
\Delta U_{ij}\propto \varepsilon_i r_j.
$$

Thus the primary learned parameters are the **higher-to-lower generative or predictive weights**. They are strengthened when the lower-level activity is underpredicted and weakened when it is overpredicted.

When the signed error is split into rectified PPE and NPE populations,

$$
\varepsilon_i = \varepsilon_i^+-\varepsilon_i^-,
$$

$$
\varepsilon_i^+=[r_i-\hat r_i]_+,
\qquad
\varepsilon_i^-=[\hat r_i-r_i]_+,
$$

then

$$
\Delta U_{ij}
\propto
(\varepsilon_i^+-\varepsilon_i^-)r_j.
$$

Consequently:

- active PPE neurons provide a teaching signal to **increase** the relevant prediction;
- active NPE neurons provide a teaching signal to **decrease** the relevant prediction.

### Ascending weights

In the original Rao--Ballard formulation, the ascending influence is effectively

$$
U_l^{\top}\varepsilon_l.
$$

The descending generative pathway uses $U_l$, while the ascending error pathway uses $U_l^{\top}$. These weights are therefore mathematically tied rather than learned as two fully independent biological pathways. A biological circuit would need either approximate reciprocal symmetry, a separate local rule that aligns the two pathways, or a formulation that does not require exact symmetry.

### Biological PPE/NPE implementations require additional balancing plasticity

A dedicated PPE neuron has the effective form

$$
r^{\mathrm{PPE}}=[w_s s-w_p p]_+,
$$

where sensory input is excitatory and prediction is inhibitory. To cancel a matched input, the circuit must learn

$$
w_s s\approx w_p p.
$$

This balancing could be implemented by plasticity at one or more of the following synapse classes:

- feedback excitation onto a prediction-driven interneuron;
- inhibitory synapses from that interneuron onto the PPE cell;
- direct sensory excitation onto the PPE cell.

Likewise, an NPE neuron has the effective form

$$
r^{\mathrm{NPE}}=[w_p p-w_s s]_+,
$$

so matched cancellation requires

$$
w_p p\approx w_s s.
$$

This could be achieved through plasticity at:

- direct feedback excitation onto the NPE cell;
- feedforward excitation onto a sensory-driven interneuron;
- inhibitory synapses from that interneuron onto the NPE cell.

The formal theory constrains the **net balance**, not the exact physical synapse at which plasticity occurs.

### What is and is not canonical

| Pathway | Formal role | Canonical plasticity status |
|---|---|---:|
| Higher representation $\rightarrow$ lower prediction pathway | Encodes the generative model | Primary learned weight |
| Lower error $\rightarrow$ higher representation pathway | Propagates signed residual | Often assumed to use $U^{\top}$; biological learning unresolved |
| Prediction-driven inhibition $\rightarrow$ PPE | Cancels expected sensory excitation | Must adapt in a biological implementation, but no single canonical rule |
| Sensory-driven inhibition $\rightarrow$ NPE | Cancels expected predictive excitation | Must adapt in a biological implementation, but no single canonical rule |
| Direct sensory excitation $\rightarrow$ PPE | Supplies bottom-up evidence | Fixed or plastic depending on model |
| Direct feedback excitation $\rightarrow$ NPE | Supplies top-down prediction | Usually part of the learned generative pathway |

This distinction matters for the present comparison. Classical HPC primarily learns a generative model that improves cancellation. Context-contrasting instead posits explicit opposing pathway plasticity:

$$
w_{\mathrm{FF}}\downarrow,
\qquad
w_{\mathrm{FB}}\uparrow,
$$

with strengthened feedback capable of gain modulation or direct drive rather than merely balancing and subtracting sensory input.

---

## 3. What canonical PPE/NPE coding can explain

### 3.1 Suppression of most familiar NO responses

If familiarization improves the prediction while the physical sensory input remains unchanged,

$$
p_i^{\mathrm{expert}}>p_i^{\mathrm{naive}},
$$

then a PPE response must decrease or remain unchanged:

$$
[s_i-p_i^{\mathrm{expert}}]_+
\leq
[s_i-p_i^{\mathrm{naive}}]_+.
$$

The dominant decrease in familiar NO responses is therefore compatible with PPE coding.

It does **not**, by itself, show that the suppression is caused by predictive feedback. Feedforward adaptation, synaptic depression, increased local inhibition, or reduced intrinsic excitability can produce the same output change.

### 3.2 Increased O responses after familiarization

For an occluded stimulus,

$$
s\approx 0, \qquad p>0,
$$

so an NPE unit produces

$$
r^{\mathrm{NPE}}_{\mathrm O}=p.
$$

A learned increase in contextual prediction therefore naturally increases O responses. Delayed O responses are also consistent with a recurrent or feedback source.

### 3.3 Separation of familiar NO and O responders

Dedicated PPE and NPE populations naturally produce distinct NO- and O-preferring cells:

- PPE-like cells respond to unpredicted feedforward evidence.
- NPE-like cells respond when predicted evidence is absent.

The strongly separated expert-familiar population can therefore be described using PPE/NPE terminology.

The harder question is why the same cells were much more mixed before learning and remain mixed for expert-novel stimuli.

### 3.4 Strong novel NO responses

A novel sensory feature can exceed its prediction:

$$
s_{\mathrm{novel}}>p_{\mathrm{novel}},
$$

producing a PPE response. Predictive coding can therefore explain strong responses to underpredicted novel inputs at a population level.

---

## 4. What the strict PPE/NPE motif cannot naturally explain

## 4.1 Selective increases in familiar NO responses with weak O responses

This is the strongest contradiction.

For a pure PPE cell, increasing prediction strength cannot increase the response to the same sensory input:

$$
r_{\mathrm{NO,expert}}^{\mathrm{PPE}}
\leq
r_{\mathrm{NO,naive}}^{\mathrm{PPE}}.
$$

A pure NPE interpretation does not solve the problem. If a cell becomes NO-responsive because the learned prediction exceeds sensory evidence,

$$
r_{\mathrm{NO}}^{\mathrm{NPE}}=[p-s]_+>0,
$$

then removing sensory evidence should make its O response at least as large:

$$
r_{\mathrm O}^{\mathrm{NPE}}=p
\geq
[p-s]_+
=r_{\mathrm{NO}}^{\mathrm{NPE}}.
$$

Therefore, the following empirical phenotype is outside the strict model:

$$
r_{\mathrm{NO,expert}}>r_{\mathrm{NO,naive}},
\qquad
r_{\mathrm O,expert}\approx 0.
$$

This is exactly the phenotype of a contextually amplified feedforward neuron: feedback enhances an existing or subthreshold NO response but is insufficient to drive the cell in the absence of feedforward input.

### Predictive-coding rescue attempts

A predictive-coding model can reproduce this by adding, for example,

$$
r^{\mathrm{PPE}}=\pi[s-p]_+,
$$

with a learned precision/gain factor $\pi$ that increases enough to overwhelm the reduction in residual error. It could also relabel the neuron as a representation unit rather than an error unit.

Both are possible, but neither result follows from PPE/NPE subtraction itself. The extra gain or representation mechanism is doing the work.

---

## 4.2 General mixed O and NO responses within the same naive L2/3 cells

A PPE neuron cannot respond positively to O because $s^{\mathrm O}\approx0$:

$$
r_{\mathrm O}^{\mathrm{PPE}}=[0-p]_+=0.
$$

An NPE neuron can respond to both O and NO, but only in the restricted O-dominant regime

$$
r_{\mathrm O}^{\mathrm{NPE}}\geq r_{\mathrm{NO}}^{\mathrm{NPE}}.
$$

Consequently:

- **NO-dominant mixed cells**, with $r_{\mathrm{NO}}>r_{\mathrm O}>0$, cannot be dedicated feature-matched PPE or NPE units under the same prediction.
- **O-dominant mixed cells** can be called NPE cells, but their NO response must be interpreted as residual overprediction, not positive feedforward drive.

The broad naive overlap therefore implies that many cells initially receive effective positive contributions from both feedforward and contextual pathways, or pool multiple feature dimensions with different signs. Either interpretation departs from a clean cell-level PPE/NPE identity.

A model may claim that PPE/NPE connectivity is learned and has not yet formed in naive animals. That is a viable hypothesis, but it concedes that the naive mixed cells are not currently operating as signed-error units. It also requires a separate plasticity rule to explain how their error identities emerge.

---

## 4.3 Mixed expert-novel responses

After familiarization, novel-image feedforward pathways have not been repeatedly activated and therefore remain strong. At the same time, strengthened contextual pathways can generalize across shared scene structure.

This creates

$$
s_{\mathrm{novel}}>0,
\qquad
p_{\mathrm{novel}}>0,
$$

for NO presentations, while O presentations retain contextual input but remove the local feedforward signal.

A strict cell-level PPE/NPE model again permits only:

- PPE: NO response without O response;
- NPE: O response at least as large as NO response.

It does not naturally generate a broad population of cells positively driven by both pathways, including NO-dominant mixed cells.

A rescue is to let a neuron be PPE-like for one feature dimension and NPE-like for another:

$$
r_j=\phi\left(\sum_k a_{jk}(s_k-p_k)\right).
$$

This can produce almost any mixed response, but then PPE/NPE sign is a property of neuron–feature pairs or dendritic subunits, not a stable identity of the L2/3 cell. The firing rate is no longer an unambiguous signed error.

---

## 4.4 Learning-dependent emergence of separation

The empirical trajectory is approximately

$$
\text{mixed naive responses}
\longrightarrow
\text{separated expert-familiar responses},
$$

while expert-novel responses remain more mixed.

If PPE and NPE neurons are fixed cell types, substantial separation should already exist whenever contextual predictions are strong enough to evoke O responses. Instead, separation develops specifically for experienced stimuli.

Predictive coding can posit learned E/I balancing that creates PPE and NPE identities. However, this makes identity:

- plastic rather than fixed;
- stimulus dependent rather than purely cell dependent;
- potentially different for familiar and novel feature combinations.

That is much closer to saying that experience-dependent opposing plasticity creates context-sensitive response categories than to saying that pre-existing PPE and NPE neurons explain the data.

---

## 4.5 Selectivity-dependent familiar suppression and enhancement

The study reports preferential suppression of broadly tuned cells and emergence or preservation of highly selective familiar NO responders.

Pure error minimization predicts suppression as a function of prediction accuracy:

$$
\Delta r_i \sim -\Delta p_i.
$$

It does not by itself predict that plasticity should depend on how many training images activate a neuron. That relationship follows naturally from activity-frequency-dependent adaptation:

$$
\Delta w_{\mathrm{FF},i}
\propto
-\text{activation frequency}_i.
$$

Broadly tuned neurons are activated often and weaken strongly. Narrowly tuned neurons are activated less frequently and preserve feedforward drive, allowing strengthened context to amplify them.

A predictive-coding network may add the same adaptation rule, but then adaptation—not signed-error subtraction—explains the tuning-breadth dependence.

---

## 5. Why context-contrasting produces all four response categories

A minimal two-compartment rate model can be written as

$$
y_i=
\left[
\gamma(u_i)\,w_{\mathrm{FF},i}x_i
+
\beta(u_i)
-
I_i x_i
-
a_i x_i
\right]_+,
$$

where

- $x_i=1$ for NO and $x_i=0$ for O;
- $u_i$ is contextual/apical activation;
- $\gamma(u_i)\geq1$ is apical gain;
- $\beta(u_i)\geq0$ is apical drive, with a higher threshold than gain;
- $I_i$ is feedforward- or surround-recruited inhibition;
- $a_i$ is familiarization-dependent feedforward adaptation.

Use separate thresholds:

$$
\theta_g \ll \theta_d,
$$

so moderate contextual activity changes gain without directly driving the cell.

One convenient choice is

$$
\gamma(u)=
\max\left\{
1,
1+\frac{g}{1+\exp[-k(u-\theta_g)]}-\frac{g}{2}
\right\},
$$

$$
\beta(u)=d[u-\theta_d]_+.
$$

This single response equation supports:

| Cell phenotype | Parameter regime | NO response | O response |
|---|---|---:|---:|
| Adapted familiar NO cell | high $a$, moderate context | decreases | weak |
| Context-amplified familiar NO cell | low $a$, $\theta_g<u<\theta_d$ | increases | absent/weak |
| O-preferring cell | strong contextual drive, strong NO-recruited inhibition | suppressed | strong |
| Mixed cell | contextual drive plus feedforward drive, modest inhibition | strong | strong |

The model therefore does not require each neuron to have a fixed PPE or NPE identity. Response category emerges from the balance of basal input, apical gain/drive, inhibition, and their plasticity.

---

# 6. Quick discriminating simulation

## 6.1 Aim

Simulate the same artificial population under:

1. a **strict PPE/NPE model** with sign-constrained connectivity;
2. a **context-contrasting model** with feedforward adaptation, contextual strengthening, apical gain/drive, and surround inhibition.

The goal is not to fit the calcium traces exactly. The goal is to determine which qualitative response configurations are reachable without adding mechanisms outside each model's defining motif.

---

## 6.2 Conditions

Use six image identities:

- four familiar images;
- two novel images.

Evaluate four principal conditions:

1. naive familiar;
2. naive novel;
3. expert familiar;
4. expert novel.

For each image, present:

- NO: local feedforward input present;
- O: local feedforward input absent, surrounding context retained.

Simulate `N = 2,000` L2/3 neurons. A static rate model is sufficient.

---

## 6.3 Model A: strict PPE/NPE population

Assign every neuron a fixed identity $z_i\in\{\mathrm{PPE},\mathrm{NPE}\}$.

For image $k$:

$$
r_{ik}^{\mathrm{PPE,NO}}
=
[w_{ik}^{\mathrm{FF}}-w_{ik}^{\mathrm P}c_k-b_i]_+,
$$

$$
r_{ik}^{\mathrm{PPE,O}}
=
[-w_{ik}^{\mathrm P}c_k-b_i]_+=0,
$$

$$
r_{ik}^{\mathrm{NPE,NO}}
=
[w_{ik}^{\mathrm P}c_k-w_{ik}^{\mathrm{FF}}-b_i]_+,
$$

$$
r_{ik}^{\mathrm{NPE,O}}
=
[w_{ik}^{\mathrm P}c_k-b_i]_+.
$$

Familiarization changes only prediction strength:

$$
w_{ik}^{\mathrm P,expert}
=
w_{ik}^{\mathrm P,naive}+\Delta w_{ik}^{\mathrm P}
$$

for familiar images, with optional partial generalization $\rho\Delta w^{\mathrm P}$ to novel images.


In the formal HPC version, this can be interpreted as learning the generative synapses:

$$
\Delta w_{ik}^{\mathrm P}
\propto
\left(r_{ik}^{\mathrm{PPE}}-r_{ik}^{\mathrm{NPE}}\right)h_k,
$$

where $h_k$ is the activity of the higher-level representation for image or feature $k$. For the quick static simulation, this learning rule can be collapsed into a direct update of $w^{\mathrm P}$.

A second version can explicitly add E/I balancing plasticity for the PPE and NPE comparison pathways. This is useful as a **predictive-coding rescue model**, but it should be labeled separately because it introduces extra learned synapses beyond the minimal fixed-sign motif.

### Mandatory predictions

Without adding precision gain, representation cells, or identity switching:

1. PPE O responses are zero.
2. NPE mixed responses satisfy
   $$
   r_{\mathrm O}\geq r_{\mathrm{NO}}.
   $$
3. Familiarization cannot increase a PPE response to the unchanged familiar NO input.
4. An increased familiar NO response in an NPE cell must be accompanied by an equal or larger O response.
5. Learning moves PPE cells leftward in the NO–O plane and NPE cells upward; it cannot create selective rightward shifts with weak O activity.

These are analytic constraints, so simulation is mainly a visualization of them.

---

## 6.4 Model B: context-contrasting population

For neuron $i$, image $k$, and condition $x\in\{0,1\}$:

$$
u_{ik}=w_{ik}^{\mathrm{FB}}c_k,
$$

$$
r_{ik}(x)=
\left[
\gamma(u_{ik})w_{ik}^{\mathrm{FF}}x
+
\beta(u_{ik})
-
I_i x
-
a_{ik}x
-b_i
\right]_+.
$$

Set

$$
x=1 \text{ for NO},
\qquad
x=0 \text{ for O}.
$$

### Plasticity

Let familiar feedforward weakening depend on tuning breadth or activation frequency:

$$
w_{ik}^{\mathrm{FF,expert}}
=
w_{ik}^{\mathrm{FF,naive}}
\left(1-\eta_{\mathrm{FF}} f_i\right),
$$

where $f_i$ is the fraction of familiar images activating neuron $i$.

Alternatively, represent this as an adaptation term

$$
a_{ik}^{\mathrm{expert}}
=
\eta_a f_i
$$

for familiar images.

Strengthen contextual input:

$$
w_{ik}^{\mathrm{FB,expert}}
=
w_{ik}^{\mathrm{FB,naive}}+
\eta_{\mathrm{FB}}q_{ik},
$$

where $q_{ik}$ represents exposure to contextual features. Allow generalization to novel scenes by correlating novel and familiar contextual feature vectors or using

$$
\Delta w_{\mathrm{FB,novel}}
=
\rho\Delta w_{\mathrm{FB,familiar}},
\qquad 0<\rho<1.
$$

### Useful heterogeneity

Sample neurons across:

- narrow versus broad feedforward tuning;
- weak versus strong contextual weights;
- weak versus strong surround inhibition;
- apical activation below the gain threshold, between gain and drive thresholds, or above the drive threshold.

No categorical cell labels are required.

---

## 6.5 Plots

### Plot 1: NO–O response scatter for each condition

Create one scatter plot per condition:

$$
x\text{-axis}=r_{\mathrm{NO}},
\qquad
y\text{-axis}=r_{\mathrm O}.
$$

Add the diagonal $r_{\mathrm O}=r_{\mathrm{NO}}$.

For strict PC, shade the analytically reachable regions:

- PPE: horizontal axis;
- NPE: region on/above the diagonal.

Points in the lower-right mixed quadrant

$$
r_{\mathrm{NO}}>r_{\mathrm O}>0
$$

are violations of the strict cell-level motif under matched predictions.

### Plot 2: matched-cell arrows from naive to expert familiar

For each neuron, draw an arrow

$$
(r_{\mathrm{NO}}^{\mathrm{naive}},r_{\mathrm O}^{\mathrm{naive}})
\rightarrow
(r_{\mathrm{NO}}^{\mathrm{expert}},r_{\mathrm O}^{\mathrm{expert}}).
$$

Highlight four classes:

1. familiar NO suppression;
2. familiar NO amplification with weak O;
3. emergence of O responses;
4. mixed-to-separated transitions.

Strict PC cannot produce class 2 without an added gain or representation mechanism.

### Plot 3: tuning breadth versus familiar NO plasticity

Plot

$$
f_i
\quad\text{versus}\quad
\Delta r_{i,\mathrm{NO}}
=
r_{i,\mathrm{NO}}^{\mathrm{expert}}
-r_{i,\mathrm{NO}}^{\mathrm{naive}}.
$$

Context-contrasting with frequency-dependent feedforward weakening should produce a negative relationship: broad cells decrease more strongly, whereas narrowly tuned cells are preserved or amplified.

The strict PPE/NPE model has no reason to generate this relationship unless prediction learning or precision is explicitly made tuning-breadth dependent.

### Plot 4: population summary

For every condition compute:

- fraction NO-only;
- fraction O-only;
- fraction mixed;
- fraction unresponsive;
- fraction with increased familiar NO response;
- fraction with increased familiar NO response **and weak O response**;
- NO–O response correlation;
- the paper's separation index, if its exact implementation is copied.

The most discriminating statistic is

$$
P\left(
\Delta r_{\mathrm{NO,familiar}}>0
\;\land\;
r_{\mathrm O,familiar}<\tau
\right).
$$

This probability should be approximately zero in strict PPE/NPE coding but positive in context-contrasting.

---

## 6.6 Minimal pseudocode
ignore

## 6.7 Strongest version: constrained model comparison

Rather than visually choosing parameters, define a qualitative target vector based on the experiment:

$$
T=
\begin{bmatrix}
\text{mean familiar NO decreases}\\
\text{mean familiar O increases}\\
\text{nonzero familiar-NO amplifier fraction}\\
\text{mixed naive fraction}\\
\text{mixed expert-novel fraction}\\
\text{expert-familiar separation increases}\\
\text{breadth--plasticity correlation is negative}
\end{bmatrix}.
$$

Optimize each model's parameters against these summary targets using random search or `scipy.optimize.differential_evolution`.

### Expected result

The strict PPE/NPE model can fit:

- familiar NO suppression;
- familiar O enhancement;
- expert-familiar separation;
- strong novel PPE responses.

It should fail simultaneously on:

- familiar NO amplification with weak O;
- general NO-dominant mixed cells;
- mixed naive and expert-novel populations combined with strong expert-familiar separation;
- tuning-breadth-dependent plasticity, unless an extra rule is introduced.

The context-contrasting model should satisfy all constraints with the same mechanisms:

- activation-frequency-dependent feedforward weakening;
- strengthened contextual input;
- low-threshold apical gain;
- higher-threshold apical drive;
- feedforward/surround-recruited inhibition.

---

# 7. Interpretation of a successful predictive-coding rescue

The comparison should distinguish **canonical PPE/NPE coding** from increasingly flexible predictive-processing models.

If predictive coding only fits after introducing

- precision-dependent amplification;
- excitatory representation neurons in L2/3;
- stimulus-dependent switching between PPE and NPE identity;
- mixed error signs across feature dimensions;
- independent feedforward adaptation;
- local surround inhibition;

then the correct conclusion is not that the data are impossible under all predictive-processing frameworks. It is:

> The fixed cell-level PPE/NPE motif does not explain the observations on its own. The successful model fits because it has incorporated mechanisms that are central to context-contrasting or related recurrent-inference accounts.

The sharpest empirical signatures are therefore not the population-average familiar suppression or O enhancement. They are:

1. **familiar NO amplification with weak O response in the same cell;**
2. **NO-dominant mixed O/NO responders;**
3. **mixed naive and expert-novel populations combined with learned expert-familiar separation;**
4. **plasticity determined by tuning breadth or activation frequency.**

---

## References

- Seignette, K. et al. (2025). *A visual occlusion paradigm uncovers a circuit mechanism for selective amplification of salient visual inputs.* In particular Figures 2–4 and Supplementary Figures S3–S8.
- Rao, R. P. N. & Ballard, D. H. (1999). *Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects.* Nature Neuroscience, 2, 79–87.
- Keller, G. B. & Mrsic-Flogel, T. D. (2018). *Predictive Processing: A Canonical Cortical Computation.* Neuron, 100, 424–435.
- Westerberg, J. A. & Roelfsema, P. R. (2025). *Hierarchical interactions between sensory cortices defy predictive coding.* Trends in Cognitive Sciences.
