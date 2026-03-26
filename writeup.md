# Variable-Length PIP + Generator Matching + `x1`-Prediction Notes

## Sources

- Generator Matching (GM): <https://arxiv.org/abs/2410.20587>
- Time dependent loss reweighting / `x1`-prediction note: <https://arxiv.org/abs/2511.16599>
- Local implementation: [src/poissonindelprocess.jl](/home/claudey/pipflows/Flowfusion.jl/src/poissonindelprocess.jl)

## Goal

The current `UniformDiscretePoissonIndelProcess` implementation in this branch trains a model to predict the conditional Doob hazards directly:

- deletion hazards
- substitution hazards
- insertion hazards

The question is whether this can be converted into an `x`-prediction / `x1`-prediction scheme, ideally with a cross-entropy loss against the terminal sequence `x1`, while staying inside the Generator Matching (GM) framework.

Short answer:

- Predicting rates or rate multipliers is immediately GM-compatible for PIP.
- Predicting the raw final sequence `x1` is **not** a simple GM-compatible linear parameterization in the same way as standard Euclidean flow matching.
- There is an **exact but likely intractable** GM formulation that predicts a full posterior distribution over terminal sequences.
- The simple practical idea "predict one endpoint sequence `x̂1`, then run the exact PIP guide on `x̂1`" is a plausible heuristic, but not an exact GM `x1`-prediction construction.
- A more promising route is a **hybrid**: keep PIP as the alignment/indel scaffold, but replace the per-lineage token mutation kernel by a DFM-style process. That does seem to recover an `x1`-prediction structure for the mutation part, and possibly for indels too if gaps are promoted to a dummy token in an extended alphabet.

## 1. GM theory that matters here

### 1.1 Linear parameterization of the generator

The key GM notion is:

`L_t^z f(x) = <K_{t,x} f, F_t^z(x)>`

where:

- `L_t^z` is the conditional generator given endpoint / latent `z`
- `K_{t,x}` is a fixed linear map acting on test functions
- `F_t^z(x)` is the target the network is allowed to predict

The important point is not "what is the state space?" but "is the generator linear in the thing I want to predict?"

GM Proposition 1 says the marginal generator is obtained by posterior averaging the conditional generator. In a linear parameterization, this means:

`F_t(x) = E[F_t^Z(x) | X_t = x]`

So if the conditional generator is linear in some target `F_t^z(x)`, then training on conditional targets is enough.

### 1.2 Why the loss has to be Bregman

GM Proposition 2 is the central training statement:

- the intractable marginal GM loss
- and the tractable conditional GM loss

have the same gradients if the loss is a Bregman divergence.

This is exactly why MSE, Poisson-style losses, BCE-style losses, etc. show up in GM.

For PIP this matters because the current loss in [src/poissonindelprocess.jl](/home/claudey/pipflows/Flowfusion.jl/src/poissonindelprocess.jl) is

`p * (log p - log q) - p + q`

which is the standard Poisson-style positive Bregman divergence, applied to nonnegative hazards.

### 1.3 What the reweighting paper adds

The 2025 note makes three extensions explicit:

- the linear parameterization may depend on both `t` and `X_t`
- the Bregman divergence may depend on both `t` and `X_t`
- the expectation over time can use a broad class of time distributions and positive reweightings

The practical message is:

- time-dependent loss scaling is theoretically fine, as long as it is positive almost everywhere
- it is fine for the thing being predicted to itself have a `t`-dependent meaning

This matters for PIP because the remaining branch length `1 - t` changes the guide sharply, and because any endpoint-based parameterization will necessarily be `t`-dependent.

### 1.4 When `x1`-prediction is valid in GM

The reweighting note gives the standard flow-matching story:

- if the conditional drift is affine in `x1`
- `u_t(x | x1) = A_{t,x} x1 + b_{t,x}`

then the generator is linear in `x1`, and a valid CGM loss can be a Bregman divergence directly between true `x1` and predicted `x̂1`.

This is why MSE-to-`x1` works for many Euclidean flow-matching constructions.

For jump models, the note also shows that if the jump kernel factors as

`Q_t(dy; x) = sum_j h_{t,j}(x) R_{t,j}(x) delta_{Gamma_{t,j}(x)}(dy)`

then one can move the time-dependent hazards `h_{t,j}` into the inner product and predict only the rate multipliers `R_{t,j}`. This is much closer to the current PIP setup.

## 2. What the local PIP code is doing

Everything below refers to [src/poissonindelprocess.jl](/home/claudey/pipflows/Flowfusion.jl/src/poissonindelprocess.jl).

### 2.1 Process definition

The state is a variable-length discrete sequence over `1:K`.

The process has:

- insertion rate `lambda`
- deletion rate `mu`
- substitution parameter `alpha`

For branch length `s`:

- survival is `exp(-mu s)`
- substitution kernel is uniform-discrete with
  - diagonal mass `exp(-alpha s) + (1 - exp(-alpha s))/K`
  - off-diagonal mass `(1 - exp(-alpha s))/K`
- insertion mass per token is `lambda * ((1 - exp(-mu s))/mu) / K`

### 2.2 Bridge construction `bridge(P, x0, x1, t)`

The bridge sampler is not doing naive ancestral sampling. It uses a very specific two-branch construction.

At time `t`:

- left branch length is `t`
- right branch length is `1 - t`

The code precomputes:

- left and right survival terms
- left and right substitution kernels
- left and right insertion masses

Then it builds a two-sequence alignment DP over `(x0, x1)` with three event types:

- `A`: symbol only on the `x0` side
- `B`: symbol only on the `x1` side
- `R`: paired symbol on both sides

For `R`, the weight depends on whether the paired letters are equal.

The sequence of steps is:

1. `make_precomp`
   - builds all branch/time constants

2. `make_kernels`
   - turns those constants into normalized alignment-step weights `pA`, `pB`, `pR_diag`, `pR_off`

3. `sample_alignment_ud`
   - forward DP to compute total alignment mass
   - backward sampling of one alignment path

4. `sample_Xt_ud`
   - given the sampled path, sample the actual ancestral symbols in `X_t`
   - for `R`, sample ancestral token from the product of left/right substitution factors
   - for `A`/`B`, mix
     - "present at time `t`"
     - "inserted later on one branch"
   - finally add ghost insertions from the immortal-line PIP construction

So the bridge is exact with respect to the chosen PIP construction, but implemented through dynamic programming and local ancestral token sampling rather than brute-force path enumeration.

### 2.3 Conditional guide / Doob target

Once `X_t` is sampled, the current implementation computes the exact conditional generator for the remaining right branch to `x1`.

Conceptually this is a Doob `h`-transform:

- let `h_t^{x1}(x) = P(X_1 = x1 | X_t = x)` under the unconditional right-branch PIP
- for any allowed event `e: x -> x^e`

`q_t^{x1}(e | x) = q_t^0(e | x) * h_t^{x1}(x^e) / h_t^{x1}(x)`

where the base hazards are:

- deletion at position `i`: `mu`
- substitution at position `i` to token `b != x_i`: `alpha / K`
- insertion of token `b` at gap `s`: `lambda / K`

The generator is then

`L_t^{x1} f(x) = sum_e q_t^{x1}(e | x) [f(x^e) - f(x)]`

This is already a perfect linear parameterization if the model predicts the hazard vector itself.

### 2.4 How the code computes the guide efficiently

The code does not explicitly recompute `h(x^e)` for every event. Instead it uses a right-branch forward-backward DP in log-space:

- `doob_ud_grouped_fast`
- `doob_ud_full_tensors_fast`

These compute:

- `del[i]`
- `sub[b, i]`
- `ins[b, s]`

from inside-outside ratios.

The structure is:

1. build right-branch constants for remaining time `1 - t`
2. compute forward table `logF`
3. compute backward table `logB`
4. combine them into event-specific hazard ratios

This is exactly the place where the dependence on the terminal sequence becomes subtle: the event hazards depend on `x1` through the entire alignment DP, not just through local token identities.

### 2.5 Current training target and loss

The current batched target is built by:

- sampling `X_t` with `bridge`
- computing exact guide tensors with `Guide(P, ts, Xts, x1s)`

Then `floss` applies a positive Bregman divergence between:

- transformed model outputs
- target hazards

for substitutions, deletions, and insertions separately.

So the current construction is already very naturally a GM jump-model construction:

- target space = nonnegative hazard tensors
- loss = Poisson-style Bregman
- `t`-dependence handled explicitly

## 3. Can PIP be converted to raw `x1`-prediction?

### 3.1 The formal requirement

To get a true GM `x1`-prediction scheme, we need a representation `psi_{t,x}(x1)` such that

`L_t^{x1} f(x) = <K_{t,x} f, psi_{t,x}(x1)>`

and we want `psi_{t,x}(x1)` to be "just the final state" or a simple encoding of it.

This is what happens in ordinary flow matching when the drift is affine in `x1`.

### 3.2 Why PIP is different

For PIP, the exact conditional hazards are Doob ratios:

`q_t^{x1}(e | x) = q_t^0(e | x) * h_t^{x1}(x^e) / h_t^{x1}(x)`

with

`h_t^{x1}(x) = P(X_1 = x1 | X_t = x)`

Now:

- `h_t^{x1}(x)` is computed by a sequence-alignment DP
- it sums over all alignments and latent insertion/deletion ancestry choices
- `q_t^{x1}(e | x)` is a ratio of two such global quantities

So as a function of the raw terminal sequence `x1`, the guide is not affine. It is a complicated rational function of the entire alignment structure.

Even before taking the ratio, the branch likelihood is already a global DP object. After taking the Doob ratio, the nonlinearity is unavoidable.

This is the central obstruction.

### 3.3 A more explicit obstruction

Suppose we encode `x1` as a padded one-hot matrix.

Then the right-branch likelihood `h_t^{x1}(x)` is obtained by summing over all alignment paths between `x` and `x1`. The DP entries are built recursively from whether:

- `x_i == x1_j`
- letters are skipped on one side
- inserted letters explain `x1_j`

Therefore:

- the DP entries are nonlinear functions of the entire padded one-hot sequence
- the hazards use ratios of sums of those DP entries

This is very different from the standard flow-matching case where the generator depends on `x1` through one affine map `A_{t,x} x1 + b_{t,x}`.

So a direct per-token cross-entropy loss to the final sequence does **not** correspond to a simple linear parameterization of the PIP guide.

### 3.4 What *is* exactly possible

There is one exact GM construction that uses endpoint prediction, but it is much bigger than usual `x1`-prediction.

Let `Z` be the space of all terminal sequences. Define a model posterior over full terminal sequences:

`pi_theta(z | t, x)`

Then define

`L_t^theta f(x) = sum_{z in Z} pi_theta(z | t, x) L_t^z f(x)`

This is linear in the predicted distribution `pi_theta(· | t, x)`. So with

- `F_t^z(x) = delta_z`
- `F_t^theta(x) = pi_theta(· | t, x)`

GM applies, and sequence-level cross-entropy against the true endpoint is a legitimate Bregman loss on the simplex over complete sequences.

This is the clean theoretical endpoint-prediction story for PIP.

### 3.5 Why the exact construction is probably intractable

The exact construction above has two serious problems.

First, the output space is huge:

- variable length
- combinatorial support
- effectively unbounded sequence space

Second, generation requires the expected guide:

`q_t^theta(e | x) = sum_z pi_theta(z | t, x) q_t^z(e | x)`

and each `q_t^z(e | x)` itself requires a PIP forward-backward computation.

So exact inference would require either:

- summing over all terminal sequences
- or Monte Carlo over endpoint sequences

Neither is remotely as clean as standard `x1`-prediction in Euclidean FM.

### 3.6 Why "decode one `x̂1` and run the exact guide" is not exact GM

A tempting scheme is:

1. train a model with cross-entropy to predict one terminal sequence `x̂1`
2. at generation time compute `Guide(P, t, X_t, x̂1)`

This is operationally simple, but it is **not** the exact GM construction corresponding to endpoint prediction.

The exact GM endpoint-posterior construction would use:

`sum_z pi_theta(z | t, x) L_t^z`

not:

`L_t^{x̂1}`

for one decoded endpoint.

These are only equal in degenerate cases. In particular, because PIP uses Doob ratios of branch likelihoods, the dependence on the endpoint does not commute with:

- argmax decoding
- taking an expected terminal token sequence
- plugging in a single sampled endpoint

So this route should be viewed as a heuristic or a Monte Carlo approximation, not as an exact `x1`-prediction derivation.

### 3.7 A more realistic exact GM reformulation: rate multipliers

A much more natural GM reformulation for PIP is to factor out the base hazards and predict only the conditional multipliers:

- `R_del(i | x, x1, t) = h(x^{-i}) / h(x)`
- `R_sub(i, b | x, x1, t) = h(x^{i->b}) / h(x)`
- `R_ins(s, b | x, x1, t) = h(x^{+s,b}) / h(x)`

Then:

- base hazards are `mu`, `alpha / K`, `lambda / K`
- the generator is linear in these multiplier tensors

This fits the jump-model discussion in the reweighting note very well.

Relative to the current code, this is only a small conceptual shift:

- current training predicts hazards directly
- multiplier prediction would predict hazards divided by the time/state dependent base hazards

That is GM-clean, tractable, and much closer to the existing implementation than raw endpoint prediction.

## 4. Practical options

### Option A: stay with hazard prediction

This is the cleanest current setup.

Pros:

- exact existing target
- exact existing guide
- exact GM interpretation
- already implemented

### Option B: switch to multiplier prediction

This is also theoretically clean.

Pros:

- closer to the jump-model formulation in the reweighting note
- base hazard schedule can be factored out explicitly
- may stabilize the scale of the learning problem

Cons:

- still not raw `x1`-prediction

### Option C: endpoint posterior prediction over full sequences

This is the exact `x1`-prediction analogue for PIP.

Pros:

- formally valid GM endpoint prediction
- cross-entropy against the true full sequence is meaningful

Cons:

- output space is huge
- exact guide requires averaging conditional guides over predicted sequence posterior
- almost certainly too expensive without aggressive approximation

### Option D: heuristic decoded-endpoint guidance

Train an endpoint predictor, decode one `x̂1`, then call the exact guide on `x̂1`.

Pros:

- easy to prototype
- reuses existing `Guide(P, t, Xt, X1)` machinery

Cons:

- not exact GM
- sensitive to decoding errors
- a single wrong insertion/deletion can change the whole alignment DP
- likely brittle early in generation when `Xt` is still far from the target

### Option E: Monte Carlo endpoint guidance

Train a posterior over endpoints, sample `M` candidate endpoints `x̂1^(m)`, compute guides for each, average them.

Pros:

- closer to the exact endpoint-distribution parameterization

Cons:

- very expensive
- high variance if `M` is small
- still substantially harder than direct hazard prediction

## 5. My current conclusion

I do **not** think there is a nice tractable raw `x1`-prediction formulation for PIP analogous to standard flow matching.

The reason is structural:

- the PIP conditional generator depends on `x1` through a global alignment / branch-likelihood DP
- the Doob transform introduces ratios of those global quantities
- this destroys the kind of simple affine dependence on `x1` that makes ordinary `x1`-prediction work

What *is* true is:

1. An exact GM endpoint-prediction formulation exists if the model predicts a full posterior over terminal sequences.
2. That exact formulation is probably too expensive to be practical.
3. Predicting hazards or Doob multipliers remains the natural tractable GM parameterization for PIP.
4. "Predict one endpoint then run the exact guide" is a heuristic worth experimenting with, but it should not be confused with an exact GM derivation.
5. If we are willing to change the mutation model while keeping PIP-style alignment marginalization, then a hybrid PIP+DFM process looks genuinely promising.

## 6. Recommendation for the next implementation step

If the goal is to stay theoretically clean and make progress quickly, I would prioritize:

1. Re-express the current target as Doob multipliers rather than raw hazards.
2. Keep the current exact bridge / exact guide machinery.
3. Optionally add a side experiment where a model predicts a candidate terminal sequence and the guide is run on that decoded sequence, but label it explicitly as heuristic.

If the goal is to pursue true endpoint prediction anyway, the least misleading research path seems to be:

1. define a posterior model `pi_theta(x1 | Xt, t)` over full variable-length sequences
2. train it with sequence-level cross-entropy
3. approximate the GM generator by averaging exact PIP guides over sampled endpoints from `pi_theta`

That is the closest thing to a principled PIP `x1`-prediction construction I currently see.

## 7. Hybrid PIP alignment + DFM mutation

The above conclusion is about the **current PIP CTMC** in [src/poissonindelprocess.jl](/home/claudey/pipflows/Flowfusion.jl/src/poissonindelprocess.jl), where the guide is built from Doob ratios of a right-branch likelihood.

There is a different and more promising question:

- keep the PIP-style marginalization over alignments / indel structure
- but replace the mutation process on aligned letters by one of the DFM processes already implemented in [src/processes.jl](/home/claudey/pipflows/Flowfusion.jl/src/processes.jl)

I think the answer here is **yes, partially**, and the cleanest way to see it is to condition on an alignment first.

### 7.1 DFM recap in the form we need

For the DFM convex paths from 2407.15595:

- `InterpolatingDiscreteFlow` uses
  - `p_t(x_i | x0_i, x1_i) = (1 - kappa_t) delta_{x0_i} + kappa_t delta_{x1_i}`
- `NoisyInterpolatingDiscreteFlow` uses
  - `p_t(x_i | x0_i, x1_i) = kappa1_t delta_{x1_i} + kappa2_t p_unif + kappa3_t delta_{x0_i}`

and the marginal velocity has the generic form

`u_t = A_t * p_{1|t}(· | X_t) + B_t * delta_{X_t} + C_t * p_unif`

So the mutation part is linear in the posterior over the terminal token. This is exactly what makes `x1`-prediction work in DFM.

### 7.2 Conditioning on an alignment makes the token process fixed-length

Suppose that for a current variable-length sequence `X_t` and terminal sequence `x1`, we condition on an alignment / ancestry object `A` saying:

- which current sites survive to which leaf positions
- which current sites are deleted
- which leaf positions arise from future insertions, and in which current gap

Once `A` is fixed, each surviving lineage is just a **single-site discrete path to a known final token**.

That is exactly the regime where the DFM mutation processes apply.

For a surviving current site `i` with current token `z = X_t[i]` and assigned leaf token `y = x1[j]`, the DFM mutation velocity can be written as

`u_t(· | z, y) = A_t * delta_y + B_t * delta_z + C_t * p_unif`

for the appropriate scheduler coefficients.

### 7.3 Marginalizing the alignment preserves the `x1`-prediction form

Now marginalize over the alignment posterior rather than fixing `A`.

Define, for each current site `i`, the alignment-marginalized posterior over the final token of the surviving descendant:

`pi_i(b | X_t, x1, t) = P(descendant token = b | site i survives, X_t, x1, t)`

Then by linearity of the DFM velocity in `delta_y`,

`E_A[u_t(· | z, y_A)] = A_t * pi_i(· | X_t, x1, t) + B_t * delta_z + C_t * p_unif`

So:

- the **mutation** part does compose cleanly with alignment marginalization
- the sufficient target is no longer the raw sequence `x1`
- it is the alignment-posterior token distribution for each current site

This is a genuine `x1`-prediction-style parameterization in the DFM sense, but at the level of **alignment-aware local token posteriors**, not the whole raw endpoint sequence.

### 7.4 What happens to deletions and insertions

The structural part does not disappear.

For each current site `i`, we still need something like

- `p_del(i | X_t, x1, t)` or a deletion hazard

and for each gap `s`, we still need something like

- `p_ins(s | X_t, x1, t)`
- plus a posterior over the token of the inserted descendant

So the most natural first hybrid is:

1. keep PIP-style structural posteriors / hazards for delete and insert
2. replace only the token mutation on surviving or newly created lineages by a DFM update driven by alignment-marginalized target-token posteriors

This would already give a partially `x1`-predictive PIP-like model.

### 7.5 Extended-alphabet view: use a dummy token for gaps

There is a stronger construction that looks even better conceptually.

Introduce an extended alphabet

`Sigma_tilde = Sigma union {gap_or_mask}`

and represent an alignment column as a pair of tokens in this extended alphabet:

- match: `(a, b)`
- deletion: `(a, mask)`
- insertion: `(mask, b)`

Now put a DFM process on each aligned column in the extended alphabet.

This is especially natural for [src/processes.jl](/home/claudey/pipflows/Flowfusion.jl/src/processes.jl):

- `InterpolatingDiscreteFlow` handles the simple convex path
- `NoisyInterpolatingDiscreteFlow(...; dummy_token=mask)` is already designed for a masked token

In this view:

- deletion is just "target token = mask"
- insertion is just "source token = mask"
- substitution is just "source token = a, target token = b"

Then:

- aligned-column mutation is a standard DFM problem on a fixed alphabet
- variable length is recovered by dropping the mask columns/tokens after marginalizing over alignments

This is the cleanest route I currently see to a true PIP-like + DFM hybrid.

### 7.6 What this buys us and what it does not

What this buys us:

- the token update rules become DFM / `x1`-prediction compatible
- alignment marginalization does **not** break that compatibility, because the local DFM velocity is linear in the target posterior
- the existing Flowfusion implementations of DFM processes are directly relevant

What it does not buy us automatically:

- a simple cross-entropy on the raw visible terminal sequence with no alignment machinery
- removal of the need to compute alignment posteriors
- an exact reduction back to the current PIP CTMC

So this is best thought of as a **new hybrid conditional path family**, not as a trivial reinterpretation of the existing PIP guide.

### 7.7 Practical implementation path

The least risky way to explore this in code would be:

1. Keep the current PIP forward-backward machinery, but reinterpret its output as alignment posteriors rather than only as hazards.
2. For each current site, compute:
   - survival / deletion posterior
   - posterior over terminal token conditional on survival
3. For each gap, compute:
   - insertion posterior
   - posterior over terminal token conditional on insertion
4. Replace the current uniform-CTMC substitution hazard update by:
   - a DFM site update using `InterpolatingDiscreteFlow` or `NoisyInterpolatingDiscreteFlow`
   - driven by those local token posteriors
5. Initially keep insert/delete as PIP structural hazards.
6. If that works, move to the stronger extended-alphabet construction with an explicit dummy token for gaps.

### 7.8 Bottom line on the hybrid idea

So my current answer to your new question is:

- **Yes**, PIP alignment marginalization and DFM-style `x1`-predictive mutation appear to compose cleanly.
- The clean object after marginalizing alignments is not the raw terminal sequence, but alignment-aware local posteriors over terminal tokens and gap/mask states.
- The most principled version is probably an extended-alphabet aligned-column model with a dummy token, plus PIP-style alignment marginalization.
- The easiest first experiment is a mixed model: PIP for structure, DFM for token mutation on surviving/born lineages.
