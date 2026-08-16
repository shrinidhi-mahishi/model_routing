# DevConf Model Routing Likely Questions

This is a talk-prep Q&A sheet based on:

- `model_routing/notebooks/bayesian_router_talk_end_to_end.ipynb`
- the current production-friendly Bayesian-router framing in the repo

Use this as:

- a rehearsal checklist before the talk
- a backup-notes sheet for Q&A
- a source for concise answers when someone asks a math-heavy question

The answers below are intentionally phrased in a way you can say out loud.

## How To Use This Sheet

For each question, there are usually two useful layers:

- `Short answer` = what you can say first, live, in 1-3 sentences
- `If they push` = the more precise or technical follow-up

## Most Likely Questions
### 0. how do beta distribution get narrower after adding more evidences
A Beta belief gets narrower because each new observation increases its effective sample size.

If your prior is Beta(alpha, beta) and you observe:

success: alpha <- alpha + 1
failure: beta <- beta + 1
then uncertainty is controlled by alpha + beta (total evidence).

For Beta:

Mean: alpha / (alpha + beta)
Variance: alpha*beta / [ (alpha+beta)^2 * (alpha+beta+1) ]
As alpha + beta grows, that denominator grows fast, so variance shrinks → the curve becomes tighter (narrower) around the mean.

Example:

Beta(1,1) (no evidence): very wide/flat
after 100 observations with ~70% success: Beta(71,31) → much narrower around ~0.70
In your router terms:

more telemetry updates (validity/reward evidence) = more concentrated belief
if you apply decay (gamma < 1), old evidence is discounted, effective alpha+beta drops, and beliefs widen again so the router can re-explore.


### 1. What problem are you actually solving?
**Short answer:**  
I am solving the problem of choosing the cheapest model that still meets the quality bar for a request, without requiring humans to label every response.

**If they push:**  
The core tension is:

- strongest model only = expensive
- cheapest model only = risky
- hand-written rules = brittle

So the router learns online from telemetry and adapts over time.

### 2. What is the key novelty here?
**Short answer:**  
The key idea is turning normal production telemetry into a reward signal, so the router can learn without human labels.

**If they push:**  
The reward comes from:

- validity / parse success
- latency
- retries

Then Thompson Sampling uses that reward to update beliefs and route future traffic.

### 3. Why not just always use `gpt-4o-mini`?
**Short answer:**  
Because some requests really do need a stronger model, and the cheap model can be too weak on those.

**If they push:**  
The goal is not “always cheapest.” The goal is “cheap when safe, strong when needed.”

### 4. Why not just always use `gpt-4o`?
**Short answer:**  
Because you pay the maximum cost on every request, including the easy ones.

**If they push:**  
If a large fraction of traffic is easy or structured, paying strongest-model prices everywhere is wasteful.

### 5. Why do you need Bayesian bandits here?
**Short answer:**  
Because this is a repeated online decision problem under uncertainty.

**If they push:**  
Bandits are a good fit because:

- each request is one choice
- reward comes back quickly
- you want exploration early and exploitation later

## Reward And Telemetry Questions

### 6. What exactly is the reward?
**Short answer:**  
It is a weighted score built from validity, latency, and retries.

**Formula:**  

```text
reward = w_validity * validity_score
       + w_latency  * latency_score
       + w_retry    * retry_score
```

### 7. Why does latency need normalization?
**Short answer:**  
Because raw milliseconds are not naturally on a `0..1` scale, but the reward needs comparable components.

**If they push:**  
Validity is already naturally bounded, but latency is not. So we map latency into a bounded score before combining it with the other terms.

### 8. Why do you use a sigmoid for latency?
**Short answer:**  
Because it gives a smooth penalty instead of a hard threshold.

**If they push:**  
With a sigmoid:

- very fast responses get high credit
- very slow responses get low credit
- responses near the midpoint are treated gradually, not as a cliff

### 9. Why not use a sigmoid for validity too?
**Short answer:**  
Because validity already means what we want on a `0..1` scale.

**If they push:**  
If `validity_score = 0.8`, that already means “mostly good.” A linear mapping is easier to explain and preserves that meaning directly.

### 10. Why not use a sigmoid for retries?
**Short answer:**  
Because retries are easier to explain as a direct penalty.

**If they push:**  
A retry count has an operational meaning:

- `0` retries = best
- `1` retry = worse
- `2+` retries = increasingly undesirable

The notebook uses a simple, interpretable penalty rather than a more abstract transform.

### 11. Where does `validity_score` come from in real systems?
**Short answer:**  
Usually from structured-output checks or task-specific validators.

**Examples:**

- JSON parses successfully
- schema validation passes
- required fields exist
- tool output is consistent
- downstream checks succeed

### 12. Is `validity_score` the same as true correctness?
**Short answer:**  
No. It is a proxy.

**If they push:**  
That is one of the most important caveats. The router learns whatever your validator measures. If the validator is weak, the router can learn the wrong thing.

### 13. How do retries affect the reward?
**Short answer:**  
Retries reduce reward because they mean the first answer was not good enough.

**If they push:**  
Retries are hidden repair work. Even if the final answer is acceptable, a model that needs retries is operationally worse.

### 14. Why does cost not appear directly in the reward?
**Short answer:**  
In the current design, cost is indirect rather than explicitly subtracted.

**If they push:**  
The notebook is careful here. Cost goes down mainly because cheaper models are often faster and good enough, so they win on the composite reward by effect, not by an explicit dollar penalty.

**Possible extension:**  

```text
reward = validity + latency + retry - lambda * cost
```

### 15. Could you make cost explicit?
**Short answer:**  
Yes, that is a natural next step.

**If they push:**  
A cost-aware reward is straightforward to define. The harder part is choosing the right trade-off coefficient so you do not accidentally optimize for cheap but low-value outputs.

## Beta / Thompson Sampling / Math Questions

### 16. What does `Beta(8, 3)` mean here?
**Short answer:**  
It means the router currently believes the model’s quality is probably around `8 / (8 + 3) = 0.727`, with some uncertainty.

### 17. What do `alpha` and `beta` mean?
**Short answer:**  
`alpha` is success-like evidence and `beta` is failure-like evidence.

**If they push:**  
In the textbook Beta-Bernoulli case:

- `alpha` behaves like good outcomes
- `beta` behaves like bad outcomes

In this project they become soft pseudo-counts because the reward is continuous.

### 18. Why use a distribution instead of just one average?
**Short answer:**  
Because the distribution captures both the current estimate and the uncertainty.

**If they push:**  
That uncertainty is what makes exploration principled instead of arbitrary.

### 19. What is the mean of a Beta distribution?
**Formula:**  

```text
mean = alpha / (alpha + beta)
```

**Example:**  

```text
Beta(8, 3) -> 8 / 11 ≈ 0.727
```

### 20. Why does a bigger `alpha + beta` make the curve narrower?
**Short answer:**  
Because it means the router has accumulated more evidence.

**If they push:**  
`Beta(8,3)` and `Beta(80,30)` have the same mean, but the second is much more confident because the total evidence is much larger.

### 21. Where do sample values like `0.65`, `0.72`, `0.80` come from?
**Short answer:**  
They are random draws from the Beta distribution.

**If they push:**  
`Beta(8,3)` defines a whole curve. `np.random.beta(8,3)` picks one random sample from that curve.

### 22. What is Thompson Sampling in one sentence?
**Short answer:**  
Sample one temporary score from each model’s belief distribution and choose the highest.

### 23. Why sample instead of using the mean?
**Short answer:**  
Because if you always used the mean, you would stop exploring.

**If they push:**  
Sampling lets uncertain models sometimes win even when their current mean is lower.

### 24. Why does exploration slow down automatically?
**Short answer:**  
Because the belief distributions get narrower as evidence accumulates.

**If they push:**  
Early on, wide curves overlap more, so weaker models sometimes win by chance. Later, narrow curves overlap less, so the better model wins more consistently.

### 25. How are Bayes theorem, Beta, and Bernoulli connected?
**Short answer:**  
Bernoulli is the data model, Beta is the belief model, and Bayes theorem is the update rule connecting them.

**Textbook story:**  

```text
x ~ Bernoulli(p)
p ~ Beta(alpha, beta)
```

### 26. What is conjugacy here?
**Short answer:**  
Beta is conjugate to Bernoulli, meaning a Beta prior plus Bernoulli data gives a Beta posterior again.

### 27. Can you give the textbook update?
**Short answer:**  
Yes.

**Formula:**  

```text
prior:     p ~ Beta(alpha, beta)
success:   Beta(alpha + 1, beta)
failure:   Beta(alpha, beta + 1)
```

### 28. But your rewards are not binary. Is this still exact Bayes?
**Short answer:**  
No, not exactly.

**If they push:**  
This implementation uses a pragmatic fractional pseudo-count update:

```text
alpha += reward
beta  += 1 - reward
```

That is inspired by Beta-Bernoulli Bayes, but it is not exact conjugate Bayes for continuous rewards.

### 29. Why is that fractional update still reasonable?
**Short answer:**  
Because it preserves the learning behavior we care about.

**If they push:**  
It keeps two important properties:

1. good outcomes move the mean up, bad outcomes move it down  
2. total evidence still grows by roughly one unit per observation

### 30. Can you show a concrete fractional update example?
**Formula:**  

```text
current: alpha = 6, beta = 4
mean    = 6 / 10 = 0.60

reward  = 0.8
update:
alpha = 6.8
beta  = 4.2

new mean = 6.8 / 11 ≈ 0.62
```

### 31. What is the Gamma-ratio construction?
**Short answer:**  
One way to generate a Beta sample is:

```text
X ~ Gamma(alpha, 1)
Y ~ Gamma(beta, 1)
return X / (X + Y)
```

**If they push:**  
You do not do this manually in the router. It is just a useful explanation for where Beta sampling comes from computationally.

## Regret / Guarantees Questions

### 32. Does Thompson Sampling have regret guarantees?
**Short answer:**  
Yes in the classical stationary Bernoulli setting. Not automatically for every engineering change in this implementation.

### 33. What is regret?
**Short answer:**  
Regret is the reward you lost while learning, compared with always picking the best arm.

**Formula:**  

```text
regret = reward_of_best_fixed_policy - reward_of_your_policy
```

### 34. What kind of regret guarantee does classical Thompson Sampling have?
**Short answer:**  
Sub-linear regret in the standard Bernoulli bandit setting.

### 35. Why can’t you claim the same guarantee here?
**Short answer:**  
Because this project intentionally changes the assumptions.

**The notebook’s reasons:**

- reward is continuous in `[0,1]`
- updates use fractional pseudo-counts
- decay handles non-stationarity
- fallback, shadowing, and circuit breakers change the policy logic

### 36. What is the safest answer if someone presses on guarantees?
**Short answer:**  
Classical Thompson Sampling is theoretically grounded, but this implementation is a production-friendly Thompson-Sampling-style router, not a claim of exact textbook regret bounds.

## Priors / Cold Start Questions

### 37. Why do expert priors help?
**Short answer:**  
They make the first few dozen decisions less wasteful.

### 38. Are priors just hand-coded bias?
**Short answer:**  
They are informed starting beliefs, not hard-coded routing rules.

**If they push:**  
The evidence can override them quickly if they are wrong.

### 39. When do priors help most?
**Short answer:**  
When they are aligned with the reward being optimized.

### 40. When can priors hurt?
**Short answer:**  
If they are wrong or mismatched to the objective.

### 41. Why did the notebook use a quality-first reward in the cold-start section?
**Short answer:**  
To isolate the prior effect clearly.

**If they push:**  
With the default reward, fast cheap models can sometimes look too good because latency gets a lot of credit. The notebook changes the reward so the stronger model is genuinely the right early choice.

## Decay / Drift Questions

### 42. Why do you need decay?
**Short answer:**  
Because provider quality is not stationary in production.

### 43. What problem does decay solve?
**Short answer:**  
It prevents the router from over-trusting old history after a model degrades.

### 44. What is the decay update?
**Short answer:**  
Every `N` queries, scale `alpha` and `beta` by `gamma`.

**Formula:**  

```text
alpha <- gamma * alpha
beta  <- gamma * beta
```

### 45. Why not just forget everything immediately?
**Short answer:**  
Because you still want the router to remember useful history. Decay gives gradual forgetting, not amnesia.

### 46. Why is non-stationarity important here?
**Short answer:**  
Because providers ship changes, latency shifts, reliability drifts, and the “best” model can change over time.

## Safety / Production Questions

### 47. What is the confidence floor fallback?
**Short answer:**  
If the selected model’s confidence is too low, route to a trusted fallback instead.

### 48. Why do you need shadow evaluation?
**Short answer:**  
To keep learning about alternatives without exposing users to unnecessary risk.

### 49. Does shadow evaluation affect learning?
**Short answer:**  
Yes. It updates beliefs, but the user does not see the shadow answer.

### 50. What is a circuit breaker doing here?
**Short answer:**  
It removes a failing model from live traffic quickly when recent failures cross a threshold.

### 51. Is safety mathematically guaranteed?
**Short answer:**  
No. Safety here is engineering-enforced, not theorem-guaranteed.

### 52. What should you monitor in production?
**Short answer:**  
At minimum:

- selection distribution
- posterior evolution
- reward quality
- fallback rate
- shadow rate
- latency and cost

## Results / Claims Questions

### 53. Are the savings numbers real?
**Short answer:**  
They are real for the simulator setup and real for a specific production deployment, but they are not universal.

### 54. Why does the notebook say 80-90% savings in simulation but the talk says 40-50% in production?
**Short answer:**  
Because the simulator is illustrative and the exact savings depend on traffic mix, model quality gaps, and validator behavior.

### 55. What does “accuracy” mean in this talk?
**Short answer:**  
Usually it means automated validity rate or task success rate under a specific validator.

### 56. Why is the “<1% accuracy drop” claim risky?
**Short answer:**  
Because that number is task-specific, not universal.

**The notebook’s caution:**

- depends on cheaper-model validity
- depends on routing mix
- depends on reward weights
- depends on fallback and confidence-floor behavior

### 57. What is the honest way to phrase the result?
**Short answer:**  
This approach can reduce cost substantially while keeping validity close to the strong-model baseline, but the exact trade-off is task-specific.

## Contextual Routing Questions

### 58. What is the current router’s biggest limitation?
**Short answer:**  
It is non-contextual.

### 59. What does non-contextual mean?
**Short answer:**  
It learns one global belief per model, not a different belief per query type.

### 60. When does contextual routing matter?
**Short answer:**  
When the best model depends strongly on the prompt type.

**Examples:**

- short vs long inputs
- code vs prose
- extraction vs open-ended reasoning

### 61. Is contextual routing already implemented?
**Short answer:**  
Not in the current production router. In the notebook it is future work and a toy simulation.

### 62. Why not start with contextual routing immediately?
**Short answer:**  
Because non-contextual routing is simpler, easier to deploy, and already useful when the main issue is drift and global model quality.

## Comparison Questions

### 63. How is this different from FrugalGPT?
**Short answer:**  
FrugalGPT is usually a deterministic cascade with offline scoring. This router is online, probabilistic, and uncertainty-aware.

### 64. How is this different from RouteLLM?
**Short answer:**  
RouteLLM is contextual and label-driven. This router is currently non-contextual and telemetry-driven.

### 65. Is speculative decoding the same thing?
**Short answer:**  
No. Speculative decoding speeds up one model’s generation. Routing chooses which model should handle the request.

## RL / Sequential Decision Questions

### 66. Is this reinforcement learning?
**Short answer:**  
It is closer to bandits than full RL.

### 67. Why is a bandit enough here?
**Short answer:**  
Because each request is mostly a one-step decision with immediate feedback.

### 68. When would you need full RL instead?
**Short answer:**  
If model choice affected later states in a multi-step or multi-turn workflow.

**Examples from the notebook:**

- multi-turn agents
- budgeted sequential routing
- multi-stage pipelines

## Code / Implementation Questions

### 69. Where does Thompson Sampling live in the repo?
**Short answer:**  
`model_routing/bayesian_router/router.py`

### 70. Where does reward design live?
**Short answer:**  
`model_routing/bayesian_router/rewards.py`

### 71. Where does the synthetic demo come from?
**Short answer:**  
`model_routing/bayesian_router/simulator.py`

### 72. Where do priors live?
**Short answer:**  
`model_routing/bayesian_router/presets.py`

### 73. Is the notebook a benchmark?
**Short answer:**  
No. It is a study and intuition-building walkthrough, not a benchmark claim about provider APIs.

### More Implementation Questions

### How would you integrate this into a real agent or API gateway?
**Short answer:**  
Put the router just before the model call and the telemetry capture just after the response.

**Simple flow:**

1. build request metadata
2. ask router for a model choice
3. call that model
4. compute telemetry-derived reward
5. update the router
6. optionally shadow-evaluate another model

### What data do you need to store per model?
**Short answer:**  
At minimum, `alpha`, `beta`, query counts, and whatever safety state you need.

**In practice:**

- current `alpha`
- current `beta`
- total query count
- recent failure window
- circuit-breaker state
- fallback / shadow counters

### Where should router state live?
**Short answer:**  
Somewhere durable and shared if you have more than one process.

**Common options:**

- in-memory for local demos
- Redis for simple shared online state
- a database table for durability and auditability

### How often should you persist updates?
**Short answer:**  
Often enough that you do not lose too much learning, but not so often that persistence dominates latency.

**Practical answer:**  
For a small system, every request is fine. At larger scale, batching or asynchronous persistence is more realistic.

### How do you handle concurrent updates safely?
**Short answer:**  
Use atomic updates or a single writer.

**If they push:**  
If many workers update the same model state at once, you need:

- transactional DB updates
- atomic Redis increments
- or an event/log-based update pipeline

Otherwise you can lose or overwrite evidence.

### How do you choose initial priors in a new deployment?
**Short answer:**  
Start from benchmarks, internal evals, or pilot traffic, then keep them weak enough that evidence can override them.

### How do you choose reward weights in practice?
**Short answer:**  
Start from the operational goal, then tune weights so the chosen reward aligns with what “good” actually means for your task.

**Examples:**

- structured extraction -> validity should dominate
- customer-facing latency-sensitive flows -> latency matters more
- retry-heavy agents -> retry penalty should matter more

### How do you validate that the reward is good?
**Short answer:**  
Check whether higher reward really correlates with better task outcomes.

**Practical checks:**

- compare reward against human review on a sample
- compare reward against downstream success metrics
- inspect failure cases where reward and real usefulness disagree

### How do you debug a router that makes bad choices?
**Short answer:**  
Break the problem into reward, beliefs, and selection behavior.

**Checklist:**

- inspect telemetry for the bad requests
- recompute reward components
- inspect `alpha` / `beta`
- inspect selection shares
- check whether fallback or shadowing is firing
- check whether priors or decay are too strong

### What is the first production metric you would look at?
**Short answer:**  
Selection distribution over time.

**Why:**  
It quickly tells you whether the router is:

- stuck on one model
- oscillating too much
- adapting after drift
- ignoring a supposedly strong model

### How do you know if decay is too aggressive?
**Short answer:**  
The router becomes too jumpy and overreacts to short-term noise.

**Signs:**

- rapid switching between models
- unstable traffic share
- confidence never really settles

### How do you know if decay is too weak?
**Short answer:**  
The router stays stuck on stale beliefs for too long.

**Signs:**

- slow recovery after provider degradation
- low sensitivity to recent failures
- traffic staying on a clearly worse model

### How much shadow traffic should you use?
**Short answer:**  
Enough to keep learning about alternatives, but not so much that you waste too much cost.

**Typical answer:**  
Small fractions like `1%` to `10%` are a reasonable starting point, then adjust based on drift risk and budget.

### When should fallback trigger?
**Short answer:**  
When the selected model’s confidence is below a floor that your team considers unsafe for user-facing traffic.

### How would you rollout this system safely?
**Short answer:**  
In phases.

**Suggested rollout:**

1. offline simulation
2. shadow-only evaluation
3. low-risk traffic slice
4. gradual percentage rollout
5. monitor and retune

### How do you explain this to non-ML engineers?
**Short answer:**  
It is a traffic controller that learns which model is good enough for each class of work, based on production signals.

### What parts of the code would you change first for contextual routing?
**Short answer:**  
The belief state and the selection/update interface.

**If they push:**  
Instead of one belief per model, you would need:

- one belief per model per context bucket
- or a contextual model such as linear Thompson Sampling

### How would you move from toy simulation to real providers?
**Short answer:**  
Replace the simulator with real API calls and real validators, but keep the same router/reward/update loop.

### What is the minimum viable real deployment?
**Short answer:**  
A small set of candidate models, one reliable validator, simple persistence for `alpha`/`beta`, and one safe fallback model.

## Hard / Skeptical Questions

### 74. Isn’t this just a heuristic?
**Short answer:**  
It is a pragmatic Bayesian-style heuristic grounded in real bandit ideas.

**If they push:**  
That is exactly the honest framing in the notebook: principled, useful, production-friendly, but not exact textbook Bayes end to end.

### 75. If the validator is weak, won’t the router learn nonsense?
**Short answer:**  
Yes. A weak validator means a weak reward signal, and the router will optimize the wrong thing.

### 76. Isn’t a fixed rule easier to explain?
**Short answer:**  
Yes, but fixed rules do not adapt well when providers drift or traffic changes.

### 77. Why not just use human feedback?
**Short answer:**  
Because in many production systems it is too slow, too expensive, or unavailable at request volume.

### 78. What if all models are bad?
**Short answer:**  
Then routing cannot save you. You need a stronger fallback, a better validator, or better prompts/models.

### 79. What if one model dominates forever?
**Short answer:**  
That can happen if it is truly better, but shadowing and decay help keep some learning pressure on alternatives.

### 80. Could the router overfit to latency?
**Short answer:**  
Yes, if you overweight latency. That is why the notebook changes weights in different sections and is explicit that reward design matters.

## Math Questions You Can Expect From Stronger Technical Audiences

### 81. What exactly is the support of the Beta distribution?
**Short answer:**  
`[0, 1]`

### 82. Why is Beta a natural belief model for probabilities?
**Short answer:**  
Because it is defined on `[0,1]` and has a simple conjugate relationship with Bernoulli observations.

### 83. What is the variance intuition for Beta?
**Short answer:**  
You do not usually need the exact formula on stage. The important intuition is: larger `alpha + beta` means smaller variance.

### 84. Do you need the exact Beta density formula in the talk?
**Short answer:**  
Probably not.

**If asked:**  

```text
Beta(p; alpha, beta) ∝ p^(alpha-1) (1-p)^(beta-1)
```

But the notebook’s advice is to explain the intuition, not the full derivation.

### 85. What is posterior sampling conceptually?
**Short answer:**  
Sample one plausible world from your current belief, act as if it were true, observe outcome, update belief.

### 86. Is this PSRL?
**Short answer:**  
Not exactly. But it belongs to the same family of posterior-sampling decision methods.

## Questions About Fit And Scope

### 87. Where does this approach fit best?
**Short answer:**  
Structured outputs, schema-constrained tasks, extraction, classification, and tool-using agents with strong telemetry.

### 88. Where does it fit poorly?
**Short answer:**  
Open-ended creative tasks with weak validators and no strong operational signals.

### 89. Can this work for subjective writing quality?
**Short answer:**  
Not very well unless you have a strong automated evaluator or a proxy that correlates with what you care about.

### 90. What is the cleanest one-line summary of the whole approach?
**Short answer:**  
This is a production-friendly, non-contextual, Thompson-Sampling-style router that learns online from telemetry instead of human labels.

## Good Closing Answers

### 91. If someone asks “what would you build next?”
**Short answer:**  
I would add contextual routing and probably make cost explicit in the reward.

### 92. If someone asks “what should I remember from this talk?”
**Short answer:**  
Telemetry can act like a reward, and once you have a reward, a bandit router can learn online without human labels.

### 93. If someone asks “what should I be careful not to overclaim?”
**Short answer:**  
Do not overclaim exact Bayes, universal regret bounds, or universal “<1% accuracy drop” numbers.

## Fast Backup Answers

Use these if you need a short, safe answer quickly:

- `Why Bayesian?`  
  Because it tracks both estimate and uncertainty.

- `Why Thompson Sampling?`  
  Because exploration is uncertainty-aware and fades automatically.

- `Why telemetry?`  
  Because labels usually do not exist in production.

- `Is this exact Bayes?`  
  Not for continuous rewards; it is Bayesian-style and pragmatic.

- `Is this RL?`  
  It is a bandit, not full RL.

- `Biggest limitation?`  
  It is non-contextual and reward quality depends on the validator.

- `Biggest production strength?`  
  It adapts online and handles drift.

- `Most honest claim?`  
  Strong cost savings are possible, but the exact trade-off is task-specific.
