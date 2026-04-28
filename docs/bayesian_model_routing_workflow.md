# Bayesian Model Routing - End-to-End Workflow Guide

This file is a compact companion to `model_routing/docs/bayesian_bandits_study_material.md`.

Its goal is simple:

- capture the most important ideas in one place
- explain the full workflow end to end
- show how the math pieces and production pieces work together

---

## 1. The Full Proposal in One Sentence

Use label-free production signals to build a Bayesian router that learns online which LLM is the best tradeoff, then add practical safeguards so the system stays useful in production.

Short version:

`label-free reward + Bayesian learning + Thompson Sampling + production safety`

---

## 2. What Problem This Solves

Suppose you have multiple LLMs:

- one is strong but expensive
- one is cheaper but less reliable
- one is fast and promising but uncertain

If you always use the strongest model:

- quality stays high
- cost stays high
- latency may stay higher than necessary

If you always use the cheapest model:

- cost goes down
- but failures, retries, or invalid outputs may rise

So the problem is:

> How do we keep choosing the best model over time when model quality, latency, and reliability are uncertain and can change?

That is what the router is solving.

---

## 3. The Big Idea

The router keeps a running belief about each model.

On each request it:

1. looks at current beliefs
2. chooses a model using Thompson Sampling
3. observes what happened
4. turns the outcome into a reward
5. updates the chosen model's belief
6. repeats

Over time, traffic shifts toward models that are:

- good enough
- faster
- cheaper by effect
- reliable under current production conditions

---

## 4. Core Concepts and Their Roles

| Concept | What it means here | Why it matters |
| --- | --- | --- |
| Multi-armed bandit | Repeated choice among multiple models | Routing is a repeated decision problem |
| Arm | One candidate LLM | Each model is one selectable option |
| Bayesian | Keep a belief distribution, not just an average | Lets the router reason about uncertainty |
| Beta distribution | `Beta(alpha, beta)` belief for one model | Stores success-like and failure-like evidence |
| Bayes theorem | Update belief after new evidence arrives | Makes the router learn from outcomes |
| Thompson Sampling | Sample from current beliefs, then choose | Turns beliefs into actions |
| Composite reward | Score built from validity, latency, and retries | Converts telemetry into learning signal |
| Expert priors | Smarter starting beliefs | Improves cold start |
| Decay | Slowly weaken old evidence | Helps adapt to model rot |
| Fallback / circuit breaker | Safety layer when confidence is too low | Prevents unsafe routing |
| Shadow evaluation / exploration | Keep testing alternatives | Prevents premature lock-in |
| Non-contextual routing | Learns overall model health, not per-prompt merit | Important limitation and framing |

---

## 5. How the Math Pieces Work Together

This is the cleanest way to understand the relationship:

- **Beta distribution** = where the belief lives
- **Bayes theorem** = how that belief gets updated
- **Thompson Sampling** = how the router uses that belief to choose a model

In plain English:

- the router stores a belief for each model as `Beta(alpha, beta)`
- after seeing a result, Bayes-style updating changes `alpha` and `beta`
- then Thompson Sampling draws one temporary score from each updated belief
- the model with the highest sampled score gets the next request

So:

`Beta distribution -> belief container`

`Bayes theorem -> belief update rule`

`Thompson Sampling -> action rule`

Another simple way to say it:

- Bayes theorem **writes** the scorecard
- Thompson Sampling **reads** the scorecard

---

## 6. End-to-End ASCII Diagram

```text
                     BAYESIAN MODEL ROUTING: END-TO-END FLOW

                  full proposal = reward + learning + safety

Incoming request
    |
    v
+---------------------------------------------------------------+
| Current router state                                          |
| - one belief per model                                        |
| - belief stored as Beta(alpha, beta)                          |
| - expert priors provide a head start                          |
+---------------------------------------------------------------+
    |
    | Beta distribution = where the belief lives
    v
+---------------------------------------------------------------+
| Thompson Sampling                                             |
| 1. sample one temporary score from each model's Beta belief   |
| 2. choose the model with the highest sampled score            |
+---------------------------------------------------------------+
    |
    | action rule: turn beliefs into one concrete decision
    v
+---------------------------------------------------------------+
| Chosen model handles the request                              |
+---------------------------------------------------------------+
    |
    v
+---------------------------------------------------------------+
| Observe production signals                                    |
| - valid output?                                               |
| - latency?                                                    |
| - retried?                                                    |
+---------------------------------------------------------------+
    |
    v
+---------------------------------------------------------------+
| Composite reward                                              |
| r = validity + latency + retry                               |
| default weights: 0.50 + 0.30 + 0.20                          |
+---------------------------------------------------------------+
    |
    | evidence for learning
    v
+---------------------------------------------------------------+
| Bayes-style update                                            |
| textbook binary version:                                      |
|   success -> alpha += 1                                       |
|   failure -> beta  += 1                                       |
| production version here:                                      |
|   alpha += reward                                             |
|   beta  += 1 - reward                                         |
+---------------------------------------------------------------+
    |
    | learning rule: revise belief after evidence
    v
+---------------------------------------------------------------+
| Updated router state                                          |
| - strong models become more trusted                           |
| - weak models become less trusted                             |
| - uncertainty changes over time                               |
+---------------------------------------------------------------+
    |
    +--------------------+--------------------+------------------+
    |                    |                    |
    v                    v                    v
+-----------+      +-------------+      +----------------------+
| Decay     |      | Fallback /  |      | Exploration /        |
| model rot |      | breaker     |      | shadow evaluation    |
| handling  |      | safety      |      | keep learning        |
+-----------+      +-------------+      +----------------------+
    |                    |                    |
    +--------------------+--------------------+
                         |
                         v
              Next request uses the updated beliefs
```

---

## 7. Step-by-Step Workflow

### Step 1: Start with beliefs

Each model starts with a Beta belief such as:

- `gpt-4o = Beta(8, 3)`
- `gpt-4o-mini = Beta(5, 4)`
- `claude-haiku = Beta(5, 4)`

Simple interpretation:

- higher `alpha` = more positive evidence
- higher `beta` = more negative evidence
- bigger `alpha + beta` = more confidence / less uncertainty

The mean belief score is:

`alpha / (alpha + beta)`

Example:

- `Beta(8, 3)` has mean `8 / 11 ~= 0.727`

That is the router's current estimate of model quality.

### Step 2: Use Thompson Sampling to choose a model

For each model:

1. draw one temporary score from its current Beta belief
2. compare all the sampled scores
3. pick the highest

Example:

- `gpt-4o`: draw `0.78`
- `gpt-4o-mini`: draw `0.74`
- `claude-haiku`: draw `0.86`

So the router chooses `claude-haiku`.

Why this is powerful:

- strong models win often
- uncertain models still get chances
- weak, well-understood models fade over time

### Step 3: Execute the request

The chosen model handles the real production request.

Example request:

- "Extract invoice fields and return valid JSON."

### Step 4: Observe production telemetry

After the response, the system measures:

- whether the output was valid
- how long it took
- whether retries were needed

This is important because the router does **not** wait for human labels.

It learns from operational signals.

### Step 5: Compute the composite reward

The reward is:

`r = r_validity + r_latency + r_retry`

Default weights:

- validity = `0.50`
- latency = `0.30`
- no-retry = `0.20`

Interpretation:

- valid output helps a lot
- fast output helps
- not needing retries helps

Example strong response:

- valid = `True`
- retried = `False`
- latency = `800 ms`

Then reward is roughly:

- validity = `0.50`
- latency = `0.264`
- retry = `0.20`
- total = `0.964`

### Step 6: Update the chosen model's belief

This is where Bayes-style learning happens.

Textbook binary version:

- success -> `alpha += 1`
- failure -> `beta += 1`

Production-oriented version in this project:

- `alpha += reward`
- `beta += 1 - reward`

So if a model gets reward `0.96`:

- `alpha` goes up a lot
- `beta` goes up only a little

That means the router trusts it more next time.

### Step 7: Apply production adaptations

The learning loop is the core, but production needs more than just the core loop.

#### Expert priors

These help during cold start.

Meaning:

- we do not begin from total ignorance
- we start with a sensible belief that stronger models are probably better
- but not so strongly that the router cannot change its mind

#### Decaying memory

This handles model rot.

Meaning:

- old evidence should not dominate forever
- recent evidence should matter more when providers change

So every `decay_interval` queries:

- `alpha <- max(1, gamma * alpha)`
- `beta <- max(1, gamma * beta)`

#### Fallback / circuit breaker logic

This is the safety layer.

Meaning:

- if a cheaper model looks too uncertain or unhealthy
- route to the safer trusted model instead

In the study guide framing, this starts with confidence-based fallback and can be extended to a fuller circuit breaker policy.

#### Shadow evaluation / exploration

This is the learning safety layer.

Meaning:

- keep gathering evidence about alternatives
- do not become overconfident too early
- safely test models that may improve or recover

### Step 8: Repeat

The next request uses the updated beliefs.

That is how the router:

- adapts online
- improves with traffic
- shifts traffic when the world changes

---

## 8. Textbook Core vs Production Version

This distinction is very important.

### Textbook core

- binary reward: success or failure
- exact Beta-Bernoulli update
- Thompson Sampling with clean theoretical assumptions
- stationary environment

### Production version in this project

- continuous reward from telemetry
- fractional pseudo-count updates
- decay for non-stationarity
- fallback / safety logic
- exploration / shadow-style testing
- non-contextual traffic-level routing

So the best description is:

> a production-friendly Thompson-Sampling-style router inspired by textbook Bayesian bandits

It is useful and principled, but not exact textbook Bayes end to end.

---

## 9. Where Cost Enters the Picture

This is another important nuance.

In the current reward, cost is **not** explicitly in the formula.

The reward directly uses:

- validity
- latency
- retries

So why does cost go down?

Because in the simulation and many practical settings:

- cheaper models are often faster
- and are often good enough for easier structured tasks

So the router favors them indirectly through better latency and adequate validity.

Best wording:

> The current design is cost-sensitive by effect, not fully cost-penalized in the objective.

Future extension:

`reward = validity + latency + retry - lambda * cost`

---

## 10. What This Router Is and Is Not

### What it is

- an online adaptive router
- label-free
- uncertainty-aware
- useful for repeated production decisions
- good at learning overall model health under current conditions

### What it is not

- not currently a fully contextual router
- not reading prompt meaning before routing
- not exact textbook Bayesian inference end to end
- not a perfect ground-truth quality system

Important nuance:

- this router is **non-contextual**
- it learns which models are generally best right now
- it does **not** yet learn "use model A for math but model B for email"

That would be a future contextual-bandit extension.

---

## 11. Best Fit

This approach works best when you have strong automatic feedback.

Best fit:

- structured outputs
- JSON / schema tasks
- extraction and classification
- tool-using agents
- systems with retries and clear operational telemetry

Harder fit:

- subjective creative writing
- open-ended tasks with weak validators
- domains where parse success is poorly correlated with correctness

---

## 12. Important Limitations

Be honest about these.

1. The policy is non-contextual.
2. The composite reward is a proxy, not ground-truth correctness.
3. Fractional Beta updates are heuristic, not exact conjugate Bayes.
4. Cost is indirect in the current objective.
5. Safety can be simplified depending on the implementation version.
6. Good validators are essential.

These are not weaknesses to hide.

They are part of a mature explanation.

---

## 13. Practical Mapping to Code

Relevant files:

- `model_routing/bayesian_router/router.py`
- `model_routing/bayesian_router/rewards.py`
- `model_routing/bayesian_router/simulator.py`
- `model_routing/examples/04_streamlit_demo.py`

High-level mapping:

- selection logic lives in `router.py`
- reward computation lives in `rewards.py`
- simulated telemetry comes from `simulator.py`
- end-to-end visualization is in the Streamlit demo

---

## 14. 30-Second Explanation

If someone asks casually, you can say:

> I treat each LLM as an arm in a multi-armed bandit. The router keeps a Bayesian belief about each model's quality, samples from those beliefs with Thompson Sampling, picks one model, observes operational signals like schema validity, latency, and retries, and updates the chosen model. That lets it shift traffic online toward cheaper models that are good enough, while using priors, decay, and safety layers to stay practical in production.

---

## 15. Final Takeaways

If you remember only a few things, remember these:

1. The router is solving repeated decision-making under uncertainty.
2. The Beta distribution stores the belief about each model.
3. Bayes theorem updates that belief after new evidence.
4. Thompson Sampling uses the updated belief to choose the next model.
5. Composite reward turns production telemetry into a learning signal.
6. Expert priors, decay, fallback, and exploration make the approach production-relevant.
7. The current design is best framed as a pragmatic, non-contextual, production-friendly adaptive router.
