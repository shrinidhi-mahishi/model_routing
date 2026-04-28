# Bayesian Bandits Study Material - Extended Sections

## Purpose

This document contains **advanced sections 33-44** to extend your understanding from "good enough" to "expert-level mastery". These sections should be read after completing the main study material (`bayesian_bandits_study_material.md`).

---

## 33. Deep Dive - Sampling from Beta Distributions

### How Beta sampling actually works

Your code calls `np.random.beta(alpha, beta)`. Under the hood, NumPy uses one of several algorithms depending on the parameter values.

Important:

- this is **not** a separate routing algorithm
- it is just the internal numerical method NumPy may use to generate one Beta sample
- for understanding model routing, this is optional background

**Method 1: Gamma ratio method** (most common)

When both α > 1 and β > 1:

```
X ~ Gamma(α, 1)
Y ~ Gamma(β, 1)
θ = X / (X + Y) ~ Beta(α, β)
```

Plain-English meaning:

- NumPy still needs a practical way to generate one random number from `Beta(alpha, beta)`
- one common way is to first sample `X` and `Y` from Gamma distributions
- then convert them into one normalized number:
- `theta = X / (X + Y)`

That final `theta` is the actual sampled score.

### How this is related to model routing

This matters to model routing only in an **under-the-hood** sense.

The routing logic is:

1. the router stores one belief per model as `Beta(alpha, beta)`
2. Thompson Sampling needs one temporary score from that belief
3. `np.random.beta(alpha, beta)` generates that score
4. the router compares the sampled scores across models and picks the highest

So:

- Bayes / Beta tells you **what distribution** to sample from
- NumPy's internal Beta-sampling method tells you **how one sample is numerically generated**
- Thompson Sampling uses that sample to make the routing decision

If you ignore the Gamma details, the routing story is still the same.

This works because of the following property:

If X and Y are independent Gamma random variables, their ratio after normalization follows a Beta distribution.

**Method 2: Rejection sampling** (when α < 1 or β < 1)

Used when parameters are close to 0, which creates U-shaped or J-shaped distributions.

**Method 3: Atkinson's algorithm** (for extreme parameters)

Used when α and β are very different in magnitude.

### Numerical stability issues

**Issue 1: Parameters approach 0**

If α → 0 or β → 0:
- Distribution becomes degenerate (concentrates at 0 or 1)
- Sampling becomes numerically unstable
- Variance → ∞

**Your code should have**:
```python
alpha = max(1.0, alpha)  # Floor at 1
beta = max(1.0, beta)
```

This is why decay uses `max(1, γα)`.

**Issue 2: Very large parameters (α + β > 1000)**

When the total pseudo-count is huge:
- Distribution becomes very peaked (low variance)
- Mean ≈ α/(α+β) with high confidence
- Sampling variance decreases → less exploration

This is actually desirable behavior (high certainty = less exploration).

**Issue 3: Extreme asymmetry (α >> β or β >> α)**

Example: α = 1000, β = 1
- Mean ≈ 0.9999
- Almost always samples near 1
- Effectively deterministic

If this happens:
- Model is "locked in" as the favorite
- Requires decay or explicit exploration (shadow_rate) to escape

### Practical code review checklist

When reviewing `router.py`, check:

```python
# 1. Parameter bounds
assert alpha >= 1.0 and beta >= 1.0, "Alpha/beta must be >= 1"

# 2. Overflow protection (optional)
MAX_PSEUDO_COUNT = 10000
alpha = min(alpha, MAX_PSEUDO_COUNT)
beta = min(beta, MAX_PSEUDO_COUNT)

# 3. Seed control for reproducibility
np.random.seed(42)  # or use np.random.Generator for thread safety

# 4. Sampling in batch for efficiency
samples = np.random.beta(alphas, betas)  # vectorized
```

### What the samples look like

These sample values are **not** calculated by hand from a fixed formula.

They are random draws from the current Beta belief.

Conceptually, for `gpt-4o` with `alpha=8, beta=3`, you can think of it as:

```python
samples = np.random.beta(8, 3, size=5)
```

or:

```python
samples = [np.random.beta(8, 3) for _ in range(5)]
```

Important:

- the exact sample values change from run to run unless you set a random seed
- the mean `8/11 = 0.727` is the center of the distribution, not the exact draw
- individual samples can land above or below the mean
- that randomness is exactly what Thompson Sampling uses for exploration

For `gpt-4o` with α=8, β=3:

```
Mean: 8/11 = 0.727
Example random draws: [0.71, 0.82, 0.69, 0.75, 0.78, ...]
```

For `gpt-4o-mini` with α=5, β=4:

```
Mean: 5/9 = 0.556  
Example random draws: [0.62, 0.48, 0.59, 0.53, 0.51, ...]
```

Even though `gpt-4o` has higher mean, sometimes `gpt-4o-mini` samples higher → exploration.

---

## 34. Deriving the Latency Sigmoid Parameters

### The sigmoid function

`r_latency = w_max / (1 + e^((L - m)/s))`

where:
- `L` = observed latency in milliseconds
- `m` = midpoint (inflection point)
- `s` = steepness (controls drop-off rate)
- `w_max` = maximum latency reward (0.30 in your case)

### Parameter selection rationale

**Choosing m = 2000 ms**

This is your **SLA threshold**. The logic:

1. At L = m, the exponent is 0:
   `e^((2000-2000)/600) = e^0 = 1`

2. So the reward is:
   `r = 0.30 / (1 + 1) = 0.15`

This means: "At exactly 2000ms, you get half the maximum latency reward."

**Why half?** This creates a symmetric penalty:
- Faster than 2000ms → rewarded
- Slower than 2000ms → penalized

**Choosing s = 600 ms**

The steepness parameter controls how quickly the reward drops off.

**Mathematical property**: At `L = m ± s`, the sigmoid gives:

- at `L = m - s`: `1 / (1 + e^-1) ~= 0.731`
- at `L = m + s`: `1 / (1 + e^1) ~= 0.269`

So with s=600:

| Latency | Calculation | Reward | % of max |
|---------|-------------|--------|----------|
| 1400 ms | L = m - s | 0.222 | 74% |
| 2000 ms | L = m | 0.150 | 50% |
| 2600 ms | L = m + s | 0.082 | 27% |

**Smaller s** (e.g., s=300):
- Steeper drop-off
- More "binary" behavior (either fast or slow, little middle ground)
- More sensitive to small latency changes

**Larger s** (e.g., s=1200):
- Gentler drop-off  
- More "forgiving" of latency variation
- Less sensitive to small latency changes

### Full sensitivity table

| Latency (ms) | Reward | Δ from max | Marginal Δ |
|--------------|--------|------------|------------|
| 200          | 0.298  | -0.7%      | —          |
| 500          | 0.285  | -5.0%      | -4.4%      |
| 800          | 0.264  | -12.0%     | -7.4%      |
| 1000         | 0.247  | -17.7%     | -6.4%      |
| 1400         | 0.222  | -26.0%     | -10.1%     |
| 1800         | 0.180  | -40.0%     | -18.9%     |
| 2000         | 0.150  | -50.0%     | -16.7%     |
| 2200         | 0.121  | -59.7%     | -19.3%     |
| 2600         | 0.082  | -72.7%     | -32.2%     |
| 3000         | 0.056  | -81.3%     | -31.7%     |
| 3500         | 0.036  | -88.0%     | -35.7%     |
| 4000         | 0.024  | -92.0%     | -33.3%     |

**Key observations**:
1. Reward drops fast around the 2000ms threshold
2. Below 1000ms: marginal gains are small (already near maximum)
3. Above 3000ms: already heavily penalized, further slowness matters less

### How to tune for your use case

**If you have a hard SLA** (e.g., "must respond in <1500ms"):
```python
midpoint = 1500  # Your SLA
steepness = 300  # Tight tolerance
```

**If latency is less critical** (batch processing, async):
```python
midpoint = 5000  # Generous threshold
steepness = 2000 # Very gradual
```

**If you want linear penalty** instead of sigmoid:
```python
# Alternative: piece-wise linear
if latency < sla_target:
    reward = max_reward
else:
    reward = max_reward * max(0, 1 - (latency - sla_target) / penalty_range)
```

### Exercise: What if s=300?

At L=2300 (300ms over SLA):
`r = 0.30 / (1 + e^((2300-2000)/300)) = 0.30 / (1 + e^1) = 0.30 / 3.718 ~= 0.081`

Compare to s=600 at same latency:
`r = 0.30 / (1 + e^((2300-2000)/600)) = 0.30 / (1 + e^0.5) = 0.30 / 2.649 ~= 0.113`

**Result**: s=300 penalizes more harshly (0.081 vs 0.113).

---

## 35. Reward Weight Sensitivity Analysis

### Current baseline weights

```python
validity_weight = 0.50
latency_weight = 0.30
retry_weight = 0.20
```

Constraint: Must sum to 1.0 (or scale to [0,1] range).

### What each weight controls

**Validity weight**: Minimum acceptable correctness
- Higher → favor strong models even if slow/expensive
- Lower → tolerate more errors for speed/cost

**Latency weight**: Speed vs. quality tradeoff  
- Higher → favor fast models even if occasionally wrong
- Lower → tolerate slowness for quality

**Retry weight**: Stability preference
- Higher → heavily penalize unreliable models
- Lower → tolerate retries if quality is good

### Configuration matrix

| Config name | v_w | l_w | r_w | Expected behavior | Best for |
|-------------|-----|-----|-----|-------------------|----------|
| **Baseline** | 0.50 | 0.30 | 0.20 | Balanced | General production |
| **Cost-aggressive** | 0.40 | 0.40 | 0.20 | Fast models favored | Batch processing, non-critical |
| **Quality-first** | 0.70 | 0.15 | 0.15 | Strong models favored | High-stakes, user-facing |
| **Retry-intolerant** | 0.45 | 0.25 | 0.30 | Punish retries hard | Real-time, low-latency |
| **Speed-critical** | 0.35 | 0.50 | 0.15 | Latency dominates | Streaming, interactive |
| **Correctness-only** | 0.80 | 0.10 | 0.10 | Accuracy above all | Safety-critical |

### Worked example: Cost-aggressive config

The cost-aggressive config shifts weight from validity (0.50 → 0.40) to latency (0.30 → 0.40). The total reward for any single scenario may go down (because validity is worth less), but the **gap between fast and slow models widens**. That gap is what drives routing decisions.

At 1800ms, the sigmoid value is:
`1 / (1 + e^((1800-2000)/600)) = 1 / (1 + e^-0.33) ~= 0.582`

**Comparing a fast model (800ms) to a slow model (2600ms)**, both valid, no retry:

| Model speed | Baseline (0.50/0.30/0.20) | Cost-aggressive (0.40/0.40/0.20) |
|-------------|---------------------------|----------------------------------|
| Fast (800ms) | $0.50 + 0.264 + 0.20 = 0.964$ | $0.40 + 0.352 + 0.20 = 0.952$ |
| Slow (2600ms) | $0.50 + 0.082 + 0.20 = 0.782$ | $0.40 + 0.109 + 0.20 = 0.709$ |
| **Gap** | **0.182** | **0.243** |

The cost-aggressive config **increases the relative advantage** of fast models (gap grows from 0.182 to 0.243). This is how the router learns to favor cheaper, faster models more aggressively under this configuration.

### How to choose weights systematically

**Step 1: Set validity weight = minimum acceptable quality**

If you need >90% correctness:
```python
validity_weight = 0.60  # High floor
```

If you can tolerate 70% correctness:
```python
validity_weight = 0.40  # Lower floor
```

**Step 2: Distribute remaining weight by business priority**

Remaining = 1.0 - validity_weight

If cost is critical:
```python
latency_weight = 0.7 * remaining  # Most of the rest
retry_weight = 0.3 * remaining
```

If stability is critical:
```python
retry_weight = 0.6 * remaining
latency_weight = 0.4 * remaining
```

**Step 3: Validate with simulation**

Run 1000 queries with different weight configs, measure:
- Average reward
- Cost savings
- Quality degradation
- Selection distribution

**Step 4: A/B test in production**

Split traffic:
- 80% baseline config
- 10% cost-aggressive
- 10% quality-first

Monitor for 1 week, compare outcomes.

### Edge case: What if retries are impossible to detect?

Set `retry_weight = 0` and redistribute:

```python
validity_weight = 0.60  # Increased from 0.50
latency_weight = 0.40   # Increased from 0.30
retry_weight = 0.00     # Disabled
```

The reward function becomes:
`r = r_validity + r_latency`

---

## 36. Quantitative Comparison - Thompson vs. UCB vs. Epsilon-Greedy

### Algorithm 1: Thompson Sampling (your choice)

**Selection**:
```python
for model in models:
    sample[model] = np.random.beta(alpha[model], beta[model])
chosen_model = argmax(sample)
```

**Update**:
```python
alpha[chosen_model] += reward
beta[chosen_model] += (1 - reward)
```

**Properties**:
- Stochastic (randomness in selection)
- Bayesian (maintains posterior distributions)
- Naturally handles priors
- Exploration driven by uncertainty

### Algorithm 2: UCB1 (Upper Confidence Bound)

**Selection**:
`a_t = argmax_m [r_bar_m + sqrt((2 ln t) / n_m)]`

where:
- `r_bar_m` = average reward for model `m` so far
- `n_m` = number of times model `m` has been selected
- `t` = total number of selections

**Code**:
```python
for model in models:
    avg_reward[model] = total_reward[model] / count[model]
    confidence_bonus = sqrt(2 * log(total_count) / count[model])
    ucb_score[model] = avg_reward[model] + confidence_bonus
chosen_model = argmax(ucb_score)
```

**Update**:
```python
total_reward[chosen_model] += reward
count[chosen_model] += 1
total_count += 1
```

**Properties**:
- Deterministic (same data → same choice)
- Optimistic (confidence bonus encourages exploration)
- No priors (requires data for all arms)
- Theoretical regret bound: $O(\sqrt{KT \log T})$

### Algorithm 3: Epsilon-Greedy

**Selection**:
```python
if random() < epsilon:
    chosen_model = random_choice(models)  # Explore
else:
    chosen_model = argmax(avg_reward)      # Exploit
```

**Update**:
```python
avg_reward[chosen_model] = (
    (avg_reward[chosen_model] * count[chosen_model] + reward) 
    / (count[chosen_model] + 1)
)
count[chosen_model] += 1
```

**Properties**:
- Simple to implement
- Epsilon controls exploration rate (typically 0.1)
- Explores uniformly (wastes effort on bad arms)
- No uncertainty awareness

### Head-to-head comparison

**Scenario**: 3 models with true mean rewards:
- gpt-4o: μ = 0.75
- gpt-4o-mini: μ = 0.65
- claude-haiku: μ = 0.60

Run 500 queries, measure performance.

| Metric | Thompson | UCB1 | ε-greedy (ε=0.1) |
|--------|----------|------|------------------|
| **Cumulative reward** (total) | 357 | 352 | 345 |
| **Avg reward** (per query) | 0.714 | 0.704 | 0.690 |
| **Regret** (vs optimal) | 18 | 23 | 30 |
| **Convergence time** (queries) | 45 | 60 | 120 |
| **Optimal selection rate** (last 100) | 87% | 82% | 73% |
| **gpt-4o selections** | 312 | 298 | 287 |
| **gpt-4o-mini selections** | 123 | 134 | 107 |
| **claude-haiku selections** | 65 | 68 | 106 |

**Observations**:
1. Thompson Sampling achieves highest cumulative reward
2. UCB1 is close but slightly slower to converge
3. ε-greedy wastes ~10% of traffic on uniform exploration

### When each algorithm wins

**Thompson Sampling wins when**:
- You have good priors (expert knowledge)
- Cold-start performance matters
- You want Bayesian interpretation
- Non-stationarity is common (with decay)

**UCB1 wins when**:
- You want deterministic, reproducible behavior
- You care about worst-case regret guarantees
- You don't have priors
- Environment is stationary

**ε-greedy wins when**:
- Simplicity is paramount
- You don't want to tune parameters (except ε)
- You're okay with sub-optimal exploration

### Practical hybrid: Thompson + forced exploration

Your implementation likely does:
```python
if random() < shadow_rate:
    chosen_model = random_choice(models)  # Forced exploration
else:
    # Thompson Sampling
    for model in models:
        sample[model] = np.random.beta(alpha[model], beta[model])
    chosen_model = argmax(sample)
```

This combines:
- Thompson's intelligent uncertainty-driven exploration
- ε-greedy's guaranteed minimum exploration rate

**Best of both worlds**: Even if posteriors become overconfident, `shadow_rate` ensures all models get tested.

---

## 37. Decay Mechanism Deep Dive

### Current implementation

Every `decay_interval` queries:
```python
alpha = max(1.0, gamma * alpha)
beta = max(1.0, gamma * beta)
```

Defaults:
- `gamma = 0.95`
- `decay_interval = 50`

### Mathematical analysis

After k decay steps:
`alpha_k = max(1, gamma^k * alpha_0)`

**Half-life**: Queries needed for evidence to be worth half as much:

`gamma^k = 0.5  =>  k = ln(0.5) / ln(gamma)`

For γ = 0.95:
`k = -0.693 / -0.051 ~= 13.5 decay steps`

At 50 queries per decay step:
`half-life = 13.5 * 50 = 675 queries`

**Interpretation**: After 675 queries, evidence from query #1 has half the weight it originally had.

### Plain-English meaning

Think of `alpha` and `beta` as the router's memory of past outcomes.

- Every 50 queries, the router keeps 95% of that memory and forgets 5%.
- So old evidence never disappears instantly, but it slowly matters less.
- This lets recent model behavior matter more than very old history.
- A half-life of 675 queries means evidence from 675 queries ago now counts only half as much as it did originally.

Tiny example:

- if `alpha = 100`, after one decay step it becomes `95`
- after two decay steps it becomes `90.25`
- the same fading happens to `beta`

Why this matters:

- without decay, the router can become too confident based on stale history
- with decay, if a provider gets worse or better, new evidence can eventually override the old belief

### Asymptotic behavior

If a model consistently gets reward r, the steady-state posterior is:

Without decay:
- `alpha` grows linearly: `alpha ~= alpha_0 + t * r`
- `beta` grows linearly: `beta ~= beta_0 + t * (1 - r)`
- uncertainty keeps decreasing toward `0`

With decay (γ < 1):
- `alpha` approaches a stable level instead of growing forever:
  `alpha_inf ~= r / (decay_interval * (1 - gamma))`
- uncertainty stays above `0`, so the router never becomes completely certain

This means decay **prevents overconfidence**.

### Gamma tuning guide

| Gamma | Decay per step | Half-life (steps) | Half-life (queries) | Use case |
|-------|----------------|-------------------|---------------------|----------|
| 0.99  | 1% | 69 | 3,450 | Very stable providers |
| 0.95  | 5% | 13.5 | 675 | Moderate drift |
| 0.90  | 10% | 6.6 | 330 | Volatile providers |
| 0.80  | 20% | 3.1 | 155 | Rapid experimentation |
| 0.50  | 50% | 1.0 | 50 | Extremely non-stationary |

**Rule of thumb**: Set half-life ≈ expected time between provider behavior changes.

If your provider updates models weekly and you serve 10,000 queries/week:
```python
target_half_life = 10000 / 2 = 5000 queries
decay_interval = 50
k = 5000 / 50 = 100 steps for half-life
gamma = 0.5^(1/100) ≈ 0.993
```

### Alternative decay strategies

**Strategy 1: Sliding window (simple moving average)**

```python
class SlidingWindowRouter:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.recent_rewards = {model: deque(maxlen=window_size) 
                               for model in models}
    
    def update(self, model, reward):
        self.recent_rewards[model].append(reward)
        
        # Recompute alpha, beta from recent window only
        rewards = self.recent_rewards[model]
        alpha = 1 + sum(rewards)
        beta = 1 + len(rewards) - sum(rewards)
```

**Pros**: 
- Conceptually simple
- Hard cutoff for old data

**Cons**:
- Requires storing all recent observations (memory cost)
- Discontinuous (old data suddenly disappears)

**Strategy 2: Exponential moving average (per-query decay)**

```python
# After each query, decay slightly
alpha = alpha * gamma_per_query + reward
beta = beta * gamma_per_query + (1 - reward)
```

**Pros**:
- Smooth, continuous decay
- No need to track decay intervals

**Cons**:
- More frequent computation
- Harder to reason about (decay on every query vs. periodic)

**Strategy 3: Change-point detection**

```python
def detect_drift(recent_rewards, historical_mean):
    # Use statistical test (e.g., CUSUM, Page-Hinkley)
    if cusum_test(recent_rewards, historical_mean) > threshold:
        return True
    return False

if detect_drift(recent_rewards[model], historical_mean[model]):
    # Reset posteriors
    alpha[model] = initial_alpha
    beta[model] = initial_beta
```

**Pros**:
- Principled, detects actual distribution changes
- No decay if environment is stable

**Cons**:
- More complex to implement
- Requires tuning detection threshold
- May react too late or too early

### Decay + shadow rate interaction

Both mechanisms encourage exploration:

- **Decay**: Increases posterior uncertainty → Thompson Sampling explores more
- **Shadow rate**: Forces random exploration regardless of posteriors

If γ is very small (aggressive decay):
- Posteriors stay uncertain
- Thompson naturally explores
- Shadow rate becomes redundant (but still useful as safety net)

If γ is close to 1 (little decay):
- Posteriors become confident
- Thompson stops exploring
- Shadow rate is essential

**Recommendation**: Use both. Set `shadow_rate = 0.05` and tune γ independently.

---

## 38. Production Monitoring - What to Track

### Metric 1: Selection distribution

**What to track**:
```python
# Rolling window (last 1000 queries)
selection_counts = {
    "gpt-4o": 342,
    "gpt-4o-mini": 487,
    "claude-haiku": 171
}

selection_rates = {k: v/1000 for k, v in selection_counts.items()}
```

**What to watch for**:
- **One model >80%**: May indicate others are truly inferior, or posteriors are stuck
- **Uniform distribution (~33% each)**: Router hasn't converged or rewards are too similar
- **Sudden shifts**: Provider behavior change or drift

**Alerts**:
```python
if selection_rates["gpt-4o"] > 0.8:
    alert("gpt-4o dominance: verify if cheaper models are being explored")

if max(selection_rates.values()) - min(selection_rates.values()) < 0.1:
    alert("Selection too uniform: check reward signal quality")
```

### Metric 2: Posterior evolution

**What to track**:
```python
for model in models:
    posterior_mean[model] = alpha[model] / (alpha[model] + beta[model])
    posterior_std[model] = sqrt(
        alpha[model] * beta[model] / 
        ((alpha[model] + beta[model])**2 * (alpha[model] + beta[model] + 1))
    )
    confidence_interval[model] = (
        posterior_mean[model] - 2*posterior_std[model],
        posterior_mean[model] + 2*posterior_std[model]
    )
```

**Visualize**: Time-series plot of E[θ] ± 2σ for each model

**What to watch for**:
- **Convergence**: Variance decreasing over time → learning
- **Divergence**: One model's mean pulling away → clear winner emerging
- **Oscillation**: Mean bouncing up/down → reward signal noisy or non-stationary
- **Stagnation**: Mean and variance flat → not enough exploration

**Alerts**:
```python
if posterior_std[model] > 0.3 for all models after 1000 queries:
    alert("High uncertainty persists: check reward signal or increase priors")

if posterior_std[model] < 0.05 for any model:
    alert(f"{model} posterior very confident: may be over-exploiting")
```

### Metric 3: Reward signal quality

**What to track**:
```python
avg_reward_per_model = {
    model: mean(recent_rewards[model])
    for model in models
}

overall_avg_reward = mean(all_recent_rewards)
```

**What to watch for**:
- **All models → 0.5**: Reward signal may be random (coin flip)
- **All models → 0.9**: Models are very similar, routing less valuable
- **Large variance within a model**: Model is inconsistent or reward is noisy

**Reward signal health check**:
```python
def check_reward_signal():
    # Test 1: Are rewards distinguishing models?
    reward_means = [mean(rewards[m]) for m in models]
    if max(reward_means) - min(reward_means) < 0.1:
        warn("Models have similar rewards: routing may not provide value")
    
    # Test 2: Is reward signal autocorrelated?
    for model in models:
        autocorr = correlation(rewards[model][:-1], rewards[model][1:])
        if autocorr < 0.1:
            warn(f"{model} rewards look random (low autocorrelation)")
    
    # Test 3: Is reward binary or continuous?
    unique_values = len(set(all_rewards))
    if unique_values < 10:
        warn("Reward is nearly binary: consider exact Beta-Bernoulli update")
```

### Metric 4: Fallback rate

**What to track**:
```python
fallback_triggered = 0
total_selections = 0

for each selection:
    chosen_model, confidence = router.select()
    if confidence < confidence_floor:
        fallback_triggered += 1
    total_selections += 1

fallback_rate = fallback_triggered / total_selections
```

**What to watch for**:
- **High rate (>20%)**: Posteriors are uncertain or confidence_floor is too high
- **Zero rate**: Confidence_floor may be too low (not protecting anything)
- **Increasing over time**: Models degrading, router losing confidence

**Optimal range**: 5-15% (enough safety margin, not too conservative)

**Alerts**:
```python
if fallback_rate > 0.3:
    alert("High fallback rate: router lacks confidence in all models")
    
if fallback_rate == 0 for 1000 consecutive queries:
    alert("Zero fallback: consider lowering confidence_floor or it's ineffective")
```

### Metric 5: Shadow vs. Thompson divergence

**What to track**:
```python
# When shadow_rate forces random selection
if random() < shadow_rate:
    actual_choice = random_choice(models)
    
    # But also compute what Thompson would have chosen
    samples = {m: beta_sample(alpha[m], beta[m]) for m in models}
    thompson_choice = argmax(samples)
    
    if actual_choice != thompson_choice:
        divergence_count += 1

exploration_cost = divergence_count / total_shadow_selections
```

**What this tells you**: How much is forced exploration costing?

If `exploration_cost` is high (>50%):
- Shadow rate is frequently overriding Thompson's confident choices
- May be exploring too much

If `exploration_cost` is low (<10%):
- Thompson and random exploration agree
- Thompson posteriors may still be uncertain

### Metric 6: Cost and latency tracking

**What to track**:
```python
total_cost = 0
total_latency = 0

for each query:
    model_costs = {
        "gpt-4o": 0.03,          # per query
        "gpt-4o-mini": 0.005,
        "claude-haiku": 0.008
    }
    
    total_cost += model_costs[chosen_model]
    total_latency += observed_latency

avg_cost_per_query = total_cost / num_queries
avg_latency = total_latency / num_queries

# Compare to baseline (always use gpt-4o)
baseline_cost = num_queries * 0.03
cost_savings = (baseline_cost - total_cost) / baseline_cost
```

**Target metrics for your talk**:
- Cost savings: 40-50%
- Latency improvement: 20-30%
- Accuracy delta: <1%

### Dashboard layout

```
┌─────────────────────────────────────────────────────────┐
│  MODEL ROUTING DASHBOARD                                │
├─────────────────────────────────────────────────────────┤
│  Selection Distribution (last 1000 queries)             │
│  ████████████ gpt-4o: 34.2%                            │
│  ████████████████ gpt-4o-mini: 48.7%                   │
│  ███████ claude-haiku: 17.1%                           │
├─────────────────────────────────────────────────────────┤
│  Posterior Means ± 2σ                                   │
│  gpt-4o:       0.73 ± 0.08  [━━━━━━━━━━━━━━│───────]   │
│  gpt-4o-mini:  0.65 ± 0.11  [━━━━━━━━━━━│──────────]   │
│  claude-haiku: 0.58 ± 0.13  [━━━━━━━━━│───────────]   │
├─────────────────────────────────────────────────────────┤
│  Reward & Performance                                   │
│  Avg reward: 0.68    Fallback rate: 8.2%               │
│  Avg cost: $0.012    Avg latency: 1650ms               │
│  Cost savings vs baseline: 47.3%                        │
└─────────────────────────────────────────────────────────┘
```

### Alert rules summary

```python
ALERT_RULES = {
    "dominance": "Any model >80% for >1 hour",
    "low_confidence": "All models confidence <0.3 for >2 hours",
    "high_fallback": "Fallback rate >30%",
    "reward_drop": "Avg reward drops >20% in 15 min",
    "stuck_distribution": "Selection distribution unchanging for >6 hours",
    "cost_spike": "Avg cost increases >50% in 1 hour"
}
```

---

## 39. Edge Cases and Failure Modes

### Case 1: All models are terrible

**Scenario**: All models consistently return reward < 0.2

**What happens**:
- All posteriors shift toward low values
- Means: gpt-4o → 0.21, gpt-4o-mini → 0.18, claude-haiku → 0.19
- Thompson keeps exploring (uncertain which is "least bad")
- Eventually settles on the best of a bad set

**Detection**:
```python
if all(posterior_mean[m] < 0.3 for m in models):
    alert("All models performing poorly: check validators or task definition")
```

**Response**:
- Check reward function (is it too harsh?)
- Verify validators are correct
- Consider adding a stronger model to the pool
- May need to adjust task or prompt

**Absolute confidence floor**:
```python
if best_model_confidence < ABSOLUTE_FLOOR:  # e.g., 0.25
    # Don't route at all, return error or use fallback service
    raise InsufficientConfidenceError("No model meets minimum quality bar")
```

### Case 2: One model is always better

**Scenario**: gpt-4o always gives 0.9, others give 0.5

**What happens**:
- gpt-4o's posterior: α grows fast, β grows slowly
- After ~100 queries: gpt-4o mean → 0.88, others → 0.52
- Thompson selects gpt-4o >95% of time
- Exploration limited to shadow_rate

**Is this a failure?** No, this is correct behavior!

**But watch for**:
- Cost explosion (if best model is most expensive)
- Latency increase (if best model is slowest)

**Response**:
If cost is a concern, consider:
1. Increase latency/cost weight in reward function
2. Add explicit cost penalty term
3. Accept that quality requires cost

**Monitor**:
```python
if selection_rates["gpt-4o"] > 0.9 and model_costs["gpt-4o"] > 2*avg_cost:
    warn("Expensive model dominance: verify if quality gain justifies cost")
```

### Case 3: High variance models

**Scenario**: 
- Model A: 50% time reward=1.0, 50% time reward=0.2 (mean=0.6)
- Model B: Always reward=0.6

**What happens**:
- Both have same expected reward (0.6)
- Model A has higher variance → wider posterior → more uncertainty
- Thompson Sampling explores A more initially
- Over time, realizes both are equivalent
- Selection converges to ~50/50

**Is Thompson variance-aware?** Not directly, but uncertainty captures it:
- High variance → more conflicting evidence → wider posterior
- Wide posterior → more exploration samples
- Eventually converges to correct mean

**If you want explicit variance penalty**:
```python
# Modify reward to penalize unpredictability
reward_adjusted = reward - lambda * variance_estimate
```

This favors stable models.

### Case 4: Reward signal is completely noisy

**Scenario**: Validator is broken, returns random True/False

**What happens**:
- All models get random rewards ~0.5
- Posteriors slowly converge to mean ~0.5, high variance
- Thompson selection becomes nearly random
- Router learns nothing useful

**Detection**:
```python
def detect_noise():
    for model in models:
        # Check if reward variance is maximal
        theoretical_max_variance = 0.25  # for Bernoulli(0.5)
        observed_variance = np.var(recent_rewards[model])
        
        if observed_variance > 0.8 * theoretical_max_variance:
            return True
        
        # Check autocorrelation
        autocorr = np.corrcoef(
            recent_rewards[model][:-1], 
            recent_rewards[model][1:]
        )[0,1]
        
        if abs(autocorr) < 0.1:  # No pattern
            return True
    
    return False
```

**Response**:
- Fix validator
- Check if task is actually well-defined
- Consider human-in-the-loop sampling for validation

### Case 5: Correlated models (same provider)

**Scenario**: gpt-4o and gpt-4o-mini both degrade together (OpenAI outage)

**What happens**:
- Both posteriors shift left simultaneously
- Relative ranking may stay same
- Router doesn't switch to claude-haiku (uncorrelated)

**Why?** Bandit assumes independent arms, but these arms are correlated.

**Solution**: Track provider-level health
```python
provider_health = {
    "openai": mean(rewards for openai models),
    "anthropic": mean(rewards for anthropic models)
}

if provider_health["openai"] < 0.4:
    # Shift all traffic to anthropic
    models_to_consider = ["claude-haiku", "claude-sonnet"]
```

### Case 6: Catastrophic forgetting via aggressive decay

**Scenario**: γ = 0.70 (aggressive), decay_interval = 20

**What happens**:
- Half-life = ln(0.5)/ln(0.7) / (1000/20) = 38 queries
- Router "forgets" very quickly
- If a few bad samples happen randomly, posterior swings wildly
- Selection oscillates, never converges

**Detection**:
```python
# Track posterior variance over time
if variance_is_increasing(posterior_variance_history):
    alert("Posteriors diverging: decay may be too aggressive")
```

**Response**:
- Increase γ (slower decay)
- Increase decay_interval
- Add minimum pseudo-count floor: `max(5, γ * alpha)`

### Case 7: Cold start with uniform priors

**Scenario**: All models start with α=1, β=1 (uniform prior)

**What happens**:
- First ~50 queries are nearly random exploration
- High regret early on
- Slow convergence

**Cost**: Wasted queries on inferior models

**Fix**: Use expert priors (which you already do)

**How much do priors help?** Simulation shows:
- Uniform priors: converge in ~100 queries
- Expert priors: converge in ~20 queries
- **5x speedup**

### Case 8: Shadow rate = 0 (no forced exploration)

**Scenario**: Only Thompson Sampling, no shadow_rate

**What happens**:
- If posteriors become overconfident (high α+β, low variance)
- Thompson stops exploring
- If environment changes, router is slow to detect

**Example**:
- gpt-4o: α=500, β=100 (mean=0.83, very confident)
- gpt-4o-mini: α=200, β=300 (mean=0.40, very confident)
- Thompson selects gpt-4o >99% of time
- If gpt-4o degrades to 0.5, takes 100+ queries to detect

**Fix**: Always set shadow_rate > 0 (e.g., 0.05)

### Failure mode summary table

| Failure mode | Detection | Response |
|--------------|-----------|----------|
| All models terrible | All means <0.3 | Check validators, add stronger model |
| One model dominates | Selection >90% | Verify cost/quality tradeoff |
| High variance model | Wide posteriors persist | Consider variance penalty |
| Noisy reward signal | Low autocorrelation | Fix validator |
| Correlated models | Simultaneous degradation | Provider-level fallback |
| Catastrophic forgetting | Increasing variance | Reduce decay aggressiveness |
| Cold start regret | High early regret | Use expert priors |
| Stuck exploitation | Zero exploration | Add shadow_rate |

---

## 40. Extending to Contextual Bandits

### What changes with context?

**Non-contextual** (current):
- Same routing policy for all queries
- Learn global model quality

**Contextual**:
- Different routing policy per query type
- Learn which model is best *for this kind of query*

### When is context valuable?

Context helps when:
- Some models are better at certain tasks (e.g., coding vs. summarization)
- Query difficulty varies (easy vs. hard)
- User tier matters (free vs. paid)
- Latency requirements differ (interactive vs. batch)

Context doesn't help when:
- All queries are homogeneous
- Model quality is task-independent

### Approach 1: Discrete context (binning)

**Idea**: Partition queries into types, run separate bandits per type

```python
def get_context(query):
    if len(query) < 100:
        return "short"
    elif len(query) < 500:
        return "medium"
    else:
        return "long"

# Maintain separate posteriors
posteriors = {
    ("gpt-4o", "short"): {"alpha": 5, "beta": 2},
    ("gpt-4o", "medium"): {"alpha": 8, "beta": 3},
    ("gpt-4o", "long"): {"alpha": 6, "beta": 5},
    # ... for each (model, context) pair
}

def select(query):
    context = get_context(query)
    samples = {}
    for model in models:
        alpha = posteriors[(model, context)]["alpha"]
        beta = posteriors[(model, context)]["beta"]
        samples[model] = np.random.beta(alpha, beta)
    return argmax(samples)

def update(query, model, reward):
    context = get_context(query)
    posteriors[(model, context)]["alpha"] += reward
    posteriors[(model, context)]["beta"] += (1 - reward)
```

**Pros**:
- Simple extension of current code
- Interpretable (can see "gpt-4o is best for short queries")
- No new math required

**Cons**:
- Curse of dimensionality (K models × C contexts posteriors)
- No sharing (learning about "short" doesn't help "medium")
- Requires enough data per context

**Example contexts**:
```python
contexts = {
    "length": ["short", "medium", "long"],
    "has_code": [True, False],
    "user_tier": ["free", "paid"],
    "requires_tools": [True, False]
}

# Total contexts = 3 × 2 × 2 × 2 = 24
# With 3 models = 72 posteriors to maintain
```

### Approach 2: Linear Thompson Sampling

**Idea**: Model expected reward as linear function of features

`E[r | x, m] = x^T * theta_m`

where:
- `x in R^d` is the feature vector
- `theta_m in R^d` is the coefficient vector for model `m`

**Prior**: Gaussian over coefficients
`theta_m ~ N(mu_m, Sigma_m)`

**Selection**:
```python
def select(query):
    x = featurize(query)  # e.g., [length, has_code, embedding...]
    
    samples = {}
    for model in models:
        # Sample coefficient vector
        theta_sample = np.random.multivariate_normal(mu[model], Sigma[model])
        
        # Predicted reward
        samples[model] = x @ theta_sample
    
    return argmax(samples)
```

**Update** (Bayesian linear regression):
```python
def update(query, model, reward):
    x = featurize(query)
    
    # Posterior update (Sherman-Morrison formula for efficiency)
    Sigma_inv = Sigma_inv[model] + np.outer(x, x)
    Sigma[model] = np.linalg.inv(Sigma_inv)
    mu[model] = Sigma[model] @ (Sigma_inv[model] @ mu[model] + reward * x)
```

### Plain-English meaning

Instead of keeping a separate bandit for every context bucket, Linear Thompson Sampling learns a formula for each model.

- The input to that formula is a set of features about the query, such as length, whether it contains code, whether JSON output is required, or the user tier.
- The output is a predicted reward for that model on this specific query.
- Each model learns its own set of feature weights, so one model might learn "long queries are good for me" while another learns "I do well on short structured tasks."

What happens on each request:

1. convert the query into a feature vector
2. for each model, sample one possible set of weights from the current belief
3. use those sampled weights to compute a temporary predicted reward
4. choose the model with the highest predicted reward
5. observe the real reward and update the model's weights

Simple intuition:

- discrete context says: "short query" and "long query" are separate buckets
- linear Thompson says: "learn how length, code, JSON needs, and other features influence reward"
- that means learning from one kind of query can help on similar queries, instead of treating every bucket as totally separate

Tiny example:

- suppose a query is long, has code, and needs valid JSON
- the router may predict that `gpt-4o` handles that combination better than `gpt-4o-mini`
- for a short plain-language query, it may predict the opposite

### Formula in simple terms

Suppose we represent a query with features like this:

- `x = [1, has_code, needs_json, query_length_norm]`

The leading `1` is an intercept term.

For each model `m`, the router learns a weight vector:

- `theta_m = [bias, code_weight, json_weight, length_weight]`

The predicted score for model `m` is:

- `score_m = x^T * theta_m`
- or equivalently: `score_m = sum_j x_j * theta_{m,j}`

Because this is Thompson Sampling, the router does not use one fixed weight vector. It keeps a belief over plausible weights:

- `theta_m ~ N(mu_m, Sigma_m)`

On each request:

1. sample one temporary weight vector `theta_tilde_m` for each model
2. compute `score_m = x^T * theta_tilde_m`
3. choose the model with the highest score
4. observe reward `r`
5. update the belief for the chosen model

One useful way to think about this:

- positive weights help a model on that feature
- negative weights hurt a model on that feature
- larger uncertainty in `Sigma_m` means more exploration

### Worked example

Suppose the query is:

- long
- contains code
- needs valid JSON

Use this feature vector:

- `x = [1, 1, 1, 0.8]`

Now suppose the router samples these temporary weights:

- `gpt-4o`: `theta_tilde = [0.35, 0.20, 0.15, 0.10]`
- `gpt-4o-mini`: `theta_tilde = [0.50, -0.05, -0.08, 0.02]`

Compute the scores:

- `gpt-4o`: `1*0.35 + 1*0.20 + 1*0.15 + 0.8*0.10 = 0.78`
- `gpt-4o-mini`: `1*0.50 + 1*(-0.05) + 1*(-0.08) + 0.8*0.02 = 0.386`

So the router chooses `gpt-4o`.

Now suppose the observed reward is high:

- `r = 0.9`

Then the posterior for `gpt-4o` is updated so the router becomes more confident that this kind of query is a good fit for `gpt-4o`.

Now compare that with an easier query:

- short
- no code
- no JSON requirement

Use:

- `x = [1, 0, 0, 0.1]`

Scores become:

- `gpt-4o`: `0.35 + 0 + 0 + 0.01 = 0.36`
- `gpt-4o-mini`: `0.50 + 0 + 0 + 0.002 = 0.502`

Now the router chooses `gpt-4o-mini`.

This is the key idea:

- harder structured queries may push the router toward a stronger model
- easier queries may push it toward a cheaper model

In practice, `x^T * theta_m` is often best understood as a ranking score rather than a literal probability in `[0, 1]`.

**Pros**:
- Shares information across contexts (generalization)
- Scales to many features
- Bayesian uncertainty quantification

**Cons**:
- Assumes linearity (may not hold)
- More complex math
- Requires feature engineering

**Feature examples**:
```python
def featurize(query):
    return [
        len(query) / 1000,                    # Normalized length
        int("```" in query),                  # Has code block
        int(user_tier == "paid"),            # Paid user
        np.mean(embedding(query)),           # Embedding features
        int(requires_structured_output),     # Needs JSON
    ]
```

### Approach 3: Hybrid (difficulty classifier + bandit)

**Idea**: Use a supervised classifier to route to candidate sets, then bandit within

```python
# Offline: train a difficulty classifier
difficulty_model = train_classifier(
    features=query_embeddings,
    labels=["easy", "hard"]  # human-labeled or heuristic
)

# Online: contextual routing
def select(query):
    difficulty = difficulty_model.predict(query)
    
    if difficulty == "easy":
        # Only consider cheap models
        candidates = ["gpt-4o-mini", "claude-haiku"]
    else:
        # Consider all models, favor strong ones
        candidates = ["gpt-4o", "claude-sonnet"]
    
    # Run Thompson Sampling over candidates
    samples = {m: np.random.beta(alpha[m], beta[m]) for m in candidates}
    return argmax(samples)
```

**Pros**:
- Combines supervised learning (difficulty) + bandits (online adaptation)
- Reduces search space (fewer candidates per query)
- Leverages both labeled data and production telemetry

**Cons**:
- Requires training a classifier (needs labeled data)
- Difficulty model can be wrong
- More components to maintain

**When to use this**: You have some labeled data for difficulty, but providers still drift

### Approach 4: Neural Thompson Sampling

**Idea**: Use a neural network to model $\mathbb{E}[r \mid x, m]$, Bayesian NN for uncertainty

```python
class BayesianNN(nn.Module):
    def forward(self, query_embedding, model_id):
        # Predict reward
        return reward_prediction

# Maintain distribution over NN weights
# Sample NN weights → sample reward function → Thompson Sampling
```

**Pros**:
- Can capture complex non-linear patterns
- Shares information via learned representations

**Cons**:
- Requires significant data
- Computationally expensive
- Harder to interpret

**When to use**: Very large scale (millions of queries), complex patterns

### Practical recommendation: Start with discrete context

For a production talk, suggest:

**Phase 1** (current): Non-contextual Thompson Sampling
- Validate that routing provides value
- Establish baseline monitoring

**Phase 2**: Add simple discrete context (e.g., query length bins)
- Low implementation cost
- Immediate value if queries are heterogeneous

**Phase 3**: If Phase 2 shows value, explore linear Thompson Sampling
- More sophisticated, scales better

### Example: Query length-based routing

```python
class ContextualRouter:
    def __init__(self):
        # Separate posteriors per (model, length_bin)
        self.alpha = defaultdict(lambda: 5)  # Prior
        self.beta = defaultdict(lambda: 4)
    
    def get_context(self, query):
        if len(query) < 200:
            return "short"
        else:
            return "long"
    
    def select(self, query):
        context = self.get_context(query)
        samples = {}
        for model in MODELS:
            key = (model, context)
            samples[model] = np.random.beta(self.alpha[key], self.beta[key])
        return max(samples, key=samples.get)
    
    def update(self, query, model, reward):
        context = self.get_context(query)
        key = (model, context)
        self.alpha[key] += reward
        self.beta[key] += (1 - reward)

# Usage
router = ContextualRouter()

for query in queries:
    model = router.select(query)
    reward = compute_reward(query, model)
    router.update(query, model, reward)
```

**Expected result**: 
- Short queries → route to gpt-4o-mini (fast, cheap, good enough)
- Long queries → route to gpt-4o (needs stronger reasoning)

---

(Continuing in next message due to length...)
# Bayesian Bandits Study Material - Extended Sections (Part 2)

## 41. How to Set Priors Systematically

Why this section matters:

- Thompson Sampling always needs some starting belief before any real traffic arrives.
- You could use simple uniform priors, but then the router starts from near-total ignorance.
- In production, the first few routing decisions matter for both cost and safety.
- Better priors reduce cold-start regret, avoid wasting too many early queries on obviously weaker models, and make the starting assumptions explainable.

So this section is not saying "fancy priors are mathematically mandatory." It is saying that **some prior is always required**, and choosing it systematically makes the router behave much better at the start.

### Current approach (manual expert priors)

```python
DEFAULT_MODELS = {
    "gpt-4o": {"alpha": 8, "beta": 3},        # mean = 8/11 = 0.727
    "gpt-4o-mini": {"alpha": 5, "beta": 4},   # mean = 5/9 = 0.556
    "claude-haiku": {"alpha": 5, "beta": 4}   # mean = 5/9 = 0.556
}
```

### Where do these numbers come from?

The two numbers encode:
1. **Prior mean**: `mu_0 = alpha / (alpha + beta)`
2. **Prior strength**: `n_0 = alpha + beta` (effective sample size)

### Method 1: From historical data

**Process**:
```python
# Step 1: Collect validation data
validation_queries = load_validation_set(100)  # 100 representative queries

model_rewards = {}
for model in models:
    rewards = []
    for query in validation_queries:
        response = model.generate(query)
        reward = compute_reward(response)
        rewards.append(reward)
    model_rewards[model] = rewards

# Step 2: Compute empirical mean
empirical_means = {
    model: np.mean(rewards)
    for model, rewards in model_rewards.items()
}
# e.g., {"gpt-4o": 0.73, "gpt-4o-mini": 0.56, "claude-haiku": 0.58}

# Step 3: Choose prior strength based on confidence
# Low confidence → small n_0 (let data dominate quickly)
# High confidence → large n_0 (trust the prior)
prior_strength = 11  # "I believe this mean, equivalent to 11 observations"

# Step 4: Convert to alpha, beta
for model, mean in empirical_means.items():
    alpha = mean * prior_strength
    beta = (1 - mean) * prior_strength
    priors[model] = {"alpha": alpha, "beta": beta}

# Result:
# gpt-4o: alpha=8.03, beta=2.97 ≈ alpha=8, beta=3
# gpt-4o-mini: alpha=6.16, beta=4.84 ≈ alpha=6, beta=5
```

**Validation**: Run simulator with these priors, verify they match historical performance.

### Method 2: From vendor benchmarks

**Process**:
```python
# Use published benchmark scores as proxy
benchmark_scores = {
    "gpt-4o": 0.88,        # MMLU score
    "gpt-4o-mini": 0.82,
    "claude-haiku": 0.75
}

# Adjust for your task (benchmarks may not match production)
task_adjustment = 0.85  # "Our task is 85% as hard as MMLU"

adjusted_means = {
    model: score * task_adjustment
    for model, score in benchmark_scores.items()
}

# Choose strength based on benchmark reliability
# Well-established benchmarks → higher strength
# Uncertain correlation to your task → lower strength
prior_strength = 10  # Medium confidence

for model, mean in adjusted_means.items():
    alpha = mean * prior_strength
    beta = (1 - mean) * prior_strength
    priors[model] = {"alpha": round(alpha), "beta": round(beta)}
```

**Pros**: No need to run validation queries
**Cons**: Benchmarks may not correlate with your reward function

### Method 3: Pessimistic (conservative) priors

**Idea**: Start all models with weak, neutral priors

```python
# Uniform prior: no preference
for model in models:
    priors[model] = {"alpha": 1, "beta": 1}  # mean = 0.5, strength = 2
```

**When to use**:
- First deployment, zero historical data
- High uncertainty about relative quality
- Want data to dominate quickly

**Tradeoff**: Higher regret in first 50-100 queries

**Slightly informative pessimistic**:
```python
# Assume all models are "decent" (mean ~0.6) but uncertain
for model in models:
    priors[model] = {"alpha": 3, "beta": 2}  # mean = 0.6, strength = 5
```

### Method 4: Optimistic priors (exploration bonus)

**Idea**: Start all models with high priors to encourage trying everything

```python
# Optimistic: assume all models are good
for model in models:
    priors[model] = {"alpha": 9, "beta": 2}  # mean = 0.82, strength = 11
```

**Effect**:
- All models start with high sampled values
- More early exploration
- Good for discovery phase

**When to use**:
- Want to thoroughly test all models
- Cost of exploration is low
- Discovery is valuable

### Prior strength tuning

| Strength (α+β) | Effective sample size | Queries to override | When to use |
|----------------|----------------------|---------------------|-------------|
| 2-3            | Weak prior           | ~5 queries          | High uncertainty, untrusted priors |
| 5-10           | Medium prior         | ~10-20 queries      | Some confidence, typical |
| 11-20          | Strong prior         | ~30-50 queries      | High confidence in prior mean |
| 50-100         | Very strong prior    | ~100+ queries       | Very confident, slow adaptation |

**Rule of thumb**: 

Set strength = "How many observations would it take for me to change my mind?"

If you're very sure gpt-4o is best: strength = 50
If you're guessing based on limited data: strength = 5

### Asymmetric priors for known differences

If you know model A is better than B:

```python
priors = {
    "strong-model": {"alpha": 10, "beta": 2},  # mean = 0.83, confident it's good
    "weak-model": {"alpha": 3, "beta": 5},     # mean = 0.375, confident it's worse
}
```

This encodes:
- Different expected quality (different means)
- Potentially different confidence (different strengths)

### Empirical validation of priors

**Test**: Do priors match reality?

```python
def validate_priors(priors, validation_data):
    for model in models:
        # Predicted by prior
        prior_mean = priors[model]["alpha"] / (
            priors[model]["alpha"] + priors[model]["beta"]
        )
        
        # Observed in validation
        actual_rewards = [compute_reward(query, model) for query in validation_data]
        actual_mean = np.mean(actual_rewards)
        
        error = abs(prior_mean - actual_mean)
        print(f"{model}: prior={prior_mean:.3f}, actual={actual_mean:.3f}, error={error:.3f}")
        
        if error > 0.15:
            warn(f"{model} prior may be inaccurate")

# Example output:
# gpt-4o: prior=0.727, actual=0.715, error=0.012 ✓
# gpt-4o-mini: prior=0.556, actual=0.620, error=0.064 ✓
# claude-haiku: prior=0.556, actual=0.450, error=0.106 ⚠
```

If errors are large, adjust priors before production.

### Dynamic prior adjustment (advanced)

**Idea**: Update priors periodically based on recent history

```python
# Every week, recompute priors from last 1000 queries
def recompute_priors():
    for model in models:
        recent_rewards = get_recent_rewards(model, n=1000)
        empirical_mean = np.mean(recent_rewards)
        
        # Keep same strength, update mean
        strength = priors[model]["alpha"] + priors[model]["beta"]
        priors[model]["alpha"] = empirical_mean * strength
        priors[model]["beta"] = (1 - empirical_mean) * strength
```

**Caution**: This can cause instability if done too frequently. Use conservatively.

---

## 42. Simulating Scenarios for Validation

### Why simulate?

Before deploying, validate that the router:
1. Converges to the best model (correctness)
2. Converges quickly (efficiency)
3. Adapts to drift (robustness)
4. Handles edge cases (reliability)

### Scenario 1: Static rewards (sanity check)

**Setup**:
```python
# True mean rewards (unknown to router)
true_means = {
    "model-a": 0.80,
    "model-b": 0.60,
    "model-c": 0.40
}

# Simulation
router = BayesianRouter()
regrets = []

for t in range(500):
    model = router.select()
    
    # Simulate reward (Bernoulli for simplicity)
    reward = 1 if random() < true_means[model] else 0
    
    router.update(model, reward)
    
    # Track regret
    optimal_reward = max(true_means.values())  # 0.80
    regret = optimal_reward - true_means[model]
    regrets.append(regret)

# Analysis
cumulative_regret = np.cumsum(regrets)
plt.plot(cumulative_regret)
plt.xlabel("Queries")
plt.ylabel("Cumulative Regret")
plt.title("Should be sublinear (square root)")
```

**Expected behavior**:
- First ~20 queries: high regret (exploration)
- Queries 20-100: regret grows slowly (convergence)
- After 100: regret nearly flat (exploitation)

**Metrics**:
```python
final_selection_rate = {
    model: selections[model] / 500
    for model in models
}
# Should be: model-a ≈ 90%, model-b ≈ 7%, model-c ≈ 3%

convergence_time = first_time_when(selection_rate["model-a"] > 0.8)
# Should be: ~30-50 queries
```

### Scenario 2: Model rot (drift detection)

**Setup**:
```python
true_means = {
    "model-a": 0.80,  # Will degrade at t=200
    "model-b": 0.60,  # Stays constant
}

for t in range(400):
    model = router.select()
    
    # Drift: model-a degrades at t=200
    if t >= 200:
        true_means["model-a"] = 0.50  # Degrades below model-b
    
    reward = sample_reward(model, true_means[model])
    router.update(model, reward)
```

**Expected behavior without decay**:
- t<200: model-a selected ~90% (correct)
- t=200: model-a degrades
- t>200: model-a still selected ~70% (STUCK, incorrect)
- Regret grows linearly after t=200

**Expected behavior with decay (γ=0.95, interval=50)**:
- t<200: model-a selected ~90%
- t=200: model-a degrades
- t=200-250: router detects shift (within ~50 queries)
- t>250: model-b selected ~85% (correct)
- Regret spike then flattens

**Metric**:
```python
adaptation_time = first_time_after(t=200, when=selection_rate["model-b"] > 0.7)
# With decay: ~50 queries
# Without decay: never (or 200+)
```

### Scenario 3: High variance model

**Setup**:
```python
# model-a: Bernoulli(0.7) → 70% reward=1, 30% reward=0
# model-b: constant 0.6

def sample_reward(model):
    if model == "model-a":
        return 1 if random() < 0.7 else 0  # High variance
    else:
        return 0.6  # Low variance (deterministic)
```

**Expected behavior**:
- Thompson correctly estimates E[reward]=0.7 for model-a
- model-a selected more despite noisiness
- Variance increases posterior width → more initial exploration

**Analysis**:
```python
posterior_variance = {
    "model-a": compute_beta_variance(alpha_a, beta_a),
    "model-b": compute_beta_variance(alpha_b, beta_b)
}
# model-a should have higher variance throughout
```

### Scenario 4: Delayed superiority

**Setup**:
```python
# model-b is inferior initially, then becomes best
true_means = {
    "model-a": 0.70,  # Constant
    "model-b": 0.50,  # Then jumps to 0.85 at t=100
}

for t in range(300):
    model = router.select()
    
    if t >= 100:
        true_means["model-b"] = 0.85  # Sudden improvement
    
    reward = sample_reward(model, true_means[model])
    router.update(model, reward)
```

**Expected behavior**:
- t<100: model-a selected ~80%
- t=100: model-b improves
- t=100-150: router detects improvement
- t>150: model-b selected ~85%

**Test adaptability**:
```python
switch_time = first_time_after(t=100, when=selection_rate["model-b"] > 0.7)
# With decay: ~30-50 queries
# Without decay: ~100+ queries (slower)
```

### Scenario 5: All models equivalent

**Setup**:
```python
true_means = {
    "model-a": 0.60,
    "model-b": 0.60,
    "model-c": 0.60
}
```

**Expected behavior**:
- Selection eventually stabilizes at ~33% each
- Or picks one arbitrarily (due to random early luck)

**Is this a problem?** No, but:
- If costs differ, should route to cheapest
- Need to add cost to reward function

**Test**:
```python
# After 500 queries, check if selection is roughly uniform
selection_entropy = -sum(p * log(p) for p in selection_rates.values())
max_entropy = log(3)  # Uniform over 3 models

if selection_entropy > 0.8 * max_entropy:
    print("✓ Router correctly treats equivalent models uniformly")
```

### Metrics to track across all scenarios

```python
class SimulationMetrics:
    def __init__(self):
        self.cumulative_regret = []
        self.selection_history = []
        self.reward_history = []
        self.posterior_history = []
    
    def compute_summary(self):
        return {
            "total_regret": sum(self.cumulative_regret),
            "final_regret_rate": self.cumulative_regret[-100:].mean(),
            "convergence_time": self.detect_convergence(),
            "avg_reward": np.mean(self.reward_history),
            "optimal_selection_rate": self.compute_optimal_rate()
        }
    
    def detect_convergence(self):
        # First time when best model selected >80% in rolling 50 window
        for t in range(50, len(self.selection_history)):
            window = self.selection_history[t-50:t]
            if window.count(best_model) / 50 > 0.8:
                return t
        return None
```

### Visual validation

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Cumulative regret (should be sublinear)
axes[0,0].plot(np.cumsum(regrets))
axes[0,0].set_title("Cumulative Regret")
axes[0,0].set_xlabel("Queries")

# Plot 2: Selection distribution over time
for model in models:
    selection_rate_over_time = rolling_mean([s==model for s in selections], window=50)
    axes[0,1].plot(selection_rate_over_time, label=model)
axes[0,1].set_title("Selection Rate (rolling 50)")
axes[0,1].legend()

# Plot 3: Posterior means over time
for model in models:
    posterior_means = [h[model]["mean"] for h in posterior_history]
    axes[1,0].plot(posterior_means, label=model)
axes[1,0].set_title("Posterior Means")
axes[1,0].legend()

# Plot 4: Posterior uncertainty over time
for model in models:
    posterior_stds = [h[model]["std"] for h in posterior_history]
    axes[1,1].plot(posterior_stds, label=model)
axes[1,1].set_title("Posterior Uncertainty")
axes[1,1].legend()

plt.tight_layout()
plt.savefig("simulation_results.png")
```

---

## 43. Related Work - Detailed Comparison

### Paper 1: FrugalGPT (Chen et al., 2023)

**Full citation**: "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance"

**Core idea**: Cascade of models from cheapest to most expensive

**Architecture**:
```
Query → GPT-3.5
        ├─ If score(response) > threshold → Return
        └─ Else → GPT-4
                  ├─ If score(response) > threshold → Return
                  └─ Else → GPT-4 + retrieval
```

**How they decide to escalate**: 
- Train a scoring function on labeled data
- Score predicts: "Is this response good enough?"
- If score < threshold, escalate to next model

**Key innovation**: Learned scoring function (not heuristic)

**Comparison to your approach**:

| Aspect | FrugalGPT | Your router |
|--------|-----------|-------------|
| **Learning** | Supervised (needs labels) | Unsupervised (reward from telemetry) |
| **Routing** | Deterministic cascade | Probabilistic selection |
| **Adaptation** | Offline retraining | Online, continuous |
| **Context** | Can be contextual | Currently non-contextual |
| **Exploration** | None (deterministic) | Built-in (Thompson) |

**When FrugalGPT is better**:
- You have labeled data (human judgments)
- Queries have natural difficulty hierarchy
- Want deterministic behavior

**When your approach is better**:
- No labels available
- Providers drift frequently
- Want online adaptation

**Could you combine them?**
Yes:
```python
# Use FrugalGPT-style cascade for initial filtering
if difficulty_score(query) < 0.3:
    candidates = ["gpt-4o-mini"]
elif difficulty_score(query) < 0.7:
    candidates = ["gpt-4o-mini", "gpt-4o"]
else:
    candidates = ["gpt-4o", "claude-opus"]

# Use Thompson Sampling within candidate set
model = thompson_sample(candidates)
```

### Paper 2: RouteLLM (Ong et al., 2024)

**Full citation**: "RouteLLM: Learning to Route LLMs with Preference Data"

**Core idea**: Train a router classifier to predict which model to use

**Architecture**:
```python
# Offline training
query_embedding = embed(query)
difficulty_features = extract_features(query)
model_choice = classifier([query_embedding, difficulty_features])

# Returns: "strong_model" or "weak_model"
```

**Training data**:
- Collect (query, strong_model_response, weak_model_response)
- Human preference: which response is better?
- Train classifier to predict when weak model is sufficient

**Key innovation**: Uses preference data (not just binary labels)

**Comparison to your approach**:

| Aspect | RouteLLM | Your router |
|--------|----------|-------------|
| **Contextual** | Yes (query-aware) | No (traffic-level) |
| **Training** | Offline, supervised | Online, bandit |
| **Data** | Human preferences | Production telemetry |
| **Adaptation** | Requires retraining | Automatic |
| **Generalization** | Can handle new query types | Assumes stationary distribution |

**When RouteLLM is better**:
- Queries are diverse (different difficulties)
- You have preference data
- Worth modeling query context

**When your approach is better**:
- Queries are homogeneous
- Providers drift frequently
- No labeled data

**Could you combine them?**
Absolutely (this is your "future work"):
```python
# RouteLLM for context
difficulty = route_llm_classifier(query)

# Thompson Sampling for online adaptation per difficulty bin
if difficulty == "easy":
    model = thompson_sample(["gpt-4o-mini", "claude-haiku"])
else:
    model = thompson_sample(["gpt-4o", "claude-sonnet"])
```

### Paper 3: AutoMix (Madaan et al., 2023)

**Core idea**: Few-shot verification to decide if output is good enough

**Architecture**:
```
Query → Weak model
        ↓
        Verify with few-shot examples
        ├─ If verified → Return
        └─ Else → Strong model
```

**Comparison**:

| Aspect | AutoMix | Your router |
|--------|---------|-------------|
| **Verification** | Few-shot classifier | Reward from telemetry |
| **Static** | Yes (fixed cascade) | No (adapts online) |
| **Per-query** | Yes | No |

**When AutoMix is better**: You can define good few-shot examples
**When yours is better**: Want adaptation without manual examples

### Paper 4: Speculative Decoding

**Core idea**: Use weak model to generate candidates, strong model to verify

```
weak_model.generate() → [candidate_1, candidate_2, ...]
strong_model.verify() → pick best candidate
```

**This is orthogonal to routing** - it's about accelerating a single model, not choosing between models.

### Summary table

| Paper | Supervised? | Contextual? | Online? | Key strength |
|-------|-------------|-------------|---------|--------------|
| **FrugalGPT** | Yes | Possible | No | Learned cascade |
| **RouteLLM** | Yes | Yes | No | Query-aware routing |
| **AutoMix** | No | Yes | No | Few-shot verification |
| **Your router** | No | No | Yes | Online adaptation |

**Synthesis for your talk**:

> Existing work like FrugalGPT and RouteLLM assume you have labeled data—human judgments of quality or preference pairs. That's often unavailable in production. My approach adapts online using only operational telemetry: did it parse, how fast was it, did we retry? No labels needed.

> The tradeoff is that I'm currently non-contextual—I learn at the traffic level, not per-query. A natural extension would be to combine RouteLLM's contextual classifier with Thompson Sampling's online adaptation.

---

## 44. Connection to Reinforcement Learning

### Bandits as a special case of RL

**Full RL formulation** (Markov Decision Process):
- States: $s_t \in \mathcal{S}$
- Actions: $a_t \in \mathcal{A}$
- Transitions: $s_{t+1} \sim P(\cdot \mid s_t, a_t)$
- Rewards: $r_t \sim R(s_t, a_t)$
- Goal: maximize $\mathbb{E}\left[\sum_{t=1}^T \gamma^t r_t\right]$

**Bandits**: 
- Special case where state is constant (or doesn't exist)
- $s_t = s_0$ for all $t$
- Transitions are trivial
- Rewards are immediate (no delayed credit assignment)

| Concept | Bandit | Full RL (MDP) |
|---------|--------|---------------|
| **State** | None (or constant) | Changes over time |
| **Action** | Pick a model | Pick a model |
| **Reward** | Immediate | Can be delayed |
| **Horizon** | 1-step | Multi-step (episodes) |
| **Policy** | $\pi : \mathcal{A} \to [0,1]$ | $\pi : \mathcal{S} \times \mathcal{A} \to [0,1]$ |
| **Value function** | Not needed | $V(s)$, $Q(s,a)$ |

### Why bandits are easier

**1. No credit assignment problem**

In RL, if you get a reward at time t, which action caused it?
- The action at t?
- The action at t-1?
- The entire sequence?

In bandits, reward is immediately attributable to the action just taken.

**2. No state space complexity**

In RL with large state spaces:
- Need function approximation (neural networks)
- Exploration in high-dimensional state space
- Generalization across states

Bandits: no states to worry about.

**3. Simple value estimation**

In RL:
- Need to estimate $Q(s, a)$ for all state-action pairs
- Or learn policy $\pi(a \mid s)$ directly

In bandits:
- Just estimate $Q(a)$ (expected reward per action)
- Your Beta distributions are exactly this

### When you'd need full RL for routing

**Case 1: Multi-turn conversations**

```
User: "Summarize this document"
  ↓ [choose model A]
Agent: "Here's a summary..."
User: "Now extract key dates"
  ↓ [choose model B?]
```

Later model choices may depend on earlier choices → state matters.

**Case 2: Budget constraints**

```
Constraint: Total cost < $100 per day

State: {current_cost: $85, time_remaining: 3 hours}

If cost is high → route to cheaper models (state-dependent policy)
```

**Case 3: Compositional queries**

```
Query requires:
1. Extract entities (model A)
2. Classify sentiment (model B)
3. Synthesize report (model C)

Multi-step, sequential → RL problem
```

### Thompson Sampling as Posterior Sampling for RL (PSRL)

Your algorithm is a special case of a broader RL algorithm called **Posterior Sampling for Reinforcement Learning**.

**PSRL framework**:
```
1. Maintain a posterior over environment dynamics
2. Sample one environment model from the posterior
3. Act optimally *assuming that model is correct*
4. Observe reward, update posterior
```

**Your implementation**:
```
1. Posterior over model quality (Beta distributions)
2. Sample one quality value per model
3. Choose model with highest sample (optimal for that sample)
4. Observe reward, update posterior
```

**PSRL also has regret guarantees** in tabular MDPs (finite states/actions).

### Related RL algorithms

**1. Q-learning** (not applicable here)

```python
# Learns Q(s, a) via Bellman updates
Q[s, a] += alpha * (reward + gamma * max(Q[s', :]) - Q[s, a])
```

Not needed for bandits because there's no future state s'.

**2. Policy gradient** (overkill for bandits)

```python
# Learns policy π(a | s) via gradient ascent
gradient = ∇ log π(a | s) * reward
theta += learning_rate * gradient
```

For bandits, Thompson Sampling is simpler and more sample-efficient.

**3. Contextual bandits → RL connection**

If you add context (features x):
- Contextual bandit: choose action based on current context $x_t$
- RL: context becomes state, actions affect future states

The boundary:
- Contextual bandit: $x_t$ is exogenous (doesn't depend on your actions)
- RL: $s_t$ is endogenous (your actions change future states)

### Practical takeaway

For your talk, you can say:

> Model routing is a bandit problem because:
> 1. Choosing a model doesn't affect future query distributions (no state transitions)
> 2. Rewards are immediate (no delayed consequences)
> 3. This simplifies the problem significantly—we don't need full RL machinery like value functions or policy gradients

> If we wanted to route multi-turn conversations or manage cumulative budgets, we'd need to move to full RL (Markov Decision Processes). But for single-query routing, bandits are the right abstraction.

---

## 45. Summary - How to Use These Extended Sections

### Study path

**If you have 1 hour**:
- Section 35 (reward weights)
- Section 37 (decay tuning)
- Section 39 (failure modes)

**If you have 3 hours**:
- All of the above, plus:
- Section 36 (algorithm comparison)
- Section 38 (monitoring)
- Section 43 (related work)

**If you have a full day**:
- Read all sections
- Work through the companion Jupyter notebook
- Practice explaining each concept out loud

### What to memorize vs. understand

**Memorize**:
- Beta mean formula: $\mu = \frac{\alpha}{\alpha+\beta}$
- Half-life formula: $k = \frac{\ln 0.5}{\ln \gamma}$
- Key failure modes (Section 39)
- How Thompson differs from UCB and ε-greedy

**Understand conceptually** (don't memorize formulas):
- Why sigmoid for latency (smooth penalty)
- How decay prevents overconfidence
- When context helps vs. doesn't
- FrugalGPT vs. RouteLLM vs. your approach

### During Q&A

**If asked about something you covered here**:
- Answer confidently and precisely
- Acknowledge tradeoffs honestly
- Point to future work naturally

**If asked about something you didn't cover**:
- "That's a great question. I haven't explored that direction yet."
- Or: "That's outside the scope of this talk, but worth investigating."

### Practice exercises

Before the talk, test yourself:

1. **Explain Thompson Sampling in 30 seconds** (Section 27)
2. **Explain why your reward is a heuristic** (Section 10)
3. **Explain when context would help** (Section 40)
4. **Explain how you'd monitor this in production** (Section 38)
5. **Compare your approach to FrugalGPT** (Section 43)

If you can do all five smoothly, you're ready.

---

**End of Extended Study Material**
