# Bayesian Model Routing — Quick Reference Card

**Purpose**: One-page cheat sheet for the DevConf talk. Print this or keep it open during Q&A.

---

## 📐 Essential Formulas

### Beta Distribution

**Mean**:
$$
\mu = \frac{\alpha}{\alpha + \beta}
$$

**Variance**:
$$
\sigma^2 = \frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
$$

**Standard Deviation**:
$$
\sigma = \sqrt{\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}}
$$

**Interpretation**:
- α = "success evidence"
- β = "failure evidence"
- α + β = "prior strength" (effective sample size)

### Thompson Sampling Update (Exact, Binary Rewards)

```
If reward = 1 (success):
    α ← α + 1

If reward = 0 (failure):
    β ← β + 1
```

### Your Implementation (Continuous Rewards)

```python
# Fractional pseudo-counts
α ← α + reward           # reward ∈ [0,1]
β ← β + (1 - reward)
```

⚠️ **This is NOT exact Beta-Bernoulli conjugacy** — it's a pragmatic heuristic.

### Composite Reward

$$
r = r_{\text{validity}} + r_{\text{latency}} + r_{\text{retry}}
$$

**Default weights**: 0.50 + 0.30 + 0.20 = 1.0

**Validity**:
```
r_validity = 0.50  if valid
             0.0   otherwise
```

**Retry**:
```
r_retry = 0.20  if not retried
          0.0   if retried
```

**Latency** (sigmoid):
$$
r_{\text{latency}} = \frac{0.30}{1 + e^{(L - 2000)/600}}
$$

where L = latency in milliseconds

### Decay Formula

Every `decay_interval` queries:
$$
\alpha \leftarrow \max(1, \gamma \cdot \alpha)
$$
$$
\beta \leftarrow \max(1, \gamma \cdot \beta)
$$

**Half-life**:
$$
\text{queries} = \frac{\ln(0.5)}{\ln(\gamma)} \times \text{decay\_interval}
$$

Example: γ=0.95, interval=50 → half-life = 675 queries

---

## 🎯 Algorithm: Thompson Sampling Step-by-Step

```python
# 1. SELECT
for each model m:
    sample[m] = np.random.beta(alpha[m], beta[m])

chosen_model = argmax(sample)

# 2. EXECUTE
response = chosen_model.generate(query)

# 3. OBSERVE
validity = validate(response)
latency = measure_latency()
retried = check_retry_flag()

# 4. COMPUTE REWARD
reward = compute_composite_reward(validity, latency, retried)

# 5. UPDATE
alpha[chosen_model] += reward
beta[chosen_model] += (1 - reward)

# 6. DECAY (every N queries)
if query_count % decay_interval == 0:
    for m in models:
        alpha[m] = max(1, gamma * alpha[m])
        beta[m] = max(1, gamma * beta[m])
```

---

## ⚙️ Parameter Tuning Guide

### Prior Strength (α + β)

| Strength | Effective Sample Size | Override Time | Use Case |
|----------|----------------------|---------------|----------|
| 2-3      | Weak                 | ~5 queries    | High uncertainty |
| 5-10     | Medium               | ~10-20 queries| Typical |
| 11-20    | Strong               | ~30-50 queries| High confidence |
| 50+      | Very strong          | 100+ queries  | Very confident |

**Rule of thumb**: "How many observations to change my mind?"

### Decay Rate (γ)

| γ    | Decay/step | Half-life (queries) | Use Case |
|------|------------|---------------------|----------|
| 0.99 | 1%         | 3,450               | Stable providers |
| 0.95 | 5%         | 675                 | **Moderate drift (default)** |
| 0.90 | 10%        | 330                 | Volatile providers |
| 0.80 | 20%        | 155                 | Rapid experimentation |

**Rule of thumb**: Set half-life ≈ expected time between provider changes

### Reward Weights

| Config          | Validity | Latency | Retry | Best For |
|-----------------|----------|---------|-------|----------|
| **Baseline**    | 0.50     | 0.30    | 0.20  | General production |
| Cost-aggressive | 0.40     | 0.40    | 0.20  | Batch, non-critical |
| Quality-first   | 0.70     | 0.15    | 0.15  | High-stakes |
| Retry-intolerant| 0.45     | 0.25    | 0.30  | Real-time |
| Speed-critical  | 0.35     | 0.50    | 0.15  | Interactive |

**Tuning process**:
1. Set validity = minimum acceptable quality
2. Distribute remaining weight by business priority
3. Validate with simulation
4. A/B test in production

---

## 🔍 Quick Diagnostics

### Is the router working?

**✅ Good signs**:
- Selection distribution changes over first 100 queries
- Posterior variance decreases over time
- One model emerges as favorite (if quality differs)
- In classical stationary simulations, cumulative regret grows roughly sublinearly

**⚠️ Warning signs**:
- Selection stays uniform (33/33/33) after 200 queries → weak reward signal
- Posterior variance increasing → decay too aggressive
- Selection distribution oscillates wildly → noisy rewards
- One model >95% forever → overconfidence or true dominance

### Health Check Queries

```python
# 1. Are posteriors converging?
if variance_is_decreasing(posterior_history):
    print("✓ Learning")
else:
    print("⚠ Check reward signal or decay")

# 2. Is reward signal meaningful?
reward_gap = max(avg_rewards) - min(avg_rewards)
if reward_gap > 0.1:
    print("✓ Distinguishing models")
else:
    print("⚠ Models too similar or reward noisy")

# 3. Is fallback triggering reasonably?
if 0.05 < fallback_rate < 0.20:
    print("✓ Appropriate fallback")
else:
    print("⚠ Tune confidence_floor")

# 4. Is exploration happening?
if all(selection_rate > 0.02 for all models):
    print("✓ All models explored")
else:
    print("⚠ Check shadow_rate or decay")
```

---

## 🚨 Common Pitfalls & Fixes

| Problem | Symptom | Likely Cause | Fix |
|---------|---------|--------------|-----|
| **Stuck on one model** | One model 95%+ forever | No decay, overconfident posterior | Add decay (γ=0.95) |
| **No convergence** | Uniform selection after 200 queries | Weak reward signal | Validate reward function, check validators |
| **Oscillating selection** | Selection swings wildly | Decay too aggressive or noisy rewards | Increase γ, smooth rewards |
| **High fallback rate** | Fallback >30% | confidence_floor too high or weak priors | Lower threshold or stronger priors |
| **Zero exploration** | Some models never selected | No shadow_rate, tight posteriors | Add shadow_rate=0.05 |
| **All models look same** | All means → 0.5 | Random validator or task too hard | Fix validator, simplify task |

---

## 🎤 Elevator Pitches

### 30 seconds
> "A Bayesian bandit treats each LLM as an arm. It keeps a probability distribution over each model's quality, samples from those distributions, picks the model with the highest sample, observes production telemetry, and updates its belief. This lets it route more traffic to cheaper models when evidence says they're good enough—no human labels needed."

### 2 minutes
> "I model each LLM as an arm in a multi-armed bandit. Instead of a single score, I maintain a Bayesian belief distribution over each model's quality. On every request, I sample once from each posterior and pick the model with the highest sample. After the response, I compute a reward from operational signals—did it parse correctly? How fast was it? Did we retry? Then I update the selected model's posterior. The system learns online without labels. I add expert priors for cold-start, decay for provider drift, and fallback when confidence is too low."

---

## 📊 Expected Performance

**Representative metrics for the current talk demo / simulation setup**:
- **Cost savings**: often around 40-50% vs. always using the strongest model
- **Latency improvement**: often around 20-30% when cheaper models are meaningfully faster
- **Validity / proxy-quality delta**: can be kept small in demo-like settings, but is task- and validator-dependent
- **Convergence time**: often around 20-50 queries with expert priors in simple simulations

**Simulation validation**:
```python
# Sanity checks for the current demo setup
# 1. In classical stationary simulations, cumulative regret
#    should grow roughly sublinearly
# 2. Best-model selection rate should increase over time
# 3. Posterior variance should usually decrease, or remain
#    bounded if decay is active
# 4. Average reward should remain consistent with your
#    simulated model qualities and reward weights
```

---

## 🔬 What's Exact vs. Heuristic

### Exact (mathematically rigorous)

✅ Beta distribution mean/variance formulas
✅ Thompson Sampling selection rule (sample → argmax)
✅ Binary Beta-Bernoulli update (if rewards were 0/1)
✅ Regret guarantees — **but only for classical stationary Bernoulli bandits, NOT for your implementation** (your continuous rewards, decay, and shadow rate violate the assumptions)

### Heuristic (pragmatic engineering)

⚠️ Continuous reward (not binary) → fractional pseudo-counts
⚠️ Decay (not pure Bayesian, addresses non-stationarity)
⚠️ Composite reward (proxy for quality, not ground truth)
⚠️ Shadow rate (forced exploration, not pure Thompson)
⚠️ Confidence fallback (lightweight, not full circuit breaker)

**How to say it**: 
> "Classical Thompson Sampling assumes binary rewards and stationary distributions. My implementation uses continuous proxy rewards and decay for non-stationarity, so it's a **production-oriented Thompson-Sampling-style router** rather than a theorem-preserving textbook bandit."

---

## 🆚 Comparison to Related Work

| Approach | Supervised? | Contextual? | Online? | When to Use |
|----------|-------------|-------------|---------|-------------|
| **FrugalGPT** | Yes (labels) | Possible | No | Have labeled data, deterministic cascade |
| **RouteLLM** | Yes (preferences) | Yes | No | Diverse queries, worth modeling context |
| **Your router** | No | No | Yes | **No labels, frequent drift, homogeneous traffic** |

**Future work**: Combine RouteLLM's contextual classifier + your Thompson adaptation

---

## 🧪 Pre-Talk Checklist

- [ ] Run simulations (notebook) to verify behavior
- [ ] Know Beta mean formula by heart
- [ ] Practice 30-second explanation
- [ ] Understand continuous reward nuance (Section 10)
- [ ] Know decay half-life formula
- [ ] Memorize failure modes table
- [ ] Can compare to FrugalGPT and RouteLLM
- [ ] Understand when context helps (future work)
- [ ] Know what's exact vs. heuristic
- [ ] Have monitoring dashboard plan ready

---

## 💬 Tough Q&A Responses (Pre-prepared)

**Q: Is the posterior mathematically exact?**
> "Not exactly. Classical Beta-Bernoulli assumes binary rewards. I use continuous rewards in [0,1], so the update is best viewed as fractional pseudo-counts—a pragmatic heuristic, not exact conjugacy."

**Q: How do you optimize cost if cost isn't in the reward?**
> "In the current demo, cost reduction emerges because cheaper models are faster and good enough. In production, I'd add an explicit cost penalty: `r = validity + latency + retry - λ·cost`."

**Q: Is this truly per-query routing?**
> "This is a non-contextual router—it learns at the traffic level. The natural next step is contextual routing using query features like length or embeddings."

**Q: What if parse success doesn't correlate with correctness?**
> "Then the proxy reward is weak and the method becomes unreliable. This works best with validators that correlate well with downstream correctness. I'd tailor the reward to the task."

**Q: Why Thompson Sampling instead of UCB?**
> "UCB is also reasonable. Thompson Sampling is Bayesian, handles priors naturally, and explores based on uncertainty rather than a deterministic confidence bound. It's also easier to explain visually."

---

**📌 Print this card. Keep it visible during the talk. You've got this.**
