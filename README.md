# Bayesian Router

Autonomous LLM model routing with Thompson Sampling — cut API costs by 40-70% with <1% accuracy drop.

## Features

* **Label-free learning** — Composite reward from 3 objective signals (validity, latency, no-retry), zero human labels
* **Model rot adaptation** — Decaying memory detects provider degradation and reroutes automatically
* **Cold-start solution** — Expert priors from benchmarks converge in ~20 queries instead of 100
* **Safety guarantees** — Confidence fallback plus automated circuit-breaker states
* **Shadow evaluation** — Mirror 5% of traffic to a hidden candidate model
* **Framework-agnostic** — Works with any LLM API (OpenAI, Anthropic, Google, local models)

## Installation

```bash
pip install bayesian-router
```

With optional dependencies:

```bash
pip install bayesian-router[demo]    # Streamlit demo + Plotly charts
pip install bayesian-router[dev]     # pytest
pip install bayesian-router[all]     # Everything
```

## Quick Start

```python
from bayesian_router import Router

# Zero-config with sensible defaults
router = Router()

# Select model for next request
result = router.select()
print(f"Route to: {result.model}")

# After LLM call, update with telemetry — no human labels needed
reward = router.update(
    result.model,
    latency_ms=450,
    is_valid=True,
    retried=False,
)
print(f"Reward: {reward.total:.2f}")

# Optional: mirror a hidden candidate model on shadow traffic
if result.shadow_model:
    router.update_shadow(
        result.shadow_model,
        latency_ms=520,
        is_valid=True,
        retried=False,
    )
```

## With Any LLM

Bayesian Router manages routing decisions — you bring your own model client:

### OpenAI

```python
from openai import OpenAI
from bayesian_router import Router

client = OpenAI()
router = Router()

result = router.select()

response = client.chat.completions.create(
    model=result.model,
    messages=[{"role": "user", "content": "Hello"}],
)

router.update(
    result.model,
    latency_ms=response.usage.completion_tokens * 10,  # rough proxy
    is_valid=True,
    retried=False,
)

if result.shadow_model:
    shadow = client.chat.completions.create(
        model=result.shadow_model,
        messages=[{"role": "user", "content": "Hello"}],
    )
    router.update_shadow(
        result.shadow_model,
        latency_ms=shadow.usage.completion_tokens * 10,
        is_valid=True,
        retried=False,
    )
```

### Anthropic

```python
from anthropic import Anthropic
from bayesian_router import Router, ModelConfig

client = Anthropic()
router = Router(models={
    "claude-sonnet": ModelConfig(alpha=8, beta=3, cost_per_1k=0.003),
    "claude-haiku":  ModelConfig(alpha=5, beta=4, cost_per_1k=0.00025),
})

result = router.select()
response = client.messages.create(model=result.model, ...)
router.update(result.model, latency_ms=..., is_valid=..., retried=...)
if result.shadow_model:
    shadow = client.messages.create(model=result.shadow_model, ...)
    router.update_shadow(
        result.shadow_model, latency_ms=..., is_valid=..., retried=...
    )
```

## Custom Reward Weights

```python
from bayesian_router import Router, CompositeReward

# Prioritise validity over latency for high-stakes applications
reward = CompositeReward(
    validity_weight=0.70,
    latency_weight=0.15,
    retry_weight=0.15,
    latency_midpoint_ms=3000,
)

router = Router(reward_fn=reward)
```

## Model Rot Handling

The router uses **decaying memory** (exponential discounting) so recent
observations weigh more than old ones.  If a provider ships a regression
overnight, the router adapts within minutes:

```python
router = Router(
    gamma=0.90,          # Stronger decay (default 0.95)
    decay_interval=30,   # Apply every 30 queries (default 50)
)
```

## Cold Start with Expert Priors

```python
from bayesian_router import Router, EXPERT_PRIORS, UNIFORM_PRIORS

# Expert priors from public benchmarks — converge fast
router_fast = Router(models=EXPERT_PRIORS)

# Uniform priors — maximum uncertainty, needs ~100 queries
router_slow = Router(models=UNIFORM_PRIORS)
```

## Health Monitoring

```python
stats = router.get_stats()
print(stats["model_share"])     # {"gpt-4o": 0.25, "gpt-4o-mini": 0.15, ...}
print(stats["distributions"])   # {"gpt-4o": "α=12.3 β=4.1", ...}

for name, state in router.get_distributions().items():
    print(f"{name}: confidence={state.confidence:.2f}, selected={state.selections}")
```

## Examples

See the `examples/` folder for complete working demos:

| Example | Description |
|---------|-------------|
| `01_basic_usage.py` | Create a router, select models, update with telemetry |
| `02_model_rot.py` | Watch the router adapt when a model degrades |
| `03_cold_start.py` | Expert priors vs uniform — convergence speed comparison |
| `04_streamlit_demo.py` | Interactive demo with live charts (DevConf talk) |

### Running Examples

```bash
git clone https://github.com/shrinidhi-mahishi/bayesian-router.git
cd bayesian-router
python -m venv venv && source venv/bin/activate
pip install -e ".[all]"

python examples/01_basic_usage.py
streamlit run examples/04_streamlit_demo.py
```

## API Reference

### Router

| Method | Description |
|--------|-------------|
| `select()` | Pick a primary model and optional shadow model → `RoutingResult` |
| `update(model, *, latency_ms, is_valid, retried)` | Update Beta distribution → `RewardResult` |
| `update_shadow(model, *, latency_ms, is_valid, retried)` | Update mirrored shadow telemetry → `RewardResult` |
| `get_distributions()` | Current α/β/confidence for every model |
| `get_stats()` | Summary statistics (JSON-serialisable) |

### CompositeReward

| Method | Description |
|--------|-------------|
| `compute(latency_ms, is_valid, retried)` | Score a single response → `RewardResult` |

### ModelSimulator

| Method | Description |
|--------|-------------|
| `call(model, tokens=500)` | Simulate an LLM call → telemetry dict |
| `degrade(model, factor)` | Inject model rot |
| `reset(model)` | Remove degradation |

## Configuration

```python
router = Router(
    models=EXPERT_PRIORS,       # Model priors (or custom dict)
    reward_fn=CompositeReward(),# Reward function
    gamma=0.95,                 # Memory decay factor
    decay_interval=50,          # Apply decay every N queries
    confidence_floor=0.50,      # Safety floor before serving a model
    shadow_rate=0.05,           # Fraction of mirrored shadow traffic
    fallback_model="gpt-4o",    # Trusted model for fallback
    circuit_window_size=5,      # Recent outcomes tracked per model
    circuit_failure_threshold=3,# Failures needed to open a breaker
    circuit_reset_queries=20,   # Cooldown before half-open probe
    half_open_max_requests=2,   # Successful probes to close breaker
)
```

## License

MIT
