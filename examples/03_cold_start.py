"""Cold start — compare expert priors vs uniform (flat) priors.

Run:
    python examples/03_cold_start.py
"""

import random

import numpy as np

from bayesian_router import EXPERT_PRIORS, UNIFORM_PRIORS, ModelSimulator, Router


def main():
    seed = 42
    n_queries = 100
    sim = ModelSimulator()

    for label, priors in [("Expert", EXPERT_PRIORS), ("Uniform", UNIFORM_PRIORS)]:
        np.random.seed(seed)
        random.seed(seed)

        router = Router(models=priors)
        total_cost = 0.0

        for _ in range(n_queries):
            result = router.select()
            t = sim.call(result.model)
            router.update(
                result.model,
                latency_ms=t["latency_ms"],
                is_valid=t["is_valid"],
                retried=t["retried"],
            )
            total_cost += t["cost"]

        share = router.get_stats()["model_share"]
        line = "  ".join(f"{m}: {v:.0%}" for m, v in share.items())
        print(f"{label:>8s} priors  ·  cost=${total_cost:.4f}  |  {line}")

    print(
        "\nExpert priors converge faster — less money wasted on random exploration."
    )


if __name__ == "__main__":
    main()
