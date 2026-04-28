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
                validity_score=t["validity_score"],
                retry_count=t["retry_count"],
            )
            total_cost += t["cost"]
            if result.shadow_model:
                shadow_t = sim.call(result.shadow_model)
                router.update_shadow(
                    result.shadow_model,
                    latency_ms=shadow_t["latency_ms"],
                    validity_score=shadow_t["validity_score"],
                    retry_count=shadow_t["retry_count"],
                )
                total_cost += shadow_t["cost"]

        share = router.get_stats()["model_share"]
        line = "  ".join(f"{m}: {v:.0%}" for m, v in share.items())
        print(f"{label:>8s} priors  ·  cost=${total_cost:.4f}  |  {line}")

    print(
        "\nExpert priors converge faster — less money wasted on random exploration."
    )


if __name__ == "__main__":
    main()
