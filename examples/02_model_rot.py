"""Model rot — watch the router adapt when a model degrades.

Run:
    python examples/02_model_rot.py
"""

from bayesian_router import ModelSimulator, Router


def _share_str(router: Router) -> str:
    share = router.get_stats()["model_share"]
    return "  ".join(f"{m}: {v:.0%}" for m, v in share.items())


def main():
    router = Router()
    sim = ModelSimulator()

    print("Phase 1  ·  Normal operation  (queries 1-75)\n")
    for i in range(75):
        result = router.select()
        t = sim.call(result.model)
        router.update(
            result.model,
            latency_ms=t["latency_ms"],
            is_valid=t["is_valid"],
            retried=t["retried"],
        )
    print(f"  Traffic share:  {_share_str(router)}")

    sim.degrade("claude-haiku", factor=3.0)
    print("\n--- claude-haiku degraded (3× latency, lower validity) ---\n")

    print("Phase 2  ·  Post-degradation  (queries 76-225)\n")
    for i in range(150):
        result = router.select()
        t = sim.call(result.model)
        router.update(
            result.model,
            latency_ms=t["latency_ms"],
            is_valid=t["is_valid"],
            retried=t["retried"],
        )
    print(f"  Traffic share:  {_share_str(router)}")
    print("\n  Router adapted — traffic shifted away from degraded model.")


if __name__ == "__main__":
    main()
