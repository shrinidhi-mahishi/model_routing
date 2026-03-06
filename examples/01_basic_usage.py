"""Basic usage — create a router, select models, update with telemetry.

Run:
    pip install -e ..
    python examples/01_basic_usage.py
"""

from bayesian_router import Router


def main():
    router = Router()  # zero-config with defaults

    for i in range(50):
        result = router.select()

        # In production you'd make the real LLM call here and capture
        # latency / validity / retry from the response.  We fake it:
        latency = 500 if "mini" in result.model or "haiku" in result.model else 2000
        router.update(
            result.model,
            latency_ms=latency,
            is_valid=True,
            retried=False,
        )

        if (i + 1) % 10 == 0:
            share = router.get_stats()["model_share"]
            line = "  ".join(f"{m}: {s:.0%}" for m, s in share.items())
            print(f"  Query {i + 1:>3d}  |  {line}")

    print("\nFinal distributions:")
    for name, state in router.get_distributions().items():
        print(
            f"  {name:<15s}  "
            f"α={state.alpha:6.1f}  β={state.beta:5.1f}  "
            f"conf={state.confidence:.2f}  "
            f"selected={state.selections}"
        )


if __name__ == "__main__":
    main()
