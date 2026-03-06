Bayesian Model Routing

Title: The 50% Cheaper Agent: Autonomous LLM Routing with Bayesian Bandits
Session type: Talk (25 minutes talk + 10 min Q&A)
Track: Future Tech and Open Research

Abstract:
You're paying GPT-4 prices for queries a smaller model could handle. Model routing promises 40-70% savings.

But there's a catch: academic approaches assume humans label every response "good" or "bad." In production, nobody does that.

This talk shows how to build a Bayesian router using Thompson Sampling that learns without labels. The key: a Composite Reward Function that scores responses automatically — Did the output parse? How fast was it? Did the agent retry? Three signals, zero human effort, one score to update routing.

We tackle production realities research skips:
- Model rot: Adapting when a provider degrades using decaying memory
- Cold-start: Converging in 20 queries instead of 100 using expert priors
- Safety: Continuous shadow evaluation and circuit breakers to guarantee accuracy

Live demo: Watch the router learn to shift traffic from expensive to cheaper models in real-time.

Result: 40-50% cost cut, <1% accuracy drop.
Prerequisites: Python, LLM API familiarity.

