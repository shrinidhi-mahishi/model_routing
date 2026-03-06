# DevConf.CZ 2026 — Proposal Drafts

CfP deadline: **March 8, 2026**
Submit at: https://pretalx.devconf.info/devconf-cz-2026/submit/

---

## Proposal 1: Bayesian Model Routing

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

Notes (for organizers only, not public):
This talk distills a production deployment pattern from a larger body of work on LLM agent optimization. It bridges the gap between AI research (RouteLLM, FrugalGPT) and production engineering by introducing label-free proxy rewards. Attendees will gain an architectural blueprint and shareable Python code to immediately cut their LLM API bills. Accuracy is maintained through three safety mechanisms: confidence-based fallback, automated circuit breakers, and continuous shadow evaluation on 5% of traffic.

---

## Submission Tips (reminders)

- Abstract limit is **1000 characters** — both drafts above are within limit
- Do NOT include names, job titles, or company references in the abstract (blind review)
- You can edit proposals after submission until March 8
- Slides don't need to be ready at submission time
- Max 3 proposals per person, max 1 accepted 35-min talk per speaker
- Consider also submitting one as a **lightning talk** (15 min) to increase overall acceptance odds
