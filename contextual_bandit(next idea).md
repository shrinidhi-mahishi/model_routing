In plain English, that means:

1. keep a running belief about each model
2. turn that belief into one temporary score for each model
3. route the request to the model with the best temporary score
4. see how that model actually performed
5. use that result to improve the next decision

### Same algorithm with a routing example

Suppose the router currently believes:

- `gpt-4o` is usually strong, but expensive and slower
- `gpt-4o-mini` is cheaper, but a bit less reliable
- `claude-haiku` is fast and promising, but still being learned

Now one new request arrives.

Step 1: the router looks at its current belief for each model.

Step 2: it draws one temporary score from each belief:

- `gpt-4o`: `0.81`
- `gpt-4o-mini`: `0.76`
- `claude-haiku`: `0.84`

Step 3: it picks `claude-haiku`, because `0.84` is the highest draw.

Step 4: it observes the real outcome:

- valid response
- no retry
- latency = 700 ms

So the final reward is high.

Step 5: it updates its belief about `claude-haiku`.

That means:

- `claude-haiku` becomes a little more trusted
- next time it becomes a little more likely to be selected again

If the result had been bad instead:

- invalid output
- retry needed
- slow latency

then the reward would be low, and the router would trust `claude-haiku` less next time.

That is the full learning loop.

### What information this router is actually using

At decision time, the current router uses:

- what it has learned so far about each model's average reward
- how confident it is in those estimates
- production signals from past requests such as validity, latency, and retries

It does **not** currently use:

- the meaning of the incoming question
- task type
- prompt embedding
- whether this specific request is easy or hard

So yes: **the current router chooses based on what it has learned about the models overall, not on the merit of the specific incoming question.**

Example:

- if `claude-haiku` has been earning the best recent overall reward, it may get chosen for many requests
- even if one particular incoming request is actually harder than average

That is why the current system is called **non-contextual**.

If you wanted routing based on the question itself, the next step would be a **contextual bandit** or another query-aware router.


refer: /Users/smahishi/Documents/micro_saas/model_routing/docs/bayesian_bandits_study_material.md


-------

13. Where Does Cost Enter the Picture?

This is another very important nuance.

In the current code

The composite reward uses:





validity



latency



retry behavior

It does not explicitly include dollar cost.

So why do we still see cost reduction?

Because in the simulation:





cheaper models are also faster



they are slightly weaker but often still good enough

So the reward indirectly favors cheaper models because:





they earn more latency reward



they remain sufficiently valid

The cost reduction emerges because:





faster, cheaper models are selected more often

If someone asks for a stricter cost-aware formulation

You can say:



In the current demo, cost is optimized indirectly through latency and model choice. In a production system, I would likely add an explicit cost penalty term to the reward, for example:

r = w_v * validity + w_l * latency + w_r * no-retry - lambda * cost

That is a very reasonable extension.