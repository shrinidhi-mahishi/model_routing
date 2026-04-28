# Bayesian Bandits Study Material for the Model Routing Talk

## Purpose of This Document

This file is for **your understanding**, not for the audience.

Think of it as a study guide for the talk, not as a script.

The goal is to help you answer questions like:

- What exactly is a Bayesian bandit?
- Why does Thompson Sampling make sense for model routing?
- Why are we using Beta distributions?
- What math is exact, and what is only a practical heuristic?
- How does the code in `model_routing/` map to the theory?
- What technical questions might the audience ask?

This guide is intentionally more detailed and more mathematical than the presentation guide in:

- `model_routing/docs/devconf_model_routing_talk_guide.md`

Relevant implementation files:

- `model_routing/bayesian_router/router.py`
- `model_routing/bayesian_router/rewards.py`
- `model_routing/bayesian_router/simulator.py`
- `model_routing/examples/04_streamlit_demo.py`

---

## Start Here If This Is Your First Read

Do **not** try to memorize the whole document on your first pass.

Use this order instead:

1. Read sections 1-3 to understand the problem and the intuition.
2. Read section 5 to understand Thompson Sampling at a high level.
3. Read sections 9-16 to understand how the theory maps to the actual router, but feel free to skip sections 10.1 and 12.1 on the first pass.
4. Read sections 27, 28, and 32 so you can explain it simply.
5. Come back to sections 4, 6, 7, 8, 10.1, and 12.1 only if you want the math details.

### The 30-second mental model

The router does five things:

1. Start with a belief about each model.
2. Sample one score from each model's current belief.
3. Pick the model with the highest sampled score.
4. Turn the result into a reward using validity, latency, and retries.
5. Update the chosen model so the next decision is better informed.

### Key terms in plain English

- **arm** = one candidate model
- **prior** = starting belief before much data arrives
- **posterior** = updated belief after observing outcomes
- **reward** = the score the router gives a response
- **exploration** = trying uncertain models to learn more
- **exploitation** = using the model that currently looks best
- **regret** = how much reward you missed compared with always picking the best option

### What is exact and what is pragmatic

- **Exact textbook idea**: Thompson Sampling with binary rewards and Beta-Bernoulli updates
- **Production adaptation**: continuous rewards, fractional pseudo-counts, decay, fallback, and shadow exploration

### Symbols you will see often

- `alpha` = accumulated success evidence
- `beta` = accumulated failure evidence
- `theta` = an unknown model quality
- `r` = reward for one request

---

## Math Words in Plain English

If words like **distribution**, **prior**, or **posterior** feel abstract, use the translations below.

### Distribution

A **distribution** is not one score. It is a picture of which scores seem plausible.

Example:

- For `gpt-4o`, you might believe the true reward is probably somewhere around `0.75` to `0.85`
- For `gpt-4o-mini`, you might believe it could be anywhere from `0.50` to `0.90`

Both models might have a similar average, but the second one is much more uncertain.

So when this guide says "distribution," read it as:

> A range of believable values, plus how confident we are about them.

### Prior

A **prior** is your starting belief before you have much data.

Example:

- You may start with the belief that `gpt-4o` is probably stronger than `gpt-4o-mini`
- That belief becomes the initial `alpha` and `beta`

So a prior is just:

> What we believe at the beginning.

### Posterior

A **posterior** is the updated belief after seeing real outcomes.

Example:

- If `gpt-4o-mini` keeps returning valid, fast answers
- your belief about it improves
- its posterior becomes better than its prior

So a posterior is just:

> What we believe now, after seeing evidence.

### Uncertainty

**Uncertainty** means how unsure the router still is.

Example:

- If a model has only handled 3 requests, you do not know much yet
- If a model has handled 300 requests, you know much more

High uncertainty means:

- "I need more evidence"

Low uncertainty means:

- "I have seen enough to trust this estimate more"

### Sample

A **sample** is one temporary score drawn from the current belief distribution.

Example:

- the router may draw `0.81` for `gpt-4o`
- and `0.84` for `gpt-4o-mini`

Even if `gpt-4o` is usually stronger, `gpt-4o-mini` can still win that round because it is uncertain and got a lucky draw.

That is how exploration happens naturally.

### Mean

The **mean** is the center of your current belief.

Example:

- if a model's current mean is `0.72`, the router currently thinks its typical reward is around `0.72`

### Variance / spread

The **variance** tells you how spread out the belief is.

Example:

- narrow spread = "I am pretty sure"
- wide spread = "I am still guessing"

You do not need to memorize the variance formula on a first read. Just remember:

> bigger spread = more uncertainty

---

## 1. The Big Picture

### The problem

Suppose you have multiple LLMs:

- one expensive but very strong
- one cheap and decent
- one cheaper and faster but slightly less reliable

If you always send traffic to the strongest model:

- quality stays high
- cost is high
- latency may be higher than necessary

If you always send traffic to the cheapest model:

- cost is low
- quality may be unstable
- retries/failures may go up

So you need a decision rule:

> For each request, which model should I choose?

### The challenge

A classic academic answer is:

- collect labels saying whether the model output was good or bad
- train a router from those labels

But in real systems:

- people do not label every response
- labels are delayed or unavailable
- providers change behavior over time
- routing policy needs to adapt online

### The solution in your talk

Use a **Bayesian bandit** approach:

- treat each model as an "arm"
- keep a probability distribution over how good each model currently is
- sample from those distributions to decide which model to try
- observe production signals
- convert those signals into a reward
- update your beliefs

Over time, the system learns which model gives the best tradeoff.

---

## 2. What Is a Multi-Armed Bandit?

The name comes from slot machines.

- A slot machine is sometimes called a "one-armed bandit"
- A casino with multiple slot machines becomes a "multi-armed bandit" problem

Each machine:

- gives rewards with some unknown probability
- may be good or bad

Your goal:

- maximize reward over repeated choices

### Exploration vs exploitation

This is the core tension.

**Exploration** means:

- try different options
- gather information
- maybe discover that a cheaper model is actually good enough

**Exploitation** means:

- use the option that currently seems best
- stop wasting time on weaker models if you already know enough

If you explore too much:

- you waste money and performance on weak models

If you exploit too much:

- you may never discover that a cheaper model is good enough
- you may fail to adapt when model quality changes

That is why bandits are useful: they solve this exploration/exploitation tradeoff.

---

## 3. What Makes a Bandit "Bayesian"?

In a Bayesian approach, you do not store just one score for each model.

You store two things:

- your current guess about how good the model is
- how confident you are in that guess

That combination is what this guide calls a **distribution**.

If the word "distribution" feels too mathematical, read it like this:

> a range of believable scores, plus how sure we are about them

### Tiny routing example

Suppose:

- `gpt-4o` has handled 500 requests and currently looks like it has an average reward(current guess of how good each model is) around `0.82`
- `gpt-4o-mini` has handled only 5 requests and currently looks like it has an average reward(current guess of how good each model is) around `0.78`

A simple average-based router says:

- `0.82` is bigger than `0.78`
- so always choose `gpt-4o`

A Bayesian router says:

- `gpt-4o` probably is better
- but I have much less evidence about `gpt-4o-mini`
- so I should still test `gpt-4o-mini` sometimes

Why?

- it is cheaper
- the evidence for it is still weak
- a few more good outcomes could change the decision

This is the important part:

- if a model has only been tried a few times, uncertainty is high
- if a model has been tried many times, uncertainty is lower

That uncertainty is what drives intelligent exploration.

### Frequentist intuition

A non-Bayesian approach might say:

- model A average reward = 0.82
- model B average reward = 0.78

So always choose model A.

### Bayesian intuition

A Bayesian approach says:

- model A looks better, and we have a lot of evidence
- model B looks slightly worse, but we do not have much evidence yet
- therefore sometimes it is still worth testing model B

That uncertainty-aware decision-making is the reason Bayesian bandits are attractive.

In plain English:

> A Bayesian bandit does not just ask "Which model looks best?"
> It asks "Which model looks best, and how sure am I about that?"

---

## 4. Formal Bandit Setup

If this is your first read, you can skim this section. It just expresses the bandit problem in compact notation.

Let there be `K` models.

**Action set**

`A = {1, 2, ..., K}`

**At time step `t`**

- choose an action `a_t in A`
- observe reward `r_t`

**Objective**

`maximize sum_{t=1..T} r_t`

**Regret formulation**

`R_T = T * mu_star - sum_{t=1..T} mu_{a_t}`

where:

- `mu_star` is the expected reward of the optimal arm
- `mu_{a_t}` is the expected reward of the chosen arm

In simple words:

- regret is how much reward you lose by not always choosing the best option

In your case:

- each arm = one LLM
- each pull = routing one request to one model
- reward = composite score from validity, latency, retry behavior

---

## 4.1 What About Regret Guarantees?

If someone asks a more theoretical question like:

> Does Thompson Sampling have a formal regret guarantee?

the short answer is:

- **yes in the classical setting**
- **not directly for every engineering modification in this implementation**

### What "formal regret guarantee" means

A **formal regret guarantee** is a mathematical statement about how much reward the algorithm is expected to lose while it is learning.

In simple terms:

- the algorithm may make some bad choices early
- but it should not keep wasting traffic forever
- over time, it should get closer and closer to the behavior of the best possible arm

Tiny example:

- imagine the best model would have earned total reward `100`
- your router earned total reward `92`
- then the regret is `100 - 92 = 8`

If an algorithm has a strong regret guarantee, that means this "lost reward while learning" grows slowly as the number of requests grows.

The important intuition is:

> regret is the price you pay for learning

### Where people use this idea

Regret guarantees are mainly used in:

- academic papers on bandits and online learning
- theoretical comparisons between algorithms like Thompson Sampling, UCB, and epsilon-greedy
- technical discussions about whether an algorithm learns efficiently over time

In real production systems, teams usually do **not** compute regret directly on every request.

Instead, they use regret guarantees as:

- a theoretical reason to trust the algorithm family
- a way to explain why the method should improve over time
- a clean benchmark in the textbook setting

### Why this matters for your problem

In your project, the real problem is:

- you must choose between multiple LLMs
- you want high quality, but also lower cost and latency
- you want the router to learn from live traffic over time

This section matters because it tells the reader:

- Thompson Sampling is not just a random heuristic
- it comes from a well-studied family of bandit algorithms
- in the clean textbook version, it has strong mathematical guarantees

That does **not** mean your exact production router has the same theorem.

It means:

- the core idea is theoretically grounded
- your implementation is an adaptation of that idea for real-world telemetry
- this is why it is reasonable to start from Thompson Sampling in the first place

### Classical theory

For standard Bernoulli bandits:

Here, **Bernoulli** just means each outcome has only two possible values:

- `1` = success
- `0` = failure

Tiny example:

- if a model returns valid JSON, call that `1`
- if it returns invalid JSON, call that `0`

So a Bernoulli bandit is the simple textbook case where every round ends in a yes/no outcome, not a continuous score like `0.73` or `0.91`.

- binary rewards
- stationary reward distributions
- exact Beta-Bernoulli posterior updates

Thompson Sampling has well-known **sub-linear regret** guarantees.

Informally, that means:

- the algorithm makes mistakes early
- but the average loss per decision shrinks over time
- it learns efficiently instead of wasting traffic forever

You do **not** need to quote a specific theorem in the talk unless someone asks.

If they do ask, the safe answer is:

> In the classical Bernoulli bandit setting, Thompson Sampling is known to achieve sub-linear regret, which is one reason it is such a popular practical choice.

What that phrase means:

- **classical** = the textbook version, not a production-modified version
- **Bernoulli** = each outcome is binary: either `1` (success) or `0` (failure)
- **bandit setting** = at each step, choose one arm, observe one reward, then learn from it

So in plain English, the classical Bernoulli bandit setting means:

- each model/arm has some fixed unknown success rate
- every round gives a simple yes/no outcome
- the success rate does not drift over time
- the update rule is the exact Beta-Bernoulli update

Tiny example:

- arm A succeeds 80% of the time
- arm B succeeds 60% of the time
- each time you choose one arm, you only observe `success` or `failure`

That clean textbook setup is what the formal regret guarantees refer to.

### Why the guarantee does not transfer cleanly here

Your current implementation differs from the textbook setting in at least three ways:

1. rewards are continuous in `[0, 1]`, not binary
2. updates use fractional pseudo-counts rather than exact Beta-Bernoulli conjugacy
3. decay is introduced to handle non-stationarity

Because of that:

- classical regret results do not apply **as-is**
- your implementation should be presented as a **pragmatic adaptive heuristic inspired by Thompson Sampling**: it uses the same core idea, but adjusts it for production realities like continuous rewards, fractional updates, and decay

Here, **heuristic** means your router is using a practical adaptive strategy inspired by Thompson Sampling, rather than the exact textbook version with all the original assumptions and guarantees.

### Best answer if someone presses on this

Use this:

> Classical Thompson Sampling has strong regret guarantees for stationary Bernoulli bandits. My implementation intentionally departs from that textbook setting because production telemetry is continuous and model quality is non-stationary. So I would not claim the same formal regret bound here. I would describe this as a production-oriented Thompson-Sampling-style router rather than a theorem-preserving textbook bandit.

That is technically correct and sounds strong, not defensive.

---

## 5. Why Thompson Sampling?

Thompson Sampling is one of the most practical Bayesian bandit algorithms.

First-time takeaway:

- it is the mechanism that decides when to trust the current winner
- and when to still give a cheaper or less-tested model a chance
- in this project, it learns from past **reward** signals, not from reading the content of the incoming question

### The algorithm

For each arm:

1. maintain a posterior distribution over its quality
2. sample one value from that posterior
3. pick the arm with the highest sample
4. observe reward
5. update your belief about the model using the newly observed reward

Here, **posterior** means the router's current belief about how good a model is after incorporating the rewards it has observed so far.

A very compact way to remember **posterior** is:

- `prior` = belief before data
- `posterior` = belief after data

Repeat.

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

### Tiny routing example

Suppose the router draws these temporary scores:

- `gpt-4o`: `0.81`
- `gpt-4o-mini`: `0.76`
- `claude-haiku`: `0.84`

It picks `claude-haiku` because `0.84` is the highest draw.

Now imagine the next request gives:

- `gpt-4o`: `0.79`
- `gpt-4o-mini`: `0.88`
- `claude-haiku`: `0.74`

This time it picks `gpt-4o-mini`.

That is the key idea:

- models that already look strong win often
- models that are still uncertain get occasional chances
- weak models slowly get fewer chances as evidence accumulates

### How Thompson Sampling solves the bandit problem here

In this project, the bandit problem is:

- if you always choose `gpt-4o`, quality is high but cost and latency stay high
- if you always choose the cheapest model, cost is low but failures and retries may increase
- you need to learn the best tradeoff **while the system is live**

Thompson Sampling helps because it gives you a practical way to balance:

- **exploitation**: use the model that currently looks best
- **exploration**: still test models that might be better than they currently appear

Here is what that looks like in this router:

1. Keep a belief for each model based on past rewards.
2. Give strong, well-tested models many chances.
3. Give uncertain models occasional chances.
4. Observe the real outcome using validity, latency, and retry behavior.
5. Update the belief and let future traffic shift automatically.

### Concrete example in this project

Imagine this situation:

- `gpt-4o` is known to be strong, but it is slower and more expensive
- `claude-haiku` is cheaper and faster, but you are less sure about it

If you always choose `gpt-4o`:

- you never learn whether `claude-haiku` is already good enough
- you miss potential savings

If you always force traffic to `claude-haiku`:

- you may hurt quality when it is not reliable enough

Thompson Sampling does the middle ground:

- `gpt-4o` still wins often because it has strong evidence behind it
- `claude-haiku` still gets chances because it is uncertain and might be better than expected
- if `claude-haiku` keeps giving valid, fast answers, its score improves and it gets more traffic
- if it performs badly, its score drops and the router sends less traffic to it

So the router does **not** need a hard-coded rule like:

- "always use the expensive model"
- "always use the cheapest model"

Instead, it learns from live traffic and gradually shifts toward the best tradeoff.

### More examples from this use case

#### Example 1: cold start on day one

At the beginning, you do not have much live traffic data yet.

- `gpt-4o` starts with a stronger prior
- so it gets trusted more early on
- but `gpt-4o-mini` and `claude-haiku` still get some chances

This means the router is cautious at first, but it can still discover that cheaper models are already good enough.

#### Example 2: stable structured-output workload

Suppose your workload is mostly:

- extraction
- JSON output
- tool-friendly structured responses

If `claude-haiku` keeps producing:

- valid outputs
- low latency
- few retries

then its reward keeps looking strong, so Thompson Sampling will naturally send it more traffic over time.

That is how cost savings emerge without hard-coding:

- "use `claude-haiku` for everything"

#### Example 3: provider degradation

Suppose `gpt-4o-mini` was doing well last week, but now:

- latency increases
- retries increase
- validity drops

Then the reward for `gpt-4o-mini` drops.

As those lower rewards accumulate:

- its posterior gets worse
- Thompson Sampling chooses it less often
- traffic shifts toward stronger alternatives

So the router can adapt when provider behavior changes.

#### Example 4: safety-oriented fallback

Suppose a cheaper model is still uncertain or starts failing.

Even if Thompson Sampling selects it sometimes:

- the confidence floor can still trigger fallback
- the request can be routed to the safer trusted model

So Thompson Sampling does not work alone. It works together with your fallback policy.

### Why this works intuitively

Suppose:

- a model looks strong and you are confident in it
- its posterior is concentrated near high values

Here, **posterior** means:

- the router's current updated belief about that model
- after taking past outcomes into account

So "posterior is concentrated near high values" means:

- the router currently believes the model is good
- and it is fairly confident about that belief

Then its random sample will often be high.

Here, **random sample** means:

- Thompson Sampling draws one temporary score from the model's current belief distribution
- that score is random, but it is guided by what the router currently believes

So if the router thinks a model is usually good, and is fairly sure about that:

- sampled values like `0.80`, `0.84`, or `0.88` are more likely
- a very low sampled value is less likely

That is why strong, well-understood models tend to win often.

If another model is uncertain:

- sometimes its sample will also be high
- so it still gets explored

That gives you a natural balance:

- strong models get used often
- uncertain models still get tested occasionally

### Why people like Thompson Sampling

- simple to implement
- uncertainty-aware
- empirically strong
- easy to explain visually

### Other bandit algorithms

Thompson Sampling is not the only option.

Some other common bandit algorithms are:

#### 1. Greedy / always pick the current best

Idea:

- always choose the model with the best current average reward

Why it is weak here:

- it may get stuck on `gpt-4o`
- it may never learn that a cheaper model became good enough

#### 2. Epsilon-greedy

Idea:

- most of the time, choose the current best model
- with a small probability `epsilon`, choose a random model

Why people use it:

- very simple

Why Thompson Sampling is often better here:

- epsilon-greedy explores randomly
- Thompson Sampling explores according to uncertainty
- so Thompson Sampling usually wastes less traffic on obviously weak models

#### 3. UCB (Upper Confidence Bound)

Idea:

- choose the model with the best optimistic score
- roughly: `average reward + exploration bonus`

Why people use it:

- strong theory
- deterministic exploration rule

Why Thompson Sampling is often preferred in this project:

- easier Bayesian story
- natural use of priors
- simpler to explain as "sample from belief, then pick the winner"

#### 4. Contextual bandits

Idea:

- use features of the incoming request before choosing a model

Example:

- one model for long reasoning prompts
- another for short structured extraction tasks

Why this matters:

- this is how you would start routing on the **merit of the question itself**
- your current router does **not** do this yet

---

## 6. Why Beta Distributions?

This section answers a very practical question:

> In the simple yes/no version of the problem, how do we keep score for each model?

In the classical bandit setup, rewards are binary:

- success = `1`
- failure = `0`

Example:

- valid JSON -> `1`
- invalid JSON -> `0`

For Bernoulli rewards, the standard prior is the **Beta distribution**.

### What a Beta distribution is

A **Beta distribution** is a probability distribution for values between `0` and `1`.

In this problem, that means:

- it is a way to represent all the success rates a model might realistically have
- it tells us which success rates seem more believable
- it also tells us how confident we are

Example:

- if the router thinks a model probably succeeds around `0.80`
- but is not fully sure yet
- the Beta distribution represents that belief

So when this guide says:

- `Beta(8, 3)`

you can read it informally as:

- "this model probably works fairly often"
- "and I have a moderate amount of evidence for that belief"

Why Beta is useful here:

- success rates naturally live between `0` and `1`
- Beta is built exactly for that range
- it can represent both **average quality** and **uncertainty** at the same time

Difference between those two phrases:

- **binary rewards** describes what you record: each result is either `0` or `1`
- **Bernoulli rewards** describes the probability model behind it: each request is treated like a yes/no trial with some unknown probability of success

Tiny example:

- if a model has a 90% chance of returning valid JSON
- then each request is a Bernoulli trial with success probability `0.90`
- the reward you actually record is still just binary: either `1` or `0`

So:

- "binary" describes the output you see
- "Bernoulli" describes how you model that output mathematically

### Beta in plain English

You do **not** need to think of Beta as a scary math object.

For a first read, think of it as:

> a scorecard that stores both "how good does this model look?" and "how sure am I?"

If you only remember one thing, remember this:

- `alpha` = good evidence
- `beta` = bad evidence
- `alpha / (alpha + beta)` = current best guess of the success rate
- larger `alpha + beta` = more confidence in that guess

### Tiny intuition examples

- `Beta(1, 1)` means: "I basically have no idea yet."
- `Beta(8, 3)` means: "This model usually looks good."
- `Beta(80, 30)` means: "This model still looks good, and now I am much more certain."

So:

- larger `alpha` relative to `beta` pushes belief upward
- larger `alpha + beta` means more confidence

### Optional notation

If `theta in [0, 1]`, then:

`theta ~ Beta(alpha, beta)`

Here `theta` just means:

- the model's true success rate
- which we do not know yet

### Most important properties

**Mean (current best guess)**

`E[theta] = alpha / (alpha + beta)`

**Variance (uncertainty)**

`Var(theta) = [alpha * beta] / [(alpha + beta)^2 * (alpha + beta + 1)]`

You do **not** need to memorize the variance formula.

Just remember:

- narrow distribution = "I have seen enough data to be more confident"
- wide distribution = "I still need more evidence"

### Why Beta is convenient

Because Beta is the **conjugate prior** for Bernoulli rewards.

Plain-English meaning:

- start with a Beta belief
- observe success/failure data
- updated belief is still Beta

So you can keep reusing the same simple two-number scorecard: `alpha` and `beta`.

### How this section fits into the full proposal

Important note:

- this section explains only the **learning engine**
- it is **not** the whole proposal by itself

What this section covers:

- beliefs
- Bayes-style belief updates
- Thompson Sampling decisions

That is the mathematical core of the router.

But the **full proposal** is broader:

1. compute a **label-free composite reward** from production signals such as validity, latency, and retries
2. use a **Bayesian bandit router** to learn from those rewards online
3. add production features such as **expert priors**, **decaying memory**, **fallback/circuit breakers**, and **shadow evaluation**

So if someone asks for the gist of the whole proposal, the shortest accurate answer is:

> Use label-free proxy rewards to drive a Bayesian router that shifts traffic toward cheaper models while staying adaptive and safe in production.

### What a belief means here

In this document, a **belief** just means:

- the router's current guess about how good a model is
- plus how sure or unsure it is about that guess

So when we say:

- "the router has a belief about `gpt-4o-mini`"

we simply mean:

- it has some current opinion about `gpt-4o-mini`
- and that opinion can become stronger or weaker as more results arrive

Tiny example:

- if `claude-haiku` has been giving valid, fast answers, the router's belief in it improves
- if it starts failing or needing retries, the router's belief in it gets worse

So you can read **belief** as:

`current learned opinion`

### What Bayes theorem means

Bayes theorem is a rule for updating a belief after seeing new evidence.

The full idea is:

`posterior = [likelihood * prior] / evidence`

You do not need to memorize that formula.

The simple meaning is:

- start with a belief
- observe new evidence
- update the belief

So Bayes theorem is basically:

`old belief + new evidence -> better updated belief`

Tiny everyday example:

- before looking outside, you think there is a `30%` chance of rain
- then you see dark clouds
- now you raise your belief, maybe to `70%`

That shift is the Bayes idea:

- you did not restart from zero
- you updated your earlier belief using new evidence

### How Bayes theorem is used in this use case

If you only remember one thing, remember this:

- Bayes theorem is the rule the router uses to **revise its belief** about a model after seeing a result
- it is not about understanding the text of the question
- it is about updating "how good do I now think this model is?"

The compact idea is:

`posterior ∝ likelihood * prior`

In simple words:

- `prior` = what the router believed before this request
- `likelihood` = how much the new outcome supports or weakens that belief
- `posterior` = the router's new belief after combining old belief and new evidence

For model routing, that means:

1. start with an initial belief about each model
2. send one request to a chosen model
3. observe how it performed
4. update the belief for that model
5. use that updated belief on the next routing decision

So Bayes theorem is the learning step behind:

`old belief + new evidence -> updated belief`

### Same idea with a concrete routing example

Suppose the router currently believes `gpt-4o-mini` is promising but not fully proven yet.

For example:

- current belief: `Beta(5, 4)`

That means:

- the router is somewhat positive about `gpt-4o-mini`
- but it is still learning

Now the router sends one real request to `gpt-4o-mini`.

#### Case 1: the result is good

Suppose the model gives:

- valid output
- no retry
- acceptable latency

Then this new result acts like evidence that:

- "yes, this model may actually be good"

So the belief shifts upward.

In the textbook binary case, you can think of it like this:

- before: `Beta(5, 4)`
- after success: `Beta(6, 4)`

Meaning:

- the router trusts `gpt-4o-mini` a little more than before
- so next time it becomes more likely to get traffic

#### Case 2: the result is bad

Suppose the model gives:

- invalid output
- retry needed
- poor latency

Then this new result acts like evidence that:

- "this model may not be as reliable as we thought"

So the belief shifts downward.

In the textbook binary case:

- before: `Beta(5, 4)`
- after failure: `Beta(5, 5)`

Meaning:

- the router trusts `gpt-4o-mini` a little less
- so it becomes less likely to be selected next time

### Why this matters in your project

This is exactly how the router learns things like:

- "maybe `claude-haiku` is already good enough for structured output"
- "maybe `gpt-4o-mini` has recently degraded"
- "maybe the expensive model is not worth using for every request"

Without Bayes-style updating, the router would need:

- fixed rules
- manual thresholds
- or human relabeling

With Bayes-style updating, the router can keep learning online from real production outcomes.

### Important practical note

In the **textbook binary version**, Bayes theorem is exact:

- success updates `alpha`
- failure updates `beta`

In **your current production-oriented router**, the reward is not pure yes/no.

Instead it uses a bounded composite reward built from:

- validity
- latency
- retries

So the code uses the **same Bayes-style idea**:

- stronger evidence pushes belief up
- weaker evidence pushes belief down

But the exact update is a **fractional pseudo-count heuristic**, not the strict textbook Beta-Bernoulli formula.

That is the simplest accurate way to say it:

> Bayes theorem is used here as the belief-update mechanism. After each request, the router combines its previous belief about a model with new evidence from that model's outcome, then uses the updated belief to make better routing decisions on later requests.

---

## 7. Bayes Update for Binary Rewards (Optional Derivation)

This section answers:

> After one yes/no result, how do `alpha` and `beta` change?

First-time takeaway:

- if the outcome is a success, increase `alpha`
- if the outcome is a failure, increase `beta`
- the derivation below just proves that this simple rule is mathematically correct

### Tiny example

Suppose a model currently has:

- `Beta(5, 4)`

If the next request is a success:

- it becomes `Beta(6, 4)`

If the next request is a failure:

- it becomes `Beta(5, 5)`

That is the whole idea:

- success adds good evidence
- failure adds bad evidence

If the symbols feel heavy, skip directly to **The simple update rule** below.

### Optional derivation

Assume:

- prior: `theta ~ Beta(alpha, beta)`
- observed rewards: `r_1, ..., r_n`, where each `r_i` is either `0` or `1`

Let:

- `S = sum_{i=1..n} r_i`
- so `S` is the number of successes

Then after combining:

- the Bernoulli likelihood
- and the Beta prior

the posterior stays in the same family:

`theta | D ~ Beta(alpha + S, beta + n - S)`

### The simple update rule

If reward is binary:

- if success: `alpha <- alpha + 1`
- if failure: `beta <- beta + 1`

This is the mathematically exact Beta-Bernoulli update.

---

## 8. Thompson Sampling Step-by-Step

This section answers:

> How does the textbook version of the router choose a model for one request?

First-time takeaway:

- draw one temporary score per model
- choose the model with the highest temporary score
- after you see the result, update only that chosen model

### How Thompson Sampling is related to Bayes theorem in this use case

If this is your first read, the simplest way to think about it is:

- **Bayes theorem** tells the router how to **learn from evidence**
- **Thompson Sampling** tells the router how to **act using what it has learned**

So they are related, but they are not the same thing.

They play two different roles:

- Bayes theorem answers: **"How should my belief change after I see a result?"**
- Thompson Sampling answers: **"Given my current beliefs, which model should I try right now?"**

In this project, the loop is:

1. start with a prior belief for each model
2. route one request
3. observe the outcome
4. use Bayes-style updating to turn the old belief into a new belief
5. use Thompson Sampling to sample from those updated beliefs for the next request

So the connection is:

`Bayes theorem updates the belief -> Thompson Sampling uses that belief to make the next choice`

### Same idea in simple words

You can think of it like this:

- Bayes theorem = the learning rule
- Thompson Sampling = the decision rule

Or even more simply:

- Bayes theorem **writes** the scorecard
- Thompson Sampling **reads** the scorecard to choose the next model

### Tiny routing example

Suppose `gpt-4o-mini` gets several good outcomes:

- valid output
- low latency
- no retry

Then Bayes-style updating makes the router's belief about `gpt-4o-mini` stronger.

Now that the belief is stronger:

- Thompson Sampling is more likely to draw a high temporary score for `gpt-4o-mini`
- so `gpt-4o-mini` becomes more likely to be selected on future requests

If later `gpt-4o-mini` starts performing badly:

- invalid output
- slow latency
- more retries

then Bayes-style updating weakens the belief.

Now Thompson Sampling will:

- more often draw lower temporary scores for `gpt-4o-mini`
- and route less traffic to it

So Bayes theorem is what makes the belief change, and Thompson Sampling is what converts that changing belief into routing behavior.

### Why both are needed

If you had Bayes updating but no Thompson Sampling:

- you would have updated beliefs
- but no clear rule for turning those beliefs into actions

If you had Thompson Sampling but no belief updates:

- you could sample numbers
- but they would not improve over time because the router would not be learning from evidence

That is why they fit together so naturally.

### One full textbook example

Suppose the current beliefs are:

- `gpt-4o`: `Beta(8, 3)`
- `gpt-4o-mini`: `Beta(5, 4)`
- `claude-haiku`: `Beta(5, 4)`

That roughly means:

- `gpt-4o` currently looks stronger
- `gpt-4o-mini` and `claude-haiku` still have a chance

Important:

Simple way to think about it:

Beta(8, 3) = “this model probably performs fairly well”
a sample like 0.81 = “for this round, the random draw from that belief came out to 0.81”
Important detail:

Beta(8, 3) has mean 8 / (8 + 3) = 0.727
but samples can still be 0.68, 0.74, 0.81, 0.77, etc.
it is a distribution over many possible values between 0 and 1
Same idea for Beta(5, 4):

mean is 5 / 9 ~= 0.556
but a sample could still occasionally be high, like 0.76 or even 0.84
that is exactly how exploration happens
So in plain English:

Beta(8, 3) says what values are plausible.
0.81 is one temporary random draw from that belief.

Now one request arrives.

Step 1: draw one temporary score from each model's belief.

Example draw:

- `gpt-4o`: `0.81`
- `gpt-4o-mini`: `0.76`
- `claude-haiku`: `0.84`

Step 2: choose the highest draw.

- the router picks `claude-haiku`

Step 3: observe the outcome.

Suppose the result is:

- success

Step 4: update only the chosen model.

- `claude-haiku` goes from `Beta(5, 4)` to `Beta(6, 4)`

If the result had been failure instead:

- `claude-haiku` would go from `Beta(5, 4)` to `Beta(5, 5)`

That is the full textbook learning loop.

### Compact algorithm notation

For each model `m`:

Symbols:

- `m` = one candidate model
- `theta_m` = the router's current belief about model `m`
- `theta_tilde_m` = one temporary sampled score for model `m`
- `a_t` = the model chosen at time `t`
- `r_t` = the reward observed after using that model

1. maintain posterior `theta_m ~ Beta(alpha_m, beta_m)`
  This means: keep the current belief about model `m` using `alpha_m` and `beta_m`.
2. sample `theta_tilde_m ~ Beta(alpha_m, beta_m)`
  This means: draw one temporary score from that belief.
3. choose `a_t = argmax_m theta_tilde_m`
  This means: pick the model with the highest temporary score.
4. observe reward `r_t`
  This means: see how well the chosen model actually performed.
5. update that arm's posterior
  This means: improve the belief for that chosen model using the new result.

That is exactly the idea implemented in your router.

### The intuition again

- high mean and low uncertainty -> often sampled high
- lower mean but high uncertainty -> sometimes sampled high
- terrible and well-known arm -> rarely sampled high

That is why Thompson Sampling explores intelligently.

### Important note for your project

Your actual router later changes this textbook version by using:

- continuous reward instead of only `0` or `1`
- reward from validity, latency, and retries

So this section is the clean foundation, not the full production behavior.

---

## 9. Mapping the Theory to Model Routing

Here is the direct mapping.


| Bandit concept   | In your router                                    |
| ---------------- | ------------------------------------------------- |
| Arm              | One LLM (`gpt-4o`, `gpt-4o-mini`, `claude-haiku`) |
| Pull an arm      | Route one request to that model                   |
| Reward           | Composite score from production telemetry         |
| Prior            | Initial `alpha, beta` values in `DEFAULT_MODELS`  |
| Posterior update | `alpha += reward`, `beta += 1 - reward`           |
| Explore          | Thompson sample + `shadow_rate`                   |
| Exploit          | Model with highest sampled utility                |


### In code

Selection happens here:

- `model_routing/bayesian_router/router.py`

Core logic:

- sample from each model's Beta distribution
- select the maximum

Update happens here:

- `Router.update(...)`

That method:

- computes reward
- updates alpha and beta
- occasionally decays memory

### Example flow of one routing decision

Suppose the incoming request is:

- "Extract the key fields from this invoice and return valid JSON."

And suppose, before this request arrives, the router's current beliefs are:

- `gpt-4o`: `Beta(8, 3)` -> current belief score `8 / (8 + 3) ~= 0.73`
- `gpt-4o-mini`: `Beta(5, 4)` -> current belief score `5 / (5 + 4) ~= 0.56`
- `claude-haiku`: `Beta(6, 3)` -> current belief score `6 / (6 + 3) ~= 0.67`

Here:

- the **Beta distribution** is the router's current belief about that model
- the **belief score** is the current mean of that belief
- the **sampled score** later is one temporary draw from that belief for this round

```text
Incoming request:
"Extract invoice fields and return valid JSON"
    |
    v
1. Read current beliefs for all models
   - gpt-4o: Beta(8, 3), belief score ~= 0.73
   - gpt-4o-mini: Beta(5, 4), belief score ~= 0.56
   - claude-haiku: Beta(6, 3), belief score ~= 0.67
    |
    | why it matters: start from what the router has learned from past traffic
    v
2. Sample one temporary score per model
   - gpt-4o: 0.78
   - gpt-4o-mini: 0.74
   - claude-haiku: 0.86
    |
    | why it matters: strong models win often, but uncertain models still get chances
    v
3. Pick the highest sampled score
   - choose claude-haiku
    |
    | why it matters: make one concrete routing decision for this request
    v
4. Send request to chosen model
   - claude-haiku returns valid JSON
   - no retry needed
   - latency = 850 ms
    |
    | why it matters: collect real production evidence instead of relying on guesses
    v
5. Measure reward
   - validity reward: 0.50
   - latency reward: ~0.26
   - retry reward: 0.20
   - total reward: ~0.96
    |
    | why it matters: turn model behavior into one comparable score
    v
6. Update claude-haiku's belief
   - before: Beta(6, 3)
   - after: Beta(6.96, 3.04)
    |
    | why it matters: claude-haiku becomes more trusted next time
    v
7. Optional safety / adaptation steps
   - fallback if confidence is too low
   - decay older evidence over time
   - exploration traffic may still test other models
    |
    | why it matters: stay safe and adapt if provider behavior changes
    v
Next request uses the updated beliefs
    |
    | continuity example:
    | because claude-haiku just performed well, its belief is now stronger
    v
Request 2 arrives
    |
    | sample example:
    | gpt-4o: 0.75
    | gpt-4o-mini: 0.70
    | claude-haiku: 0.88
    v
claude-haiku wins again
    |
    | outcome:
    | valid JSON, no retry, latency = 780 ms
    v
reward is high again
    |
    | why it matters: repeated good outcomes make the router trust claude-haiku more
    v
belief updates again
    |
    | example:
    | Beta(6.96, 3.04) -> Beta(7.92, 3.08)
    v
Request 3 arrives
    |
    | sample example:
    | gpt-4o: 0.82
    | gpt-4o-mini: 0.71
    | claude-haiku: 0.79
    v
gpt-4o wins this round
    |
    | why it matters: Thompson Sampling still gives other strong models chances
    | it does not permanently lock onto one winner
    v
router keeps learning over repeated requests
```

### How those sampled scores are generated

The numbers in the sample examples:

- `gpt-4o: 0.75`
- `gpt-4o-mini: 0.70`
- `claude-haiku: 0.88`

are **not** hand-written fixed scores.

They are random draws from each model's current Beta belief.

For example:

- `gpt-4o` might currently be `Beta(8, 3)`
- `gpt-4o-mini` might currently be `Beta(5, 4)`
- `claude-haiku` might currently be `Beta(6.96, 3.04)`

Then Thompson Sampling draws one temporary value from each of those three distributions.

Important:

- yes, **any** model can get a high sampled score on a given round
- but not all models are equally likely to get one

In practice:

- a strong, well-tested model gets high samples often
- an uncertain model can sometimes get a surprisingly high sample
- a weak, well-understood model rarely gets a high sample

That is exactly why Thompson Sampling balances:

- exploitation of models that already look strong
- exploration of models that might be better than we currently think

### How the belief is calculated and updated

In this guide, the router's **belief** about a model is represented as:

- `Beta(alpha, beta)`

And the current **belief score** is the mean of that belief:

- `belief score = alpha / (alpha + beta)`

Example from this flow:

- before request 1, `claude-haiku = Beta(6, 3)`
- belief score = `6 / (6 + 3) = 6 / 9 ~= 0.67`

After request 1, the reward was about `0.96`, so the update is:

- `alpha += 0.96`
- `beta += 0.04`

So:

- before: `Beta(6, 3)`
- after: `Beta(6.96, 3.04)`

New belief score:

- `6.96 / (6.96 + 3.04) = 6.96 / 10 ~= 0.70`

After request 2, suppose reward is high again, around `0.96`:

- before: `Beta(6.96, 3.04)`
- after: `Beta(7.92, 3.08)`

New belief score:

- `7.92 / (7.92 + 3.08) = 7.92 / 11 ~= 0.72`

So when line `because claude-haiku just performed well, its belief is now stronger` appears, it means:

- the belief distribution shifted upward
- the average belief score increased
- the router is now a bit more likely to choose `claude-haiku` again

### What each step is doing for model routing

1. **Read beliefs**
   The router starts with what it already knows about each model from past traffic.

2. **Sample scores**
   This creates a temporary competition for this one request.

3. **Pick the highest score**
   This is the actual routing decision.

4. **Observe the real outcome**
   The router sees whether the chosen model was actually good on this request.

5. **Measure reward**
   The router converts that outcome into one number it can learn from.

6. **Update belief**
   The chosen model becomes more or less trusted depending on what happened.

7. **Use updated beliefs next time**
   The router slowly shifts traffic toward the best real-world tradeoff.

In one sentence:

- the router uses past evidence, makes one live routing decision, learns from the result, and then uses that updated knowledge on the next request

---

## 10. Very Important Nuance: Your Implementation Uses a Continuous Reward

This is one of the most important things for you to understand clearly.

### Classical theory

Beta-Bernoulli Thompson Sampling is exact when rewards are binary:

- 1 = success
- 0 = failure

### Your implementation

Your reward is continuous:

`r in [0, 1]`

It is not binary.

Your code does:

- `alpha <- alpha + r`
- `beta <- beta + (1 - r)`

This is **not** the strict Beta-Bernoulli conjugate update anymore.

### So what is it?

It is best understood as a practical **soft pseudo-count update**:

- reward 0.9 means "almost a success"
- reward 0.2 means "mostly a failure"

This is a pragmatic engineering heuristic.

### Why that is okay

Because:

- your goal is practical online adaptation
- the reward is a meaningful bounded utility signal
- the update remains stable and interpretable

### How to say this if someone asks

Use this answer:

> Strictly speaking, classical Thompson Sampling with a Beta posterior assumes binary rewards. In this implementation, I use a bounded composite reward in [0,1] and update alpha and beta with fractional pseudo-counts. So it is a pragmatic generalized Bayesian-style heuristic rather than a textbook conjugate update. I chose it because production telemetry is naturally continuous and I want online learning without human labels.

That is a strong and honest answer.

---

## 10.1 Why Fractional Pseudo-Counts Actually Work

Section 10 says the continuous update is "heuristic." This section explains **why that heuristic is reasonable**, so you can say more than just "it works in practice."

If this is your first read, the main message is simple: even though the update is not exact textbook Bayes, it still moves the belief in the right direction and reduces uncertainty over time.

### Same idea in simple words

Your router does **not** see only:

- full success
- or full failure

Instead, it sees a reward between `0` and `1`.

So each request gives the router **one unit of evidence**, but that evidence can be split:

- reward `1.0` = treat it like a full success
- reward `0.0` = treat it like a full failure
- reward `0.8` = treat it like "mostly success"
- reward `0.3` = treat it like "mostly failure"

That is why the update is:

- `alpha += r`
- `beta += (1 - r)`

So:

- good outcomes push belief upward
- bad outcomes push belief downward
- mixed outcomes partially help and partially hurt

### Concrete example

Suppose the router currently has:

- `alpha = 6`
- `beta = 4`

Current belief score:

- `6 / (6 + 4) = 0.60`

Now one request gets reward `0.8`.

Update:

- `alpha = 6 + 0.8 = 6.8`
- `beta = 4 + 0.2 = 4.2`

New belief score:

- `6.8 / (6.8 + 4.2) = 6.8 / 11 ~= 0.62`

So the model becomes a bit more trusted.

Now imagine the next request gets reward `0.2`.

Update again:

- `alpha = 6.8 + 0.2 = 7.0`
- `beta = 4.2 + 0.8 = 5.0`

New belief score:

- `7 / (7 + 5) = 7 / 12 ~= 0.58`

So the belief moves back down.

That is the whole intuition:

- a good reward pushes the score up
- a bad reward pushes the score down
- every request adds one more piece of evidence

If the formulas below feel heavy, the main takeaway is:

> The update behaves like a soft version of success/failure counting.

### The posterior mean still moves in the right direction

After observing reward `r in [0, 1]` and applying the update:

- `alpha' = alpha + r`
- `beta' = beta + (1 - r)`

The new posterior mean is:

`mu' = (alpha + r) / (alpha + beta + 1)`

Compare this to the old mean:

`mu = alpha / (alpha + beta)`

The new mean `mu'` is a weighted average that pulls toward `r`:

- If `r > mu`, the mean increases
- If `r < mu`, the mean decreases

This is exactly the same directional behavior as the exact Bernoulli update. The posterior mean always moves toward the observed evidence.

### The effective sample size increases by exactly 1

Notice the denominator: `alpha + beta + 1`. Every update adds exactly `1` to the total pseudo-count regardless of the reward value. So the posterior narrows at the same rate as the exact binary case. The uncertainty reduction is correct.

### What is different from exact Bernoulli?

The **shape** of the posterior. In exact Beta-Bernoulli, the posterior is always a valid Beta distribution because `alpha` and `beta` only increase by integers. With fractional updates, the parameters are still valid Beta parameters (any `alpha, beta > 0` gives a valid Beta distribution), but the posterior is no longer the true Bayesian posterior for any well-defined likelihood model.

Same idea in simple words:

- in the textbook case, each request is either full success or full failure
- so you add whole numbers like `+1` or `+0`
- that matches the exact math perfectly

In your router:

- a request can be partly good and partly bad
- so you add fractional values like `+0.8` and `+0.2`

Example:

- textbook success: `alpha += 1`, `beta += 0`
- textbook failure: `alpha += 0`, `beta += 1`
- your soft update for reward `0.8`: `alpha += 0.8`, `beta += 0.2`

So what is the difference?

- the update is still sensible
- the Beta distribution still works as a practical scorecard
- but it is no longer the exact textbook Bayesian formula for a pure yes/no world

The easiest way to think about it is:

> exact Bernoulli = strict yes/no counting  
> your router = soft yes/no counting

So the math is no longer "perfectly textbook," but the learning behavior is still useful and intuitive.

In practice this does not matter because:

1. You only use the posterior for Thompson Sampling (sampling and argmax)
2. The mean and variance of the Beta distribution remain interpretable
3. The convergence behavior is preserved

### The strongest intuitive justification

Think of reward `r = 0.7` as:

- "70% of a success and 30% of a failure"
- You split your evidence proportionally

For Thompson Sampling, what matters is:

- Does the posterior mean track the true expected reward? **Yes.**
- Does uncertainty decrease with more observations? **Yes.**
- Does the sampling distribution concentrate around the true mean? **Yes.**

### How to say this if someone pushes beyond "it is a heuristic"

> The fractional update preserves the two properties Thompson Sampling needs: the posterior mean moves toward observed rewards, and the posterior narrows at one pseudo-count per observation. The posterior shape is no longer the exact Bayesian posterior for a Bernoulli likelihood, but the mean and variance remain correct enough for sampling-based exploration to work well.

---

## 11. The Composite Reward Function

This is implemented in:

- `model_routing/bayesian_router/rewards.py`

First-time takeaway:

- the router rewards outputs that are valid
- it also rewards being fast
- it gives a bonus when the response did not need a retry

The reward is:

`r = r_validity + r_latency + r_retry`

with default weights:

- validity = 0.50
- latency = 0.30
- no-retry = 0.20

### Validity term

- `r_validity = 0.50` if the response is valid
- `r_validity = 0` otherwise

### Retry term

- `r_retry = 0.20` if no retry was needed
- `r_retry = 0` if the system had to retry

### Latency term

In code:

`r_latency = 0.30 / (1 + exp((L - m) / s))`

where:

- `L` = latency in milliseconds
- `m` = midpoint = 2000 ms
- `s` = steepness = 600

### Interpretation

If latency is much smaller than 2000 ms:

- latency reward is close to 0.30

If latency is exactly 2000 ms:

`r_latency = 0.30 / 2 = 0.15`

If latency is much larger than 2000 ms:

- latency reward moves toward 0

### Why sigmoid instead of hard threshold?

Where it is used:

- in `model_routing/bayesian_router/rewards.py`
- inside `CompositeReward.compute(...)`
- specifically for the **latency reward** term

In code, this is the part:

- `l = latency_weight / (1 + exp((latency_ms - midpoint_ms) / steepness))`

So the sigmoid is **not** used for:

- validity reward
- retry reward

It is only used to turn raw latency into a smooth score between:

- almost `0.30` for fast responses
- down toward `0` for slow responses

Because a hard cutoff would be brittle.

A sigmoid gives:

- smooth penalty
- easier tuning
- more realistic behavior around the SLA boundary

---

## 12. Worked Reward Examples

These examples matter more than the formulas. If you understand the three examples below, you understand how the reward behaves in practice.

### Example A: strong response

Assume:

- valid = True
- retried = False
- latency = 800 ms

Latency term:

`r_latency = 0.30 / (1 + exp((800 - 2000) / 600))`

`r_latency ~= 0.264`

Total reward:

`r ~= 0.50 + 0.264 + 0.20 = 0.964`

That is almost a full success.

### Example B: valid but slow

Assume:

- valid = True
- retried = False
- latency = 3500 ms

Latency term:

`r_latency = 0.30 / (1 + exp((3500 - 2000) / 600))`

`r_latency ~= 0.023`

Total:

`r ~= 0.50 + 0.023 + 0.20 = 0.723`

Still decent, but clearly less attractive than the fast model.

### Example C: invalid and retried

Assume:

- valid = False
- retried = True
- latency = 3500 ms

Then:

`r ~= 0 + 0.023 + 0 = 0.023`

That model gets heavily penalized.

---

## 12.1 Worked Examples Using the Actual Simulator Profiles (Optional Deep Dive)

Sections 12's examples use made-up numbers. This section uses the **real simulator defaults** from `model_routing/bayesian_router/simulator.py` so you know what your demo actually produces.

### Simulator profiles


| Model          | Validity | Latency range (ms) | Retry rate | Cost per 1k tokens |
| -------------- | -------- | ------------------ | ---------- | ------------------ |
| `gpt-4o`       | 0.96     | 1500-3500          | 4%         | $0.005             |
| `gpt-4o-mini`  | 0.89     | 400-1200           | 9%         | $0.00015           |
| `claude-haiku` | 0.91     | 300-900            | 7%         | $0.00025           |


### Expected reward per model (average case)

To compute expected reward, use the midpoint of each model's latency range and the base rates for validity and retry.

**gpt-4o** (midpoint latency = 2500 ms):

- `r_validity = 0.50 * 0.96 = 0.48`
- `r_latency = 0.30 / (1 + exp((2500 - 2000) / 600)) ~= 0.091`
- `r_retry = 0.20 * (1 - 0.04) = 0.192`
- `r_gpt-4o ~= 0.48 + 0.091 + 0.192 = 0.763`

**gpt-4o-mini** (midpoint latency = 800 ms):

- `r_validity = 0.50 * 0.89 = 0.445`
- `r_latency = 0.30 / (1 + exp((800 - 2000) / 600)) ~= 0.264`
- `r_retry = 0.20 * (1 - 0.09) = 0.182`
- `r_gpt-4o-mini ~= 0.445 + 0.264 + 0.182 = 0.891`

**claude-haiku** (midpoint latency = 600 ms):

- `r_validity = 0.50 * 0.91 = 0.455`
- `r_latency = 0.30 / (1 + exp((600 - 2000) / 600)) ~= 0.273`
- `r_retry = 0.20 * (1 - 0.07) = 0.186`
- `r_claude-haiku ~= 0.455 + 0.273 + 0.186 = 0.914`

### What this tells you

How to read this table:

- **Model** = the candidate LLM
- **Expected reward** = the average total reward the router expects from that model under the simulator defaults
- **Cost per 1k** = the model's dollar cost per 1,000 tokens
- **Reward per dollar** = `expected reward / cost per 1k tokens`

In simple terms:

- **Expected reward** tells you how good the model looks overall
- **Reward per dollar** tells you how much value you get for the money

So a model can have:

- slightly lower raw quality
- but much better value for cost

That is exactly what happens here.

Important clarification:

- in the **current router**, the main thing that gets optimized is **reward**
- `Cost per 1k` and `Reward per dollar` in this table are mainly for **analysis and explanation**
- they help you understand why cheaper models may look attractive
- but the router does **not** directly sample from `reward per dollar`

So, in simple terms:

- **Expected reward** gets more weight in the actual routing decision
- **Reward per dollar** helps you interpret the business value of that decision

If you wanted the router to give more direct weight to dollar cost, you would need to add an explicit cost term to the reward function.


| Model          | Expected reward | Cost per 1k | Reward per dollar |
| -------------- | --------------- | ----------- | ----------------- |
| `gpt-4o`       | ~0.76           | $0.005      | 152               |
| `gpt-4o-mini`  | ~0.89           | $0.00015    | 5,940             |
| `claude-haiku` | ~0.91           | $0.00025    | 3,656             |


**This is why the router shifts traffic to cheaper models.** Despite slightly lower validity, the cheaper models earn **higher total reward** because their speed advantage in the latency term more than compensates.

`claude-haiku` earns the highest reward (~0.91) and is 20x cheaper than `gpt-4o`. The router will converge toward it.

### This is the core insight of the demo

The expensive model (`gpt-4o`) has the highest validity (0.96 vs 0.91) but the **lowest total reward** (0.76 vs 0.91) because it is slow. The composite reward function naturally discovers that the cheaper, faster models provide a better overall tradeoff.

---

## 13. Where Does Cost Enter the Picture?

This is another very important nuance.

### In the current code

The composite reward uses:

- validity
- latency
- retry behavior

It does **not** explicitly include dollar cost.

### So why do we still see cost reduction?

Because in the simulation:

- cheaper models are also faster
- they are slightly weaker but often still good enough

So the reward indirectly favors cheaper models because:

- they earn more latency reward
- they remain sufficiently valid

The cost reduction emerges because:

- faster, cheaper models are selected more often

### If someone asks for a stricter cost-aware formulation

You can say:

> In the current demo, cost is optimized indirectly through latency and model choice. In a production system, I would likely add an explicit cost penalty term to the reward, for example:
>
> `r = w_v * validity + w_l * latency + w_r * no-retry - lambda * cost`

That is a very reasonable extension.

### Be precise when you speak

The current code is best described as:

- a **cost-sensitive** router by effect
- not a fully explicit **cost-penalized** optimizer

That distinction is useful.

---

## 13.1 What Does "Accuracy" Mean in the Proposal?

The proposal claims:

> Result: 40-50% cost cut, <1% accuracy drop.

You need to know exactly what "accuracy" refers to here and how the <1% number is measured.

### In the current simulation

"Accuracy" means **validity rate**: the fraction of responses that pass the schema/JSON validator.

This is tracked through the `is_valid` field in the simulator (`model_routing/bayesian_router/simulator.py`).

### How to measure the <1% claim

Compare two configurations:

1. **Baseline**: always route to `gpt-4o` (the strongest model)
  - `gpt-4o` has `base_validity = 0.96` in the simulator
2. **Router**: let the Bayesian bandit route traffic
  - After convergence, the router sends most traffic to cheaper models
  - Cheaper models have `base_validity` of 0.89 (`gpt-4o-mini`) and 0.91 (`claude-haiku`)
  - But the router still sends a share of traffic to `gpt-4o`

The weighted average validity under the router depends on the selection distribution. If the router converges to roughly 30% gpt-4o, 45% gpt-4o-mini, 25% claude-haiku:

- `validity_router ~= 0.30 * 0.96 + 0.45 * 0.89 + 0.25 * 0.91 = 0.917`
- `validity_baseline = 0.96`
- `accuracy drop = 0.96 - 0.917 = 0.043` (approx 4.3%)

That is **not** <1%.

### Important: the <1% claim depends on specific conditions

The <1% claim holds when:

- the cheaper models have validity very close to the expensive model
- the confidence floor forces fallback for degraded models
- the reward function heavily weights validity

With the default simulator profiles as written, the validity gap between models is larger than 1%. This means:

- The <1% claim may come from a specific tuned configuration, not the defaults
- Or it refers to a scenario where cheaper models happen to be nearly as accurate

### How to be honest about this in the talk

Use this:

> The <1% accuracy drop depends on the specific models and task. In the demo, "accuracy" means schema validity rate. The actual drop depends on how close the cheaper models are to the expensive one. If validity is similar across models, the drop is minimal. If not, you can tune the validity weight upward to tighten the accuracy guarantee at the cost of less routing savings.

Practical note:

- yes, prompt engineering and structured-output techniques can help close this gap
- this is especially true when "accuracy" mostly means **schema validity**

Examples:

- stronger instructions like "return only valid JSON"
- few-shot examples
- JSON mode / function calling / constrained decoding
- validation + repair loop

Why this helps:

- it can improve validity
- it can reduce retries
- both of those improve the reward of cheaper models

Important limit:

- prompt engineering helps most when the gap is about formatting, control, or output discipline
- it helps less when the gap is about deeper reasoning or model capability

### If someone asks you to be precise

> Accuracy here refers to the fraction of responses that pass automated validation. The <1% claim is achievable when cheaper model validity is within a few percentage points of the baseline. In the general case, accuracy drop is a function of the validity gap between models and the router's selection distribution.

---

## 14. Expert Priors and Cold Start

First-time takeaway:

- we don't start from zero
- we give the router a "head start" by telling it `gpt-4o` is probably better than `gpt-4o-mini`
- but we don't make that belief so strong that the router can't change its mind

Cold start means:

- the router has little or no observed data
- you still need good early decisions

### In your code

Default priors are:

- `gpt-4o`: `alpha=8, beta=3`
- `gpt-4o-mini`: `alpha=5, beta=4`
- `claude-haiku`: `alpha=5, beta=4`

### Prior mean

The prior mean is:

`mu_0 = alpha / (alpha + beta)`

So:

- `gpt-4o`: `8/11 ~= 0.727`
- `gpt-4o-mini`: `5/9 ~= 0.556`
- `claude-haiku`: `5/9 ~= 0.556`

### Prior strength

The total pseudo-count is:

`alpha + beta`

This tells you how strongly the prior resists change.

In your setup:

- `gpt-4o` starts with a better prior mean
- but the cheaper models are not crushed
- the router can still discover they are good enough

### Why this matters

If you started from uniform priors:

- all models look identical at first
- you need more exploration
- early traffic may be worse or more expensive

This is why expert priors help.

---

## 15. Decaying Memory and Model Rot

First-time takeaway:

- LLM quality changes over time (model updates, server load)
- if we remember everything forever, the router reacts too slowly to new problems
- "decay" means we slowly shrink `alpha` and `beta` so old history matters less than new history

### Same idea in simple words

Imagine this:

- last week, `claude-haiku` was fast and reliable
- so the router learned to trust it
- but today the provider changed something, and now it is slower or less reliable

If the router remembers all old evidence equally forever:

- it will keep trusting `claude-haiku` too much
- it will take too long to notice that the situation changed

That is the problem this section is solving.

Decay is the fix:

- old evidence slowly fades
- new evidence matters more

So the router can change its mind faster when the world changes.

The classical bandit setting often assumes the reward distribution of each arm is stationary.

That means:

- if arm A has average reward 0.8 today
- it also has average reward 0.8 tomorrow

Real LLM systems are not like that.

### Why model rot happens

Provider behavior changes:

- model version updates
- load and latency shifts
- infrastructure behavior changes
- prompt distribution changes

So the old evidence can become misleading.

### Concrete example

Suppose the router has learned this from old traffic:

- `claude-haiku = Beta(40, 10)`

That means:

- the router has a lot of evidence
- and it currently trusts `claude-haiku` quite a lot

Now imagine the model degrades:

- latency gets worse
- retries go up
- validity drops

If you do **not** use decay:

- the huge old evidence `Beta(40, 10)` dominates
- a few bad recent outcomes do not change the belief much
- the router reacts slowly

If you **do** use decay:

- before the next update, beliefs are shrunk a bit
- for example with `gamma = 0.95`:
- `alpha = 40 -> 38`
- `beta = 10 -> 9.5`

Now recent bad outcomes have more influence.

So decay is basically:

- "trust recent history more than old history"

### Your solution

Every `decay_interval` queries:

- `alpha <- max(1, gamma * alpha)`
- `beta <- max(1, gamma * beta)`

with default:

- `gamma = 0.95`
- `decay_interval = 50`

What `decay_interval = 50` means:

- the router does **not** decay beliefs on every single request
- it waits until 50 update calls have happened
- then it applies decay once to all models

In code, the logic is:

- increase query count after each update
- if `query_count % 50 == 0`, run decay

So:

- after query 1, no decay
- after query 2, no decay
- ...
- after query 49, no decay
- after query 50, decay happens
- then again after query 100, 150, 200, and so on

Tiny example:

Suppose before the 50th query:

- `gpt-4o = Beta(20, 8)`

At query 50, decay runs:

- `alpha = 20 * 0.95 = 19`
- `beta = 8 * 0.95 = 7.6`

So the router keeps most of the old belief, but makes it slightly weaker.

Why this design is useful:

- decaying every request would make the router forget too aggressively
- never decaying would make it adapt too slowly
- decaying every 50 requests is a middle ground

### What decay does

- discounts stale evidence
- recent observations matter more
- posterior becomes more adaptive

In plain English:

- the router forgets slowly
- not instantly, and not completely
- just enough to stay responsive when model behavior changes

### Is this fully Bayesian?

Not in the strict classical sense.

It is a practical heuristic for a **non-stationary bandit** environment.

That is completely fine for a production talk. Just be honest about it.

### How to explain it simply

> If a provider degrades, I do not want the router to remain overconfident based on old history. Decay is a way to let the system forget slowly.

---

## 16. Confidence Floor, Fallback, and the Circuit Breaker Question

First-time takeaway:

- if the router is very uncertain about a cheap model, or if the model has been failing a lot, the router's confidence drops
- if confidence drops below 50%, the router stops using it and falls back to the safest, most expensive model
- this is a simple safety net

What if that model improves later?

- the router does **not** permanently ban it
- but it usually needs new good evidence before it will trust it again

In the current implementation, that new evidence mainly comes from:

- exploration traffic (`shadow_rate`)
- or future controlled testing

So the recovery story is:

1. model confidence drops below the floor
2. normal traffic falls back to the trusted model
3. exploration traffic still gives the weak model occasional chances
4. if it starts performing well again, its belief improves
5. once its confidence rises enough, it can re-enter normal competition

Important nuance:

- decay helps old bad history matter less over time
- but decay alone does not prove the model is good again
- the model still needs fresh successful outcomes

The router computes:

`confidence(m) = alpha_m / (alpha_m + beta_m)`

If the chosen model's confidence is below a threshold, it falls back to the trusted model.

In code:

- `confidence_floor = 0.50`
- fallback model defaults to the first model, typically `gpt-4o`

### What this does

If a cheap model is too uncertain or degraded:

- do not trust it
- route to the safer expensive model

### Important nuance

Your current implementation is **confidence-based fallback**.

It is **not** a full circuit breaker state machine in the strict Fowler sense.

A full circuit breaker usually has:

- closed state
- open state
- half-open state
- recent failure threshold
- reset timeout

Same idea in simple words:

- **closed state** = everything is normal, requests are allowed through
- **open state** = stop sending traffic to that model because it looks unhealthy
- **half-open state** = cautiously try a small number of requests again to see if the model recovered
- **recent failure threshold** = how many recent failures are enough to trigger the breaker
- **reset timeout** = how long to wait before testing the model again

Tiny example:

- suppose `gpt-4o-mini` fails 5 times in a short window
- that crosses the recent failure threshold
- the breaker moves to **open**
- the router stops sending normal traffic to `gpt-4o-mini`
- after the reset timeout, the system tries one or two test requests
- if those succeed, it moves back toward normal use
- if they fail again, it stays open

So a full circuit breaker is basically:

> a safety system that temporarily stops using a model when it looks unhealthy, then carefully checks later whether it has recovered

Your code currently does not implement that full state machine.

### Best way to say this in the talk

Say:

> The current implementation includes confidence-based fallback, which serves as a lightweight safety guard. In a production deployment, I would likely wrap this with a fuller circuit breaker policy over recent failure windows.

That is accurate and strong.

---

## 17. Shadow Evaluation vs Forced Exploration

First-time takeaway:

- the demo code randomly routes 10% of traffic just to explore
- this is "forced exploration"
- true "shadow evaluation" would mean running the cheap model in the background without showing the user the result, which this code does not do

Your talk mentions shadow evaluation.

### Strict definition of shadow evaluation

Shadow evaluation means:

- copy a production request
- send it to a candidate model in parallel
- do not show its answer to the user
- log results for comparison

### What your current code does

The `shadow_rate` in `Router.select()` actually does:

- route a fraction of traffic randomly

The default value in code is `shadow_rate = 0.10` (10% of traffic). Note that the proposal says "5% of traffic" — this is a discrepancy. The code is the source of truth. If you change it for the demo, know which value you are actually using.

If you want the implementation to match the proposal more closely, two things still need to be implemented as full production features:

- **automated circuit breakers**
- **continuous shadow evaluation on 5% of traffic**

Right now, the package has:

- confidence-based fallback instead of a full automated circuit breaker state machine
- 10% exploration traffic instead of true hidden shadow evaluation on 5% of traffic

So the proposal describes the stronger target architecture, while the current code implements a lighter version of that idea.

That is closer to:

- forced exploration
- exploration traffic

not a full hidden shadow inference pipeline.

### Why this matters

If someone asks:

> Are you really doing shadow evaluation in the code?

the honest answer is:

> The current code uses explicit exploration traffic. A stricter production shadow deployment would mirror requests and compare hidden candidate outputs against the baseline path.

That answer makes you sound careful and technically credible.

---

## 17.1 How To Reconcile the Proposal Wording with the Current Code

Your accepted proposal uses strong production language:

- "continuous shadow evaluation"
- "circuit breakers"
- "40-50% cost cut"

That is fine as a talk framing, but when discussing the current package you should be slightly more precise.

### Best accurate phrasing

Use language like this:

- **shadow-style evaluation / exploration traffic** rather than claiming a full hidden mirroring pipeline
- **confidence-based fallback** rather than claiming a complete circuit breaker state machine
- **cost reduction emerges from routing toward fast, cheaper models** rather than claiming the reward directly optimizes dollars

### One good concise sentence

> The current implementation captures the core production pattern: online adaptation, safety-oriented fallback, and exploration traffic, while leaving room for stricter shadow mirroring and fuller circuit breaker policies in a hardened production deployment.

If you use that framing, you will stay both honest and persuasive.

---

## 18. Is This Really "Model Routing"?

First-time takeaway:

- yes, it is routing
- but it routes based on **overall model health**, not based on **reading the specific prompt**

This is another question you might get.

### Short answer

Yes, but with an important nuance.

### The nuance

Your current implementation is a **non-contextual online router**.

It learns:

- overall utility of each model across traffic

It does **not** currently use query features such as:

- prompt length
- embeddings
- request type
- tool requirements
- user tier

So it is not yet a **contextual bandit router**.

### How to phrase it safely

Use this:

> This implementation is a lightweight online router at the traffic level. It learns which models are generally the best tradeoff under current production conditions. A natural next step would be contextual routing, where the posterior or policy also conditions on query features.

That is exactly the right framing.

---

## 19. Non-Contextual vs Contextual Bandits

First-time takeaway:

- **Non-contextual**: The router learns "Model A is generally better than Model B right now." (This is what your code does.)
- **Contextual**: The router learns "Model A is better for math questions, but Model B is better for writing emails." (This is a future extension.)

### Non-contextual bandit

Choose arm using only historical rewards.

Mathematically:

- no input feature vector `x_t`
- same decision policy for all requests

### Contextual bandit

At each time `t`, observe context `x_t` and choose action:

`a_t = pi(x_t)`

In model routing, context might include:

- query embedding
- prompt length
- estimated difficulty
- task type
- whether tools are required
- whether structured output is needed

### Why your current design is still useful

Because even non-contextual adaptation solves real problems:

- drift
- provider instability
- global cost/latency shifts
- online learning without labels

### If you want a future extension

You could combine:

- a contextual classifier for "difficulty"
- a Bayesian bandit controller for online adaptation

That would be a very nice future-work answer.

---

## 20. Worked Example of One Full Routing Cycle

First-time takeaway:

- this is the step-by-step story of exactly what happens when one request comes in
- read this to understand the sequence of events

Suppose current state is:

- `gpt-4o`: `alpha=8, beta=3`
- `gpt-4o-mini`: `alpha=5, beta=4`
- `claude-haiku`: `alpha=5, beta=4`

### Step 1: sample one value from each posterior

Example random samples:

- `gpt-4o`: 0.71
- `gpt-4o-mini`: 0.62
- `claude-haiku`: 0.77

Router picks:

- `claude-haiku`

### Step 2: send request

Suppose the response is:

- valid
- no retry
- latency = 900 ms

### Step 3: compute reward

Validity term:

`0.50`

Retry term:

`0.20`

Latency term:

`0.30 / (1 + exp((900 - 2000) / 600)) ~= 0.259`

Total:

`r ~= 0.959`

### Step 4: update posterior

For `claude-haiku`:

- `alpha <- 5 + 0.959 = 5.959`
- `beta <- 4 + 0.041 = 4.041`

Its posterior mean becomes:

`5.959 / (5.959 + 4.041) approx 0.596`

So the router becomes more favorable toward that model.

---

## 21. What If the Model Does Poorly?

Suppose the selected model returns:

- invalid output
- retried
- latency = 3500 ms

Then reward is approximately:

`r approx 0.023`

Update:

- `alpha <- alpha + 0.023`
- `beta <- beta + 0.977`

So:

- alpha barely moves
- beta grows a lot
- posterior shifts left
- model becomes less likely to be chosen

That is exactly what you want.

---

## 22. What the Demo Simulates

The simulator in:

- `model_routing/bayesian_router/simulator.py`

defines synthetic model profiles.

Defaults:

- `gpt-4o`
  - higher cost
  - higher validity
  - slower
- `gpt-4o-mini`
  - very cheap
  - faster
  - slightly lower validity
- `claude-haiku`
  - cheap
  - fast
  - decent validity

When the demo shows savings, it is because:

- cheaper models are fast
- validity is still acceptable
- the reward function gives enough value to speed and successful outputs

The simulator also supports degradation:

- multiply latency
- reduce validity
- increase retry rate

That is how the "model rot" demo works.

---

## 23. Key Theoretical Limitations

First-time takeaway:

- no system is perfect
- admitting these limitations makes you look like a mature engineer who understands tradeoffs, not a salesperson
- the biggest limitation is that it doesn't read the prompt before routing

You should know these and be comfortable admitting them.

### 1. Non-contextual policy

It does not route based on the specific query.

### 2. Composite reward is a proxy

It is not a ground-truth quality measure.

### 3. Fractional Beta updates are heuristic

Not exact conjugate Bayesian updating.

### 4. Cost is indirect in the current reward

Not explicitly penalized in the objective.

### 5. Safety is simplified

Fallback is implemented.
Full circuit breaker and true shadow mirroring are not fully implemented.

### 6. Good validators are essential

If parse success is weakly correlated with correctness, the reward may mislead.

These do not weaken your talk. They make it more mature.

---

## 24. When This Approach Works Best

First-time takeaway:

- use this when you have clear "right/wrong" signals (like JSON validation)
- don't use this for subjective tasks like "write a poem" because the router won't know how to score it

Best fit:

- structured outputs
- JSON / schema-based tasks
- tool-using agents
- extraction / classification tasks
- workflows with retries and strong operational telemetry

Harder fit:

- open-ended creative writing
- subjective quality tasks
- domains with weak automatic validators
- high-risk domains without strong fallback policies

---

## 25. Comparing with Related Approaches

### Always use the best model

Pros:

- simple
- high quality

Cons:

- expensive
- often slower than necessary

### Static rule-based router

Pros:

- simple
- predictable

Cons:

- brittle
- does not adapt to drift
- requires manual maintenance

### Learned offline router with labels

Pros:

- potentially query-aware
- can be powerful

Cons:

- needs labeled data
- often difficult to keep fresh

### Your Bayesian bandit router

Pros:

- online adaptation
- no human labels
- uncertainty-aware exploration
- simple implementation

Cons:

- currently non-contextual
- proxy reward quality matters

---

## 26. Possible Audience Questions and Good Answers

### Q: Why Thompson Sampling instead of epsilon-greedy?

Answer:

> Epsilon-greedy explores in a flat random way. Thompson Sampling explores according to uncertainty. If a model is uncertain but promising, it gets explored more. If it is both poor and well-understood, it gets explored less.

### Q: Why not UCB?

Answer:

> UCB is also reasonable. I chose Thompson Sampling because it is simple, Bayesian, easy to implement with priors, and gives an intuitive uncertainty-aware story for routing.

### Q: Is the posterior mathematically exact here?

Answer:

> Not exactly, because my reward is continuous rather than binary. The update is best viewed as fractional pseudo-counts. If I used binary success/failure rewards, the Beta-Bernoulli posterior would be exact.

### Q: How do you optimize cost if cost is not in the reward?

Answer:

> In the current demo, cost reduction emerges because cheaper models are also faster and good enough on many queries. In a stricter production system, I would likely include an explicit cost penalty in the reward or optimize under a cost budget.

### Q: Is this truly per-query routing?

Answer:

> This implementation is an online non-contextual router, so it learns at the traffic level rather than conditioning on query features. The next natural extension would be a contextual bandit or hybrid router using query embeddings or task features.

### Q: What if parse success is not strongly correlated with correctness?

Answer:

> Then the proxy reward is weak and the method becomes less reliable. This works best when you have validators that correlate well with downstream correctness. In practice I would tailor the reward to the task.

### Q: What does "Bayesian" really buy us here?

Answer:

> It gives us uncertainty. We do not just keep averages; we keep distributions over model quality. That makes exploration more principled and lets priors matter during cold start.

### Q: Why not just do A/B tests?

Answer:

> A/B tests are good for offline or one-time decisions, but routing is an ongoing online decision problem. Model quality and traffic change, so we need something adaptive rather than a one-shot experiment.

### Q: What do priors mean in practice?

Answer:

> Priors encode both initial belief and initial confidence. A strong model can start with a better prior mean, while weaker models still get enough probability mass to be explored.

### Q: How do you pick decay parameters?

Answer:

> Gamma close to 1 means slow forgetting; smaller gamma means faster adaptation. I would tune it based on how volatile provider behavior is and how quickly I need to react to changes.

### Q: Could you use this with more than three models?

Answer:

> Yes. The bandit formulation scales naturally to any number of candidate models.

### Q: What does the fallback protect against?

Answer:

> It protects against overconfident routing to a weak model when the posterior mean confidence is still too low. It is a simple safety brake.

---

## 27. How To Explain Bayesian Bandits in 30 Seconds

If someone asks informally, say this:

> A Bayesian bandit is a decision system for repeated choices under uncertainty. In my case, each LLM is an arm. The system keeps a probability distribution over how good each model currently is, samples from those distributions, picks a model, observes the outcome, and updates its belief. That lets it gradually route more traffic to the cheaper models when evidence says they are good enough.

---

## 28. How To Explain It in 2 Minutes

> I model each candidate LLM as an arm in a multi-armed bandit. Instead of storing a single fixed score, I keep a Bayesian belief distribution over each model's quality. On each request, I sample once from every model's posterior and pick the model with the highest sample. After the response comes back, I compute a reward from production telemetry such as schema validity, latency, and whether retries were needed. Then I update the selected model's posterior. So the router keeps learning online without human labels. I also add expert priors for cold start, decay for provider drift, and fallback when confidence is too low.

---

## 29. What You Should and Should Not Claim

### Safe claims

- "This is a production-friendly adaptive router."
- "It learns without human labels."
- "It uses a composite reward from operational telemetry."
- "It adapts online."
- "It includes practical safety measures such as fallback and exploration."

### Claims to avoid or qualify

- "This is a fully contextual router."
- "This is exact Bayesian inference end-to-end."
- "This fully implements strict shadow evaluation."
- "This fully implements a classical circuit breaker state machine."

Better phrasing:

- "lightweight"
- "pragmatic"
- "production-friendly"
- "online adaptive"
- "heuristic but useful"

---

## 30. Suggested Study Order

If you want to get comfortable quickly, study in this order.

### Step 1: Understand the intuition

Read:

- this file sections 1-9

Goal:

- know what bandits are
- know why Thompson Sampling fits routing

### Step 2: Understand the math

Read:

- sections 6-12

Goal:

- know Beta mean
- know posterior update
- understand fractional reward nuance

### Step 3: Understand the production adaptations

Read:

- sections 13-18

Goal:

- know cold start
- know drift
- know safety nuances

### Step 4: Map theory to code

Read:

- `model_routing/bayesian_router/router.py`
- `model_routing/bayesian_router/rewards.py`
- `model_routing/bayesian_router/simulator.py`

Goal:

- be able to point to implementation details

### Step 5: Practice audience questions

Read:

- sections 26-29

Goal:

- answer confidently
- sound honest and precise

---

## 31. Recommended External Reading

### Foundational / closest to your topic

- [FrugalGPT](https://arxiv.org/abs/2305.05176)
- [RouteLLM](https://arxiv.org/abs/2406.18665)
- [A Tutorial on Thompson Sampling](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)

### Easier practical reading

- [Solving The Multi-Armed Bandit Problem with Thompson Sampling](https://oren0e.github.io/2020/04/27/mab_thompson/)
- [Martin Fowler on Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Shadow Traffic for LLM Testing](https://www.codeant.ai/blogs/shadow-traffic-llm-testing)
- [Databricks: Deployment to Drift](https://databricks.com/blog/2019/09/18/productionizing-machine-learning-from-deployment-to-drift-detection.html)

---

## 32. Final Summary

If you remember only a few things, remember these:

1. A bandit is for repeated decision-making under uncertainty.
2. Thompson Sampling chooses actions by sampling from posterior beliefs.
3.
A Beta distribution is written as Beta(alpha, beta).

In simple terms:

alpha = positive evidence / success-like evidence
beta = negative evidence / failure-like evidence
So:

Beta(5, 4) means the distribution is parameterized by alpha = 5 and beta = 4
they control the shape of the belief distribution
they are not the mean themselves
Useful intuition:

larger alpha pushes belief toward higher quality
larger beta pushes belief toward lower quality
larger alpha + beta means more confidence / less uncertainty
The mean is:

alpha / (alpha + beta)

Example:

Beta(8, 3) has mean 8 / 11 ~= 0.727
In your router:

the Beta distribution = the belief about a model
alpha and beta = the two numbers that store that belief


Beta distributions store the belief about model quality
Bayes updates alpha and beta after new evidence
Thompson Sampling draws one temporary score from that updated Beta belief
Simple version:

Beta distribution = where the belief lives
Bayes theorem = how that belief gets updated
Thompson Sampling = how the router uses that belief to choose a model

4. Your implementation uses a **continuous proxy reward**, so the update is a practical pseudo-count heuristic, not exact Beta-Bernoulli conjugacy.
5. The router solves a real production problem: adapting model choice online without human labels.
6. Cold start, drift, fallback, and exploration are what make the talk production-relevant.
7. The current package is best described as a **non-contextual adaptive router** with pragmatic production heuristics.
8. Classical Thompson Sampling theory motivates the design, but the exact regret guarantees do not carry over unchanged to the continuous-reward, decayed production version.

If you are clear on those eight points, you will be able to deliver the talk well and handle most audience questions confidently.