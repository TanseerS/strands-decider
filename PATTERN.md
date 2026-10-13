# Decision pattern: context relevance gate

**Tags:** `#strands-decider-2b` `#context-memory`

A drop-in gate that scores which memory or context chunks are relevant before
injecting them into an agent's prompt.

## The problem

Retrieval ranks by *aboutness*. A note explaining unexpected card charges and a
note explaining failed payout batches share almost all their vocabulary, so an
embedding search returns both for either question. The irrelevant one costs
tokens and invites the generative model to use it.

Similarity cannot separate them, because they genuinely are similar. The question
you actually want answered is *does this note answer the question*, which is a
typed decision, not a distance.

## The decision handed to Decider 2B

One `ask()` per candidate chunk, three axes:

| Axis | Type | Role |
|---|---|---|
| `answers_it` | noul | The decision. Calibrated probability the note answers the question. |
| `same_topic` | noul | Catches same-domain, different-problem decoys. |
| `specificity` | score (3 levels) | Diagnostic: vague background → concrete procedure. |

The state given to the model:

```
USER QUESTION:
{query}

REFERENCE NOTE:
{chunk}

Decide whether this reference note should be given to an assistant that is
about to answer the user's question.
```

## Why not a cross-encoder reranker?

A reranker does score query-document pairs, and if all you need is "put the best
chunk first", use one. This pattern exists for the three things a reranker cannot
do:

| | reranker | typed decision model |
|---|---|---|
| Output | relative ordering | calibrated absolute probability |
| Threshold | re-tuned per corpus | 0.45 means the same thing everywhere |
| "None of these" | cannot express it | `answers_it` low across the board |
| Why it chose | an opaque score | named axes you wrote |

The practical difference is the cut, not the ranking. A reranker tells you kb01
beats kb02; it does not tell you whether kb01 is good enough to use at all. With
six candidates and no relevant answer among them, a reranker still returns a
confident ordering of six irrelevant notes. A calibrated `answers_it` says 0.2
six times, and the gate injects nothing.

The axes are the other half. `same_topic` and `specificity` are not reranking
signals; they are questions chosen for this corpus, and their values are legible
in a log. You can add an axis for anything your domain cares about -- is it
current, is it authoritative, does it apply to this customer's plan -- without
training anything.

## Wiring it in

```python
from patterns import ContextGate
from strands_decider.infer import load_engine

gate = ContextGate(load_engine(CHECKPOINT, device="mps"))
kept = gate.select(user_question, candidate_chunks, k=3)
prompt = build_prompt(user_question, [c.text for c in kept])
```

Any object exposing `.ask(state, questions)` works as the engine. Chunks need
only `.id` and `.text`; `as_chunks()` adapts plain dicts.

For a Strands agent, call `select()` from an `InterventionHandler` before the
model call, or in your retriever's post-processing step.

## Letting the agent explain itself

`select()` returns survivors. `report()` returns a `GateReport` that keeps the
rejections too, which is what lets the agent account for its own context window:

```python
report = gate.report(user_question, candidate_chunks, k=3)

report.grounding      # "strong" | "partial" | "none"
report.kept           # what went into the prompt
report.near_misses    # the best of what did not, for when you decline
print(report.trace())
```

```
grounding: strong
  used    kb01  0.79  (a concrete procedure, answers it (0.79))
  ignored kb05  0.55  (does not answer it (0.55))
  ignored kb04  0.34  (does not answer it (0.34))
```

`grounding` is the agent-visible consequence of calibration, and it only works
because `answers_it` is an absolute probability rather than a relative score: the
strength of the best surviving chunk means something on its own. Feed it into
your system prompt and the agent matches its tone to its evidence instead of
always sounding certain — confident at `strong`, explicitly hedged at `partial`.

At `none`, show `near_misses` rather than declining silently. A silent decline
looks identical whether your memory store is empty or your gate just rejected
the right note, and you want to be able to tell those apart.

## Why calibration matters here

`threshold` is applied to a calibrated probability, so 0.45 means the same thing
across different corpora — you are not re-tuning a magic number per dataset.

`min_confidence` makes the gate **abstain** rather than guess, and defaults to
0.0 — off. A noul answer of 0.5 is maximum doubt, so confidence is
`abs(p - 0.5) * 2`, and anything below the floor is dropped.

Abstention sounds like the safe default, and it measured badly. Shipping this
with a floor of 0.35 scored P 1.00 / R 0.58 on the bundled eval: it discarded
five of twelve correct answers, four by vetoing notes the threshold had already
scored correctly. Removing it took F1 from 0.74 to 0.92. Raise it to ~0.10 only
if a wrong chunk costs you more than a missing one, and sweep it on your own
data with `sweep.py` rather than trusting either number.

## `answers_it` means "answers all of it"

The axis is strict, and on compound questions that bites. A question asking about
two things — "we were charged twice **for the same annual plan**" — scores 0.27
against a note explaining duplicate charges generically, because the note says
nothing about plans. Remove those two words and the same note scores 0.65.

```
0.27  we were charged twice for the same annual plan
0.65  we got a duplicate charge
```

Softening the wording barely helps (`"would this be useful"` reaches 0.39). If
your queries are compound, either split them into separate axes and combine the
results yourself, or accept that the gate will decline rather than half-answer.
`probe.py` in the repo is the tool for diagnosing this on your own corpus.

## Knobs

| | default | |
|---|---|---|
| `threshold` | 0.45 | Minimum `answers_it` to keep a chunk |
| `min_confidence` | 0.0 | Below this the gate abstains and drops. Off by default — see above |
| `questions` | `RELEVANCE_QUESTIONS` | Swap in your own typed question set |

## What you get back

Every decision carries its own explanation, so the agent can show its work:

```
GateResult(id, text, keep, relevance, confidence, axes, reason)
```

which prints as `KEEP kb01 relevance=0.91 answers it (0.91)`.
