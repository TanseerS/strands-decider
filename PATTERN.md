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

## Wiring it in

```python
from patterns import ContextGate
from strands_decider.infer import load_engine

gate = ContextGate(load_engine(CHECKPOINT, device="mps"), threshold=0.6)
kept = gate.select(user_question, candidate_chunks, k=3)
prompt = build_prompt(user_question, [c.text for c in kept])
```

Any object exposing `.ask(state, questions)` works as the engine. Chunks need
only `.id` and `.text`; `as_chunks()` adapts plain dicts.

For a Strands agent, call `select()` from an `InterventionHandler` before the
model call, or in your retriever's post-processing step.

## Why calibration matters here

`threshold` is applied to a calibrated probability, so 0.6 means the same thing
across different corpora — you are not re-tuning a magic number per dataset.

`min_confidence` makes the gate **abstain** rather than guess. A noul answer of
0.5 is maximum doubt, so confidence is `abs(p - 0.5) * 2`, and anything below the
floor is dropped. Dropping is the safe direction: a missing note makes the agent
say "I don't know", while a wrong note makes it answer confidently and wrongly.

## Knobs

| | default | |
|---|---|---|
| `threshold` | 0.6 | Minimum `answers_it` to keep a chunk |
| `min_confidence` | 0.35 | Below this the gate abstains and drops |
| `questions` | `RELEVANCE_QUESTIONS` | Swap in your own typed question set |

## What you get back

Every decision carries its own explanation, so the agent can show its work:

```
GateResult(id, text, keep, relevance, confidence, axes, reason)
```

which prints as `KEEP kb01 relevance=0.91 answers it (0.91)`.
