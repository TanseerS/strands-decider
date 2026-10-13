# The agent that shows you the memory it threw away

**Agent Experience Hackathon submission** · `#strands-decider-2b` `#context-memory`

| | |
|---|---|
| **Decision pattern** | Context and memory — score which chunks are relevant before injecting them |
| **Decision model** | [Strands Decider 2B](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19) (`hobson-v19`), local |
| **Generative model** | [Amazon Bedrock](https://aws.amazon.com/bedrock/) `us.amazon.nova-lite-v1:0` |
| **Reusable pattern** | [`patterns/context_relevance.py`](patterns/context_relevance.py) — typed questions, `ContextGate`, `GateReport` |
| **Wiring** | [`patterns/strands_wiring.py`](patterns/strands_wiring.py) — a Strands `InterventionHandler`; example in [`strands_agent.py`](strands_agent.py) |
| **How to use it** | [`PATTERN.md`](PATTERN.md) |
| **Write-up** | [`ARTICLE.md`](ARTICLE.md) |
| **Demo** | [`demo.mp4`](demo.mp4) — 2560×1440, 1:54 |

```
Memory trace:
  grounding: none
    ignored kb03  0.27  (does not answer it (0.27))

Nothing in memory answers this, so the agent declines rather than
inventing an answer. The closest it looked at:

  kb03  0.27  Double charges: duplicate charges occur when a payment is retried...
```

kb03 is the right note, and the agent threw it away — then said so, with the
number attached. That is the point of this: not a filter that silently drops
context, but an agent that can account for what it read, what it refused, and how
sure it was, so a bad call surfaces as a bug report instead of a shrug.

## Why there is a gate at all

Retrieval is good at finding notes that are *about* your question. It is bad at
finding notes that *answer* it. Those are different things, and the gap is where
RAG agents quietly go wrong.

A note about unexpected card charges and a note about failed payout batches share
a vocabulary — payment, charge, billing, failure. Embedding search returns both
for either question, and it is not malfunctioning: they really are similar. Only
one answers the question.

Similarity cannot separate them, because similarity is not the thing you want to
measure. *Does this note answer this question* is a typed decision, not a
distance. So that decision goes to Decider 2B.

## Run it

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python agent.py --compare "our nightly payout batch keeps failing"
```

First run downloads ~5 GB to `~/.cache/huggingface`: the checkpoint (a LoRA
adapter plus a pointer head) and its base model, `Qwen/Qwen3.5-2B-Base`.
Decider 2B runs locally and needs no AWS access; Bedrock needs credentials with
`bedrock:InvokeModel` in `us-east-1`.

```
retriever proposed 6: kb01 kb05 kb02 kb03 kb04 kb06

Decider 2B gate:
  [keep] kb01  answers_it=0.79 same_topic=0.81 specificity=1.54  -- answers it (0.79)
  [keep] kb05  answers_it=0.55 same_topic=0.62 specificity=1.26  -- answers it (0.55)
  [drop] kb02  answers_it=0.19 same_topic=0.19 specificity=1.00  -- does not answer it (0.19)
  ...

Memory trace:
  grounding: strong
    used    kb01  0.79  (a concrete procedure, answers it (0.79))
    used    kb05  0.55  (general guidance, answers it (0.55))
    ignored kb04  0.34  (does not answer it (0.34))

Bedrock answered from kb01 kb05, grounding strong, 160 input tokens
Ungated: all 6 chunks, 340 input tokens (53% saved)
```

The retriever deliberately has no relevance floor, because that is how top-k
vector search behaves: it always returns k candidates, relevant or not.
Supplying that floor is the gate's job.

## The agent explains its own memory

`grounding` is the agent-visible consequence of calibration, and it changes how
the agent speaks rather than only what it reads:

| | |
|---|---|
| `strong` (best ≥ 0.7) | answer plainly |
| `partial` | answer what the notes support, name what they do not cover |
| `none` | decline, and show the near-misses |

The decline path makes the gate's own mistakes visible:

```
grounding: none
Nothing in memory answers this, so the agent declines rather than
inventing an answer. The closest it looked at:

  kb03  0.27  Double charges: duplicate charges occur when a payment is retried...
```

kb03 is the correct note, sitting right there at 0.27. A silent decline looks
identical whether the store is empty or the gate blundered.

It turns out not to be a blunder. `python probe.py` isolates it: the question
asks about a duplicate charge **and** an annual plan, kb03 covers only the first,
and `answers_it` means *answers all of it*. Drop the words "annual plan" and the
same note scores 0.65. Strictness, not confusion — see
[`data/probe_result.txt`](data/probe_result.txt).

## Does the gate help?

Twelve labelled queries, one correct note each:

```sh
.venv/bin/python eval.py     # labelled eval; caches raw scores
.venv/bin/python sweep.py    # sweep thresholds offline, instantly
```

```
             precision   recall      F1
retriever         0.17     1.00    0.29
+ gate            0.92     0.92    0.92

context shrunk 15321 -> 2455 chars (84% less)
```

### The abstention floor was pure loss

The gate originally shipped with `threshold=0.6, min_confidence=0.35`, reasoning
that refusing to guess is safer than guessing. Measured, that scored
**P 1.00 / R 0.58** — five of twelve correct answers discarded, four by
abstaining on notes it had already scored correctly.

```
thresh  minconf   prec  recall     F1  context
  0.45     0.00   0.92    0.92   0.92      16%   <- current default
  0.45     0.10   1.00    0.83   0.91      13%   <- precision-first
  0.60     0.35   1.00    0.58   0.74       9%   <- original guess
```

Abstention is not free safety. `sweep.py` exists so you can find your own
operating point rather than inheriting a guess.

### Does it make the agent *right*? No.

```sh
.venv/bin/python contrast.py --candidates 12
```

Five questions, each with a plausibly-misleading note in the store, run both
ways. **The ungated agent answered correctly 5 of 5**, with six candidates and
again with the full twelve-note store. Nova-lite ignores a card-charge note when
asked about payout batches. On one query the gate was actively harmful.

What the gate measurably buys: **cost** (160 input tokens vs 340, and 105 vs 586
with the full store), **the ability to decline**, and **an auditable context
window**. Not accuracy, at this scale. That may change as a store grows into the
thousands — unmeasured, so unclaimed.

Raw output: [`data/contrast_6.txt`](data/contrast_6.txt), [`data/contrast_12.txt`](data/contrast_12.txt).

## Layout

| | |
|---|---|
| `agent.py` | the agent — retrieve, gate, answer via Bedrock |
| `patterns/context_relevance.py` | **the reusable pattern**: `ContextGate`, typed questions, `GateReport` |
| `patterns/strands_wiring.py` | **the wiring**: a Strands `InterventionHandler` |
| `strands_agent.py` | the same gate inside a real Strands agent |
| `eval.py` / `sweep.py` | labelled eval, and offline threshold sweep |
| `contrast.py` | gated vs ungated answers |
| `probe.py` | diagnosing the kb03 rejection |
| `data/` | memory store, labelled queries, raw results |
| `ARTICLE.md` / `PATTERN.md` / `VIDEO.md` | write-up, pattern docs, shot list |
| `demo.sh` / `demo.tape` / `render.sh` / `shots.sh` | demo, recording, images |

## Appendix: the experiments this grew out of

Two earlier explorations, kept because they are what led to the pattern.

**[`route.py`](route.py)** — the first thing built: routing a ticket to one of
three teams with a single `ChoiceQuestion`. Shows the `#policy` shape.

**[`fingerprint.py`](fingerprint.py) + [`analyze.py`](analyze.py)** —
interpretable embeddings. Scoring a document against a rubric of named questions
gives a short vector where every axis has a name and a calibrated value, so
clustering, anomaly detection and drift all explain themselves:

```
cluster 0  (2):  security +0.49, billing -0.29     t20 t26
billing     0.28 -> 0.59   up 0.30      (drift, q1 -> q2)
```

That idea — named, calibrated axes instead of one opaque score — is what the
context gate is built on.

**[`bench_prefix.py`](bench_prefix.py)** — batching benchmark. One pass with 8
questions is ~2.4× cheaper than 8 separate calls, not 8×: the shared prefix cache
amortises encoding the state, not scoring the questions.
