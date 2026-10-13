# strands-decider

Routing tickets to teams with [Strands Decider 2B](https://strandsagents.com/blog/introducing-strands-decider/) —
a 2B decision model that scores a fixed set of options instead of generating text.

It runs locally. It is Apache-2.0 open weights on Hugging Face, not a Bedrock model,
so no AWS account or model access request is involved.

## Setup

Python 3.14 has no torch wheels yet, so the venv is pinned to 3.12:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

First run downloads ~5-6 GB to `~/.cache/huggingface`, as two repos: the
checkpoint itself (a LoRA adapter + pointer head) and its base model,
`Qwen/Qwen3.5-2B-Base`.

## CLI

```sh
.venv/bin/strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "Help! My payouts have been failing for 3 days!" \
  --choice "Which team should handle this?=billing,sales,retail"
```

Other question types: `--noul "question?"` for calibrated yes/no, and
`--score "question?=low,mid,high"` for an ordinal scale. `--json` prints the raw
response, `--device mps|cuda|cpu` overrides auto-detection.

## Python

```sh
.venv/bin/python route.py
```

`route.py` loads the engine once and routes several tickets, printing the chosen
option, its confidence, and the full probability spread. Unlike the CLI, it attaches
a description to each option via `ChoiceQuestion.criteria`.

## API shape

```python
from strands_decider.infer import load_engine
from strands_decider.schema import ChoiceQuestion

engine = load_engine("StrandsAgents/strands-decider-2B-hobson-v19", device="mps")
response = engine.ask(
    "Help! My payouts have been failing for 3 days!",
    {"team": ChoiceQuestion(
        instructions="Which team should handle this?",
        criteria={"billing": "payouts, refunds", "sales": "", "retail": ""},
    )},
)
answer = response.answers["team"]
answer.choice         # "billing"
answer.confidence     # calibrated confidence
answer.probabilities  # {"billing": ..., "sales": ..., "retail": ...}
```

`engine.ask` takes a dict of named questions, so one state can be asked several
questions in a single pass.

There is also `strands-decider serve`, which exposes `POST /v1/systemone`.

## Measured on an M1 (8 GB), device=mps

```
loaded on mps in 10.5s

Help! My payouts have been failing for 3 days!
  -> billing  (confidence 0.902, 4632ms)
     billing=0.935  sales=0.034  retail=0.031
Do you offer volume discounts for 500+ seats?
  -> sales  (confidence 0.718, 790ms)
     sales=0.812  billing=0.150  retail=0.038
The size chart on the product page is wrong.
  -> retail  (confidence 0.943, 575ms)
     retail=0.962  billing=0.022  sales=0.016
```

The first call carries MPS graph warmup; steady state is ~575-790 ms. The blog's
~153 ms figure is an M3, so budget several times that on an M1.

Two notes:

- `causal_conv1d is not installed` on startup is harmless. It is CUDA-only and cannot
  be installed on macOS, and the package's own profiling puts it at 32 ms of a 1,739 ms
  forward. The part that actually matters, the gated-delta chunk rule, is replaced by
  `strands_decider.mps_kernels`, which `SystemOneEngine` installs automatically
  whenever `device` starts with `mps`.
- Describing the options raises confidence. The bare CLI call, which sends `""` for
  every option description, returned `billing` at 0.769 confidence / 0.846 probability
  on the payouts ticket; `route.py`, with one line of description per team, returned
  0.902 / 0.935 on the same ticket.

---

# Interpretable embeddings

A conventional embedding is ~1,536 opaque floats. Score a document against a rubric
of named questions instead and you get a short vector where **every axis has a name
and a calibrated value** — so clustering, nearest-neighbour search, anomaly detection
and drift monitoring all become self-explaining.

```sh
.venv/bin/python fingerprint.py          # documents -> data/fingerprints.csv
.venv/bin/python analyze.py --k 4 --neighbours t16
```

| File | |
|---|---|
| `rubric.py` | The 11 axes. This is the design surface — edit this first. |
| `fingerprint.py` | Corpus -> `fingerprints.csv` + `confidence.csv` |
| `analyze.py` | Clusters, anomalies, neighbours, drift. Pure numpy, no sklearn. |
| `data/tickets.jsonl` | 30 sample support tickets, in two periods |
| `bench_prefix.py` | Batching benchmark (see below) |

## Why the axes are the whole design

`RUBRIC` maps a name to a `NoulQuestion` (yes/no) or `ScoreQuestion` (ordinal).
`normalise()` puts every answer on [0, 1] so the axes are mutually comparable: a noul
contributes its probability, a score contributes its expected level index divided by
`levels - 1`.

Keep the questions independent. Two axes asking the same thing in different words
double-count that trait in every distance downstream.

## What it produces

Clusters name themselves by the axes where they depart most from the corpus mean:

```
cluster 0  (2):  security +0.49, billing -0.29, angry -0.19     t20 t26
cluster 1  (7):  bug +0.40, angry +0.17, presales -0.11         t04 t10 t12 t16 ...
cluster 3  (10): angry -0.26, presales +0.21, billing -0.20     t01 t02 t03 t05 ...
```

Drift reports *which trait* moved, not merely that the distribution shifted:

```
billing     0.28 -> 0.59   up 0.30
angry       0.39 -> 0.57   up 0.18
presales    0.41 -> 0.27   down 0.14
```

That last table is the point. Cosine distance between embedding centroids can tell you
something changed; it cannot tell you your inbox got angrier about billing.

## Cost

Batching questions into one `ask()` is sublinear but not free. Measured on an M1 with
an 828-character state:

| | total | per question |
|---|---|---|
| 1 question | 978 ms | 978 ms |
| 8 questions, one pass | 3,274 ms | 409 ms |
| 8 questions, separate calls | 8,009 ms | 1,001 ms |

So one pass is ~2.4x cheaper than separate calls, not ~8x. The shared prefix cache
amortises encoding the state, not scoring the questions. Budget ~4.2 s per document
for an 11-axis rubric on an M1; a CUDA box will be several times faster.

## Adapting it

1. Rewrite `RUBRIC` for your domain — that is the real work.
2. Point `--corpus` at your own JSONL of `{"id", "text", "period"}`.
3. `confidence.csv` is written alongside the values. Filter on it before trusting a
   row: a fingerprint built from low-confidence answers is noise with a name.

---

# The agent: a context window chosen by Decider 2B

**Pattern:** `#context-memory` — see [PATTERN.md](PATTERN.md) for the reusable asset.

Retrieval proposes, Decider 2B disposes, Amazon Bedrock writes the answer.

```sh
.venv/bin/python agent.py --compare "our nightly payout batch keeps failing"
```

A lexical retriever pulls six candidate notes from memory. Decider 2B scores each
one with a typed question and keeps only those that actually answer the question.
Bedrock Nova drafts the reply from the survivors.

The decoys are the point. Ask about failed payouts and the retriever happily
returns notes about unexpected card charges and duplicate charges: same
vocabulary, same domain, wrong answer. Similarity cannot separate them because
they genuinely are similar.

```
retriever proposed 6: kb01 kb05 kb02 kb03 kb04 kb06

Decider 2B gate:
  [keep] kb01  answers_it=0.79 same_topic=0.81 specificity=1.54  -- answers it (0.79)
  [drop] kb05  answers_it=0.55 same_topic=0.62 specificity=1.26  -- abstained, confidence 0.10
  [drop] kb04  answers_it=0.34 same_topic=0.39 specificity=0.98  -- abstained, confidence 0.32
  [drop] kb06  answers_it=0.28 same_topic=0.28 specificity=0.91  -- does not answer it (0.28)
  [drop] kb03  answers_it=0.23 same_topic=0.25 specificity=1.14  -- does not answer it (0.23)
  [drop] kb02  answers_it=0.19 same_topic=0.19 specificity=1.00  -- does not answer it (0.19)

Bedrock answered from kb01, 106 input tokens
Ungated: all 6 chunks, 329 input tokens (68% saved)
```

Both answers are correct here — the gated one just costs a third as much and
cannot be led astray by kb02. On a larger memory store, where the decoys
outnumber the answer, that second property matters more than the tokens.

Note `kb05`: a real payout note, about a *different* payout problem. The gate
abstained at 0.55 rather than guessing. Dropping is the safe direction — a
missing note makes the agent say "I don't know", a wrong one makes it answer
confidently and wrongly.

## Models

Decider 2B runs locally (~4 s for 6 chunks on an M1; the blog reports ~115 ms per
decision on a 3090). The generative half is Bedrock `us.amazon.nova-lite-v1:0` in
`us-east-1` — change `BEDROCK_MODEL` in `agent.py` to use another.

## Does the gate actually help?

Twelve labelled queries over the memory store, one correct note each:

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

The retriever's precision is capped at 1/k by construction — it returns six
candidates whether or not six are relevant. The gate supplies the relevance floor
it lacks.

### The abstention floor was pure loss

The gate originally shipped with `threshold=0.6, min_confidence=0.35`, on the
reasoning that refusing to guess is safer than guessing. Measured, that scored
**P 1.00 / R 0.58** — it discarded five of twelve correct answers, four of them by
abstaining on notes it had already scored correctly.

```
thresh  minconf   prec  recall     F1  context
  0.45     0.00   0.92    0.92   0.92      16%   <- current default
  0.45     0.10   1.00    0.83   0.91      13%   <- precision-first
  0.60     0.35   1.00    0.58   0.74       9%   <- original guess
```

Abstention is not free safety. It bought no precision here and cost a third of the
recall. `sweep.py` exists so you can find your own operating point rather than
inheriting a guess — `eval.py` caches the raw per-chunk scores, so sweeping is
instant and never re-runs the model.

One genuine miss remains at the current settings: kb03 scores 0.27 on "we were
charged twice for the same annual plan" despite being exactly the right note. That
one is unexplained.

## Submission assets

| File | |
|---|---|
| `ARTICLE.md` | Draft of the Builder Center article |
| `PATTERN.md` | The reusable decision pattern, with usage notes |
| `VIDEO.md` | Timed shot list for the demo video |
| `demo.sh` | Scripted demo, paced for screen recording |
