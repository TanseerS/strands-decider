# Article images

Every image referenced by [`../ARTICLE.md`](../ARTICLE.md). Regenerate them all
with [`../shots.sh`](../shots.sh).

| File | What it shows | Source |
|---|---|---|
| `01-architecture.png` | retriever → Decider 2B gate → Bedrock | rendered from `docs/architecture.txt` |
| `02-decoy-notes.png` | the three notes that share a vocabulary | live `data/memory.jsonl` |
| `03-gate-decisions.png` | the gate keeping kb01, dropping five decoys, Bedrock answering | frame from `demo.mp4` @ 62s |
| `04-eval-results.png` | labelled eval, retriever vs gate | frame from `demo.mp4` @ 93s |
| `05-sweep-abstention.png` | the abstention floor measured against the tuned setting | frame from `demo.mp4` @ 97s |
| `06-pattern-usage.png` | the reusable pattern in three lines | frame from `demo.mp4` @ 110s |
| `07-git-history.png` | commit history and authorship | live `git log` |
| `08-repo-structure.png` | project layout | live `ls` |
| `09-agent-declines.png` | the agent declining, surfacing the note its gate wrongly rejected | live `agent.py` run |

Shots 03–06 are pulled from `demo.mp4` rather than re-run, so the article and the
video show the **same** run instead of two different ones with different numbers.

Nothing here is mocked up. Each image is a real terminal rendered by `vhs` against
the real repository, or a frame of the real recorded demo.
