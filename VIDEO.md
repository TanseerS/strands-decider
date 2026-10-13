# Demo video script

Target: under 3 minutes. Run `./demo.sh` and narrate over it, or record the
beats below. Everything here is real output — nothing is staged.

| Time | Show | Say |
|---|---|---|
| 0:00–0:15 | Title / the memory store | "My agent answers support questions from a memory store. The hard part isn't finding notes — it's knowing which ones deserve to be in the prompt." |
| 0:15–0:40 | `data/memory.jsonl`, highlight kb01 vs kb02/kb03 | "These three notes all use the same words: payout, charge, billing. Only one answers a question about failed payout batches. Embedding similarity can't tell them apart, because they genuinely are similar." |
| 0:40–1:20 | `agent.py --compare` running, gate output on screen | "Retrieval proposes six. Decider 2B scores each with a typed question — does this note actually answer the question — and keeps one. Every decision comes with a calibrated number and a reason." |
| 1:20–1:45 | Point at the kb05 line | "This is the part I didn't expect. kb05 is a real payout note, about a *different* payout problem. Decider scored it 0.55 — maximum doubt — and the gate abstained rather than guessing. Dropping is the safe direction: a missing note makes the agent say I don't know, a wrong note makes it answer confidently and wrongly." |
| 1:45–2:05 | Bedrock answer + token counts | "Bedrock Nova writes the reply from the one surviving note. 106 input tokens instead of 329 — 68% fewer — and it can't be led astray by the card-charge note." |
| 2:05–2:35 | `eval.py` results table | "Twelve labelled queries. Here's what the gate does to precision and recall." |
| 2:35–3:00 | `patterns/context_relevance.py` + PATTERN.md | "The reusable piece is one class and one typed question set. Point it at any engine, any chunks, and it drops into your retriever's post-processing or a Strands InterventionHandler." |

## Recording notes

- `PAUSE=4 ./demo.sh` slows the beats if you're narrating live.
- The first run loads 4.3 GB of weights (~10 s). Run it once before recording so
  the model is warm and the demo starts instantly.
- Gate latency is ~4 s per chunk on an M1. Either narrate over it or cut. Don't
  speed up the video to hide it — the honest number is in the article.
