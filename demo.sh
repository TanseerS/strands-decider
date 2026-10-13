#!/usr/bin/env bash
# Scripted demo for the submission video. Paced to land under 3 minutes.
#
#   ./demo.sh            run it
#   ./demo.sh --fast     no pauses, for a dry run
#
# Timing note: eval.py takes ~3.5 minutes because it scores 72 chunks with a 2B
# model. That does not fit in a 3-minute video, so this shows the cached result
# and runs sweep.py, which is instant because eval.py caches the raw scores.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
PAUSE=${PAUSE:-2.5}
[[ "${1:-}" == "--fast" ]] && PAUSE=0

say()  { printf '\n\033[1;36m%s\033[0m\n' "$*"; sleep "$PAUSE"; }
run()  { printf '\033[2m$ %s\033[0m\n' "$*"; "$@"; sleep "$PAUSE"; }

say "Strands Decider 2B decides which memories my agent is allowed to see."

say "Three notes. Same vocabulary -- payout, charge, billing. One answers the question."
run sed -n '1,3p' data/memory.jsonl

say "Ask about failed payouts. Lexical search returns six candidates, no relevance floor."
say "Decider 2B scores each one, then Bedrock answers from what survives."
run $PY agent.py --compare "our nightly payout batch keeps failing"

say "One kept, every decoy dropped, each with a reason. 68% fewer input tokens."

say "Does it hold up? Twelve labelled queries (cached -- the live run takes 3.5 min)."
run sed -n '/precision/,/84% less/p' data/eval_result.txt

say "I shipped it with an abstention floor, thinking refusing to guess was safer."
say "It cost a third of my recall and bought no precision. Here is the sweep:"
run $PY sweep.py

say "The reusable pattern: one class, one typed question set. Drop it in your agent."
run sed -n '/## Wiring it in/,/post-processing step./p' PATTERN.md
