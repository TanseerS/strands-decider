#!/usr/bin/env bash
# Scripted demo for the submission video. Paced for screen recording.
#
#   ./demo.sh            run it
#   ./demo.sh --fast     no pauses, for a dry run
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
PAUSE=${PAUSE:-2.5}
[[ "${1:-}" == "--fast" ]] && PAUSE=0

say()  { printf '\n\033[1;36m%s\033[0m\n' "$*"; sleep "$PAUSE"; }
run()  { printf '\033[2m$ %s\033[0m\n' "$*"; "$@"; sleep "$PAUSE"; }

say "Strands Decider 2B decides which memories my agent is allowed to see."
say "The memory store. Twelve support notes -- several are decoys."
run cat data/memory.jsonl

say "Ask about failed payouts. Lexical search returns six candidates."
say "Notes about card charges and duplicate charges score high: same words, wrong answer."
run $PY agent.py --compare "our nightly payout batch keeps failing"

say "Decider 2B kept one note and dropped every decoy -- and said why."
say "It also abstained on kb05: a real payout note about a different payout problem."

say "Does it hold up? Twelve labelled queries."
run $PY eval.py

say "The reusable pattern is in patterns/context_relevance.py -- drop it in your own agent."
run sed -n '1,30p' PATTERN.md
