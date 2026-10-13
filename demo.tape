# VHS tape for the submission demo video.
#   vhs demo.tape   ->  demo.mp4 (3840x2160)
#
# Captions are printed by the script itself, so the video explains itself
# without narration. Everything shown is real output, recorded live.
#
# Timing: loading 4.3 GB of weights under ttyd takes ~40s, then six chunk
# decisions at ~4s each. The agent block needs ~95s. Do not trim it below that
# or the gate output gets clipped.

Output demo.mp4

Set Shell "bash"
# 2560x1440. Measured ceiling for this machine: at 3840x2160 the headless
# capture cannot keep up, frames drop, and 150s of content encodes as 62s.
# Font is scaled with the canvas to keep ~164 columns, which the agent's gate
# output needs to avoid wrapping.
Set FontSize 26
Set Width 2560
Set Height 1440
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms
Set Padding 40

Hide
Type "export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error"
Enter
Type "clear"
Enter
Sleep 1s
Show

Type "# Strands Decider 2B decides what my agent is allowed to remember"
Enter
Sleep 2s

Type "# Three notes. Same words -- payout, charge, billing. One answers the question."
Enter
Sleep 1500ms
Type "cut -c1-160 data/memory.jsonl | sed -n '1,3p'"
Enter
Sleep 6s

Type "clear"
Enter
Sleep 500ms

Type "# Retrieval returns 6 candidates with no relevance floor. The gate supplies it."
Enter
Sleep 2s
Type "./.venv/bin/python agent.py --compare 'our nightly payout batch keeps failing'"
Enter
Sleep 95s

Type "clear"
Enter
Sleep 500ms

Type "# Twelve labelled queries. Cached -- the live run scores 72 chunks, 3.5 minutes."
Enter
Sleep 1500ms
Type "sed -n '/precision/,/84% less/p' data/eval_result.txt"
Enter
Sleep 7s

Type "# I shipped an abstention floor, thinking refusing to guess was safer."
Enter
Sleep 2s
Type "# It cost a third of my recall and bought no precision at all:"
Enter
Sleep 1500ms
Type "./.venv/bin/python sweep.py | tail -8"
Enter
Sleep 10s

Type "clear"
Enter
Sleep 500ms
Type "# The reusable pattern: one class, one typed question set."
Enter
Sleep 1500ms
Type "sed -n '/^from patterns import/,/build_prompt/p' PATTERN.md"
Enter
Sleep 8s
