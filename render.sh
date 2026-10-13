#!/usr/bin/env bash
# Render the submission video end to end.
#
#   ./render.sh   ->  demo.mp4  (2560x1440, ~1:53)
#
# Two steps, because the raw capture has a dead patch in it:
#
# 1. vhs runs demo.tape, driving a real terminal against the real agent. The
#    model load is ~14s of identical frames with nothing on screen.
# 2. ffmpeg speeds that one window 5x. Every frame of actual output is kept at
#    real time -- only the wait is compressed.
#
# Resolution note: 3840x2160 does not work on this machine. The headless capture
# cannot keep up, frames drop, and 150s of content encodes as 62s. Measured:
# 1920x1080 accurate, 2560x1440 within ~5%, 4K 2.2x too fast. QHD is the ceiling.
set -euo pipefail
cd "$(dirname "$0")"

DEAD_FROM=${DEAD_FROM:-29}   # first second of the loading stall
DEAD_TO=${DEAD_TO:-43}       # first second of real gate output
SPEED=${SPEED:-5}

echo "1/2  vhs demo.tape"
vhs demo.tape

echo "2/2  compressing the ${DEAD_FROM}-${DEAD_TO}s load stall ${SPEED}x"
ffmpeg -v error -i demo.mp4 -filter_complex \
  "[0:v]trim=0:${DEAD_FROM},setpts=PTS-STARTPTS[a];
   [0:v]trim=${DEAD_FROM}:${DEAD_TO},setpts=(PTS-STARTPTS)/${SPEED}[b];
   [0:v]trim=${DEAD_TO},setpts=PTS-STARTPTS[c];
   [a][b][c]concat=n=3:v=1:a=0[out]" \
  -map "[out]" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p demo.trimmed.mp4 -y

mv demo.trimmed.mp4 demo.mp4
ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
        -show_entries format=duration -of default=nw=1 demo.mp4

# If vhs timing drifts, re-measure the stall before trusting DEAD_FROM/DEAD_TO:
#   for t in $(seq 24 2 50); do ffmpeg -v error -ss $t -i demo.mp4 \
#     -frames:v 1 /tmp/$t.png -y; done; ls -l /tmp/*.png
# A plateau of near-identical file sizes is the dead window.
