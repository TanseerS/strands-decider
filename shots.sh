#!/usr/bin/env bash
# Regenerate every image referenced by ARTICLE.md into images/.
#
#   ./shots.sh
#
# Terminal shots are rendered by vhs and the final frame extracted with ffmpeg.
# Shots of the agent running are pulled straight from demo.mp4, so the article
# and the video show the identical run rather than two different ones.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p images

# render <output.png> <height> <command...>
render() {
  out=$1; h=$2; shift 2
  cat > /tmp/shot.tape <<TAPE
Output "/tmp/shot.mp4"
Set Shell "bash"
Set FontSize 22
Set Width 2000
Set Height $h
Set Theme "Catppuccin Mocha"
Set Padding 30
Hide
Type "export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error"
Enter
Type "clear"
Enter
Show
Type "$*"
Enter
Sleep 4s
TAPE
  vhs /tmp/shot.tape >/dev/null 2>&1
  ffmpeg -v error -sseof -1 -i /tmp/shot.mp4 -frames:v 1 "images/$out" -y
  echo "  images/$out"
}

# grab <output.png> <seconds-into-demo.mp4>
grab() {
  ffmpeg -v error -ss "$2" -i demo.mp4 -frames:v 1 "images/$1" -y
  echo "  images/$1 (from demo.mp4 @ ${2}s)"
}

echo "rendering:"
render 01-architecture.png   1150 "cat docs/architecture.txt"
render 02-decoy-notes.png     700 "cut -c1-170 data/memory.jsonl | sed -n '1,3p'"
render 07-git-history.png     620 "git --no-pager log --format='%h  %an <%ae>  %ad  %s' --date=short"
render 08-repo-structure.png  900 "ls -1 *.py *.md *.sh *.tape data/ patterns/"

echo "extracting from demo.mp4:"
grab 03-gate-decisions.png     62
grab 04-eval-results.png       93
grab 05-sweep-abstention.png   97
grab 06-pattern-usage.png     110

echo
ls -la images/
