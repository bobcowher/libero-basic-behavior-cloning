#!/usr/bin/env bash
#
# download_models.sh [RUN_ID] — fetch trained weights off the Beekeeper box.
#
#   ./download_models.sh          # latest run
#   ./download_models.sh 427      # a specific run
#
# Downloads every checkpoint the run produced, named checkpoints/run<ID>_<name>.
# The run id is baked into the filename on purpose: this project has lost
# checkpoints twice to blind overwrites, and a download that clobbered
# checkpoints/bc_network would be the third. Nothing here overwrites weights
# from a different run.
set -euo pipefail

BEEKEEPER_HOST="${BEEKEEPER_HOST:-http://lab.local:5000}"
PROJECT="libero-behavior-cloning"
RUN_ID="${1:-latest}"
DEST="$(dirname "$0")/checkpoints"

API="${BEEKEEPER_HOST}/api/v1/projects/${PROJECT}/runs/${RUN_ID}"

echo "Listing checkpoints for run '$RUN_ID' on $BEEKEEPER_HOST..."
LISTING="$(curl -fsSL "${API}/files/checkpoints")"

# Resolve "latest" to the real run id -- the listing reports which run it served,
# so the saved filenames say what they actually are.
RESOLVED="$(printf '%s' "$LISTING" | grep -oE '"run_id":[0-9]+' | head -1 | cut -d: -f2)"
NAMES="$(printf '%s' "$LISTING" | grep -oE '"name":"[^"]+"' | cut -d'"' -f4)"

if [ -z "$NAMES" ]; then
  echo "No checkpoints found for run '$RUN_ID'." >&2
  exit 1
fi

mkdir -p "$DEST"
for name in $NAMES; do
  out="$DEST/run${RESOLVED}_${name}"
  echo "  $name -> $out"
  curl -fsSL "${API}/files/checkpoints/${name}" -o "$out"
done

echo
echo "Done. Score one with:"
for name in $NAMES; do
  echo "  ./build.sh --ckpt checkpoints/run${RESOLVED}_${name} --n-eval 50 --env-num 10"
done
echo
ls -lh "$DEST"
