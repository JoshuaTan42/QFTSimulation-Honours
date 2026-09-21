#!/bin/sh
# Type-check every proof with a plain Lean 4 toolchain (no Mathlib, no lake).
#   LEAN=/path/to/lean ./check.sh
set -e
LEAN="${LEAN:-lean}"
cd "$(dirname "$0")"
"$LEAN" -o Lattice.olean Lattice.lean
for f in GaugeInvariance Colouring Casimir PlaquettePauli; do
  LEAN_PATH=. "$LEAN" "$f.lean" && echo "ok  $f.lean"
done
echo "ok  Lattice.lean"
rm -f Lattice.olean
