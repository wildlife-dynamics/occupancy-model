#!/usr/bin/env bash
# Dev compile for occupancy-model, with a minor-version bump per revision.
#
#   ./dev/compile.sh            # compile (--clobber --install) and set VERSION = committed MIN + 1
#   ./dev/compile.sh --no-bump  # compile only; restore the committed VERSION.yaml
#
# A dev compile resets <WF>/VERSION.yaml to 0.0.0, and the compiler's --update bumps MAJ
# whenever params change, so the version is set here instead: the minor version of the
# last *committed* VERSION.yaml plus one (PATCH reset to 0). Re-running within the same
# revision gives the same version; committing the revision makes the next one bump again.
# The publish compile (/ecoscope:publish) follows the repo's CI and must keep this version.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WF="ecoscope-workflows-occupancy-model-workflow"
VERSION_FILE="$WF/VERSION.yaml"

BUMP=1
[[ "${1:-}" == "--no-bump" ]] && BUMP=0

committed="$(git show "HEAD:$VERSION_FILE" 2>/dev/null || echo '{MAJ: 0, MIN: 0, PATCH: 0}')"
read -r MAJ MIN PATCH < <(python3 -c "import re,sys; s=sys.argv[1]; print(*[re.search(k+r': *(\d+)', s).group(1) for k in ('MAJ','MIN','PATCH')])" "$committed")

pixi run --manifest-path pixi.toml wt-compiler compile \
  --spec spec.yaml \
  --pkg-name-prefix=ecoscope-workflows \
  --results-env-var=ECOSCOPE_WORKFLOWS_RESULTS \
  --clobber --install --no-progress

if [[ "$BUMP" == 1 ]]; then
  MIN=$((MIN + 1))
  PATCH=0
fi
echo "{MAJ: $MAJ, MIN: $MIN, PATCH: $PATCH}" > "$VERSION_FILE"
echo "VERSION: $MAJ.$MIN.$PATCH (committed was $(tr -d '\n' <<<"$committed"))"
