#!/usr/bin/env bash
# Regenerate CHANGELOG.md from git tags + their commit subjects.
# Idempotent — safe to run on every release.
set -euo pipefail
cd "$(dirname "$0")/.."

cat > CHANGELOG.md <<'HEADER'
# Changelog

All notable changes per release. The "Unreleased" section accumulates
work on `main` since the last tag; everything below is auto-generated
from `git tag` + commit subjects (see `scripts/gen_changelog.sh`).

## [Unreleased]

HEADER

for t in $(git tag --sort=-creatordate | grep -E '^v[0-9]'); do
  DATE=$(git log -1 --format=%ad --date=short "$t")
  SUBJ=$(git log -1 --format=%s "$t" | sed 's/^v[0-9.]*[: ]*//' | sed 's/^v[0-9.]* //')
  printf "## [%s] - %s\n\n%s\n\n" "$t" "$DATE" "$SUBJ"
done >> CHANGELOG.md

echo "wrote CHANGELOG.md ($(wc -l < CHANGELOG.md) lines)"
