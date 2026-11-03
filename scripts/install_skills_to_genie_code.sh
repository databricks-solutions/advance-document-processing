#!/usr/bin/env bash
#
# Install this repo's skills (skills/<name>/SKILL.md + supporting files) into a
# Databricks workspace so Genie Code loads them.
#
# Usage:
#   bash scripts/install_skills_to_genie_code.sh --profile <profile>                 # all skills, personal
#   bash scripts/install_skills_to_genie_code.sh --profile <profile> parse-large-pdfs
#   bash scripts/install_skills_to_genie_code.sh --profile <profile> --workspace     # /Workspace/.assistant/skills (admins)
#   bash scripts/install_skills_to_genie_code.sh --profile <profile> --dest /Users/<you>/tmp_skills  # any folder (testing)
#
# Targets (per Genie Code docs):
#   personal:  /Users/<you>/.assistant/skills/<name>/
#   workspace: /Workspace/.assistant/skills/<name>/
#
# Only SKILL.md, *.md, *.py, *.sql, *.sh, *.json, *.yaml files are uploaded;
# __pycache__ and compiled files are skipped. Helper .py files must not start with
# "# Databricks notebook source", or the workspace converts them to notebooks.

set -euo pipefail

PROFILE=""
SCOPE="personal"
NAMES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --workspace) SCOPE="workspace"; shift ;;
    --dest) SCOPE="custom"; DEST_ROOT="$2"; shift 2 ;;
    -h|--help) sed -n 2,18p "$0"; exit 0 ;;
    *) NAMES+=("$1"); shift ;;
  esac
done
[[ -n "$PROFILE" ]] || { echo "error: --profile <name> is required (see: databricks auth profiles)" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILLS_DIR="$REPO_ROOT/skills"

if [[ "$SCOPE" == "workspace" ]]; then
  DEST_ROOT="/Workspace/.assistant/skills"
elif [[ "$SCOPE" == "personal" ]]; then
  USER_NAME="$(databricks current-user me --profile "$PROFILE" -o json | python3 -c 'import json,sys; print(json.load(sys.stdin)["userName"])')"
  DEST_ROOT="/Users/$USER_NAME/.assistant/skills"
fi

if [[ ${#NAMES[@]} -eq 0 ]]; then
  for d in "$SKILLS_DIR"/*/; do
    [[ -f "$d/SKILL.md" ]] && NAMES+=("$(basename "$d")")
  done
fi

echo "Profile: $PROFILE"
echo "Target:  $DEST_ROOT"
for name in "${NAMES[@]}"; do
  src="$SKILLS_DIR/$name"
  [[ -f "$src/SKILL.md" ]] || { echo "  SKIP $name (no SKILL.md)"; continue; }
  count=0
  while IFS= read -r -d '' f; do
    rel="${f#"$src"/}"
    dest="$DEST_ROOT/$name/$rel"
    databricks workspace mkdirs "$(dirname "$dest")" --profile "$PROFILE"
    databricks workspace import "$dest" --file "$f" --format AUTO --overwrite --profile "$PROFILE"
    count=$((count + 1))
  done < <(find "$src" -type f \
             \( -name '*.md' -o -name '*.py' -o -name '*.sql' -o -name '*.sh' -o -name '*.json' -o -name '*.yaml' \) \
             -not -path '*/__pycache__/*' -print0)
  echo "  OK   $name ($count files)"
done
echo "Done. Genie Code loads skills by description match, or @-mention them."
