#!/usr/bin/env bash
#
# Upload PDF files to a Unity Catalog volume using the Databricks CLI.
#
# Requires: databricks CLI v0.205+ (the unified CLI).
#
set -euo pipefail

# ----- defaults ----------------------------------------------------------------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
SOURCE="${repo_root}/document-embedding-chart-analysis/sample_data"
PROFILE=""; CATALOG=""; SCHEMA=""; VOLUME=""; FOLDER=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --profile <p> --catalog <c> --schema <s> --volume <v> --folder <f> [--source <path>]

Uploads PDF files into /Volumes/<catalog>/<schema>/<volume>/<folder>/.

Required:
  -p, --profile   Databricks CLI profile (from ~/.databrickscfg)
  -c, --catalog   Unity Catalog name
  -s, --schema    Schema name
  -v, --volume    Volume name
  -f, --folder    Subfolder inside the volume (e.g. scanned_pdfs)

Optional:
  -i, --source    Local PDF file or directory of PDFs
                  (default: ../document-embedding-chart-analysis/sample_data)
  -h, --help      Show this help

Example:
  $(basename "$0") -p DEFAULT -c fins_genai -s unstructured_documents \\
                   -v pdf_examples -f scanned_pdfs
EOF
}

# ----- parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--profile) PROFILE="$2"; shift 2 ;;
    -c|--catalog) CATALOG="$2"; shift 2 ;;
    -s|--schema)  SCHEMA="$2";  shift 2 ;;
    -v|--volume)  VOLUME="$2";  shift 2 ;;
    -f|--folder)  FOLDER="$2";  shift 2 ;;
    -i|--source)  SOURCE="$2";  shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

missing=()
[[ -z "$PROFILE" ]] && missing+=("--profile")
[[ -z "$CATALOG" ]] && missing+=("--catalog")
[[ -z "$SCHEMA"  ]] && missing+=("--schema")
[[ -z "$VOLUME"  ]] && missing+=("--volume")
[[ -z "$FOLDER"  ]] && missing+=("--folder")
if (( ${#missing[@]} > 0 )); then
  echo "Error: missing required arguments: ${missing[*]}" >&2
  usage; exit 1
fi

# ----- pre-flight --------------------------------------------------------------
command -v databricks >/dev/null 2>&1 || {
  echo "Error: 'databricks' CLI not found in PATH." >&2
  echo "       Install: https://docs.databricks.com/en/dev-tools/cli/install.html" >&2
  exit 1
}

[[ -e "$SOURCE" ]] || { echo "Error: source path does not exist: $SOURCE" >&2; exit 1; }

VOLUME_ROOT="dbfs:/Volumes/${CATALOG}/${SCHEMA}/${VOLUME}"
TARGET_DIR="${VOLUME_ROOT}/${FOLDER}"

echo "Profile : $PROFILE"
echo "Source  : $SOURCE"
echo "Target  : $TARGET_DIR"
echo

# ----- verify auth + volume access --------------------------------------------
echo "Verifying volume access..."
if ! databricks --profile "$PROFILE" fs ls "$VOLUME_ROOT" >/dev/null 2>&1; then
  echo "Error: cannot list '$VOLUME_ROOT' with profile '$PROFILE'." >&2
  echo "       Check the volume exists, auth is valid, and the profile has READ VOLUME." >&2
  echo "       Re-auth with:  databricks --profile $PROFILE auth login" >&2
  exit 1
fi

# ----- ensure target folder exists --------------------------------------------
echo "Ensuring target folder exists..."
databricks --profile "$PROFILE" fs mkdirs "$TARGET_DIR"

# ----- collect PDFs ------------------------------------------------------------
files=()
if [[ -d "$SOURCE" ]]; then
  # immediate children only; flat upload preserves filenames
  while IFS= read -r f; do files+=("$f"); done < <(
    find "$SOURCE" -maxdepth 1 -type f -iname '*.pdf' | sort
  )
elif [[ -f "$SOURCE" ]]; then
  files=("$SOURCE")
fi

if (( ${#files[@]} == 0 )); then
  echo "Error: no PDF files found under: $SOURCE" >&2
  exit 1
fi

# ----- upload ------------------------------------------------------------------
echo
echo "Uploading ${#files[@]} file(s)..."
for f in "${files[@]}"; do
  fname="$(basename "$f")"
  echo "  -> $fname"
  databricks --profile "$PROFILE" fs cp --overwrite "$f" "$TARGET_DIR/$fname"
done

echo
echo "Listing target folder:"
databricks --profile "$PROFILE" fs ls "$TARGET_DIR"
