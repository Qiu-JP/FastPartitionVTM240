#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/data"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --input-dir DIR --output-file FILE [--frames N] [--fps N]

Generate a DIV2K-style sequence list by scanning image files and inferring width/height via ffprobe.
EOF
}

input_dir=""
output_file="${SCRIPT_DIR}/Training_Sequences_DIV2K.txt"
frames=1
fps=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      input_dir="$2"
      shift 2
      ;;
    --output-file)
      output_file="$2"
      shift 2
      ;;
    --frames)
      frames="$2"
      shift 2
      ;;
    --fps)
      fps="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${input_dir}" ]]; then
  echo "--input-dir is required." >&2
  usage >&2
  exit 1
fi

: > "${output_file}"

shopt -s nullglob
for img in "${input_dir}"/*.png; do
  base="$(basename "${img}" .png)"
  width="$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of default=nw=1:nk=1 "${img}")"
  height="$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=nw=1:nk=1 "${img}")"
  echo "${base},${base}_${width}x${height}.yuv,${width},${height},${frames},${fps}" >> "${output_file}"
done

echo "Generated sequence list: ${output_file}"
