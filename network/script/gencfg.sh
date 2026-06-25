#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/data"
DEFAULT_TEMPLATE="${SCRIPT_DIR}/encoder_intra_vtm.cfg"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") --dataset DATASET --qp QP [options]

Options:
  --dataset NAME           Sequence set name, e.g. DIV2K / HEVC / VVC.
  --qp QP                  Quantization parameter.
  --type TYPE              Sequence list type: train or test. Default: train.
  --sequence-list PATH     Explicit sequence list path. Overrides --dataset + --type lookup.
  --template PATH          Encoder cfg template. Default: network/script/encoder_intra_vtm.cfg.
  --video-root PATH        Video root. Default: data/video/<dataset>.
  --output-root PATH       Output cfg root. Default: data/CodecTrainCfg/<dataset>/qp_<qp>.

Sequence list lookup rule:
  --type train -> network/script/Training_Sequences_<dataset>.txt
  --type test  -> network/script/Testing_Sequences_<dataset>.txt

Example:
  bash network/script/gencfg.sh --dataset DIV2K --qp 22 --type train
  bash network/script/gencfg.sh --dataset HEVC --qp 32 --type test
USAGE
}

dataset=""
qp=""
seq_type="train"
sequence_list=""
template_path="${DEFAULT_TEMPLATE}"
video_root=""
output_root=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      dataset="$2"
      shift 2
      ;;
    --qp)
      qp="$2"
      shift 2
      ;;
    --type)
      seq_type="$2"
      shift 2
      ;;
    --sequence-list)
      sequence_list="$2"
      shift 2
      ;;
    --template)
      template_path="$2"
      shift 2
      ;;
    --video-root)
      video_root="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
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

if [[ -z "${dataset}" || -z "${qp}" ]]; then
  echo "--dataset and --qp are required." >&2
  usage >&2
  exit 1
fi

case "${seq_type}" in
  train|test)
    ;;
  *)
    echo "Unsupported --type: ${seq_type}. Use train or test." >&2
    exit 1
    ;;
esac

if [[ -z "${sequence_list}" ]]; then
  if [[ "${seq_type}" == "train" ]]; then
    sequence_list="${SCRIPT_DIR}/Training_Sequences_${dataset}.txt"
  else
    sequence_list="${SCRIPT_DIR}/Testing_Sequences_${dataset}.txt"
  fi
fi

video_root="${video_root:-${DATA_ROOT}/video/${dataset}}"
output_root="${output_root:-${DATA_ROOT}/CodecTrainCfg/${dataset}/qp_${qp}}"

mkdir -p "${output_root}"

if [[ ! -f "${template_path}" ]]; then
  echo "Template not found: ${template_path}" >&2
  exit 1
fi

if [[ ! -f "${sequence_list}" ]]; then
  echo "Sequence list not found: ${sequence_list}" >&2
  exit 1
fi

while IFS= read -r line || [[ -n "${line}" ]]; do
  [[ -z "${line}" ]] && continue
  [[ "${line}" =~ ^# ]] && continue
  [[ "${line}" == *"end!!!!"* ]] && break

  IFS=',' read -r str_name file_name str_sizX str_sizY str_framenum str_fps <<< "${line}"

  if [[ -z "${file_name:-}" ]]; then
    echo "Skip invalid line: ${line}" >&2
    continue
  fi

  cfg_name="${str_name}_intra_vtm.cfg"
  cfg_path="${output_root}/${cfg_name}"
  cp "${template_path}" "${cfg_path}"

  if [[ "${video_root}" == ~* ]]; then
    input_file="${video_root/#\~/$HOME}/${file_name}"
  else
    input_file="${video_root}/${file_name}"
  fi
  sed -i "s|^InputFile[[:space:]]*:.*|InputFile                     : ${input_file}|" "${cfg_path}"
  sed -i "s|^FramesToBeEncoded[[:space:]]*:.*|FramesToBeEncoded             : ${str_framenum}|" "${cfg_path}"
  sed -i "s|^FrameRate[[:space:]]*:.*|FrameRate                     : ${str_fps}|" "${cfg_path}"
  sed -i "s|^SourceWidth[[:space:]]*:.*|SourceWidth                   : ${str_sizX}|" "${cfg_path}"
  sed -i "s|^SourceHeight[[:space:]]*:.*|SourceHeight                  : ${str_sizY}|" "${cfg_path}"
  sed -i "s|^BitstreamFile[[:space:]]*:.*|BitstreamFile                 : ${str_name}.bin|" "${cfg_path}"
  sed -i "s|^QP[[:space:]]*:.*|QP                            : ${qp}|" "${cfg_path}"
done < "${sequence_list}"

echo "Sequence list : ${sequence_list}"
echo "Video root    : ${video_root}"
echo "Output cfg dir: ${output_root}"
echo "Generated cfg files in ${output_root}"
