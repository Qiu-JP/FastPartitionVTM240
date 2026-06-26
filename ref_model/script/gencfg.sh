#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REF_MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/data"
DEFAULT_SEQUENCE_TEMPLATE="${REF_MODEL_DIR}/cfg/sequence.cfg"
DEFAULT_TEMPLATE="${REF_MODEL_DIR}/cfg/encoder_intra_vtm.cfg"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") --dataset DATASET --qp QP [options]

Options:
  --dataset NAME           Sequence set name, e.g. DIV2K / HEVC / VVC.
  --qp QP                  Quantization parameter.
  --type TYPE              Sequence list type: train, valid, or test. Default: train.
  --sequence-list PATH     Explicit sequence list path. Overrides --dataset + --type lookup.
  --template PATH          Encoder cfg template. Default: ref_model/cfg/encoder_intra_vtm.cfg.
  --sequence-template PATH Sequence cfg header. Default: ref_model/cfg/sequence.cfg.
  --video-root PATH        Video root. Default: data/video/<dataset>.
  --output-root PATH       Output cfg root. Default: data/CodecTrainCfg/<dataset>/qp_<qp>.

Sequence list lookup rule:
  --type train -> ref_model/script/Training_Sequences_<dataset>.txt
  --type valid -> ref_model/script/Validating_Sequences_<dataset>.txt
  --type test  -> ref_model/script/Testing_Sequences_<dataset>.txt

Example:
  bash ref_model/script/gencfg.sh --dataset DIV2K --qp 22 --type train
  bash ref_model/script/gencfg.sh --dataset HEVC --qp 32 --type test
USAGE
}

set_cfg_value() {
  local cfg_path="$1"
  local key="$2"
  local value="$3"
  if ! grep -qE "^${key}[[:space:]]*:" "${cfg_path}"; then
    echo "Template field not found: ${key} in ${cfg_path}" >&2
    exit 1
  fi
  sed -i "s|^${key}[[:space:]]*:.*|$(printf '%-30s' "${key}") : ${value}|" "${cfg_path}"
}

dataset=""
qp=""
seq_type="train"
sequence_list=""
template_path="${DEFAULT_TEMPLATE}"
sequence_template_path="${DEFAULT_SEQUENCE_TEMPLATE}"
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
    --sequence-template)
      sequence_template_path="$2"
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
  train|training)
    sequence_list_prefix="Training"
    ;;
  valid|validate|validating)
    sequence_list_prefix="Validating"
    ;;
  test|testing)
    sequence_list_prefix="Testing"
    ;;
  *)
    echo "Unsupported --type: ${seq_type}. Use train, valid, or test." >&2
    exit 1
    ;;
esac

if [[ -z "${sequence_list}" ]]; then
  sequence_list="${SCRIPT_DIR}/${sequence_list_prefix}_Sequences_${dataset}.txt"
fi

video_root="${video_root:-${DATA_ROOT}/video/${dataset}}"
output_root="${output_root:-${DATA_ROOT}/CodecTrainCfg/${dataset}/qp_${qp}}"

mkdir -p "${output_root}"

if [[ ! -f "${template_path}" ]]; then
  echo "Template not found: ${template_path}" >&2
  exit 1
fi

if [[ ! -f "${sequence_template_path}" ]]; then
  echo "Sequence template not found: ${sequence_template_path}" >&2
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
  {
    cat "${sequence_template_path}"
    echo
    cat "${template_path}"
  } > "${cfg_path}"

  if [[ "${video_root}" == ~* ]]; then
    input_file="${video_root/#\~/$HOME}/${file_name}"
  else
    input_file="${video_root}/${file_name}"
  fi
  set_cfg_value "${cfg_path}" "InputFile" "${input_file}"
  set_cfg_value "${cfg_path}" "FramesToBeEncoded" "${str_framenum}"
  set_cfg_value "${cfg_path}" "FrameRate" "${str_fps}"
  set_cfg_value "${cfg_path}" "SourceWidth" "${str_sizX}"
  set_cfg_value "${cfg_path}" "SourceHeight" "${str_sizY}"
  set_cfg_value "${cfg_path}" "BitstreamFile" "${str_name}.bin"
  set_cfg_value "${cfg_path}" "QP" "${qp}"
  set_cfg_value "${cfg_path}" "TemporalSubsampleRatio" "20         #set the ratio of Sampled Encoding Frames"
done < "${sequence_list}"

echo "Sequence list : ${sequence_list}"
echo "Video root    : ${video_root}"
echo "Output cfg dir: ${output_root}"
echo "Generated cfg files in ${output_root}"
