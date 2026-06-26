#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REF_MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/data"
DEFAULT_ENCODER="${REF_MODEL_DIR}/bin/dumpPartition/EncoderAppStatic"
DEFAULT_DECODER="${REF_MODEL_DIR}/bin/dumpPartition/DecoderAppStatic"
ENCODER="${DEFAULT_ENCODER}"
DECODER="${DEFAULT_DECODER}"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") --dataset DATASET --qp QP [options]

Workflow:
  1. Run ref_model/script/gencfg.sh first to generate cfg files.
  2. Run this script to execute encode/decode using those cfg files.

Options:
  --dataset NAME           Dataset name, e.g. HEVC_CTC / DIV2K / HEVC / VVC.
  --qp QP                  Quantization parameter.
  --type TYPE              Sequence list type: train, valid, or test. Default: train.
  --sequence-list PATH     Explicit sequence list path. Overrides automatic lookup.
  --cfg-root PATH          Config root. Default: data/CodecTrainCfg/<dataset>/qp_<qp>.
  --partition-root PATH    Partition output root. Default: data/partition/<dataset>/<split>.
  --log-root PATH          Codec log root. Default: data/logs/<dataset>/qp_<qp>.
  --work-root PATH         Codec output root. Default: data/codec_run/<dataset>/qp_<qp>.
  --encoder PATH           Override encoder binary. Default: ref_model/bin/dumpPartition/EncoderAppStatic.
  --decoder PATH           Override decoder binary. Default: ref_model/bin/dumpPartition/DecoderAppStatic.
  --skip-decode            Only run encoder, skip decode and MD5 check.

Sequence list lookup rule:
  --type train -> ref_model/script/Training_Sequences_<dataset>.txt
  --type valid -> ref_model/script/Validating_Sequences_<dataset>.txt
  --type test  -> ref_model/script/Testing_Sequences_<dataset>.txt
USAGE
}

dataset=""
qp=""
seq_type="train"
sequence_list=""
cfg_root=""
partition_root=""
log_root=""
work_root=""
skip_decode="0"

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
    --cfg-root)
      cfg_root="$2"
      shift 2
      ;;
    --partition-root)
      partition_root="$2"
      shift 2
      ;;
    --log-root)
      log_root="$2"
      shift 2
      ;;
    --work-root)
      work_root="$2"
      shift 2
      ;;
    --encoder)
      ENCODER="$2"
      shift 2
      ;;
    --decoder)
      DECODER="$2"
      shift 2
      ;;
    --skip-decode)
      skip_decode="1"
      shift 1
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
    split_dir="training"
    sequence_list_prefix="Training"
    ;;
  valid|validate|validating)
    split_dir="validating"
    sequence_list_prefix="Validating"
    ;;
  test|testing)
    split_dir="testing"
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

cfg_root="${cfg_root:-${DATA_ROOT}/CodecTrainCfg/${dataset}/qp_${qp}}"
partition_root="${partition_root:-${DATA_ROOT}/partition/${dataset}/${split_dir}}"
log_root="${log_root:-${DATA_ROOT}/logs/${dataset}/qp_${qp}}"
work_root="${work_root:-${DATA_ROOT}/codec_run/${dataset}/qp_${qp}}"

mkdir -p "${partition_root}" "${log_root}" "${work_root}"

if [[ ! -x "${ENCODER}" ]]; then
  echo "Encoder not found or not executable: ${ENCODER}" >&2
  exit 1
fi

if [[ "${skip_decode}" != "1" && ! -x "${DECODER}" ]]; then
  echo "Decoder not found or not executable: ${DECODER}" >&2
  exit 1
fi

if [[ ! -f "${sequence_list}" ]]; then
  echo "Sequence list not found: ${sequence_list}" >&2
  exit 1
fi

if ! command -v md5sum >/dev/null 2>&1; then
  echo "md5sum not found in PATH." >&2
  exit 1
fi

num=1
while IFS= read -r line || [[ -n "${line}" ]]; do
  [[ -z "${line}" ]] && continue
  [[ "${line}" =~ ^# ]] && continue
  [[ "${line}" == *"end!!!!"* ]] && break

  IFS=',' read -r str_name file_name str_sizX str_sizY str_framenum str_fps <<< "${line}"

  cfg_path="${cfg_root}/${str_name}_intra_vtm.cfg"
  if [[ ! -f "${cfg_path}" ]]; then
    echo "Cfg not found: ${cfg_path}" >&2
    echo "Please run ref_model/script/gencfg.sh first." >&2
    exit 1
  fi

  seq_work_dir="${work_root}/${str_name}"
  mkdir -p "${seq_work_dir}" "${partition_root}"

  bitstream_path="${seq_work_dir}/${str_name}.bin"
  recon_path="${seq_work_dir}/${str_name}_rec.yuv"
  dec_path="${seq_work_dir}/${str_name}_dec.yuv"
  log_path="${log_root}/${str_name}.log"

  : > "${log_path}"

  echo "[run] (${num}) encoding ${str_name}"
  echo "[run] cfg: ${cfg_path}"
  echo "[run] partition dir: ${partition_root}"

  export FASTPARTITION_DEPTH_DIR="${partition_root}"
  export FASTPARTITION_DEPTH_PREFIX="${str_name}"

  {
    echo "[encode]"
    echo "cfg=${cfg_path}"
    echo "bitstream=${bitstream_path}"
    echo "recon=${recon_path}"
  } >> "${log_path}"

  "${ENCODER}" \
    -c "${cfg_path}" \
    -b "${bitstream_path}" \
    -o "${recon_path}" \
    >> "${log_path}" 2>&1

  if [[ "${skip_decode}" != "1" ]]; then
    echo "[run] (${num}) decoding ${str_name}"
    {
      echo
      echo "[decode]"
      echo "bitstream=${bitstream_path}"
      echo "decoded=${dec_path}"
    } >> "${log_path}"

    "${DECODER}" \
      -b "${bitstream_path}" \
      -o "${dec_path}" \
      >> "${log_path}" 2>&1

    recon_md5="$(md5sum "${recon_path}" | awk '{print $1}')"
    dec_md5="$(md5sum "${dec_path}" | awk '{print $1}')"

    {
      echo
      echo "[md5]"
      echo "recon_md5 ${recon_md5}"
      echo "decode_md5 ${dec_md5}"
    } >> "${log_path}"

    if [[ "${recon_md5}" != "${dec_md5}" ]]; then
      echo "MD5 mismatch for ${str_name}" >&2
      echo "recon_md5=${recon_md5}" >&2
      echo "decode_md5=${dec_md5}" >&2
      exit 1
    fi

    rm -f "${bitstream_path}" "${recon_path}" "${dec_path}"

    echo "[run] (${num}) md5 verified: ${str_name}"
  else
    rm -f "${bitstream_path}" "${recon_path}"
  fi

  echo "[run] (${num}) done: ${str_name}"
  num=$((num + 1))
done < "${sequence_list}"

echo "Sequence list : ${sequence_list}"
echo "Cfg root      : ${cfg_root}"
echo "Partition root: ${partition_root}"
echo "Log root      : ${log_root}"
echo "Work root     : ${work_root}"
echo "Encoder       : ${ENCODER}"
echo "Decoder       : ${DECODER}"
echo "Run completed."
