#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VTM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${VTM_DIR}/.." && pwd)"
REF_DIR="${REPO_ROOT}/ref_model"

YUV_ROOT="${HOME}/qiujp/HEVC_CTC_YUV"
SEQ_LIST="${REPO_ROOT}/network/script/Testing_Sequences_HEVC.txt"
QPS_STR="22 27 32 37"
MAX_FRAMES=10
FULL_FRAMES=0
WORKDIR=""
FAST_MODEL=""
CLASSIFIER_MODEL=""
CHROMA_FAST_MODEL=""
CHROMA_CLASSIFIER_MODEL=""
FAST_PARTITION_PRESET="all"
FAST_PARTITION_THRESHOLD=""
FAST_PARTITION_THRESHOLDS=""
DUMP_FIRST_LCU=0
DUMP_BOUNDARY_CTU=0
ANCHOR_BIN="${REF_DIR}/bin/vtm240/EncoderAppStatic"
TEST_BIN="${VTM_DIR}/bin/umake/gcc-12.2/x86_64/release/EncoderApp"
TEST_BIN_FALLBACK="${VTM_DIR}/bin/EncoderAppStatic"
MAIN_CFG="${REF_DIR}/cfg/encoder_intra_vtm.cfg"
PYTHON_BIN="${PYTHON:-python3}"
BUILD_DIR="${VTM_DIR}/build"
BUILD_JOBS="$(nproc 2>/dev/null || echo 8)"
SKIP_BUILD=0
TEST_BIN_USER_SET=0

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [options]

Options:
  --yuv-root DIR       HEVC CTC YUV root. Default: ${YUV_ROOT}
  --seq-list FILE      Sequence list. Default: ${SEQ_LIST}
  --qps "QP ..."       QP list. Default: "${QPS_STR}"
  --max-frames N       Encode at most N frames per sequence. Default: ${MAX_FRAMES}
  --full-frames        Use each sequence's original frame count.
  --workdir DIR        Output directory. Default: ${SCRIPT_DIR}/output/ai_eval_<timestamp>
  --fast-model FILE    TorchScript Swin model passed to test encoder as --FastPartitionSwinModel.
  --classifier-model FILE
                       Native JSON Classifier_I model passed as --FastPartitionClassifierModel.
  --chroma-fast-model FILE
                       2x48x48 Chroma TorchScript model passed as --FastPartitionChromaSwinModel.
  --chroma-classifier-model FILE
                       Chroma native JSON Classifier_I model passed as --FastPartitionChromaClassifierModel.
                       Omit either chroma model to keep the original chroma RDO flow.
  --preset NAME        FastPartitionPreset passed to test encoder. Default: ${FAST_PARTITION_PRESET}
  --threshold VALUE    Optional FastPartitionThreshold override passed to test encoder.
  --thresholds LIST    FastPartitionTh list, e.g. "[0.1,0.1,0.1,0.1,0.1,0.1]".
  --dump-first-lcu     Write first-LCU gridmaps and classifier decisions to workdir/stat.log.
  --dump-boundary-ctu  Write POC 0 boundary-CTU gridmaps and classifier decisions to workdir/stat.log.
  --anchor-bin FILE    Anchor EncoderAppStatic path. Default: ${ANCHOR_BIN}
  --test-bin FILE      Test EncoderApp path after build. Default: ${TEST_BIN}, fallback: ${TEST_BIN_FALLBACK}
  --build-dir DIR      CMake build directory for the modified test encoder. Default: ${BUILD_DIR}
  --build-jobs N       Parallel build jobs. Default: ${BUILD_JOBS}
  --skip-build         Do not build the modified test encoder before evaluation.
  -h, --help           Show this help.
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

trim() {
  local value="$1"
  value="${value%%#*}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yuv-root)
      YUV_ROOT="$2"
      shift 2
      ;;
    --seq-list)
      SEQ_LIST="$2"
      shift 2
      ;;
    --qps)
      QPS_STR="$2"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --full-frames)
      FULL_FRAMES=1
      shift
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --fast-model)
      FAST_MODEL="$2"
      shift 2
      ;;
    --classifier-model)
      CLASSIFIER_MODEL="$2"
      shift 2
      ;;
    --chroma-fast-model)
      CHROMA_FAST_MODEL="$2"
      shift 2
      ;;
    --chroma-classifier-model)
      CHROMA_CLASSIFIER_MODEL="$2"
      shift 2
      ;;
    --preset)
      FAST_PARTITION_PRESET="$2"
      shift 2
      ;;
    --threshold)
      FAST_PARTITION_THRESHOLD="$2"
      shift 2
      ;;
    --thresholds)
      FAST_PARTITION_THRESHOLDS="$2"
      shift 2
      ;;
    --dump-first-lcu)
      DUMP_FIRST_LCU=1
      shift
      ;;
    --dump-boundary-ctu)
      DUMP_BOUNDARY_CTU=1
      shift
      ;;
    --anchor-bin)
      ANCHOR_BIN="$2"
      shift 2
      ;;
    --test-bin)
      TEST_BIN="$2"
      TEST_BIN_USER_SET=1
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --build-jobs)
      BUILD_JOBS="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "$YUV_ROOT" in
  "~"/*) YUV_ROOT="${HOME}/${YUV_ROOT#~/}" ;;
esac

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="${SCRIPT_DIR}/output/ai_eval_$(date +%Y%m%d_%H%M%S)"
fi

build_test_encoder() {
  echo "[build] configuring test encoder: ${BUILD_DIR}"
  cmake -S "$VTM_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  echo "[build] building test encoder: ${BUILD_DIR}"
  cmake --build "$BUILD_DIR" --config Release --target EncoderApp --parallel "$BUILD_JOBS"
}

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  build_test_encoder
fi

if [[ "$TEST_BIN_USER_SET" -eq 0 && ! -x "$TEST_BIN" && -x "$TEST_BIN_FALLBACK" ]]; then
  TEST_BIN="$TEST_BIN_FALLBACK"
fi

[[ -x "$ANCHOR_BIN" ]] || die "anchor encoder is not executable: $ANCHOR_BIN"
[[ -x "$TEST_BIN" ]] || die "test encoder is not executable: $TEST_BIN"
[[ -f "$MAIN_CFG" ]] || die "main cfg not found: $MAIN_CFG"
[[ -f "$SEQ_LIST" ]] || die "sequence list not found: $SEQ_LIST"
[[ -d "$YUV_ROOT" ]] || die "YUV root not found: $YUV_ROOT"
if [[ -n "$FAST_MODEL" && ! -f "$FAST_MODEL" ]]; then
  die "fast model not found: $FAST_MODEL"
fi
if [[ -n "$CLASSIFIER_MODEL" && ! -f "$CLASSIFIER_MODEL" ]]; then
  die "classifier model not found: $CLASSIFIER_MODEL"
fi
if [[ -n "$CHROMA_FAST_MODEL" && ! -f "$CHROMA_FAST_MODEL" ]]; then
  die "chroma fast model not found: $CHROMA_FAST_MODEL"
fi
if [[ -n "$CHROMA_CLASSIFIER_MODEL" && ! -f "$CHROMA_CLASSIFIER_MODEL" ]]; then
  die "chroma classifier model not found: $CHROMA_CLASSIFIER_MODEL"
fi

mkdir -p "$WORKDIR/logs/anchor" "$WORKDIR/logs/test" "$WORKDIR/tmp"
if [[ "$DUMP_FIRST_LCU" -eq 1 || "$DUMP_BOUNDARY_CTU" -eq 1 ]]; then
  : > "$WORKDIR/stat.log"
fi

RD_ANCHOR="${WORKDIR}/rd_anchor.log"
RD_TEST="${WORKDIR}/rd_test.log"
RESULT_LOG="${WORKDIR}/result.log"

write_rd_header() {
  local out="$1"
  {
    printf "%-57s %-57s %-57s %s\n" "average" "I frame" "P frame" "B frame"
    for _ in 1 2 3 4; do
      printf "%-12s  %-12s  %-12s  %-18s  " "bitrate(kb/s)" "psnr(Y)" "psnr(U)" "psnr(V)"
    done
    printf "%s\n" "elapsed(s)"
  } > "$out"
}

append_rd_row() {
  local out="$1"
  local seq="$2"
  local qp="$3"
  local a_bitrate="$4"
  local a_y="$5"
  local a_u="$6"
  local a_v="$7"
  local i_bitrate="$8"
  local i_y="$9"
  local i_u="${10}"
  local i_v="${11}"
  local elapsed="${12}"

  printf "%-12.4f %-12.4f %-12.4f %-12.4f " "$a_bitrate" "$a_y" "$a_u" "$a_v" >> "$out"
  printf "%-12.4f %-12.4f %-12.4f %-12.4f " "$i_bitrate" "$i_y" "$i_u" "$i_v" >> "$out"
  printf "%-12.4f %-12.4f %-12.4f %-12.4f " 0 0 0 0 >> "$out"
  printf "%-12.4f %-12.4f %-12.4f %-12.4f " 0 0 0 0 >> "$out"
  printf "%-18.4f " "$elapsed" >> "$out"
  printf "%s_%s\n" "$seq" "$qp" >> "$out"
}

parse_cfg_value() {
  local key="$1"
  local cfg="$2"
  awk -F ':' -v key="$key" '
    $1 ~ "^[[:space:]]*" key "[[:space:]]*$" {
      v=$2
      sub(/#.*/, "", v)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
      print v
      exit
    }
  ' "$cfg"
}

find_yuv_file() {
  local cfg_input="$1"
  local seq="$2"
  local width="$3"
  local height="$4"
  local fps="$5"
  local found=""

  if [[ -n "$cfg_input" ]]; then
    found="$(find "$YUV_ROOT" -type f -name "$cfg_input" -print -quit)"
  fi
  if [[ -z "$found" ]]; then
    found="$(find "$YUV_ROOT" -type f -name "${seq}_${width}x${height}_${fps}.yuv" -print -quit)"
  fi
  if [[ -z "$found" ]]; then
    found="$(find "$YUV_ROOT" -type f -name "${seq}*.yuv" -print -quit)"
  fi

  printf '%s' "$found"
}

parse_encode_log() {
  local log_file="$1"
  "$PYTHON_BIN" "$SCRIPT_DIR/parseLog.py" "$log_file"
}

run_encoder() {
  local kind="$1"
  local encoder="$2"
  local seq="$3"
  local qp="$4"
  local frames="$5"
  local yuv="$6"
  local seq_cfg="$7"
  local log_file="$8"
  local bitstream="${WORKDIR}/tmp/${kind}_${seq}_QP${qp}.vvc"
  local recon="${WORKDIR}/tmp/${kind}_${seq}_QP${qp}.yuv"

  local cmd=(
    "$encoder"
    -c "$MAIN_CFG"
    -c "$seq_cfg"
    -i "$yuv"
    -b "$bitstream"
    -o "$recon"
    --QP="$qp"
    --FramesToBeEncoded="$frames"
    --Verbosity=6
  )

  if [[ "$kind" == "test" && -n "$FAST_MODEL" ]]; then
    cmd+=( --FastPartitionSwinModel="$FAST_MODEL" )
  fi
  if [[ "$kind" == "test" && -n "$CLASSIFIER_MODEL" ]]; then
    cmd+=( --FastPartitionClassifierModel="$CLASSIFIER_MODEL" )
  fi
  if [[ "$kind" == "test" && -n "$CHROMA_FAST_MODEL" ]]; then
    cmd+=( --FastPartitionChromaSwinModel="$CHROMA_FAST_MODEL" )
  fi
  if [[ "$kind" == "test" && -n "$CHROMA_CLASSIFIER_MODEL" ]]; then
    cmd+=( --FastPartitionChromaClassifierModel="$CHROMA_CLASSIFIER_MODEL" )
  fi
  if [[ "$kind" == "test" ]]; then
    cmd+=( --FastPartitionPreset="$FAST_PARTITION_PRESET" )
    if [[ -n "$FAST_PARTITION_THRESHOLD" ]]; then
      cmd+=( --FastPartitionThreshold="$FAST_PARTITION_THRESHOLD" )
    fi
    if [[ -n "$FAST_PARTITION_THRESHOLDS" ]]; then
      cmd+=( --FastPartitionTh="$FAST_PARTITION_THRESHOLDS" )
    fi
  fi

  echo "[run] ${kind} ${seq} QP${qp}, frames=${frames}"
  local status=0
  if [[ "$kind" == "test" && ( "$DUMP_FIRST_LCU" -eq 1 || "$DUMP_BOUNDARY_CTU" -eq 1 ) ]]; then
    if FASTPARTITION_DUMP_FIRST_LCU="$DUMP_FIRST_LCU" FASTPARTITION_DUMP_BOUNDARY_CTU="$DUMP_BOUNDARY_CTU" \
      FASTPARTITION_STAT_LOG="$WORKDIR/stat.log" \
      bash -c 'exec "$@"' _ "${cmd[@]}" > "$log_file" 2>&1; then
      status=0
    else
      status=$?
    fi
  elif bash -c 'exec "$@"' _ "${cmd[@]}" > "$log_file" 2>&1; then
    status=0
  else
    status=$?
  fi

  rm -f "$bitstream" "$recon"
  return "$status"
}

write_rd_header "$RD_ANCHOR"
write_rd_header "$RD_TEST"

IFS=' ' read -r -a QPS <<< "$QPS_STR"
if [[ "${#QPS[@]}" -eq 0 ]]; then
  die "empty QP list"
fi

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="$(trim "$raw_line")"
  [[ -z "$line" || "$line" == \#* ]] && continue

  IFS=',' read -r seq list_file width height seq_frames fps _ <<< "$line"
  seq="$(trim "$seq")"
  list_file="$(trim "$list_file")"
  width="$(trim "$width")"
  height="$(trim "$height")"
  seq_frames="$(trim "$seq_frames")"
  fps="$(trim "$fps")"

  seq_cfg="${VTM_DIR}/cfg/per-sequence/${seq}.cfg"
  if [[ ! -f "$seq_cfg" ]]; then
    echo "WARNING: ${seq}: missing sequence cfg: ${seq_cfg}" >&2
    continue
  fi

  cfg_input="$(parse_cfg_value "InputFile" "$seq_cfg")"
  if [[ -z "$cfg_input" ]]; then
    cfg_input="$list_file"
  fi

  yuv_file="$(find_yuv_file "$cfg_input" "$seq" "$width" "$height" "$fps")"
  if [[ -z "$yuv_file" ]]; then
    echo "WARNING: ${seq}: missing YUV file: ${cfg_input}" >&2
    continue
  fi

  frames="$seq_frames"
  if [[ "$FULL_FRAMES" -eq 0 && "$frames" -gt "$MAX_FRAMES" ]]; then
    frames="$MAX_FRAMES"
  fi

  for qp in "${QPS[@]}"; do
    anchor_log="${WORKDIR}/logs/anchor/${seq}_QP${qp}.log"
    test_log="${WORKDIR}/logs/test/${seq}_QP${qp}.log"

    if ! run_encoder "anchor" "$ANCHOR_BIN" "$seq" "$qp" "$frames" "$yuv_file" "$seq_cfg" "$anchor_log"; then
      echo "WARNING: ${seq} QP${qp}: anchor encode failed: ${anchor_log}" >&2
      continue
    fi
    if ! run_encoder "test" "$TEST_BIN" "$seq" "$qp" "$frames" "$yuv_file" "$seq_cfg" "$test_log"; then
      echo "WARNING: ${seq} QP${qp}: test encode failed: ${test_log}" >&2
      continue
    fi

    if ! anchor_metrics="$(parse_encode_log "$anchor_log")"; then
      echo "WARNING: ${seq} QP${qp}: anchor log parse failed: ${anchor_log}" >&2
      continue
    fi
    if ! test_metrics="$(parse_encode_log "$test_log")"; then
      echo "WARNING: ${seq} QP${qp}: test log parse failed: ${test_log}" >&2
      continue
    fi

    IFS=$'\t' read -r a_frames a_br a_y a_u a_v i_frames i_br i_y i_u i_v a_time <<< "$anchor_metrics"
    IFS=$'\t' read -r t_a_frames t_br t_y t_u t_v t_i_frames t_i_br t_i_y t_i_u t_i_v t_time <<< "$test_metrics"

    append_rd_row "$RD_ANCHOR" "$seq" "$qp" "$a_br" "$a_y" "$a_u" "$a_v" "$i_br" "$i_y" "$i_u" "$i_v" "$a_time"
    append_rd_row "$RD_TEST" "$seq" "$qp" "$t_br" "$t_y" "$t_u" "$t_v" "$t_i_br" "$t_i_y" "$t_i_u" "$t_i_v" "$t_time"
  done
done < "$SEQ_LIST"

if [[ "${#QPS[@]}" -ge 3 ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/getBdRate.py" "$RD_ANCHOR" "$RD_TEST" > "$RESULT_LOG" 2>&1
else
  echo "BD-rate skipped because fewer than 3 QP points were requested." > "$RESULT_LOG"
fi

cat "$RESULT_LOG"
echo
echo "[run] workdir: $WORKDIR"
echo "[run] result:  $RESULT_LOG"
