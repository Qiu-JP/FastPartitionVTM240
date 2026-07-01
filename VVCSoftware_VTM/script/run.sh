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
                       TorchScript Classifier_I model passed as --FastPartitionClassifierModel.
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

mkdir -p "$WORKDIR/logs/anchor" "$WORKDIR/logs/test" "$WORKDIR/tmp"
if [[ "$DUMP_FIRST_LCU" -eq 1 || "$DUMP_BOUNDARY_CTU" -eq 1 ]]; then
  : > "$WORKDIR/stat.log"
fi

RD_ANCHOR="${WORKDIR}/rd_anchor.log"
RD_TEST="${WORKDIR}/rd_test.log"
BD_LOG="${WORKDIR}/bd_rate.log"
PER_QP_CSV="${WORKDIR}/per_qp_results.csv"
RESULT_TXT="${WORKDIR}/result.txt"
FAILURES="${WORKDIR}/failures.txt"

: > "$FAILURES"

write_rd_header() {
  local out="$1"
  {
    printf "%-57s %-57s %-57s %s\n" "average" "I frame" "P frame" "B frame"
    for _ in 1 2 3 4; do
      printf "%-12s  %-12s  %-12s  %-18s  " "bitrate(kb/s)" "psnr(Y)" "psnr(U)" "psnr(V)"
    done
    printf "\n"
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

  printf "%-12.4f %-12.4f %-12.4f %-12.4f " "$a_bitrate" "$a_y" "$a_u" "$a_v" >> "$out"
  printf "%-12.4f %-12.4f %-12.4f %-12.4f " "$i_bitrate" "$i_y" "$i_u" "$i_v" >> "$out"
  printf "%-12.4f %-12.4f %-12.4f %-12.4f " 0 0 0 0 >> "$out"
  printf "%-12.4f %-12.4f %-12.4f %-12.4f " 0 0 0 0 >> "$out"
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
  "$PYTHON_BIN" - "$log_file" <<'PY'
import math
import re
import sys

log_path = sys.argv[1]
lines = open(log_path, "r", errors="replace").read().splitlines()
summary = {}
elapsed = None

num = r"(?:[-+]?nan|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
metric_re = re.compile(r"^\s*(\d+)\s+([aipb])\s+(" + num + r")\s+(" + num + r")\s+(" + num + r")\s+(" + num + r")")
time_re = re.compile(r"Total Time:\s*([0-9.]+)\s*sec\.\s*\[user\]\s*([0-9.]+)\s*sec\.\s*\[elapsed\]")

for line in lines:
    m = metric_re.match(line)
    if m:
        frame_count = int(m.group(1))
        frame_type = m.group(2)
        values = [float(m.group(i)) for i in range(3, 7)]
        summary[frame_type] = [frame_count] + values
        continue
    m = time_re.search(line)
    if m:
        elapsed = float(m.group(2))

if "a" not in summary:
    raise SystemExit("missing average summary row")
if elapsed is None:
    raise SystemExit("missing elapsed time")

a = summary["a"]
i = summary.get("i", a)
required = a[1:] + i[1:] + [elapsed]
if any((not math.isfinite(x)) for x in required):
    raise SystemExit("summary contains non-finite values")

print("\t".join(str(x) for x in [
    a[0], a[1], a[2], a[3], a[4],
    i[0], i[1], i[2], i[3], i[4],
    elapsed,
]))
PY
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
printf "sequence,qp,frames,width,height,fps,anchor_bitrate,anchor_y,anchor_u,anchor_v,anchor_time,test_bitrate,test_y,test_u,test_v,test_time,time_saving_percent,time_ratio\n" > "$PER_QP_CSV"

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
    echo "${seq},ALL,missing sequence cfg: ${seq_cfg}" >> "$FAILURES"
    continue
  fi

  cfg_input="$(parse_cfg_value "InputFile" "$seq_cfg")"
  if [[ -z "$cfg_input" ]]; then
    cfg_input="$list_file"
  fi

  yuv_file="$(find_yuv_file "$cfg_input" "$seq" "$width" "$height" "$fps")"
  if [[ -z "$yuv_file" ]]; then
    echo "${seq},ALL,missing YUV file: ${cfg_input}" >> "$FAILURES"
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
      echo "${seq},QP${qp},anchor encode failed: ${anchor_log}" >> "$FAILURES"
      continue
    fi
    if ! run_encoder "test" "$TEST_BIN" "$seq" "$qp" "$frames" "$yuv_file" "$seq_cfg" "$test_log"; then
      echo "${seq},QP${qp},test encode failed: ${test_log}" >> "$FAILURES"
      continue
    fi

    if ! anchor_metrics="$(parse_encode_log "$anchor_log")"; then
      echo "${seq},QP${qp},anchor log parse failed: ${anchor_log}" >> "$FAILURES"
      continue
    fi
    if ! test_metrics="$(parse_encode_log "$test_log")"; then
      echo "${seq},QP${qp},test log parse failed: ${test_log}" >> "$FAILURES"
      continue
    fi

    IFS=$'\t' read -r a_frames a_br a_y a_u a_v i_frames i_br i_y i_u i_v a_time <<< "$anchor_metrics"
    IFS=$'\t' read -r t_a_frames t_br t_y t_u t_v t_i_frames t_i_br t_i_y t_i_u t_i_v t_time <<< "$test_metrics"

    time_saving="$("$PYTHON_BIN" - "$a_time" "$t_time" <<'PY'
import sys
a = float(sys.argv[1])
t = float(sys.argv[2])
print((a - t) / a * 100.0 if a else 0.0)
PY
)"
    time_ratio="$("$PYTHON_BIN" - "$a_time" "$t_time" <<'PY'
import sys
a = float(sys.argv[1])
t = float(sys.argv[2])
print(t / a if a else 0.0)
PY
)"

    append_rd_row "$RD_ANCHOR" "$seq" "$qp" "$a_br" "$a_y" "$a_u" "$a_v" "$i_br" "$i_y" "$i_u" "$i_v"
    append_rd_row "$RD_TEST" "$seq" "$qp" "$t_br" "$t_y" "$t_u" "$t_v" "$t_i_br" "$t_i_y" "$t_i_u" "$t_i_v"
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "$seq" "$qp" "$frames" "$width" "$height" "$fps" \
      "$a_br" "$a_y" "$a_u" "$a_v" "$a_time" \
      "$t_br" "$t_y" "$t_u" "$t_v" "$t_time" \
      "$time_saving" "$time_ratio" >> "$PER_QP_CSV"
  done
done < "$SEQ_LIST"

if [[ "${#QPS[@]}" -ge 3 ]]; then
  set +e
  "$PYTHON_BIN" "$SCRIPT_DIR/getBdRate.py" "$RD_ANCHOR" "$RD_TEST" YUV420 > "$BD_LOG" 2>&1
  BD_STATUS=$?
  set -e
  if [[ "$BD_STATUS" -ne 0 ]]; then
    echo "getBdRate.py failed; see ${BD_LOG}" >> "$FAILURES"
  fi
else
  echo "BD-rate skipped because fewer than 3 QP points were requested." > "$BD_LOG"
fi

"$PYTHON_BIN" - "$SCRIPT_DIR" "$PER_QP_CSV" "$FAILURES" "$RESULT_TXT" "$BD_LOG" <<'PY'
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

script_dir, per_qp_csv, failures_path, result_path, bd_log = sys.argv[1:]
sys.path.insert(0, script_dir)
_bd_core = None
_np = None

def load_bd_core():
    global _bd_core, _np
    if _bd_core is None:
        from getBdRateCore import getBdRateCore
        import numpy as np
        _bd_core = getBdRateCore
        _np = np
    return _bd_core, _np

rows = []
with open(per_qp_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

by_seq = defaultdict(list)
for row in rows:
    by_seq[row["sequence"]].append(row)

def f(row, key):
    return float(row[key])

def calc_bdrate(seq_rows, comp):
    seq_rows = sorted(seq_rows, key=lambda r: int(r["qp"]))
    if len(seq_rows) < 3:
        return math.nan
    try:
        getBdRateCore, np = load_bd_core()
        anchor_bitrate = np.array([f(r, "anchor_bitrate") for r in seq_rows], dtype=float)
        test_bitrate = np.array([f(r, "test_bitrate") for r in seq_rows], dtype=float)
        anchor_psnr = np.array([f(r, f"anchor_{comp}") for r in seq_rows], dtype=float)
        test_psnr = np.array([f(r, f"test_{comp}") for r in seq_rows], dtype=float)
        if not all(np.isfinite(x).all() for x in (anchor_bitrate, test_bitrate, anchor_psnr, test_psnr)):
            return math.nan
        return float(getBdRateCore(anchor_bitrate, anchor_psnr, test_bitrate, test_psnr))
    except Exception:
        return math.nan

def fmt(x):
    return "nan" if x is None or not math.isfinite(x) else f"{x:.3f}"

seq_summaries = []
for seq in sorted(by_seq):
    seq_rows = by_seq[seq]
    bdr_y = calc_bdrate(seq_rows, "y")
    bdr_u = calc_bdrate(seq_rows, "u")
    bdr_v = calc_bdrate(seq_rows, "v")
    if all(math.isfinite(x) for x in (bdr_y, bdr_u, bdr_v)):
        bdr_avg = (4 * bdr_y + bdr_u + bdr_v) / 6.0
    else:
        bdr_avg = math.nan
    avg_ts = sum(f(r, "time_saving_percent") for r in seq_rows) / len(seq_rows)
    seq_summaries.append((seq, bdr_y, bdr_u, bdr_v, bdr_avg, avg_ts))

def mean_valid(values):
    vals = [x for x in values if math.isfinite(x)]
    return sum(vals) / len(vals) if vals else math.nan

avg_bdr_y = mean_valid([x[1] for x in seq_summaries])
avg_bdr_u = mean_valid([x[2] for x in seq_summaries])
avg_bdr_v = mean_valid([x[3] for x in seq_summaries])
avg_bdr_avg = mean_valid([x[4] for x in seq_summaries])
avg_ts = mean_valid([x[5] for x in seq_summaries])

failures = Path(failures_path).read_text().strip()

with open(result_path, "w") as out:
    out.write("FastPartition all-intra evaluation\n")
    out.write(f"per_qp_csv: {per_qp_csv}\n")
    out.write(f"bd_rate_log: {bd_log}\n\n")

    out.write("[Per-sequence per-QP results]\n")
    out.write("sequence qp frames anchor_bitrate anchor_y anchor_u anchor_v anchor_time ")
    out.write("test_bitrate test_y test_u test_v test_time time_saving(%) time_ratio\n")
    for r in sorted(rows, key=lambda x: (x["sequence"], int(x["qp"]))):
        out.write(
            f"{r['sequence']} {r['qp']} {r['frames']} "
            f"{fmt(f(r, 'anchor_bitrate'))} {fmt(f(r, 'anchor_y'))} {fmt(f(r, 'anchor_u'))} {fmt(f(r, 'anchor_v'))} {fmt(f(r, 'anchor_time'))} "
            f"{fmt(f(r, 'test_bitrate'))} {fmt(f(r, 'test_y'))} {fmt(f(r, 'test_u'))} {fmt(f(r, 'test_v'))} {fmt(f(r, 'test_time'))} "
            f"{fmt(f(r, 'time_saving_percent'))} {fmt(f(r, 'time_ratio'))}\n"
        )

    out.write("\n[Per-sequence BD-rate and time saving]\n")
    out.write("sequence bdrate_y(%) bdrate_u(%) bdrate_v(%) bdrate_avg(%) avg_time_saving(%)\n")
    for seq, y, u, v, avg, ts in seq_summaries:
        out.write(f"{seq} {fmt(y)} {fmt(u)} {fmt(v)} {fmt(avg)} {fmt(ts)}\n")

    out.write("\n[Average]\n")
    out.write(f"average_bdrate_y(%) {fmt(avg_bdr_y)}\n")
    out.write(f"average_bdrate_u(%) {fmt(avg_bdr_u)}\n")
    out.write(f"average_bdrate_v(%) {fmt(avg_bdr_v)}\n")
    out.write(f"average_bdrate_avg(%) {fmt(avg_bdr_avg)}\n")
    out.write(f"average_time_saving(%) {fmt(avg_ts)}\n")

    out.write("\n[Failures]\n")
    out.write(failures + "\n" if failures else "none\n")
PY

cat "$RESULT_TXT"
echo
echo "[run] workdir: $WORKDIR"
echo "[run] result:  $RESULT_TXT"
