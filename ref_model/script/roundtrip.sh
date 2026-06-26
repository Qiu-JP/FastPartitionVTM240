#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash ref_model/script/roundtrip.sh \
#     --input ~/qiujp/HEVC_CTC_YUV/416x240/BasketballPass_416x240_50.yuv \
#     --cfg ref_model/cfg/encoder_intra_vtm.cfg \
#     --seq-cfg VVCSoftware_VTM/cfg/per-sequence/BasketballPass.cfg \
#     --width 416 \
#     --height 240 \
#     --frames 10 \
#     --framerate 50 \
#     --qp 32

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN_DIR="${REF_MODEL_DIR}/bin/vtm240"
ENCODER="${BIN_DIR}/EncoderAppStatic"
DECODER="${BIN_DIR}/DecoderAppStatic"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") \
    --input INPUT_YUV \
    --cfg MAIN_CFG \
    --width WIDTH \
    --height HEIGHT \
    --frames FRAMES \
    --framerate FPS \
    --qp QP \
    [--seq-cfg SEQ_CFG] \
    [--bitdepth BITDEPTH] \
    [--chroma-format FORMAT] \
    [--workdir WORKDIR]

Description:
  Run a full VTM roundtrip test:
    1. Encode the input YUV sequence.
    2. Decode the generated bitstream.
    3. Compare encoder reconstruction and decoder output byte-by-byte.
  Exit with non-zero status if any step fails or if the two YUV outputs differ.

Arguments:
  --input          Input YUV sequence path.
  --cfg            Main encoder cfg file, e.g. ref_model/cfg/encoder_randomaccess_vtm.cfg
  --width          Source width.
  --height         Source height.
  --frames         Number of frames to encode.
  --framerate      Frame rate.
  --qp             QP value.
  --seq-cfg        Optional sequence-specific cfg file.
  --bitdepth       Input bit depth. Default: not explicitly set.
  --chroma-format  Input chroma format. Default: not explicitly set.
  --workdir        Output working directory. Default: ref_model/script/output/<timestamp>
USAGE
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: file not found: $path" >&2
    exit 1
  fi
}

require_bin() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "ERROR: executable not found: $path" >&2
    exit 1
  fi
}

INPUT=""
MAIN_CFG=""
SEQ_CFG=""
WIDTH=""
HEIGHT=""
FRAMES=""
FRAMERATE=""
QP=""
BITDEPTH=""
CHROMA_FORMAT=""
WORKDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --cfg)
      MAIN_CFG="$2"
      shift 2
      ;;
    --seq-cfg)
      SEQ_CFG="$2"
      shift 2
      ;;
    --width)
      WIDTH="$2"
      shift 2
      ;;
    --height)
      HEIGHT="$2"
      shift 2
      ;;
    --frames)
      FRAMES="$2"
      shift 2
      ;;
    --framerate)
      FRAMERATE="$2"
      shift 2
      ;;
    --qp)
      QP="$2"
      shift 2
      ;;
    --bitdepth)
      BITDEPTH="$2"
      shift 2
      ;;
    --chroma-format)
      CHROMA_FORMAT="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT" || -z "$MAIN_CFG" || -z "$WIDTH" || -z "$HEIGHT" || -z "$FRAMES" || -z "$FRAMERATE" || -z "$QP" ]]; then
  echo "ERROR: missing required arguments." >&2
  usage >&2
  exit 1
fi

require_bin "$ENCODER"
require_bin "$DECODER"
require_file "$INPUT"
require_file "$MAIN_CFG"
if [[ -n "$SEQ_CFG" ]]; then
  require_file "$SEQ_CFG"
fi

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="${SCRIPT_DIR}/output/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$WORKDIR"

BITSTREAM="${WORKDIR}/bitstream.vvc"
ENC_RECON="${WORKDIR}/encoder_recon.yuv"
DEC_RECON="${WORKDIR}/decoder_recon.yuv"
ENC_LOG="${WORKDIR}/encode.log"
DEC_LOG="${WORKDIR}/decode.log"
ROUNDTRIP_LOG="${WORKDIR}/roundtrip.log"

encode_cmd=(
  "$ENCODER"
  -c "$MAIN_CFG"
  -i "$INPUT"
  -b "$BITSTREAM"
  -o "$ENC_RECON"
  -wdt "$WIDTH"
  -hgt "$HEIGHT"
  --FrameRate="$FRAMERATE"
  --FramesToBeEncoded="$FRAMES"
  --QP="$QP"
)

if [[ -n "$BITDEPTH" ]]; then
  encode_cmd+=( --InputBitDepth="$BITDEPTH" )
fi

if [[ -n "$CHROMA_FORMAT" ]]; then
  encode_cmd+=( --InputChromaFormat="$CHROMA_FORMAT" )
fi

if [[ -n "$SEQ_CFG" ]]; then
  encode_cmd+=( -c "$SEQ_CFG" )
fi

decode_cmd=(
  "$DECODER"
  -b "$BITSTREAM"
  -o "$DEC_RECON"
)

exec > >(tee -a "$ROUNDTRIP_LOG") 2>&1

echo "[roundtrip] workdir: $WORKDIR"
echo "[roundtrip] input:   $INPUT"
echo "[roundtrip] encode:  ${encode_cmd[*]}"
"${encode_cmd[@]}" 2>&1 | tee "$ENC_LOG"

echo "[roundtrip] decode:  ${decode_cmd[*]}"
"${decode_cmd[@]}" 2>&1 | tee "$DEC_LOG"

require_file "$ENC_RECON"
require_file "$DEC_RECON"

if command -v md5sum >/dev/null 2>&1; then
  ENC_MD5="$(md5sum "$ENC_RECON" | awk '{print $1}')"
  DEC_MD5="$(md5sum "$DEC_RECON" | awk '{print $1}')"
  echo "[roundtrip] encoder recon md5: $ENC_MD5"
  echo "[roundtrip] decoder recon md5: $DEC_MD5"
fi

if cmp -s "$ENC_RECON" "$DEC_RECON"; then
  echo "[roundtrip] SUCCESS: encoder reconstruction matches decoder output."
  exit 0
fi

echo "[roundtrip] ERROR: encoder reconstruction and decoder output differ." >&2

if command -v sha256sum >/dev/null 2>&1; then
  echo "[roundtrip] encoder recon sha256: $(sha256sum "$ENC_RECON" | awk '{print $1}')" >&2
  echo "[roundtrip] decoder recon sha256: $(sha256sum "$DEC_RECON" | awk '{print $1}')" >&2
fi

if command -v cmp >/dev/null 2>&1; then
  set +e
  cmp -l "$ENC_RECON" "$DEC_RECON" | head -n 10 >&2
  set -e
fi

exit 2
