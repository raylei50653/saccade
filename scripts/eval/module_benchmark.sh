#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="all"
DETECTOR="SDP"
SEQUENCES="MOT17-04-SDP,MOT17-10-SDP"
PROFILE_SEQUENCES=""
VALIDATE_SEQUENCES=""
ENGINE="models/yolo/yolo26s_960_batch1.engine"
TILING="native_960"
MAX_FRAMES="100"
OUTPUT_ROOT="results/module_benchmark/$(date +%Y%m%d_%H%M%S)"
ABLATION_CATEGORIES="detection,association,geometry,reid,semantic,trigger,lifecycle"
REID_MODE=""
TILE_DIAGNOSTICS="auto"
DRY_RUN=0
EXTRA_MOT17_ARGS=()
COMMANDS_FILE=""
NOTES_FILE=""
SUMMARY_FILE=""
MATRIX_FILE=""

usage() {
    cat <<'EOF'
Usage:
  scripts/eval/module_benchmark.sh [options] [-- <extra mot17 args>]

Modes:
  all        Run stage profile, grouped ablation, then non-profile validation.
  profile    Run mot17.py with --profile-stages.
  ablation   Run ablation_mot17.py with grouped categories.
  validate   Run mot17.py without --profile-stages for end-to-end comparison.

Options:
  --mode MODE                    One of: all, profile, ablation, validate.
  --detector NAME                Sequence detector suffix filter. Default: SDP.
  --sequences CSV                Default sequence set for profile/validate.
  --profile-sequences CSV        Override sequence set used by profile mode.
  --validate-sequences CSV       Override sequence set used by validate mode.
  --engine PATH                  Detector engine path for mot17.py runs.
  --tiling NAME                  One of: native_960, 960p_2x2, 960p_3x2.
  --max-frames N                 Per-sequence frame cap.
  --output-root PATH             Root directory for this experiment batch.
  --ablation-categories CSV      Categories for ablation_mot17.py.
  --reid-mode NAME               Optional mot17.py override.
  --tile-diagnostics             Force tile diagnostics on.
  --no-tile-diagnostics          Force tile diagnostics off.
  --dry-run                      Print commands without executing them.
  --help                         Show this help.

Examples:
  scripts/eval/module_benchmark.sh
  scripts/eval/module_benchmark.sh --mode profile --sequences MOT17-09-SDP --max-frames 80
  scripts/eval/module_benchmark.sh --mode ablation --ablation-categories detection,geometry
  scripts/eval/module_benchmark.sh --mode validate --tiling 960p_2x2 --engine models/yolo/yolo26s_batch6.engine
  scripts/eval/module_benchmark.sh --mode profile -- --async-reid
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:?missing value for --mode}"
            shift 2
            ;;
        --detector)
            DETECTOR="${2:?missing value for --detector}"
            shift 2
            ;;
        --sequences)
            SEQUENCES="${2:?missing value for --sequences}"
            shift 2
            ;;
        --profile-sequences)
            PROFILE_SEQUENCES="${2:?missing value for --profile-sequences}"
            shift 2
            ;;
        --validate-sequences)
            VALIDATE_SEQUENCES="${2:?missing value for --validate-sequences}"
            shift 2
            ;;
        --engine)
            ENGINE="${2:?missing value for --engine}"
            shift 2
            ;;
        --tiling)
            TILING="${2:?missing value for --tiling}"
            shift 2
            ;;
        --max-frames)
            MAX_FRAMES="${2:?missing value for --max-frames}"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="${2:?missing value for --output-root}"
            shift 2
            ;;
        --ablation-categories)
            ABLATION_CATEGORIES="${2:?missing value for --ablation-categories}"
            shift 2
            ;;
        --reid-mode)
            REID_MODE="${2:?missing value for --reid-mode}"
            shift 2
            ;;
        --tile-diagnostics)
            TILE_DIAGNOSTICS="on"
            shift
            ;;
        --no-tile-diagnostics)
            TILE_DIAGNOSTICS="off"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_MOT17_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "${MODE}" in
    all|profile|ablation|validate) ;;
    *)
        echo "Unsupported mode: ${MODE}" >&2
        exit 1
        ;;
esac

PROFILE_SEQUENCES="${PROFILE_SEQUENCES:-${SEQUENCES}}"
VALIDATE_SEQUENCES="${VALIDATE_SEQUENCES:-${SEQUENCES}}"

mkdir -p "${PROJECT_ROOT}/${OUTPUT_ROOT}"

PYTHONPATH_ENTRIES=("${PROJECT_ROOT}")
if [[ -d "${PROJECT_ROOT}/src" ]]; then
    PYTHONPATH_ENTRIES+=("${PROJECT_ROOT}/src")
fi
if [[ -d "${PROJECT_ROOT}/build" ]]; then
    PYTHONPATH_ENTRIES+=("${PROJECT_ROOT}/build")
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
    PYTHONPATH_ENTRIES+=("${PYTHONPATH}")
fi
export PYTHONPATH
PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")"

if [[ -d "${PROJECT_ROOT}/build" ]]; then
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        export LD_LIBRARY_PATH="${PROJECT_ROOT}/build:${LD_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="${PROJECT_ROOT}/build"
    fi
fi

run_cmd() {
    printf '>>>'
    for arg in "$@"; do
        printf ' %q' "${arg}"
    done
    printf '\n'
    if [[ -n "${COMMANDS_FILE}" ]]; then
        {
            printf '>>>'
            for arg in "$@"; do
                printf ' %q' "${arg}"
            done
            printf '\n'
        } >> "${COMMANDS_FILE}"
    fi
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "$@"
    fi
}

build_mot17_cmd() {
    local output_dir="$1"
    local seqs="$2"
    local profile_flag="$3"
    local -a cmd=(
        uv run python scripts/eval/mot17.py
        --output "${output_dir}"
        --detector "${DETECTOR}"
        --engine "${ENGINE}"
        --tiling "${TILING}"
        --split train
        --sequences "${seqs}"
        --max-frames "${MAX_FRAMES}"
    )
    if [[ -n "${REID_MODE}" ]]; then
        cmd+=(--reid-mode "${REID_MODE}")
    fi
    case "${TILE_DIAGNOSTICS}" in
        on)
            cmd+=(--tile-diagnostics)
            ;;
        off)
            ;;
        auto)
            if [[ "${TILING}" != "native_960" ]]; then
                cmd+=(--tile-diagnostics)
            fi
            ;;
    esac
    if [[ "${profile_flag}" == "yes" ]]; then
        cmd+=(--profile-stages)
    fi
    if [[ "${#EXTRA_MOT17_ARGS[@]}" -gt 0 ]]; then
        cmd+=("${EXTRA_MOT17_ARGS[@]}")
    fi
    printf '%s\n' "${cmd[@]}"
}

run_profile() {
    mapfile -t cmd < <(build_mot17_cmd "${OUTPUT_ROOT}/profile" "${PROFILE_SEQUENCES}" "yes")
    echo "== Stage profile =="
    run_cmd "${cmd[@]}"
}

run_ablation() {
    local -a cmd=(
        uv run python scripts/eval/ablation_mot17.py
        --category "${ABLATION_CATEGORIES}"
        --detector "${DETECTOR}"
        --sequences "${VALIDATE_SEQUENCES}"
        --max-frames "${MAX_FRAMES}"
        --output-root "${OUTPUT_ROOT}/ablation"
    )
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        cmd+=(--dry-run)
    fi
    echo "== Grouped ablation =="
    run_cmd "${cmd[@]}"
}

run_validate() {
    mapfile -t cmd < <(build_mot17_cmd "${OUTPUT_ROOT}/validate" "${VALIDATE_SEQUENCES}" "no")
    echo "== End-to-end validation =="
    run_cmd "${cmd[@]}"
}

write_run_metadata() {
    COMMANDS_FILE="${PROJECT_ROOT}/${OUTPUT_ROOT}/commands.txt"
    NOTES_FILE="${PROJECT_ROOT}/${OUTPUT_ROOT}/notes.md"
    SUMMARY_FILE="${PROJECT_ROOT}/${OUTPUT_ROOT}/summary.txt"
    MATRIX_FILE="${PROJECT_ROOT}/${OUTPUT_ROOT}/experiment_matrix.md"

    cat > "${SUMMARY_FILE}" <<EOF
mode=${MODE}
detector=${DETECTOR}
profile_sequences=${PROFILE_SEQUENCES}
validate_sequences=${VALIDATE_SEQUENCES}
engine=${ENGINE}
tiling=${TILING}
max_frames=${MAX_FRAMES}
ablation_categories=${ABLATION_CATEGORIES}
reid_mode=${REID_MODE:-default}
tile_diagnostics=${TILE_DIAGNOSTICS}
dry_run=${DRY_RUN}
output_root=${OUTPUT_ROOT}
timestamp=$(date -Iseconds)
EOF

    {
        echo "# Module Benchmark Notes"
        echo
        echo "- Output root: \`${OUTPUT_ROOT}\`"
        echo "- Mode: \`${MODE}\`"
        echo "- Detector: \`${DETECTOR}\`"
        echo "- Engine: \`${ENGINE}\`"
        echo "- Tiling: \`${TILING}\`"
        echo "- Profile sequences: \`${PROFILE_SEQUENCES}\`"
        echo "- Validate sequences: \`${VALIDATE_SEQUENCES}\`"
        echo "- Max frames: \`${MAX_FRAMES}\`"
        echo
        echo "## Hypothesis"
        echo
        echo "- Fill in the module or bottleneck you are testing."
        echo
        echo "## Findings"
        echo
        echo "- Profile:"
        echo "- Ablation:"
        echo "- Validate:"
        echo
        echo "## Decision"
        echo
        echo "- Keep / revert / investigate further."
    } > "${NOTES_FILE}"

    {
        echo "# Experiment Matrix"
        echo
        echo "| Experiment | Module | Sequence | Key Flags | IDF1 | MOTA | FP | FN | IDs | detect ms | post ms | track ms | reid ms |"
        echo "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        echo "| ${MODE} | fill-me | ${VALIDATE_SEQUENCES} | engine=${ENGINE};tiling=${TILING};max_frames=${MAX_FRAMES} |  |  |  |  |  |  |  |  |  |"
        echo
        echo "Rules:"
        echo "- \`detect ms\` = \`detect\`"
        echo "- \`post ms\` = \`postprocess\` or split into filter/nms/merge in notes"
        echo "- \`reid ms\` = \`reid_crop + reid_extract\`"
        echo "- Keep category and key flags explicit so runs remain comparable"
    } > "${MATRIX_FILE}"

    : > "${COMMANDS_FILE}"
}

echo "Module benchmark template"
echo "  mode: ${MODE}"
echo "  output: ${OUTPUT_ROOT}"
echo "  detector: ${DETECTOR}"
echo "  profile sequences: ${PROFILE_SEQUENCES}"
echo "  validate sequences: ${VALIDATE_SEQUENCES}"
echo "  engine: ${ENGINE}"
echo "  tiling: ${TILING}"
echo "  max frames: ${MAX_FRAMES}"
echo "  ablation categories: ${ABLATION_CATEGORIES}"

cd "${PROJECT_ROOT}"
write_run_metadata

case "${MODE}" in
    all)
        run_profile
        run_ablation
        run_validate
        ;;
    profile)
        run_profile
        ;;
    ablation)
        run_ablation
        ;;
    validate)
        run_validate
        ;;
esac
