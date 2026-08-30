#!/usr/bin/env bash
set -uo pipefail

ROOT=${ROOT:-/mnt/afs_1/charles/codes/LightX2V_fastwam_20step_f8164573}
EVAL_ROOT=${EVAL_ROOT:-/mnt/afs_1/charles/codes/LightX2V_fastwam_eval_fcf10f6a}
PYTHON=${PYTHON:-/mnt/afs_1/charles/env/miniconda3/envs/lightx2v_libero_plus/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/afs_1/charles/codes/LIBERO-plus/eval_results/fastwam_20step_50trials}
PROTOCOL_DIRECTORY=${PROTOCOL_DIRECTORY:-official_protocol_shared_policy_nvidia_egl_550_90_07_cuda5_egl4_cuda7_egl6_12w_frozen_fcf10f6a_20step}
PREREQUISITE_SESSION=${PREREQUISITE_SESSION:-fastwam_libero_plus_shared_watchdog_v9}
RUN_SESSION=${RUN_SESSION:-fastwam_libero_plus_20step_official}
RUN_SCRIPT="$ROOT/lightx2v_train/scripts/run_fastwam_libero_plus_shared_eval_20step.sh"
PROTOCOL_ROOT="$OUTPUT_ROOT/native_20step/$PROTOCOL_DIRECTORY"
SUMMARY="$PROTOCOL_ROOT/summary.json"
LAUNCH_LOG="$OUTPUT_ROOT/launcher.log"
WATCHDOG_LOG="$OUTPUT_ROOT/watchdog.log"
CHECK_SECONDS=${CHECK_SECONDS:-300}
STALL_TIMEOUT_SECONDS=${STALL_TIMEOUT_SECONDS:-3600}

mkdir -p "$OUTPUT_ROOT"

summary_complete() {
    [[ -f "$SUMMARY" ]] || return 1
    "$PYTHON" - "$SUMMARY" <<'PY' >/dev/null 2>&1
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
protocol = payload.get("protocol", {})
overall = payload.get("overall", payload.get("summary", {}).get("overall", {}))
valid = (
    int(protocol.get("action_infer_steps", -1)) == 20
    and int(protocol.get("actions_per_plan", -1)) == 10
    and int(protocol.get("seed", -1)) == 0
    and int(overall.get("episodes", -1)) == 501500
)
raise SystemExit(0 if valid else 1)
PY
}

latest_progress_epoch() {
    local latest
    latest=$(find "$PROTOCOL_ROOT/shards" -type f -name '*.json' -printf '%T@\n' 2>/dev/null \
        | sort -nr | head -n 1)
    if [[ -n "$latest" ]]; then
        printf '%s\n' "${latest%%.*}"
    else
        printf '0\n'
    fi
}

last_progress_at=$(date +%s)
while ! summary_complete; do
    if tmux has-session -t "$PREREQUISITE_SESSION" 2>/dev/null; then
        sleep "$CHECK_SECONDS"
        continue
    fi

    if tmux has-session -t "$RUN_SESSION" 2>/dev/null; then
        now=$(date +%s)
        latest=$(latest_progress_epoch)
        if (( latest > last_progress_at )); then
            last_progress_at=$latest
        elif (( now - last_progress_at >= STALL_TIMEOUT_SECONDS )); then
            printf '[%s] no shard progress for %ss; restarting %s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STALL_TIMEOUT_SECONDS" "$RUN_SESSION" \
                >> "$WATCHDOG_LOG"
            tmux kill-session -t "$RUN_SESSION" 2>/dev/null || true
            last_progress_at=$now
        fi
        sleep "$CHECK_SECONDS"
        continue
    fi

    printf '[%s] starting or resuming %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_SESSION" >> "$WATCHDOG_LOG"
    tmux new-session -d -s "$RUN_SESSION" \
        "/usr/bin/env LAUNCHER_ROOT=$ROOT EVAL_ROOT=$EVAL_ROOT FASTWAM_EVALUATION_ROOT=$EVAL_ROOT OUTPUT_ROOT=$OUTPUT_ROOT PROTOCOL_DIRECTORY=$PROTOCOL_DIRECTORY /bin/bash -lc 'exec $RUN_SCRIPT >> $LAUNCH_LOG 2>&1'" \
        || true
    last_progress_at=$(date +%s)
    sleep "$CHECK_SECONDS"
done

printf '[%s] 20-step evaluation complete\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCHDOG_LOG"
