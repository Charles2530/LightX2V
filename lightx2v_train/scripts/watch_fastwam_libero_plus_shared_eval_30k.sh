#!/usr/bin/env bash
set -uo pipefail

ROOT=/mnt/afs_1/charles/codes/LightX2V_fastwam
RESULTS=/mnt/afs_1/charles/codes/LIBERO-plus/eval_results/fastwam_1step_30k
REPORT_ROOT=${REPORT_ROOT:-$ROOT}
RUN_SCRIPT="$REPORT_ROOT/lightx2v_train/scripts/run_fastwam_libero_plus_shared_eval_30k.sh"
RUN_SESSION=fastwam_libero_plus_shared_official
LAUNCH_LOG="$RESULTS/shared_official_launcher.log"
WATCHDOG_LOG="$RESULTS/shared_watchdog.log"
PYTHON=${PYTHON:-/mnt/afs_1/charles/env/miniconda3/envs/lightx2v_libero_plus/bin/python}
PROTOCOL_DIRECTORY=${PROTOCOL_DIRECTORY:-official_protocol_shared_policy}
LORA_SUMMARY="$RESULTS/lora_only_30k/$PROTOCOL_DIRECTORY/summary.json"
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$ROOT}
JOINT_ADAPTER="$ARTIFACT_ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_joint/exports/checkpoint-000030000-student.pt"
COMPARISON_SUMMARY="$RESULTS/comparison_summary.json"
FINAL_AGGREGATOR="$REPORT_ROOT/lightx2v_train/tools/aggregate_fastwam_libero_shared_results.py"
FINAL_AGGREGATOR_LOG="$RESULTS/final_aggregation.log"
SNAPSHOT_AUDIT=${SNAPSHOT_AUDIT:-"$RESULTS/FROZEN_SNAPSHOT_PROTOCOL_AUDIT.json"}
TASK_CATALOG_AUDIT=${TASK_CATALOG_AUDIT:-"$RESULTS/TASK_CATALOG_RESOURCE_AUDIT.json"}
PROGRESS_TOOL="$REPORT_ROOT/lightx2v_train/tools/report_fastwam_libero_shared_progress.py"
PROGRESS_OUTPUT="$RESULTS/LIVE_PROGRESS.json"
STALL_TIMEOUT_SECONDS=${STALL_TIMEOUT_SECONDS:-3600}
PROGRESS_CHECK_SECONDS=${PROGRESS_CHECK_SECONDS:-60}
PROGRESS_REPORT_INTERVAL_SECONDS=${PROGRESS_REPORT_INTERVAL_SECONDS:-3600}

latest_progress_epoch() {
    local latest
    latest=$(find "$RESULTS" -type f \
        -path "*/$PROTOCOL_DIRECTORY/shards/*.json" \
        -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1)
    if [[ -n "$latest" ]]; then
        printf '%s\n' "${latest%%.*}"
    else
        printf '0\n'
    fi
}

stop_one_policy_server() {
    local pid pgid
    pid=$(pgrep -f "[e]val_fastwam_libero_shared_policy.py.*$PROTOCOL_DIRECTORY" | head -n 1)
    [[ -n "$pid" ]] || return 1
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    [[ -n "$pgid" ]] || return 1
    kill -TERM -- "-$pgid"
}

comparison_complete() {
    [[ -f "$COMPARISON_SUMMARY" ]] || return 1
    "$PYTHON" - "$COMPARISON_SUMMARY" "$PROTOCOL_DIRECTORY" <<'PY' >/dev/null 2>&1
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
protocol_directory = sys.argv[2]
verification = payload.get("verification", {})
weights = payload.get("weights", {})
expected_weights = {"native", "old_success_baseline_30k", "lora_only_30k", "joint_30k"}
weight_outputs_valid = set(weights) == expected_weights and all(
    int(item.get("total_episodes", -1)) == 501500
    and f"/{protocol_directory}/" in item.get("summary_json", "")
    and f"/{protocol_directory}/" in item.get("commands_json", "")
    for item in weights.values()
)
raise SystemExit(
    0
    if verification.get("all_suites_and_tasks_complete")
    and verification.get("same_episode_catalog_seed_and_initial_states")
    and verification.get("shared_policy_lazy_prompt_cache_verified")
    and weight_outputs_valid
    else 1
)
PY
}

enhanced_comparison_complete() {
    [[ -f "$COMPARISON_SUMMARY" ]] || return 1
    "$PYTHON" - "$COMPARISON_SUMMARY" <<'PY' >/dev/null 2>&1
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
weights = payload.get("weights", {})
verification = payload.get("verification", {})
valid = len(weights) == 4 and verification.get(
    "checkpoint_and_dependency_hashes_match_snapshot"
)
valid = (
    valid
    and verification.get("evaluation_and_official_hashes_match_snapshot")
    and verification.get("libero_task_resources_match_baseline")
    and verification.get("episode_outcomes_match_official_horizon")
)
for item in weights.values():
    checkpoint = item.get("checkpoint", {})
    logs = item.get("server_log_evidence", [])
    valid = (
        valid
        and len(checkpoint.get("sha256", "")) == 64
        and checkpoint.get("matches_snapshot_audit")
        and len(logs) == 7
    )
    valid = valid and all(
        log.get("commands_match_manifest")
        and int(log.get("action_infer_steps", -1)) == 1
        and int(log.get("error_markers", -1)) == 0
        for log in logs
    )
raise SystemExit(0 if valid else 1)
PY
}

run_final_aggregator() {
    "$PYTHON" "$FINAL_AGGREGATOR" \
        --results-root "$RESULTS" \
        --protocol-directory "$PROTOCOL_DIRECTORY" \
        --snapshot-audit "$SNAPSHOT_AUDIT" \
        --task-catalog-audit "$TASK_CATALOG_AUDIT" \
        --weight native native /mnt/afs_1/charles/models/fastwam/libero_uncond_2cam224.pt \
        --weight old_success_baseline_30k old_success_baseline_30k \
            "$ARTIFACT_ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_16gpu_mbs48_nogc/exports/checkpoint-000030000-student.pt" \
        --weight lora_only_30k lora_only_30k \
            "$ARTIFACT_ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_only/exports/checkpoint-000030000-student.pt" \
        --weight joint_30k joint_30k \
            "$ARTIFACT_ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_joint/exports/checkpoint-000030000-student.pt" \
        >> "$FINAL_AGGREGATOR_LOG" 2>&1
}

last_progress_at=$(date +%s)
last_report_at=0

while ! comparison_complete; do
    if tmux has-session -t "$RUN_SESSION" 2>/dev/null; then
        now=$(date +%s)
        if (( now - last_report_at >= PROGRESS_REPORT_INTERVAL_SECONDS )); then
            if "$PYTHON" "$PROGRESS_TOOL" \
                --results-root "$RESULTS" \
                --protocol-directory "$PROTOCOL_DIRECTORY" \
                --output "$PROGRESS_OUTPUT" >> "$WATCHDOG_LOG" 2>&1; then
                last_report_at=$now
            else
                printf '[%s] failed to refresh %s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PROGRESS_OUTPUT" >> "$WATCHDOG_LOG"
            fi
        fi
        latest=$(latest_progress_epoch)
        if (( latest > last_progress_at )); then
            last_progress_at=$latest
        elif (( now - last_progress_at >= STALL_TIMEOUT_SECONDS )); then
            printf '[%s] no shard progress for %ss; restarting %s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STALL_TIMEOUT_SECONDS" "$RUN_SESSION" \
                >> "$WATCHDOG_LOG"
            if ! stop_one_policy_server; then
                tmux kill-session -t "$RUN_SESSION" 2>/dev/null || true
            fi
            last_progress_at=$now
        fi
        sleep "$PROGRESS_CHECK_SECONDS"
        continue
    fi

    if [[ -f "$LORA_SUMMARY" && ! -f "$JOINT_ADAPTER" ]]; then
        sleep 300
        continue
    fi

    printf '[%s] restarting %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_SESSION" >> "$WATCHDOG_LOG"
    tmux new-session -d -s "$RUN_SESSION" \
        "/bin/bash -lc 'exec $RUN_SCRIPT >> $LAUNCH_LOG 2>&1'" \
        || true
    last_progress_at=$(date +%s)
    sleep "$PROGRESS_CHECK_SECONDS"
done

printf '[%s] base comparison complete; generating auditable final comparison\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCHDOG_LOG"
if ! run_final_aggregator || ! enhanced_comparison_complete; then
    printf '[%s] auditable final comparison failed; see %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$FINAL_AGGREGATOR_LOG" >> "$WATCHDOG_LOG"
    exit 1
fi
printf '[%s] four-weight comparison complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCHDOG_LOG"
