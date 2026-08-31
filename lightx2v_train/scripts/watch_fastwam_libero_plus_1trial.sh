#!/usr/bin/env bash
set -uo pipefail

PYTHON=/mnt/afs_1/charles/env/miniconda3/envs/lightx2v_libero_plus/bin/python
ONE_STEP_ROOT=/mnt/afs_1/charles/codes/LIBERO-plus/eval_results/fastwam_1trial_official
TWENTY_STEP_ROOT=/mnt/afs_1/charles/codes/LIBERO-plus/eval_results/fastwam_20step_1trial_official
ONE_STEP_PROTOCOL=official_protocol_shared_policy_1trial_fcf10f6a
TWENTY_STEP_PROTOCOL=official_protocol_shared_policy_20step_1trial_fcf10f6a
ONE_STEP_SCRIPT=/mnt/afs_1/charles/codes/LightX2V_fastwam_report_f8164573/lightx2v_train/scripts/run_fastwam_libero_plus_shared_eval_1trial.sh
TWENTY_STEP_SCRIPT=/mnt/afs_1/charles/codes/LightX2V_fastwam_20step_f8164573/lightx2v_train/scripts/run_fastwam_libero_plus_shared_eval_20step_1trial.sh
AGGREGATOR=/mnt/afs_1/charles/codes/LightX2V_fastwam_20step_f8164573/lightx2v_train/tools/aggregate_fastwam_libero_five_model_1trial_results.py
TASK_CATALOG_AUDIT=/mnt/afs_1/charles/codes/LIBERO-plus/eval_results/fastwam_1step_30k/TASK_CATALOG_RESOURCE_AUDIT.json
FINAL_JSON="$TWENTY_STEP_ROOT/five_model_1trial_comparison_summary.json"
FINAL_CSV="$TWENTY_STEP_ROOT/five_model_1trial_comparison_summary.csv"
LOG_ROOT=/mnt/afs_1/charles/codes/LIBERO-plus/eval_results/fastwam_1trial_official
CHECK_SECONDS=${CHECK_SECONDS:-60}
RETRY_SECONDS=${RETRY_SECONDS:-60}
EXPECTED_EPISODES=10030
mkdir -p "$ONE_STEP_ROOT" "$TWENTY_STEP_ROOT"

summary_complete() {
    local summary=$1
    local steps=$2
    [[ -f "$summary" ]] || return 1
    "$PYTHON" - "$summary" "$steps" "$EXPECTED_EPISODES" <<'PY' >/dev/null 2>&1
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
overall = payload.get("overall", payload.get("summary", {}).get("overall", {}))
protocol = payload.get("protocol", {})
raise SystemExit(0 if (
    int(overall.get("episodes", -1)) == int(sys.argv[3])
    and int(protocol.get("action_infer_steps", -1)) == int(sys.argv[2])
    and int(protocol.get("actions_per_plan", -1)) == 10
    and int(protocol.get("seed", -1)) == 0
) else 1)
PY
}

all_one_step_complete() {
    local label
    for label in native old_success_baseline_30k lora_only_30k joint_30k; do
        summary_complete "$ONE_STEP_ROOT/$label/$ONE_STEP_PROTOCOL/summary.json" 1 || return 1
    done
}

run_until_complete() {
    local kind=$1
    local script=$2
    local log=$3
    while true; do
        if [[ "$kind" == one_step ]] && all_one_step_complete; then
            return 0
        fi
        if [[ "$kind" == twenty_step ]] && \
            summary_complete "$TWENTY_STEP_ROOT/native_20step/$TWENTY_STEP_PROTOCOL/summary.json" 20; then
            return 0
        fi
        printf '[%s] starting/resuming %s official 1-trial evaluation\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$kind" >> "$LOG_ROOT/official_1trial_watchdog.log"
        /bin/bash "$script" >> "$log" 2>&1
        status=$?
        printf '[%s] %s launcher exited status=%s; validating outputs\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$kind" "$status" >> "$LOG_ROOT/official_1trial_watchdog.log"
        sleep "$RETRY_SECONDS"
    done
}

run_until_complete one_step "$ONE_STEP_SCRIPT" "$ONE_STEP_ROOT/official_1trial_launcher.log"
run_until_complete twenty_step "$TWENTY_STEP_SCRIPT" "$TWENTY_STEP_ROOT/official_1trial_launcher.log"
until "$PYTHON" "$AGGREGATOR" \
    --model native_20step "$TWENTY_STEP_ROOT/native_20step" "$TWENTY_STEP_PROTOCOL" \
        /mnt/afs_1/charles/models/fastwam/libero_uncond_2cam224.pt 20 10 \
    --model native_1step "$ONE_STEP_ROOT/native" "$ONE_STEP_PROTOCOL" \
        /mnt/afs_1/charles/models/fastwam/libero_uncond_2cam224.pt 1 10 \
    --model joint_30k "$ONE_STEP_ROOT/joint_30k" "$ONE_STEP_PROTOCOL" \
        /mnt/afs_1/charles/codes/LightX2V_fastwam/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_joint/exports/checkpoint-000030000-student.pt 1 10 \
    --model lora_only_30k "$ONE_STEP_ROOT/lora_only_30k" "$ONE_STEP_PROTOCOL" \
        /mnt/afs_1/charles/codes/LightX2V_fastwam/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_only/exports/checkpoint-000030000-student.pt 1 10 \
    --model old_joint_30k "$ONE_STEP_ROOT/old_success_baseline_30k" "$ONE_STEP_PROTOCOL" \
        /mnt/afs_1/charles/codes/LightX2V_fastwam/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_16gpu_mbs48_nogc/exports/checkpoint-000030000-student.pt 1 10 \
    --task-catalog-audit "$TASK_CATALOG_AUDIT" \
    --output-json "$FINAL_JSON" \
    --output-csv "$FINAL_CSV" \
    >> "$TWENTY_STEP_ROOT/final_aggregation.log" 2>&1; do
    printf '[%s] final aggregation failed; retrying in %ss\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RETRY_SECONDS" >> "$LOG_ROOT/official_1trial_watchdog.log"
    sleep "$RETRY_SECONDS"
done
printf '[%s] all five official 1-trial evaluations complete\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_ROOT/official_1trial_watchdog.log"
