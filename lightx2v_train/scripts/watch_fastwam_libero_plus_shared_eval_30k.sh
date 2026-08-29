#!/usr/bin/env bash
set -uo pipefail

ROOT=/mnt/afs_1/charles/codes/LightX2V_fastwam
RESULTS=/mnt/afs_1/charles/codes/LIBERO-plus/eval_results/fastwam_1step_30k
RUN_SCRIPT="$ROOT/lightx2v_train/scripts/run_fastwam_libero_plus_shared_eval_30k.sh"
RUN_SESSION=fastwam_libero_plus_shared_official
LAUNCH_LOG="$RESULTS/shared_official_launcher.log"
WATCHDOG_LOG="$RESULTS/shared_watchdog.log"
PROTOCOL_DIRECTORY=official_protocol_shared_policy
LORA_SUMMARY="$RESULTS/lora_only_30k/$PROTOCOL_DIRECTORY/summary.json"
JOINT_ADAPTER="$ROOT/lightx2v_train/runs/fastwam_libero_action_1step_dmd_lora_joint/exports/checkpoint-000030000-student.pt"
COMPARISON_SUMMARY="$RESULTS/comparison_summary.json"

comparison_complete() {
    [[ -f "$COMPARISON_SUMMARY" ]] || return 1
    /mnt/afs_1/charles/env/miniconda3/envs/lightx2v_libero_plus/bin/python - "$COMPARISON_SUMMARY" <<'PY' >/dev/null 2>&1
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
verification = payload.get("verification", {})
raise SystemExit(
    0
    if verification.get("all_suites_and_tasks_complete")
    and verification.get("same_episode_catalog_seed_and_initial_states")
    and verification.get("shared_policy_lazy_prompt_cache_verified")
    else 1
)
PY
}

while ! comparison_complete; do
    if tmux has-session -t "$RUN_SESSION" 2>/dev/null; then
        sleep 60
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
    sleep 60
done

printf '[%s] four-weight comparison complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WATCHDOG_LOG"
