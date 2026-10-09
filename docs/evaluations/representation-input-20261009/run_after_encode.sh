#!/usr/bin/env bash
# Task-owned durable follow-on runner: representation-input-20261009.
#
# Waits for the exact encode job (PID 3989858, no duplicate encode), then runs
# compare --arm baseline and --arm challenger in fresh processes on separate
# task-owned snapshot copies, then report. Sequential, one CPU, niced.
#
# Exits (no restart loops) with a factual status on: encode failure,
# resource pause (load > 4 or available RAM < 2 GB), config hash change,
# or any phase failure. Gate failure in report is recorded, not retried.
# No secrets, no story texts, no production/live-DB paths in this file.
set -u

TASK=/tmp/opencode/representation-input-20261009
ENCODE_PID=3989858
PROJ=/home/dev/hn-rewrite/main
UV=/home/dev/.local/bin/uv
HARNESS=$PROJ/docs/evaluations/representation-input-20261009/harness.py
SNAP_SRC=/tmp/opencode/cc-replay/snapshot.db
SAMPLE=/home/dev/hn-rewrite/llm-labels/facet-residual-20261009/sample.json
OOF_REF=$TASK/diag_threads1/oof_scores.json
MANIFEST=$TASK/manifest_threads1.json
LIVE_CFG=$PROJ/config.toml
FROZEN_CFG=$TASK/config_frozen.toml
NPZ=$TASK/challenger/challenger_vectors.npz
DONE_JSON=$TASK/challenger/challenger_done.json
ENCODE_REPORT=$TASK/challenger/encode_report.json
STATUS_JSON=$TASK/followon_status.json

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if command -v taskset >/dev/null 2>&1; then AFF="taskset -c 1"; else AFF=""; fi

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

write_status() {
    # args: event, code, detail
    python3 - "$STATUS_JSON" "$1" "$2" "$3" <<'EOF'
import json, sys, datetime
path, event, code, detail = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
try:
    st = json.load(open(path))
except Exception:
    st = {"events": []}
st["events"].append({
    "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "event": event, "code": code, "detail": detail[:500],
})
st["last"] = st["events"][-1]
json.dump(st, open(path, "w"), indent=1, sort_keys=True)
EOF
}

headroom_ok() {
    awk '{ if ($1 > 4) exit 1; exit 0; }' /proc/loadavg || return 1
    awk '/MemAvailable/ { if ($2 < 2097152) exit 1; exit 0; }' /proc/meminfo || return 1
    return 0
}

log "follow-on runner start; waiting on encode PID $ENCODE_PID"
write_status "runner_start" 0 "waiting on encode PID $ENCODE_PID"

# 1. Freeze config copy inside task dir after verifying source hash.
H=$(sha256sum "$LIVE_CFG" | cut -c1-16)
if [ "$H" != "99a2ad40f7f9973f" ]; then
    log "CONFIG HASH CHANGED ($H); refusing"
    write_status "config_hash_changed" 3 "$H"
    exit 3
fi
cp "$LIVE_CFG" "$FROZEN_CFG"
log "config frozen: $FROZEN_CFG sha16=$H"

# 2. Encoder file hashes for the record (compare re-records them per arm).
sha16() { sha256sum "$1" | cut -c1-16; }
MODEL_DIR=/home/dev/hn-rewrite/shared/mxbai-embed-xsmall-v1
log "encoder model.onnx=$(sha16 "$MODEL_DIR/model.onnx") tokenizer.json=$(sha16 "$MODEL_DIR/tokenizer.json")"

# 3. Wait for the exact encode job (max ~6 h), with headroom checks.
WAIT_N=0
while kill -0 "$ENCODE_PID" 2>/dev/null; do
    if ! headroom_ok; then
        log "RESOURCE PAUSE (load>4 or availRAM<2GB); stopping, no restart"
        write_status "resource_pause" 4 "$(uptime; free -m | head -n 2 | tr '\n' ' ')"
        exit 4
    fi
    WAIT_N=$((WAIT_N + 1))
    if [ "$WAIT_N" -gt 720 ]; then
        log "wait timeout; stopping"
        write_status "wait_timeout" 7 "encode still alive after ~6h"
        exit 7
    fi
    sleep 30
done
log "encode PID gone after ~$((WAIT_N / 2)) min of waiting"

# 4. Verify encode completed (report flag + vector count), else stop.
OK=$(python3 - "$ENCODE_REPORT" <<'EOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
    print("OK" if (r.get("complete") is True and r.get("n_vectors") == 3237) else "BAD")
except Exception:
    print("MISSING")
EOF
)
NDONE=$(python3 -c "import json; print(len(json.load(open('$DONE_JSON'))))" 2>/dev/null || echo "?")
log "encode_report=$OK done_rows=$NDONE"
if [ "$OK" != "OK" ]; then
    log "ENCODE FAILED OR INCOMPLETE; stopping, no restart"
    write_status "encode_failed" 5 "report=$OK done_rows=$NDONE"
    exit 5
fi
write_status "encode_verified" 0 "n_vectors=3237"
[ -f "$NPZ" ] || { log "npz missing; stopping"; write_status "npz_missing" 5 "$NPZ"; exit 5; }

# 5. Compare arms in fresh processes (dataset/source-hash checks inside).
if ! headroom_ok; then
    log "RESOURCE PAUSE before compare; stopping"
    write_status "resource_pause" 4 "before compare"
    exit 4
fi
log "compare baseline start"
# shellcheck disable=SC2086
$AFF nice -n 10 "$UV" run --project "$PROJ" python "$HARNESS" compare \
    --arm baseline --snapshot-src "$SNAP_SRC" \
    --work-snapshot "$TASK/snapshot_arm_baseline.db" \
    --sample "$SAMPLE" --oof-scores "$OOF_REF" \
    --config "$FROZEN_CFG" --out-dir "$TASK/compare_baseline"
CB=$?
log "compare baseline exit=$CB"
write_status "compare_baseline" "$CB" "exit=$CB"
if [ "$CB" -ne 0 ]; then log "BASELINE FAILED; stopping"; exit 6; fi

if ! headroom_ok; then
    log "RESOURCE PAUSE before challenger; stopping"
    write_status "resource_pause" 4 "before challenger"
    exit 4
fi
log "compare challenger start"
# shellcheck disable=SC2086
$AFF nice -n 10 "$UV" run --project "$PROJ" python "$HARNESS" compare \
    --arm challenger --snapshot-src "$SNAP_SRC" \
    --work-snapshot "$TASK/snapshot_arm_challenger.db" \
    --sample "$SAMPLE" --manifest "$MANIFEST" --challenger-npz "$NPZ" \
    --config "$FROZEN_CFG" --out-dir "$TASK/compare_challenger"
CC=$?
log "compare challenger exit=$CC"
write_status "compare_challenger" "$CC" "exit=$CC"
if [ "$CC" -ne 0 ]; then log "CHALLENGER FAILED; stopping"; exit 6; fi

# 6. Report (exit 1 on gate failure is a result, recorded factually).
log "report start"
nice -n 10 "$UV" run --project "$PROJ" python "$HARNESS" report \
    --baseline-rows "$TASK/compare_baseline/baseline_rows.json" \
    --challenger-rows "$TASK/compare_challenger/challenger_rows.json" \
    --sample "$SAMPLE" --out "$TASK/metrics_report.json"
CR=$?
log "report exit=$CR (1 means frozen gates failed)"
write_status "report" "$CR" "exit=$CR"
log "follow-on runner done"
exit "$CR"
