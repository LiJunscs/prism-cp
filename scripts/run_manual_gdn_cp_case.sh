#!/usr/bin/env bash
# Shared native DUPC recipe for baseline N and the installed joint executor.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: [environment overrides] bash run_manual_gdn_cp_case.sh [DUPC] [RUN_ID] [MASTER_PORT]
Current path: dynamic CP with contiguous GDN blocks; the joint replay hook is opt-in.
Set MEGATRON_ROOT, PYTHON, NNODES, NODE_RANK and MASTER_ADDR for your cluster.
Use gpu_smoke.py with an explicit frozen input/replay for full-update experiments.
DRY_RUN=1 prints the command. This script does not authorize a GPU run.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

warn() {
  echo "WARNING: $*" >&2
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  usage
  exit 0
fi

# ---------------------------------------------------------------------------
# Editable configuration. Every value can also be overridden as an environment
# variable, so the file itself does not need to be copied for every experiment.
# ---------------------------------------------------------------------------

MODE=${1:-${MODE:-DUPC}}
RUN_ID=${2:-${RUN_ID:-}}
MASTER_PORT=${3:-${MASTER_PORT:-29500}}
CASE_NAME=${CASE_NAME:-manual_case}

PRISM_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MEGATRON_ROOT=${MEGATRON_ROOT:-$PRISM_ROOT/../Megatron-LM}
PYTHON=${PYTHON:-$(command -v python3)}
TORCHRUN=${TORCHRUN:-$(dirname -- "$PYTHON")/torchrun}
OUTPUT_ROOT=${OUTPUT_ROOT:-$MEGATRON_ROOT/outputs/manual_gdn_cp}
FAKE_DNS_SO=${FAKE_DNS_SO:-}

NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MIN_DYNAMIC_CP_SIZE=${MIN_DYNAMIC_CP_SIZE:-1}

LENGTHS_CSV=${LENGTHS_CSV:-66560,31744,16384,8192,8192}
PROMPT_LENGTHS_CSV=${PROMPT_LENGTHS_CSV:-}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-0}
MAX_TOKENS_PER_RANK=${MAX_TOKENS_PER_RANK:-16384}

NUM_LAYERS=${NUM_LAYERS:-16}
HIDDEN_SIZE=${HIDDEN_SIZE:-4096}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-$((4 * HIDDEN_SIZE))}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-32}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-8}
KV_CHANNELS=${KV_CHANNELS:-$((HIDDEN_SIZE / NUM_ATTENTION_HEADS))}
LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-4}
LINEAR_KEY_HEAD_DIM=${LINEAR_KEY_HEAD_DIM:-128}
LINEAR_VALUE_HEAD_DIM=${LINEAR_VALUE_HEAD_DIM:-128}
LINEAR_NUM_KEY_HEADS=${LINEAR_NUM_KEY_HEADS:-16}
LINEAR_NUM_VALUE_HEADS=${LINEAR_NUM_VALUE_HEADS:-16}
LINEAR_CONV_KERNEL_DIM=${LINEAR_CONV_KERNEL_DIM:-4}
VOCAB_SIZE=${VOCAB_SIZE:-32768}

RUN_ITERS=${RUN_ITERS:-40}
WARMUP_STEPS=${WARMUP_STEPS:-10}
EVAL_ITERS=${EVAL_ITERS:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-10000000}
SEED=${SEED:-1234}
RECOMPUTE=${RECOMPUTE:-full}
ENABLE_OVERLAP=${ENABLE_OVERLAP:-1}
DDP_BUCKET_SIZE=${DDP_BUCKET_SIZE:-64000000}
LOG_MEMORY_INTERVAL=${LOG_MEMORY_INTERVAL:-$RUN_ITERS}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-0}
TRACE=${TRACE:-0}
CP_PARTITION_LOG=${CP_PARTITION_LOG:-$TRACE}
DRY_RUN=${DRY_RUN:-0}
NSYS_PROFILE=${NSYS_PROFILE:-0}
PROFILE_STEP_START=${PROFILE_STEP_START:-$WARMUP_STEPS}
PROFILE_STEP_END=${PROFILE_STEP_END:-$((WARMUP_STEPS + 3))}
PROFILE_RANKS=${PROFILE_RANKS:-0}
POST_RUN_SLEEP=${POST_RUN_SLEEP:-0}

[[ -n $RUN_ID ]] || die "Set RUN_ID, for example RUN_ID=pair01"
[[ $CASE_NAME =~ ^[A-Za-z0-9._-]+$ ]] || die "CASE_NAME may contain only letters, digits, dot, underscore, and dash"
[[ $RUN_ID =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_ID may contain only letters, digits, dot, underscore, and dash"
[[ $MODE == DUPC ]] || die "Only the current DUPC baseline/joint recipe is retained"

for item in \
  "NNODES=$NNODES" "GPUS_PER_NODE=$GPUS_PER_NODE" "NODE_RANK=$NODE_RANK" \
  "MASTER_PORT=$MASTER_PORT" "MIN_DYNAMIC_CP_SIZE=$MIN_DYNAMIC_CP_SIZE" \
  "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE" "MAX_TOKENS_PER_RANK=$MAX_TOKENS_PER_RANK" \
  "NUM_LAYERS=$NUM_LAYERS" "HIDDEN_SIZE=$HIDDEN_SIZE" \
  "FFN_HIDDEN_SIZE=$FFN_HIDDEN_SIZE" "NUM_ATTENTION_HEADS=$NUM_ATTENTION_HEADS" \
  "NUM_QUERY_GROUPS=$NUM_QUERY_GROUPS" "KV_CHANNELS=$KV_CHANNELS" \
  "LINEAR_KEY_HEAD_DIM=$LINEAR_KEY_HEAD_DIM" \
  "LINEAR_VALUE_HEAD_DIM=$LINEAR_VALUE_HEAD_DIM" \
  "LINEAR_NUM_KEY_HEADS=$LINEAR_NUM_KEY_HEADS" \
  "LINEAR_NUM_VALUE_HEADS=$LINEAR_NUM_VALUE_HEADS" \
  "LINEAR_CONV_KERNEL_DIM=$LINEAR_CONV_KERNEL_DIM" "VOCAB_SIZE=$VOCAB_SIZE" \
  "RUN_ITERS=$RUN_ITERS" "WARMUP_STEPS=$WARMUP_STEPS" \
  "EVAL_ITERS=$EVAL_ITERS" "EVAL_INTERVAL=$EVAL_INTERVAL" "SEED=$SEED" \
  "DDP_BUCKET_SIZE=$DDP_BUCKET_SIZE" "LOG_MEMORY_INTERVAL=$LOG_MEMORY_INTERVAL" \
  "ENABLE_OVERLAP=$ENABLE_OVERLAP" "SAVE_CHECKPOINT=$SAVE_CHECKPOINT" \
  "TRACE=$TRACE" "CP_PARTITION_LOG=$CP_PARTITION_LOG" \
  "DRY_RUN=$DRY_RUN" "NSYS_PROFILE=$NSYS_PROFILE" \
  "POST_RUN_SLEEP=$POST_RUN_SLEEP"; do
  name=${item%%=*}
  value=${item#*=}
  [[ $value =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer, got: $value"
done

(( NNODES >= 1 && GPUS_PER_NODE >= 1 )) || die "NNODES and GPUS_PER_NODE must be positive"
(( NODE_RANK >= 0 && NODE_RANK < NNODES )) || die "NODE_RANK=$NODE_RANK is invalid for NNODES=$NNODES"
(( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || die "MASTER_PORT must be in 1..65535"
(( NUM_LAYERS > 0 && HIDDEN_SIZE > 0 && FFN_HIDDEN_SIZE > 0 )) || die "Invalid model dimensions"
(( NUM_ATTENTION_HEADS > 0 && HIDDEN_SIZE % NUM_ATTENTION_HEADS == 0 )) || die "HIDDEN_SIZE must be divisible by NUM_ATTENTION_HEADS"
(( NUM_QUERY_GROUPS > 0 && NUM_ATTENTION_HEADS % NUM_QUERY_GROUPS == 0 )) || die "NUM_QUERY_GROUPS must divide NUM_ATTENTION_HEADS"
(( RUN_ITERS > WARMUP_STEPS )) || die "RUN_ITERS must be greater than WARMUP_STEPS"
if (( RUN_ITERS - WARMUP_STEPS < 30 )); then
  warn "Only $((RUN_ITERS - WARMUP_STEPS)) measured iterations remain after warmup; use at least 30 for evidence runs"
fi
(( MAX_TOKENS_PER_RANK > 0 && LOG_MEMORY_INTERVAL > 0 && EVAL_INTERVAL > 0 )) || \
  die "Token and logging/evaluation intervals must be positive"
[[ $RECOMPUTE == full || $RECOMPUTE == none ]] || die "RECOMPUTE must be full or none"
[[ $ENABLE_OVERLAP == 0 || $ENABLE_OVERLAP == 1 ]] || die "ENABLE_OVERLAP must be 0 or 1"
[[ $SAVE_CHECKPOINT == 0 || $SAVE_CHECKPOINT == 1 ]] || die "SAVE_CHECKPOINT must be 0 or 1"
[[ $TRACE == 0 || $TRACE == 1 ]] || die "TRACE must be 0 or 1"
[[ $CP_PARTITION_LOG == 0 || $CP_PARTITION_LOG == 1 ]] || die "CP_PARTITION_LOG must be 0 or 1"
[[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || die "DRY_RUN must be 0 or 1"
[[ $NSYS_PROFILE == 0 || $NSYS_PROFILE == 1 ]] || die "NSYS_PROFILE must be 0 or 1"
if (( NSYS_PROFILE && RUN_ITERS < WARMUP_STEPS + 2 )); then
  die "NSYS_PROFILE requires RUN_ITERS >= WARMUP_STEPS + 2"
fi
if (( NSYS_PROFILE )); then
  [[ $PROFILE_STEP_START =~ ^[0-9]+$ && $PROFILE_STEP_END =~ ^[0-9]+$ ]] ||
    die "Profile step boundaries must be non-negative integers"
  (( PROFILE_STEP_START >= WARMUP_STEPS && PROFILE_STEP_END > PROFILE_STEP_START && PROFILE_STEP_END <= RUN_ITERS )) ||
    die "Profile window must be after warmup and within RUN_ITERS"
  [[ $PROFILE_RANKS =~ ^[0-9]+([,\ ][0-9]+)*$ ]] || die "PROFILE_RANKS must list integer ranks"
fi

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
(( WORLD_SIZE >= 2 && (WORLD_SIZE & (WORLD_SIZE - 1)) == 0 )) || \
  die "NNODES*GPUS_PER_NODE must be a power of two and at least 2"
(( MIN_DYNAMIC_CP_SIZE >= 1 && WORLD_SIZE % MIN_DYNAMIC_CP_SIZE == 0 )) || \
  die "MIN_DYNAMIC_CP_SIZE must divide WORLD_SIZE"
if (( NNODES > 1 )) && [[ $MASTER_ADDR == 127.0.0.1 || $MASTER_ADDR == localhost ]]; then
  die "For multiple nodes, set MASTER_ADDR to the reachable address of node 0"
fi

[[ -d $MEGATRON_ROOT ]] || die "MEGATRON_ROOT does not exist: $MEGATRON_ROOT"
[[ -f $MEGATRON_ROOT/pretrain_gpt.py ]] || die "pretrain_gpt.py not found below MEGATRON_ROOT"
[[ -x $PYTHON ]] || die "PYTHON is not executable: $PYTHON"
[[ -x $TORCHRUN ]] || die "TORCHRUN is not executable: $TORCHRUN"

if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  CUDA_VISIBLE_DEVICES=$(printf '%s,' $(seq 0 $((GPUS_PER_NODE - 1))))
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%,}
fi
IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
(( ${#visible_devices[@]} == GPUS_PER_NODE )) || \
  die "CUDA_VISIBLE_DEVICES contains ${#visible_devices[@]} devices, expected GPUS_PER_NODE=$GPUS_PER_NODE"

# Build and validate the exact packed-sequence fixture. GLOBAL_BATCH_SIZE=0
# means one copy of every listed sequence. A larger batch must contain an
# integer number of complete length cycles; the fixture is expanded explicitly
# so the saved command fully describes the global step.
if [[ -n ${SEQUENCE_CP_PACKET_MANIFEST:-} ]]; then
  mapfile -t META < <(
    "$PYTHON" - "$SEQUENCE_CP_PACKET_MANIFEST" "$WORLD_SIZE" "$MAX_TOKENS_PER_RANK" "${BASH_SOURCE[0]}" <<'PY_PACKET'
import json, sys
from pathlib import Path
m = json.load(open(sys.argv[1]))
sys.path.insert(0,str(Path(sys.argv[4]).resolve().parents[1]/'experiments/sequence_dynamic_cp'))
from joint_cp.inputs import packet_launch_metadata
metadata=packet_launch_metadata(m,int(sys.argv[2]),int(sys.argv[3]))
for value in metadata:
    print(json.dumps(value,separators=(',',':')) if isinstance(value,dict) else value)
PY_PACKET
  )
else
mapfile -t META < <(
  "$PYTHON" - "$LENGTHS_CSV" "$PROMPT_LENGTHS_CSV" "$WORLD_SIZE" \
    "$GLOBAL_BATCH_SIZE" "$MAX_TOKENS_PER_RANK" <<'PY'
import json
import sys


def parse_csv(raw, name, allow_zero=False):
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated integer list: {exc}")
    invalid = any(value < 0 if allow_zero else value <= 0 for value in values)
    if not values or invalid:
        requirement = "non-negative" if allow_zero else "positive"
        raise SystemExit(f"{name} must contain {requirement} integers")
    return values


lengths = parse_csv(sys.argv[1], "LENGTHS_CSV")
prompt_raw = sys.argv[2].strip()
prompts = parse_csv(prompt_raw, "PROMPT_LENGTHS_CSV", allow_zero=True) if prompt_raw else [n // 2 for n in lengths]
world, global_batch, cap = map(int, sys.argv[3:])
if len(prompts) != len(lengths):
    raise SystemExit("PROMPT_LENGTHS_CSV must contain exactly one value per sequence length")
if any(prompt < 0 or prompt >= length for prompt, length in zip(prompts, lengths)):
    raise SystemExit("Every prompt length must satisfy 0 <= prompt < sequence length")
if global_batch == 0:
    global_batch = len(lengths)
if global_batch < len(lengths) or global_batch % len(lengths):
    raise SystemExit("GLOBAL_BATCH_SIZE must be zero or a positive multiple of the sequence count")

repeat = global_batch // len(lengths)
lengths = lengths * repeat
prompts = prompts * repeat
alignment = 2 * world
padded = [((length + alignment - 1) // alignment) * alignment for length in lengths]
if max(padded) > world * cap:
    raise SystemExit(
        f"Longest padded sequence {max(padded)} exceeds parent capacity {world * cap}; "
        "increase WORLD_SIZE/MAX_TOKENS_PER_RANK or shorten the sequence"
    )
total = sum(padded)
if total % world:
    raise SystemExit("Internal error: total padded tokens are not divisible by WORLD_SIZE")

fixture = {
    "mode": "packed_cp_case",
    "lengths": lengths,
    "prompt_lengths": prompts,
}
print(global_batch)
# seq-length is a single packed-buffer bound, not the sum of all optimizer
# microbatches. Keep every sample in the fixture and let the native scheduler
# split a larger input cycle into as many microbatches as required.
print(min(total, world * cap))
print(total // world)
print(sum(lengths))
print(sum(length - prompt for length, prompt in zip(lengths, prompts)))
print(",".join(map(str, padded)))
print(json.dumps(fixture, separators=(",", ":")))
PY
)
fi
# Native Windows Python emits CRLF even when called from Git Bash. Linux does
# not need this, but stripping a trailing carriage return makes dry-run checks
# portable and keeps generated numeric arguments clean.
for index in "${!META[@]}"; do
  META[$index]=${META[$index]%$'\r'}
done
(( ${#META[@]} == 7 )) || die "Failed to build packed-sequence metadata"
GLOBAL_BATCH_SIZE=${META[0]}
SEQ_LENGTH=${META[1]}
PARENT_LOCAL_TOKENS=${META[2]}
INPUT_TOKENS=${META[3]}
SUPERVISED_TOKENS=${META[4]}
PADDED_LENGTHS_CSV=${META[5]}
FIXTURE_JSON=${META[6]}

declare -a POLICY=(--dynamic-context-parallel --min-dynamic-context-parallel-size "$MIN_DYNAMIC_CP_SIZE"
  --sequence-packing-scheduler default_dynamic_cp --gdn-preserve-parent-contiguous)

declare -a EXTRA_ARGS=()
if [[ $RECOMPUTE == full ]]; then
  EXTRA_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
fi
if (( ENABLE_OVERLAP )); then
  EXTRA_ARGS+=(--overlap-grad-reduce --overlap-param-gather)
fi

RUN_DIR=$OUTPUT_ROOT/$CASE_NAME/$RUN_ID/$MODE
LOG_FILE=$RUN_DIR/train_node${NODE_RANK}.log
COMMAND_FILE=$RUN_DIR/command_node${NODE_RANK}.txt
CONFIG_FILE=$RUN_DIR/config_node${NODE_RANK}.txt

declare -a PROFILE_ARGS=() PROFILE_CMD=()
if (( NSYS_PROFILE )); then
  PROFILE_ARGS=(
    --profile
    --profile-step-start "$PROFILE_STEP_START"
    --profile-step-end "$PROFILE_STEP_END"
    --profile-ranks ${PROFILE_RANKS//,/ }
    --nvtx-ranges
  )
  if [[ ${HIDDEN_NSYS_MANUAL:-0} == 1 ]]; then
    # The hidden adapter controls cudaProfilerStart/Stop without emit_nvtx's
    # autograd-wide instrumentation; its explicit layer/transport ranges remain.
    PROFILE_ARGS=()
  fi
  PROFILE_CMD=(
    nsys profile
    -s none
    -t nvtx,cuda
    -o "$RUN_DIR/nsys_profile_node${NODE_RANK}"
    --force-overwrite true
    --capture-range cudaProfilerApi
    --capture-range-end stop
  )
  if [[ ${NSYS_CUDA_MEMORY_USAGE:-0} == 1 ]]; then
    PROFILE_CMD+=(--cuda-memory-usage=true)
  fi
fi

declare -a CMD=(
  "$TORCHRUN"
  --nnodes "$NNODES"
  --nproc_per_node "$GPUS_PER_NODE"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  --max_restarts 0
  pretrain_gpt.py
  --num-layers "$NUM_LAYERS"
  --hidden-size "$HIDDEN_SIZE"
  --ffn-hidden-size "$FFN_HIDDEN_SIZE"
  --num-attention-heads "$NUM_ATTENTION_HEADS"
  --num-query-groups "$NUM_QUERY_GROUPS"
  --kv-channels "$KV_CHANNELS"
  --group-query-attention
  --experimental-attention-variant gdn
  --linear-attention-freq "$LINEAR_ATTENTION_FREQ"
  --linear-key-head-dim "$LINEAR_KEY_HEAD_DIM"
  --linear-value-head-dim "$LINEAR_VALUE_HEAD_DIM"
  --linear-num-key-heads "$LINEAR_NUM_KEY_HEADS"
  --linear-num-value-heads "$LINEAR_NUM_VALUE_HEADS"
  --linear-conv-kernel-dim "$LINEAR_CONV_KERNEL_DIM"
  --linear-cp-mode chunkwise
  --cp-partition-mode zigzag
  --cp-comm-type p2p
  --context-parallel-size "$WORLD_SIZE"
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  "${POLICY[@]}"
  --max-seqlen-per-dp-cp-rank "$MAX_TOKENS_PER_RANK"
  --micro-batch-size 1
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --seq-length "$SEQ_LENGTH"
  --max-position-embeddings "$SEQ_LENGTH"
  --use-varlen-dataset
  --mock-data
  --varlen-mock-dataset-config-json "$FIXTURE_JSON"
  --tokenizer-type NullTokenizer
  --vocab-size "$VOCAB_SIZE"
  --null-tokenizer-pad-id 0
  --split 100,0,0
  --dataloader-type single
  --num-workers 0
  --no-create-attention-mask-in-dataloader
  --normalization RMSNorm
  --position-embedding-type rope
  --transformer-impl transformer_engine
  --attention-backend flash
  --swiglu
  --disable-bias-linear
  --untie-embeddings-and-output-weights
  --bf16
  --attention-dropout 0
  --hidden-dropout 0
  --optimizer adam
  --lr 1e-4
  --min-lr 1e-5
  --lr-decay-style cosine
  --train-iters "$RUN_ITERS"
  --lr-decay-iters "${LR_DECAY_ITERS:-$RUN_ITERS}"
  --lr-warmup-iters 0
  --weight-decay 0.1
  --clip-grad 1.0
  --seed "$SEED"
  --use-distributed-optimizer
  --disable-gloo-process-groups
  --ckpt-format torch_dist
  --ddp-bucket-size "$DDP_BUCKET_SIZE"
  --calculate-per-token-loss
  --no-gradient-accumulation-fusion
  --log-interval 1
  --log-memory-interval "$LOG_MEMORY_INTERVAL"
  --log-throughput
  --eval-iters "$EVAL_ITERS"
  --eval-interval "$EVAL_INTERVAL"
  --distributed-backend nccl
  --distributed-timeout-minutes 30
  --cross-entropy-loss-fusion
  --cross-entropy-fusion-impl native
  "${EXTRA_ARGS[@]}"
  "${PROFILE_ARGS[@]}"
)
if (( SAVE_CHECKPOINT )); then
  CMD+=(--save "$RUN_DIR/checkpoints" --save-interval "$RUN_ITERS")
fi
CMD=("${PROFILE_CMD[@]}" "${CMD[@]}")

if [[ -n ${SEQUENCE_CP_PACKET_MANIFEST:-} ]]; then
  echo "Mode=$MODE world=$WORLD_SIZE model=${NUM_LAYERS}L/${HIDDEN_SIZE}H packet_count=$GLOBAL_BATCH_SIZE logical_samples=variable_per_manifest"
else
  echo "Mode=$MODE world=$WORLD_SIZE model=${NUM_LAYERS}L/${HIDDEN_SIZE}H sequence_count=$GLOBAL_BATCH_SIZE"
fi
echo "Sequence buffer=$SEQ_LENGTH input_tokens=$INPUT_TOKENS supervised_tokens=$SUPERVISED_TOKENS parent_local_tokens=$PARENT_LOCAL_TOKENS measurement_steps=$((RUN_ITERS - WARMUP_STEPS))"

if (( DRY_RUN )); then
  echo "Dry run: no output files were created."
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if (( NSYS_PROFILE )) && ! command -v nsys >/dev/null 2>&1; then
  die "NSYS_PROFILE=1 requires nsys in PATH on every node"
fi

if [[ -e $LOG_FILE || -e $COMMAND_FILE || -e $CONFIG_FILE ]]; then
  die "This node/mode already has output; choose a fresh RUN_ID or remove the intended run directory: $RUN_DIR"
fi
mkdir -p "$RUN_DIR"

if [[ -n ${SEQUENCE_CP_PACKET_MANIFEST:-} ]]; then
  GIT_HEAD=omitted_user_no_digest_operations
else
  GIT_HEAD=$(git -C "$MEGATRON_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
fi
{
  printf 'case_name=%s\n' "$CASE_NAME"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'mode=%s\n' "$MODE"
  printf 'git_head=%s\n' "$GIT_HEAD"
  printf 'nnodes=%s\n' "$NNODES"
  printf 'gpus_per_node=%s\n' "$GPUS_PER_NODE"
  printf 'world_size=%s\n' "$WORLD_SIZE"
  printf 'node_rank=%s\n' "$NODE_RANK"
  printf 'master_addr=%s\n' "$MASTER_ADDR"
  printf 'master_port=%s\n' "$MASTER_PORT"
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  if [[ -n ${SEQUENCE_CP_PACKET_MANIFEST:-} ]]; then
    printf 'global_batch_unit=packet\npacket_manifest=%s\nlogical_samples=variable_per_manifest\n' "$SEQUENCE_CP_PACKET_MANIFEST"
    printf 'fixture_scope=initialization_only_actual_inputs_from_packet_manifest\n'
  fi
  printf 'lengths_csv=%s\n' "$LENGTHS_CSV"
  printf 'prompt_lengths_csv=%s\n' "${PROMPT_LENGTHS_CSV:-auto_half}"
  printf 'padded_lengths_csv=%s\n' "$PADDED_LENGTHS_CSV"
  printf 'global_batch_size=%s\n' "$GLOBAL_BATCH_SIZE"
  printf 'seq_length=%s\n' "$SEQ_LENGTH"
  printf 'parent_local_tokens=%s\n' "$PARENT_LOCAL_TOKENS"
  printf 'input_tokens=%s\n' "$INPUT_TOKENS"
  printf 'supervised_tokens=%s\n' "$SUPERVISED_TOKENS"
  printf 'max_tokens_per_rank=%s\n' "$MAX_TOKENS_PER_RANK"
  printf 'num_layers=%s\n' "$NUM_LAYERS"
  printf 'hidden_size=%s\n' "$HIDDEN_SIZE"
  printf 'ffn_hidden_size=%s\n' "$FFN_HIDDEN_SIZE"
  printf 'num_attention_heads=%s\n' "$NUM_ATTENTION_HEADS"
  printf 'num_query_groups=%s\n' "$NUM_QUERY_GROUPS"
  printf 'kv_channels=%s\n' "$KV_CHANNELS"
  printf 'linear_attention_freq=%s\n' "$LINEAR_ATTENTION_FREQ"
  printf 'linear_num_key_heads=%s\n' "$LINEAR_NUM_KEY_HEADS"
  printf 'linear_num_value_heads=%s\n' "$LINEAR_NUM_VALUE_HEADS"
  printf 'linear_key_head_dim=%s\n' "$LINEAR_KEY_HEAD_DIM"
  printf 'linear_value_head_dim=%s\n' "$LINEAR_VALUE_HEAD_DIM"
  printf 'recompute=%s\n' "$RECOMPUTE"
  printf 'enable_overlap=%s\n' "$ENABLE_OVERLAP"
  printf 'run_iters=%s\n' "$RUN_ITERS"
  printf 'warmup_steps_for_analysis=%s\n' "$WARMUP_STEPS"
  printf 'nsys_profile=%s\n' "$NSYS_PROFILE"
  if (( NSYS_PROFILE )); then
    printf 'nsys_output=%s\n' "$RUN_DIR/nsys_profile_node${NODE_RANK}"
    printf 'profile_step_start=%s\n' "$PROFILE_STEP_START"
    printf 'profile_step_end=%s\n' "$PROFILE_STEP_END"
    printf 'profile_ranks=%s\n' "$PROFILE_RANKS"
  fi
  printf 'eval_iters=%s\n' "$EVAL_ITERS"
  printf 'eval_interval=%s\n' "$EVAL_INTERVAL"
  printf 'seed=%s\n' "$SEED"
  printf 'cp_partition_log=%s\n' "$CP_PARTITION_LOG"
  printf 'fixture_json=%s\n' "$FIXTURE_JSON"
} > "$CONFIG_FILE"
printf '%q ' "${CMD[@]}" > "$COMMAND_FILE"
printf '\n' >> "$COMMAND_FILE"

echo "Run directory: $RUN_DIR"
echo "Configuration: $CONFIG_FILE"
echo "Command:       $COMMAND_FILE"

declare -a ENV_CMD=(env
  "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
  "PYTHONDONTWRITEBYTECODE=1"
  "MCORE_PACKED_CP_NVTX=$TRACE"
  "MCORE_DYNAMIC_CP_LOG=$CP_PARTITION_LOG"
  "MCORE_DYNAMIC_GDN_CP_LOG=$CP_PARTITION_LOG"
  "PYTHONPATH=$MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
)
if [[ -n $FAKE_DNS_SO ]]; then
  if [[ -f $FAKE_DNS_SO ]]; then
    ENV_CMD+=("LD_PRELOAD=$FAKE_DNS_SO${LD_PRELOAD:+:$LD_PRELOAD}")
  else
    warn "FAKE_DNS_SO does not exist and will not be preloaded: $FAKE_DNS_SO"
  fi
fi

set +e
(
  cd "$MEGATRON_ROOT" || exit 1
  "${ENV_CMD[@]}" "${CMD[@]}"
) 2>&1 | tee -a "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e
printf 'torchrun_exit_code=%s\n' "$rc" | tee -a "$LOG_FILE"
if (( POST_RUN_SLEEP > 0 )); then
  sleep "$POST_RUN_SLEEP"
fi
exit "$rc"
