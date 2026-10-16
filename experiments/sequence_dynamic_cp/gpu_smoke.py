"""Run a frozen native N or joint CP plan through the full training path."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from numerical_contracts import contract
from paths import HERE, ROOT
from two_dimensional_inputs import validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--two-dimensional-input', type=Path, required=True)
    parser.add_argument('--two-dimensional-config', type=Path)
    parser.add_argument('--native-n', action='store_true')
    parser.add_argument('--packet-role', choices=('N', 'S'))
    parser.add_argument('--pattern', default='GGG*GGG*')
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--cap', type=int, default=16384)
    parser.add_argument('--nnodes', type=int, default=int(os.environ.get('NNODES', '2')))
    parser.add_argument('--node-rank', type=int, default=int(os.environ.get('NODE_RANK', '0')))
    parser.add_argument('--master-addr', default=os.environ.get('MASTER_ADDR', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('MASTER_PORT', '29500')))
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--reuse-source-evidence', required=True)
    # Retained CLI spellings for existing frozen drivers. Their values are fixed.
    parser.add_argument('--native-fallback', action='store_true')
    parser.add_argument('--mixed-cp', action='store_true')
    parser.add_argument('--skip-gpu-preflight', action='store_true')
    parser.add_argument('--eval-iters', type=int, choices=(0,), default=0)
    parser.add_argument('--numerics', choices=('off',), default='off')
    parser.add_argument('--logical', choices=('off',), default='off')
    args = parser.parse_args()
    if args.native_n == bool(args.two_dimensional_config):
        parser.error('Select native N or an explicit joint executor configuration')
    if args.num_layers <= 0 or args.num_layers % 4 or args.pattern != 'GGG*' * (args.num_layers // 4):
        parser.error('Current experiments require repeated GGG* blocks (GDN:GQA=3:1)')
    if not 0 <= args.warmup < args.iterations or not 0 <= args.node_rank < args.nnodes:
        parser.error('Invalid update window or node rank')
    role = 'N' if args.native_n else 'S'
    if args.packet_role and args.packet_role != role:
        parser.error('Packet role differs from executor')
    manifest = json.loads(args.two_dimensional_input.read_text())
    validate_manifest(manifest)
    if manifest.get('world', 16) != args.nnodes * 8:
        parser.error('Input world differs from the 8-GPU-per-node launch')
    rows = json.loads(args.replay.read_text())
    joint = None
    if not args.native_n:
        from joint_cp.replay import validate_replay
        joint = json.loads(args.two_dimensional_config.read_text())
        if joint.get('mode') != 'joint':
            parser.error('Retired adapter; executor mode must be joint')
        if not rows:
            parser.error('Empty replay')
        for row in rows:
            validate_replay(row, args.nnodes * 8, args.pattern)
        args.cap = max(row['capacity_tokens'] for row in rows)
    if len(rows) != args.iterations or any(
        row['step'] != step or row['stream_id'] != 'TWO_DIMENSIONAL'
        or row['capacity_tokens'] != args.cap or row['role'] != role
        for step, row in enumerate(rows, 1)
    ):
        parser.error('Indexed replay steps, stream, role or capacity differ from launch')
    frequency = [int(c == 'G') for c in args.pattern]
    settings = dict(
        native_n=args.native_n, accumulation_units=1, mixed_cp=True,
        input_transport='native', allow_compute_proxy=False, record_training_steps=True,
        model_config=dict(num_layers=args.num_layers, hidden_size=4096, ffn_hidden_size=16384,
                          num_attention_heads=32, num_query_groups=8, linear_attention_freq=frequency),
        expected_attention_classes=['GatedDeltaNet' if c == 'G' else 'SelfAttention' for c in args.pattern],
        two_dimensional_input=str(args.two_dimensional_input.resolve()), packet_updates=args.iterations,
        packet_role=role, packet_architecture='gdngqa_gpt', packet_warmup=max(1, args.warmup),
        packet_execution_cap=args.cap, replay=rows,
    )
    if joint:
        settings['two_dimensional'] = joint
    output = args.output.resolve()
    if args.node_rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'admission_init.json').write_text('{"ready": true}\n')
    else:
        for _ in range(600):
            if (output / 'admission_init.json').exists():
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Node 0 did not prepare the shared output directory')
    config_dest = output / 'config.json'
    if args.node_rank == 0:
        temporary = output / 'config_node0.tmp'
        temporary.write_text(json.dumps(settings, indent=2) + '\n')
        temporary.replace(config_dest)
    else:
        for _ in range(600):
            if config_dest.exists():
                if json.loads(config_dest.read_text()) != settings:
                    raise RuntimeError('Peer packet configuration differs')
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Peer configuration timed out')
    numerical = contract('cp_bf16_v2', 'gdngqa_gpt')
    if args.node_rank == 0:
        (output / 'numerical_contract.json').write_text(json.dumps(numerical, indent=2) + '\n')
    python = os.environ.get('PYTHON', sys.executable)
    environment = dict(
        PYTHON=python, TORCHRUN=os.environ.get('TORCHRUN', str(Path(python).with_name('torchrun'))),
        NUM_LAYERS=str(args.num_layers), HIDDEN_SIZE='4096', FFN_HIDDEN_SIZE='16384',
        LINEAR_ATTENTION_FREQ=json.dumps(frequency, separators=(',', ':')),
        NUM_ATTENTION_HEADS='32', NUM_QUERY_GROUPS='8', LINEAR_NUM_KEY_HEADS='16',
        LINEAR_NUM_VALUE_HEADS='16', LINEAR_KEY_HEAD_DIM='128', LINEAR_VALUE_HEAD_DIM='128',
        LINEAR_CONV_KERNEL_DIM='4', SEED='1234', ENABLE_OVERLAP='1', SAVE_CHECKPOINT='0',
        RECOMPUTE='none', RUN_ITERS=str(args.iterations), LR_DECAY_ITERS='50',
        WARMUP_STEPS=str(args.warmup), EVAL_ITERS='0', NNODES=str(args.nnodes),
        NODE_RANK=str(args.node_rank), MASTER_ADDR=args.master_addr, MASTER_PORT=str(args.port),
        GPUS_PER_NODE='8', MODE='DUPC', RUN_ID=output.name, CASE_NAME='sequence_joint_cp',
        MAX_TOKENS_PER_RANK=str(args.cap), LENGTHS_CSV='32', GLOBAL_BATCH_SIZE='1',
        OMP_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='2',
        NCCL_NVLS_ENABLE=os.environ.get('NCCL_NVLS_ENABLE', '0'),
        TORCH_SYMM_MEM_DISABLE_MULTICAST=os.environ.get('TORCH_SYMM_MEM_DISABLE_MULTICAST', '1'),
        NVTE_ALLOW_NONDETERMINISTIC_ALGO='0', PYTHONPATH=str(HERE / 'hooks'),
        SEQUENCE_CP_CONFIG=str(config_dest), SEQUENCE_CP_NO_DIGEST='1',
        SEQUENCE_CP_PACKET_MANIFEST=str(args.two_dimensional_input.resolve()),
        SEQUENCE_CP_OUTPUT=str(output / 'records'), SEQUENCE_CP_NUMERICS='off', SEQUENCE_CP_LOGICAL='off',
        SEQUENCE_CP_NUMERICAL_CONTRACT=str(output / 'numerical_contract.json'),
        SEQUENCE_CP_COMM_AUDIT='0', SEQUENCE_CP_DIAGNOSTIC='0', SEQUENCE_CP_CHECK_INIT='0',
        PET_LOG_DIR=str(output / f'rank_logs_node{args.node_rank}'), PET_TEE='3',
        NSYS_PROFILE='0', TRACE='0', OUTPUT_ROOT=str(output / 'training'),
        FAKE_DNS_SO=os.environ.get('FAKE_DNS_SO', ''),
    )
    command = ['timeout', *(['--foreground'] if os.environ.get('PRISM_LAUNCHER_GROUP') == '1' else []),
               '--signal=TERM', '--kill-after=30s', f'{args.timeout}s',
               'bash', str(ROOT / 'scripts/run_manual_gdn_cp_case.sh')]
    if args.node_rank == 0:
        (output / 'protocol.json').write_text(json.dumps(dict(
            arguments=vars(args), environment=environment, command=command,
            source_evidence_reference=args.reuse_source_evidence,
            scope='frozen complete training updates; no GPU preflight, snapshots or profiling',
        ), indent=2, default=str) + '\n')
    with (output / f'launcher_node{args.node_rank}.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=os.environ | environment,
                                stdout=log, stderr=subprocess.STDOUT)
    (output / f'result_node{args.node_rank}.json').write_text(json.dumps(
        dict(returncode=result.returncode, status='ok' if result.returncode == 0 else 'failed')) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
