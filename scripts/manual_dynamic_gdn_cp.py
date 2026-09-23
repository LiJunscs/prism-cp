#!/usr/bin/env python3
"""Manual, sequential Dynamic GDN CP experiments; standard library only.

Run the coordinator on node0. Both nodes must share the checkout and output path.
Dry-run performs CPU geometry preview only. It never connects to another node.
"""
import argparse
import csv
import errno
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
K = 1024
POST_RUN_WAIT_SECONDS = 60
# lengths are one cycle; gbs counts independent samples, not packed microbatches.
CASES = {
    'D8-P1': (8, [17, 7, 5, 3], 16, 18, ['U', 'Smax']),
    'D8-P2': (8, [65, 31, 16, 8, 8], 5, 17, ['U', 'Smax', 'Scap']),
    'D8-P3': (8, [33, 15, 8, 8], 8, 17, ['U', 'Smax', 'Scap']),
    'D8-P4': (8, [65] + [7] * 9, 10, 17, ['U', 'Smax', 'Scap']),
    'D8-C1': (8, [16] * 8, 8, None, ['U', 'Smax']),
    'D8-C2': (8, [128], 1, None, ['U', 'Smax']),
    'D16-P1': (16, [17, 7, 5, 3], 32, 18, ['U', 'Smax']),
    'D16-P2': (16, [129, 63, 32, 16, 8, 8], 6, 17, ['U', 'Smax', 'Scap']),
    'D16-P3': (16, [65, 31, 16, 8, 8], 10, 17, ['U', 'Smax', 'Scap']),
    'D16-P4': (16, [129] + [7] * 17 + [4, 4], 20, 17, ['U', 'Smax', 'Scap']),
    'D16-C1': (16, [16] * 16, 16, None, ['U', 'Smax']),
    'D16-C2': (16, [256], 1, None, ['U', 'Smax']),
}
CORE = 'D8-P1,D8-P2,D8-C1,D8-C2,D16-P2,D16-P3'
MODELS = {
    # layers, hidden, FFN, GQA heads, query groups, head dim, GDN key/value heads
    'M0': (5, 512, 2048, 8, 2, 64, 8, 8),
    'M1': (16, 2048, 8192, 16, 4, 128, 16, 16),
    'M2-R24': (24, 4096, 16384, 32, 8, 128, 16, 32),
}
SOURCE_FILES = [
    'pretrain_gpt.py', 'megatron/core/packed_sequence_cp.py',
    'megatron/core/attention_execution_plan.py', 'megatron/core/packed_seq_params.py',
    'megatron/core/parallel_state.py', 'megatron/core/process_groups_config.py',
    'megatron/core/datasets/data_schedule_utils.py', 'megatron/core/datasets/data_schedule.py',
    'megatron/core/transformer/transformer_block.py', 'megatron/core/transformer/transformer_layer.py',
    'megatron/core/transformer/transformer_config.py', 'megatron/core/ssm/gated_delta_net/gdn.py',
    'megatron/core/distributed/param_and_grad_buffer.py',
    'megatron/core/distributed/distributed_data_parallel.py',
    'megatron/training/arguments.py', 'megatron/training/training.py',
    'megatron/training/datasets/varlen_dataset.py',
    'megatron/training/datasets/data_samplers.py',
]
ITER = re.compile(r'iteration\s+(\d+)/\s*(\d+)\s*\|.*?consumed samples:\s*(\d+).*?'
                  r'elapsed time per iteration \(ms\):\s*([\d.]+).*?lm loss:\s*([^\s|]+)')
MEM = re.compile(r'\[Rank\s+(\d+)\]\s*\(after (\d+) iterations?\) memory \(MB\)\s*'
                 r'\|\s*allocated:\s*([\d.]+)\s*\|\s*max allocated:\s*([\d.]+)\s*'
                 r'\|\s*reserved:\s*([\d.]+)\s*\|\s*max reserved:\s*([\d.]+)')


def dump(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def capture(cmd, cwd=None):
    return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT,
                                   timeout=30).strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(root):
    return {
        'head': capture(['git', 'rev-parse', 'HEAD'], root),
        'tracked_diff_sha256': hashlib.sha256(subprocess.check_output(
            ['git', 'diff', 'HEAD', '--binary'], cwd=root, timeout=30)).hexdigest(),
        'files': {f: sha(Path(root) / f) for f in SOURCE_FILES},
        'runner_sha256': sha(HERE),
        'preview_sha256': sha(HERE.with_name('preview_dynamic_gdn_cp.py')),
    }


def recipes(args):
    result = {}
    for name, (world, lengths, gbs, cap, policies) in CASES.items():
        result[name] = dict(case=name, world=world, lengths=[n*K for n in lengths],
                            gbs=gbs, parent_cap=16*K, child_cap=cap*K if cap else None,
                            policies=policies)
    if args.matrix:
        for case in read(args.matrix):
            result[case['case']] = case
    chosen = args.cases.split(',')
    if len(chosen) != len(set(chosen)):
        raise ValueError('Duplicate case IDs')
    for name in chosen:
        c = result[name]
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Invalid case ID')
        if c['world'] not in (8, 16) or c['gbs'] % len(c['lengths']):
            raise ValueError(f'Invalid world/GBS: {name}')
        if any(type(n) is not int or n <= 0 for n in c['lengths']):
            raise ValueError(f'Invalid lengths: {name}')
        if c['parent_cap'] <= 0 or max(c['lengths']) > c['world'] * c['parent_cap']:
            raise ValueError(f'Invalid parent capacity: {name}')
        yield c


def variants(case, args):
    if args.phase == 'noise':
        return [('U', False)]
    if args.phase == 'defer':
        if case['child_cap'] is None:
            raise ValueError('defer phase needs a case with Scap')
        return [('U', False), ('U', True), ('Scap', False), ('Scap', True)]
    policies = args.policies.split(',') if args.policies else case['policies']
    if len(policies) != len(set(policies)):
        raise ValueError('Duplicate policies')
    if any(p not in ('U', 'Smax', 'Scap') for p in policies):
        raise ValueError('Only U, Smax, Scap exist; dynamic U-block is not implemented')
    if 'Scap' in policies and not case['child_cap']:
        raise ValueError('Scap requires child_cap')
    return [(p, False) for p in policies]


def command(req, node):
    c = req['case']; layers, hidden, ffn, heads, groups, dim, kh, vh = MODELS[req['model']]
    pattern = [0, 1, 1, 1, 0] if req['model'] == 'M0' else [1, 1, 1, 0] * (layers//4)
    padded = [(n+2*c['world']-1)//(2*c['world'])*(2*c['world']) for n in c['lengths']]
    fixture = dict(mode='packed_cp_case', lengths=c['lengths'],
                   prompt_lengths=[n//2 for n in c['lengths']])
    cmd = [req['python'], '-m', 'torch.distributed.run', '--nnodes', str(c['world']//8),
           '--nproc_per_node', '8', '--node_rank', str(node), '--master_addr', req['master_addr'],
           '--master_port', str(req['master_port']), '--max_restarts', '0', 'pretrain_gpt.py']
    flags = {
        'num-layers': layers, 'hidden-size': hidden, 'ffn-hidden-size': ffn,
        'num-attention-heads': heads, 'num-query-groups': groups, 'kv-channels': dim,
        'experimental-attention-variant': 'gdn', 'linear-attention-freq': json.dumps(pattern),
        'linear-key-head-dim': dim, 'linear-value-head-dim': dim,
        'linear-num-key-heads': kh, 'linear-num-value-heads': vh, 'linear-conv-kernel-dim': 4,
        'linear-cp-mode': 'chunkwise', 'cp-partition-mode': 'zigzag', 'cp-comm-type': 'p2p',
        'context-parallel-size': c['world'], 'tensor-model-parallel-size': 1,
        'pipeline-model-parallel-size': 1, 'min-dynamic-context-parallel-size': 1,
        'sequence-packing-scheduler': 'default_dynamic_cp', 'max-seqlen-per-dp-cp-rank': c['parent_cap'],
        'micro-batch-size': 1, 'global-batch-size': c['gbs'], 'seq-length': sum(padded),
        'max-position-embeddings': sum(padded), 'varlen-mock-dataset-config-json': json.dumps(fixture),
        'tokenizer-type': 'NullTokenizer', 'vocab-size': 32768, 'null-tokenizer-pad-id': 0,
        'split': '100,0,0', 'dataloader-type': 'single', 'num-workers': 0,
        'normalization': 'RMSNorm', 'position-embedding-type': 'rope',
        'transformer-impl': 'transformer_engine', 'attention-backend': 'flash',
        'attention-dropout': 0, 'hidden-dropout': 0, 'optimizer': 'adam',
        'lr': '1e-4', 'min-lr': '1e-5', 'lr-decay-style': 'cosine',
        'train-iters': req['iters'], 'lr-decay-iters': req['iters'], 'lr-warmup-iters': 0,
        'weight-decay': 0.1, 'clip-grad': 1.0, 'seed': req['seed'], 'log-interval': 1,
        # Only initial automatic reports and the final step; outside timed 6-20.
        'log-memory-interval': req['iters'], 'eval-iters': req.get('eval_iters', 1),
        'eval-interval': req.get('eval_interval', max(10000000, req['iters'] + 1)),
        'distributed-backend': 'nccl', 'distributed-timeout-minutes': 10,
        'ddp-bucket-size': req['bucket_size'],
    }
    for key, value in flags.items():
        cmd += ['--'+key, str(value)]
    for flag in ['dynamic-context-parallel', 'use-varlen-dataset', 'mock-data',
                 'no-create-attention-mask-in-dataloader', 'group-query-attention', 'swiglu',
                 'disable-bias-linear', 'untie-embeddings-and-output-weights', 'bf16',
                 'use-distributed-optimizer', 'calculate-per-token-loss',
                 'no-gradient-accumulation-fusion', 'log-throughput']:
        cmd.append('--'+flag)
    if req['policy'] != 'U':
        cmd.append('--gdn-dynamic-context-parallel')
        if req['policy'] == 'Scap':
            cmd += ['--packed-sequence-cp-max-tokens', str(c['child_cap'])]
    if req['overlap'] == 'on':
        cmd += ['--overlap-grad-reduce', '--overlap-param-gather']
    if req['defer']:
        cmd.append('--defer-grad-reduce-dispatch')
    if req['recompute'] == 'full':
        cmd += ['--recompute-granularity', 'full', '--recompute-method', 'uniform',
                '--recompute-num-layers', '1']
    return cmd


def make_plan(args):
    from preview_dynamic_gdn_cp import preview, load_planners
    root = Path(args.megatron_root).resolve()
    planners = load_planners(root)
    model = args.model or ('M0' if args.phase == 'smoke' else 'M1')
    repeats = args.repeats if args.repeats is not None else {'smoke': 1, 'screen': 2, 'confirm': 5, 'noise': 3, 'defer': 2}[args.phase]
    iters = args.iters if args.iters is not None else (5 if args.phase == 'smoke' else 25)
    if repeats < 1 or iters < 1 or (args.phase != 'smoke' and not 1 < args.window_start <= args.window_end < iters):
        raise ValueError('Need 1 < window-start <= window-end < iters, and positive repeats')
    if args.phase == 'defer' and args.overlap != 'on':
        raise ValueError('defer requires overlap=on; fixed bucketization')
    if args.bucket_size <= 0 or not 1024 <= args.master_port <= 65535:
        raise ValueError('Invalid bucket size or port')
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + args.phase
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('run-id must contain only letters, digits, underscore or hyphen')
    base = root / 'outputs/manual_dynamic_gdn_cp' / run_id
    plan = []
    for case in recipes(args):
        expanded = case['lengths'] * (case['gbs']//len(case['lengths']))
        previews = {p: preview(root, expanded, case['world'], case['parent_cap'],
                              case['child_cap'] if p == 'Scap' else None, planners=planners)
                    for p in ('Smax', 'Scap')}
        for pair in range(1, repeats+1):
            order = variants(case, args)
            if pair % 2 == 0:
                order = list(reversed(order))
            for policy, deferred in order:
                tag = policy + ('-defer' if deferred else '-eager') if args.phase == 'defer' else policy
                req = dict(case=case, phase=args.phase, model=model, pair=pair, policy=policy,
                           variant=tag, defer=deferred, overlap=args.overlap, iters=iters,
                           seed=args.seed if args.seed is not None else (20260920 if args.phase == 'confirm' else 1234),
                           window=[args.window_start, args.window_end], recompute=args.recompute,
                           bucket_size=args.bucket_size, megatron_root=str(root), python=args.python,
                           eval_iters=args.eval_iters, eval_interval=max(10000000, iters + 1),
                           master_addr=args.master_addr, master_port=args.master_port,
                           preload=args.preload, output=str(base / case['case'] / f'pair{pair}' / tag),
                           preview=previews, ready_timeout=args.ready_timeout,
                           first_step_timeout=args.first_step_timeout, step_timeout=args.step_timeout,
                           teardown_timeout=args.teardown_timeout, run_timeout=args.run_timeout,
                           expected_input_tokens=sum(expanded),
                           expected_supervised_tokens=sum(n-n//2 for n in expanded))
                plan.append(req)
    return base, plan


def gpu_state():
    fields = 'index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm'
    rows = list(csv.reader(io.StringIO(capture(['nvidia-smi', '--query-gpu='+fields,
                                               '--format=csv,noheader,nounits']))))
    if len(rows) != 8 or {int(r[0]) for r in rows} != set(range(8)):
        raise RuntimeError('This runner reserves a whole 8-GPU node; expected physical GPUs 0-7')
    return [dict(zip(fields.split(','), [v.strip() for v in r])) for r in rows]


def compute_pids():
    text = capture(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    return {int(x.strip()) for x in text.splitlines() if x.strip()}


def proc_identity(pid, proc_root=Path('/proc')):
    try:
        text = (proc_root/str(pid)/'stat').read_text()
        # comm may contain spaces and parentheses; fields after its final ')' start at field 3.
        fields = text[text.rfind(')')+2:].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]),
                    sid=int(fields[3]), start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def proc_snapshot():
    rows = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                row = proc_identity(int(entry.name))
                if row:
                    rows[row['pid']] = row
            except PermissionError:
                continue
    return rows


def check_process_tracking():
    if proc_identity(os.getpid()) is None:
        raise RuntimeError('Container /proc does not expose this worker process')
    os.kill(os.getpid(), 0)


class OwnedProcesses:
    """Track only this launch's session/descendants, identified by PID and birth time."""
    def __init__(self, pid, out, node):
        self.root_pid = pid
        self.out = Path(out)
        self.node = node
        self.known = {}
        root = proc_identity(pid)
        if root is None:
            raise RuntimeError('torchrun exited before its process identity could be recorded')
        if root['sid'] != pid:
            raise RuntimeError('torchrun did not enter its own session')
        self.known[(pid, root['start_ticks'])] = root
        self.refresh()

    def refresh(self):
        rows = proc_snapshot()
        anchors = {pid for pid, born in self.known if pid in rows and rows[pid]['start_ticks'] == born}
        session_owned = any(rows[pid]['sid'] == self.root_pid for pid in anchors)
        while True:
            added = {pid for pid,row in rows.items() if pid not in anchors and
                     (row['ppid'] in anchors or (session_owned and row['sid'] == self.root_pid))}
            if not added:
                break
            for pid in added:
                self.known[(pid, rows[pid]['start_ticks'])] = rows[pid]
            anchors.update(added)
        live = []
        for pid in anchors:
            row = rows[pid]
            self.known[(pid, row['start_ticks'])] = row
            if row['state'] not in ('Z', 'X'):
                live.append(row)
        dump(self.out/f'processes_node{self.node}.json', dict(
            root_pid=self.root_pid, node=self.node, hostname=socket.gethostname(),
            updated_at=datetime.now().isoformat(), processes=list(self.known.values()),
            live_pids=[r['pid'] for r in live]))
        return live


def signal_identity(row, sig):
    def same_process():
        current = proc_identity(row['pid'])
        return (current is not None and current['start_ticks'] == row['start_ticks']
                and current['state'] not in ('Z', 'X'))

    # Prefer pidfd when available; older Python/container environments use kill().
    if callable(getattr(os, 'pidfd_open', None)) and callable(getattr(signal, 'pidfd_send_signal', None)):
        fd = None
        try:
            fd = os.pidfd_open(row['pid'], 0)
            if not same_process():
                return False
            signal.pidfd_send_signal(fd, sig)
            return True
        except ProcessLookupError:
            return False
        except OSError as exc:
            if exc.errno not in (errno.ENOSYS, errno.EINVAL, errno.EPERM, errno.EACCES):
                raise
        finally:
            if fd is not None:
                os.close(fd)
    # Best-effort identity check immediately before signalling without pidfd.
    if not same_process():
        return False
    try:
        os.kill(row['pid'], sig)
    except ProcessLookupError:
        return False
    return True


def cleanup_owned(proc, tracker, out, node, grace_seconds):
    report = dict(node=node, started_at=datetime.now().isoformat(), grace_seconds=grace_seconds,
                  signals=[], status='FAILED', remaining_pids=[])
    try:
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            proc.poll()
            tracker.refresh()
            if (Path(out)/'cancel.json').exists():
                report['grace_interrupted_by_cancel'] = True
                break
            time.sleep(min(1, max(0, deadline-time.monotonic())))
        for sig in (signal.SIGTERM, signal.SIGKILL):
            deadline = time.monotonic() + 5
            sent = set()
            while True:
                proc.poll()
                live = tracker.refresh()
                if not live:
                    break
                for row in live:
                    identity = (row['pid'], row['start_ticks'])
                    if identity not in sent:
                        if signal_identity(row, sig):
                            report['signals'].append(dict(pid=row['pid'], start_ticks=row['start_ticks'], signal=int(sig)))
                        sent.add(identity)
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
        proc.poll()
        report['remaining_pids'] = [r['pid'] for r in tracker.refresh()]
        if report['remaining_pids']:
            raise RuntimeError(f"Owned processes still alive after cleanup: {report['remaining_pids']}")
        report['status'] = 'CLEAN'
        return report
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['finished_at'] = datetime.now().isoformat()
        dump(Path(out)/f'cleanup_node{node}.json', report)


def cancel_once(out, payload):
    """Keep the first cancellation cause instead of replacing it with peer fallout."""
    try:
        with (Path(out)/'cancel.json').open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2)
            stream.write('\n')
    except FileExistsError:
        pass


def node_diagnostics(out, world):
    rows = []
    for node in range(world//8):
        row = dict(node=node, ready=(out/f'ready_node{node}.json').exists())
        exit_path = out/f'exit_node{node}.json'
        if exit_path.exists():
            try:
                row['exit'] = read(exit_path)
            except (ValueError, OSError) as exc:
                row['exit_read_error'] = str(exc)
        log = out/f'worker_node{node}.log'
        if log.exists():
            row['worker_log_tail'] = '\n'.join(log.read_text(errors='replace').splitlines()[-15:])[-3000:]
        rows.append(row)
    return dict(start_exists=(out/'start.json').exists(), nodes=rows)


def readiness_error(out, world, reason):
    diagnostic = node_diagnostics(out, world)
    diagnostic['reason'] = reason
    dump(out/'startup_failure.json', diagnostic)
    details = []
    for row in diagnostic['nodes']:
        status = row.get('exit', {})
        detail = status.get('error') or row.get('worker_log_tail') or 'no exit/error record yet'
        details.append(f"node{row['node']}: ready={row['ready']}, phase={status.get('phase', 'unknown')}: {detail}")
    return RuntimeError(f'{reason}: {out}\n'+'\n'.join(details))


def check_master_port(port, wait_seconds=10):
    """Probe like a reusable TCP listener; never share an active listening port."""
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(('0.0.0.0', port))
                probe.listen(1)
            return
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            if time.monotonic() >= deadline:
                raise RuntimeError(f'MASTER_PORT_BUSY: {port} is not available after {wait_seconds}s; '
                                   f'check processes_node0.json and cleanup_node0.json from the preceding run. '
                                   f'Last error: {exc}') from exc
            time.sleep(0.25)


def worker(request, node):
    import fcntl  # Linux server only; never imported by dry-run or local tests.
    req = read(request); out = Path(req['output']); root = Path(req['megatron_root']).resolve()
    if not out.resolve().is_relative_to(root / 'outputs/manual_dynamic_gdn_cp'):
        raise ValueError('Output must be inside Megatron outputs/manual_dynamic_gdn_cp')
    proc = None; tracker = None; lock = None; cleanup_attempted = False
    result = dict(node=node, hostname=socket.gethostname(), status='FAILED')
    try:
        result['phase'] = 'node_lock'
        lockdir = root / 'outputs/manual_dynamic_gdn_cp/.locks'
        lockdir.mkdir(parents=True, exist_ok=True)
        lock = (lockdir / (socket.gethostname()+'.lock')).open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result['phase'] = 'gpu_idle_check'
        gpus = gpu_state()
        if compute_pids() or any(float(g['memory.used']) >= 500 or float(g['utilization.gpu']) > 5 for g in gpus):
            raise RuntimeError('GPU busy; refusing to launch. No foreign process will be killed.')
        result['phase'] = 'source_check'
        source = fingerprint(root)
        if source != req['source']:
            raise RuntimeError('Source changed or differs between nodes')
        result['phase'] = 'preload_check'
        if req['preload'] and not Path(req['preload']).is_file():
            raise RuntimeError('Requested preload library is missing')
        result['phase'] = 'process_tracking_check'
        check_process_tracking()
        if node == 0:
            result['phase'] = 'master_port_check'
            check_master_port(req['master_port'], wait_seconds=min(10, req['ready_timeout']/3))
        result['phase'] = 'dependency_and_topology_check'
        versions = {}
        for package in ['torch', 'transformer-engine', 'flash-linear-attention', 'triton']:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        dump(out/f'ready_node{node}.json', dict(node=node, hostname=socket.gethostname(),
             gpus=gpus, source=source, versions=versions, topology=capture(['nvidia-smi', 'topo', '-m'])))
        result['phase'] = 'waiting_for_start'
        deadline = time.monotonic() + req['ready_timeout']
        while not (out/'start.json').exists():
            if (out/'cancel.json').exists():
                raise RuntimeError('Cancelled before start; original cause is in cancel.json')
            if time.monotonic() > deadline:
                raise RuntimeError(f"Timed out after {req['ready_timeout']}s waiting for coordinator start")
            time.sleep(0.5)
        result['phase'] = 'launch_gpu_idle_check'
        if compute_pids():
            raise RuntimeError('GPU claimed by another process during readiness handshake')
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', CUDA_DEVICE_MAX_CONNECTIONS='1',
                   PYTHONDONTWRITEBYTECODE='1', MCORE_PACKED_CP_NVTX='0',
                   MCORE_DYNAMIC_GDN_CP_LOG='1' if req['phase']=='smoke' else '0',
                   PYTHONPATH=str(root)+os.pathsep+env.get('PYTHONPATH', ''))
        if req['preload']:
            env['LD_PRELOAD'] = req['preload'] + (':'+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
        dump(out/f'environment_node{node}.json', {k: v for k, v in env.items()
             if k.startswith(('NCCL_', 'TORCH_', 'CUDA_', 'MCORE_')) or k in ('LD_PRELOAD', 'PYTHONPATH')})
        cmd = command(req, node)
        (out/f'command_node{node}.txt').write_text(shlex.join(cmd)+'\n', encoding='utf-8')
        with (out/f'train_node{node}.log').open('w', encoding='utf-8') as log:
            result['phase'] = 'training'
            proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            tracker = OwnedProcesses(proc.pid, out, node)
            result['torchrun_pid'] = proc.pid
            started = last_progress = time.monotonic(); last_step = 0; complete_at = None
            result['started_at'] = datetime.now().isoformat()
            while proc.poll() is None:
                tracker.refresh()
                if (out/'cancel.json').exists():
                    raise RuntimeError('Cancelled by coordinator or failed peer')
                now = time.monotonic()
                metric = out/f"train_node{req['case']['world']//8-1}.log"
                text = metric.read_text(errors='replace') if metric.exists() else ''
                step = max((int(m[1]) for m in ITER.finditer(text)), default=0)
                if step > last_step:
                    last_step = step; last_progress = now
                if step == req['iters'] and complete_at is None:
                    complete_at = now
                if complete_at and now-complete_at > min(req['teardown_timeout'], POST_RUN_WAIT_SECONDS):
                    raise RuntimeError('TEARDOWN_TIMEOUT: steps completed but torchrun did not exit within the cleanup deadline')
                limit = req['step_timeout'] if last_step else req['first_step_timeout']
                if not complete_at and now-last_progress > limit:
                    raise RuntimeError(f'NO_STEP_PROGRESS: last step={last_step}')
                if now-started > req['run_timeout']:
                    raise RuntimeError('RUN_TIMEOUT')
                foreign = []
                for pid in compute_pids():
                    try:
                        if os.getpgid(pid) != proc.pid:
                            foreign.append(pid)
                    except ProcessLookupError:
                        pass
                telemetry = dict(time=datetime.now().isoformat(), step=step,
                                 gpus=gpu_state(), foreign_gpu_pids=foreign)
                with (out/f'gpu_node{node}.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(telemetry)+'\n')
                if foreign:
                    raise RuntimeError(f'GPU_CONTENTION: external processes {foreign}')
                time.sleep(5)
            result['returncode'] = proc.returncode
            if proc.returncode:
                raise RuntimeError(f'Training process exited {proc.returncode}')
        result['phase'] = 'post_run_source_check'
        if fingerprint(root) != req['source']:
            raise RuntimeError('Source changed during training')
        result['phase'] = 'post_run_cleanup'
        cleanup_attempted = True
        result['cleanup'] = cleanup_owned(proc, tracker, out, node, POST_RUN_WAIT_SECONDS)
        result['status'] = 'EXIT_OK'
        result['phase'] = 'complete'
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        cancel_once(out, result)
        try:
            if tracker is not None and not cleanup_attempted:
                result['cleanup'] = cleanup_owned(proc, tracker, out, node, 0)
            elif tracker is None and proc is not None and proc.poll() is None:
                # Popen still owns this unreaped direct child even if tracking setup failed.
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except BaseException as cleanup_error:
            result['cleanup_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'
    finally:
        result['finished_at'] = datetime.now().isoformat()
        dump(out/f'exit_node{node}.json', result)
        if lock is not None:
            lock.close()
    return 0 if result['status']=='EXIT_OK' else 1


def execute(req, args):
    out = Path(req['output']); out.mkdir(parents=True, exist_ok=False)
    dump(out/'request.json', req)
    workers = []; streams = []
    try:
        for node in reversed(range(req['case']['world']//8)):
            cmd = [req['python'], '-B', str(HERE), '_worker', str(out/'request.json'), str(node)]
            if node:
                cmd = ['ssh', '-p', str(args.ssh_port), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                       args.node1, shlex.join(cmd)]
            stream = (out/f'worker_node{node}.log').open('w', encoding='utf-8'); streams.append(stream)
            workers.append(subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT))
        deadline = time.monotonic()+req['ready_timeout']
        while not all((out/f'ready_node{n}.json').exists() for n in range(len(workers))):
            if any(p.poll() is not None for p in workers) or time.monotonic()>deadline:
                raise readiness_error(out, req['case']['world'], 'Node readiness failed')
            time.sleep(0.5)
        ready = [read(out/f'ready_node{n}.json') for n in range(len(workers))]
        if len({x['hostname'] for x in ready}) != len(workers):
            raise RuntimeError('Two node ranks resolved to the same physical host')
        uuids = [g['uuid'] for x in ready for g in x['gpus']]
        if len(uuids) != 8*len(workers) or len(set(uuids)) != len(uuids):
            raise RuntimeError('GPU UUIDs overlap between nodes or GPU inventory is incomplete')
        if len({json.dumps(x['versions'], sort_keys=True) for x in ready}) != 1:
            raise RuntimeError('Dependency versions differ between nodes')
        dump(out/'start.json', {'start': datetime.now().isoformat()})
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in (None, 0) for p in workers):
                raise readiness_error(out, req['case']['world'], 'Worker failed')
            time.sleep(1)
        if any(p.returncode for p in workers):
            raise readiness_error(out, req['case']['world'], 'Worker failed')
        summary = parse_run(out)
        dump(out/'result.json', summary)
        if summary['execution_status'] != 'COMPLETE_LOGS':
            details = '\n'.join('  - '+error for error in summary['errors'])
            raise RuntimeError(f"Log validation failed (process status: {summary['process_exit_status']}): "
                               f'{out}\n{details}')
    finally:
        if any(p.poll() is None for p in workers):
            cancel_once(out, {'reason': 'coordinator stopping'})
            for p in workers:
                try:
                    p.wait(timeout=55)
                except subprocess.TimeoutExpired:
                    p.terminate()  # Remote worker also sees cancel.json on shared storage.
        for stream in streams:
            stream.close()


def evaluation_expectation(req, out, errors):
    """Read legacy eval settings from saved commands, never from observed args."""
    recorded = {}
    for node in range(req['case']['world']//8):
        path = out / f'command_node{node}.txt'
        if not path.exists():
            continue
        try:
            tokens = shlex.split(path.read_text(encoding='utf-8'))
            values = [int(tokens[i+1]) for i, token in enumerate(tokens) if token == '--eval-iters']
            values += [int(token.split('=', 1)[1]) for token in tokens if token.startswith('--eval-iters=')]
            if len(values) != 1 or values[0] < 0:
                raise ValueError('need exactly one nonnegative --eval-iters')
            recorded[str(node)] = values[0]
        except (ValueError, IndexError) as exc:
            errors.append(f'Cannot verify eval_iters from {path.name}: {exc}')
    if 'eval_iters' in req:
        expected, source = req['eval_iters'], 'request.json'
    elif recorded:
        expected, source = next(iter(recorded.values())), 'saved_command_legacy_request'
    else:
        # Initial runner requested eval_iters=0 but did not store it in request.json.
        expected, source = 0, 'legacy_default_no_saved_command'
    if any(value != expected for value in recorded.values()):
        errors.append(f'Saved command eval_iters differs from expected {expected}: {recorded}')
    return dict(eval_iters=expected, source=source, recorded_by_node=recorded)


def parse_run(out):
    out = Path(out); req = read(out/'request.json'); world = req['case']['world']
    logs = {p: p.read_text(errors='replace') for p in out.glob('train_node*.log')}
    metric, matches = max(((p, list(ITER.finditer(t))) for p,t in logs.items()),
                          key=lambda v: len(v[1]), default=(None, []))
    steps = [dict(step=int(m[1]), consumed_samples=int(m[3]), ms=float(m[4]), loss=float(m[5]))
             for m in matches]
    errors = []
    evaluation = evaluation_expectation(req, out, errors)
    if [s['step'] for s in steps] != list(range(1, req['iters']+1)):
        errors.append('Missing, duplicated, or unexpected iterations')
    if any(not math.isfinite(s['loss']) or s['ms'] <= 0 or
           s['consumed_samples'] != s['step']*req['case']['gbs'] for s in steps):
        errors.append('Invalid loss, timing or consumed sample count')
    expected = dict(num_layers=str(MODELS[req['model']][0]), hidden_size=str(MODELS[req['model']][1]),
                    world_size=str(world), seed=str(req['seed']),
                    gdn_dynamic_context_parallel=str(req['policy'] != 'U'),
                    dynamic_context_parallel='True', overlap_grad_reduce=str(req['overlap']=='on'),
                    overlap_param_gather=str(req['overlap']=='on'),
                    defer_grad_reduce_dispatch=str(req['defer']),
                    recompute_granularity='full' if req['recompute']=='full' else 'None',
                    eval_iters=str(evaluation['eval_iters']))
    if 'eval_interval' in req:
        expected['eval_interval'] = str(req['eval_interval'])
    resolved = {}
    # Megatron prints args on global rank0, but iteration metrics on last rank.
    config_text = logs.get(out/'train_node0.log', '')
    for key, value in expected.items():
        match = re.search(r'^\s*'+key+r'\s+\.+\s+(.*)$', config_text, re.M)
        resolved[key] = match[1].strip() if match else None
        if resolved[key] != value:
            errors.append(f'Resolved {key}: expected {value}, got {resolved[key]}')
    process_exits = {}
    for n in range(world//8):
        exit_path = out/f'exit_node{n}.json'
        exit_data = read(exit_path) if exit_path.exists() else {}
        process_exits[str(n)] = exit_data.get('status', 'MISSING')
        if exit_data.get('status') != 'EXIT_OK' or exit_data.get('returncode', 0) != 0:
            process_exits[str(n)] = 'FAILED_OR_INCOMPLETE'
            errors.append(f'Node {n} did not exit successfully')
    memory = {}
    for text in logs.values():
        if re.search(r'number of (?:nan|skipped) iterations:\s*[1-9]|Traceback \(most recent call last\)', text):
            errors.append('NaN, skipped update or traceback')
        for m in MEM.finditer(text):
            row = memory.setdefault(m[1], {'allocated_MiB': 0, 'reserved_MiB': 0})
            row['allocated_MiB'] = max(row['allocated_MiB'], float(m[4]))
            row['reserved_MiB'] = max(row['reserved_MiB'], float(m[6]))
    if set(memory) != {str(i) for i in range(world)}:
        errors.append('Missing per-rank memory records')
    lo, hi = req['window']; window = [s['ms'] for s in steps if lo<=s['step']<=hi]
    perf = req['phase'] != 'smoke'
    if perf and len(window) != hi-lo+1:
        errors.append('Measurement window incomplete')
    mean = statistics.mean(window) if window and perf else None
    times = {int(m[2]): datetime.fromisoformat(m[1]) for m in re.finditer(
        r'\[([\d-]+ [\d:.]+)\] iteration\s+(\d+)/', logs.get(metric, ''))}
    wall = (times[hi]-times[lo-1]).total_seconds() if perf and hi in times and lo-1 in times else None
    not_started = not logs and not (out/'start.json').exists()
    startup = node_diagnostics(out, world) if not_started else None
    if not_started:
        errors = ['Training not started: node readiness/start handshake did not complete']
        for row in startup['nodes']:
            detail = row.get('exit', {}).get('error') or row.get('worker_log_tail')
            if detail:
                errors.append(f"Node {row['node']}: {detail}")
    return dict(case=req['case']['case'], pair=req['pair'], variant=req['variant'], phase=req['phase'],
                output=str(out), execution_status='NOT_STARTED' if not_started else ('INVALID' if errors else 'COMPLETE_LOGS'), errors=errors,
                process_exit_status='EXIT_OK' if all(s=='EXIT_OK' for s in process_exits.values()) else 'FAILED_OR_INCOMPLETE',
                log_validation_status='NOT_RUN' if not_started else ('INVALID' if errors else 'VALID'),
                startup_diagnostics=startup,
                evaluation_protocol=evaluation,
                correctness='NOT_RUN' if not_started else 'SMOKE_ONLY_NOT_FULL_PARITY', steps=steps, memory_by_rank=memory,
                resolved_config=resolved,
                mean_step_ms=mean, median_step_ms=statistics.median(window) if mean else None,
                log_timestamp_window_s=wall, input_tokens_s_fixture=req['expected_input_tokens']*1000/mean if mean else None,
                supervised_tokens_s_fixture=req['expected_supervised_tokens']*1000/mean if mean else None,
                token_count_status='FIXTURE_DERIVED_NOT_RUNTIME_TOKEN_AUDIT',
                memory_scope='Process lifetime high-water marks including warmup', formal_verdict='NOT_EVALUATED')


def checked_output(base, req):
    base = Path(base).resolve()
    path = Path(req['output']).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError(f'Run output is outside campaign: {path}')
    return path


def grouped_plan(plan):
    groups = {}
    for req in plan:
        key = (req['case']['case'], req['pair'])
        groups.setdefault(key, []).append(req)
    return groups


def source_compatible(saved, current):
    # Runner upgrades are allowed; training code and the geometry helper are frozen.
    for key in ('head', 'tracked_diff_sha256', 'preview_sha256'):
        if key not in saved or saved[key] != current.get(key):
            raise ValueError(f'Cannot resume: Megatron/preview source changed ({key}); use a new run-id')
    if not saved.get('files') or any(current.get('files', {}).get(k) != v for k,v in saved['files'].items()):
        raise ValueError('Cannot resume: Megatron source files changed; use a new run-id')


def prepare_resume(base, requested, current_source):
    """Read-only resume plan. Keep complete pairs and retry incomplete pairs in new directories."""
    base = Path(base).resolve()
    original = read(base/'plan.json')
    saved_source = read(base/'source_manifest.json')
    source_compatible(saved_source, current_source)
    active_path = base/'active_plan.json'
    active = read(active_path) if active_path.exists() else original
    identity = lambda req: (req['case']['case'], req['pair'], req['variant'])
    if ([identity(r) for r in original] != [identity(r) for r in requested]
            or [identity(r) for r in original] != [identity(r) for r in active]):
        raise ValueError('Cannot resume: case/pair/policy order differs; repeat the original matrix options')
    # Legacy requests omitted eval settings. Recover them from recorded commands,
    # so the eval_iters=1 workaround can be resumed without rewriting old evidence.
    legacy_eval = set()
    for req in active:
        out = checked_output(base, req)
        if (out/'request.json').exists():
            evidence = read(out/'request.json')
            errors = []
            info = evaluation_expectation(evidence, out, errors)
            if errors:
                raise ValueError('Cannot resume: '+ '; '.join(errors))
            if info['recorded_by_node'] or 'eval_iters' in evidence:
                legacy_eval.add(info['eval_iters'])
    if len(legacy_eval) > 1:
        raise ValueError('Cannot resume: existing runs use different eval settings')
    for old, wanted, selected in zip(original, requested, active):
        checked_output(base, old)
        out = checked_output(base, selected)
        normalized = dict(old)
        normalized.setdefault('eval_iters', next(iter(legacy_eval), 0))
        normalized.setdefault('eval_interval', 10000000)
        ignored = {'source', 'preview', 'output'}
        differences = [k for k in set(normalized) | set(wanted)
                       if k not in ignored and normalized.get(k) != wanted.get(k)]
        if differences:
            raise ValueError(f'Cannot resume {identity(old)}: settings differ: {", ".join(sorted(differences))}; '
                             'repeat the original options or use a new run-id')
    attempt = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + f'_{os.getpid()}'
    groups = []
    selected_plan = []
    desired_groups = grouped_plan(requested)
    for key, old_group in grouped_plan(active).items():
        existing = False
        complete = True
        reasons = []
        for req in old_group:
            out = checked_output(base, req)
            existing = existing or out.exists()
            try:
                result = parse_run(out)
                if result['execution_status'] != 'COMPLETE_LOGS':
                    complete = False
                    reasons.extend(f"{req['variant']}: {e}" for e in result['errors'])
                # Complete evidence must include the commands on every participating node.
                if any(not (out/f'command_node{n}.txt').exists() for n in range(req['case']['world']//8)):
                    complete = False
                    reasons.append(f"{req['variant']}: saved command missing")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                complete = False
                reasons.append(f"{req['variant']}: {type(exc).__name__}")
        action = 'SKIP_COMPLETE_PAIR' if complete else ('RETRY_WHOLE_PAIR' if existing else 'RUN_PENDING_PAIR')
        group = []
        for old_req, wanted in zip(old_group, desired_groups[key]):
            req = dict(wanted)
            req['output'] = old_req['output']
            if existing and not complete:
                req['output'] = str(base/'retries'/attempt/key[0]/f'pair{key[1]}'/req['variant'])
            checked_output(base, req)
            group.append(req)
        selected_plan.extend(group)
        groups.append(dict(case=key[0], pair=key[1], action=action, reasons=reasons,
                           outputs=[r['output'] for r in group]))
    return dict(attempt=attempt, plan=selected_plan, groups=groups,
                previous_source=saved_source, current_source=current_source)


def campaign_outputs(base):
    active = base/'active_plan.json'
    if active.exists():
        return [checked_output(base, req) for req in read(active)
                if (checked_output(base, req)/'request.json').exists()]
    return [p.parent for p in sorted(base.glob('*/*/*/request.json'))]


def summarize(base):
    base = Path(base)
    results = []
    for out in campaign_outputs(base):
        result = parse_run(out)
        dump(out/'result.json', result)
        results.append(result)
    lookup = {(r['case'],r['pair'],r['variant']):r for r in results}; comparisons=[]
    for r in results:
        if r['variant'] in ('U', 'U-eager', 'U-defer'):
            continue
        baseline = 'U-defer' if r['variant'].endswith('-defer') else ('U-eager' if r['phase']=='defer' else 'U')
        u = lookup.get((r['case'],r['pair'],baseline))
        if not u or any(x['execution_status']!='COMPLETE_LOGS' or not x['mean_step_ms'] for x in (r,u)):
            continue
        if u['evaluation_protocol']['eval_iters'] != r['evaluation_protocol']['eval_iters']:
            continue
        comparisons.append(dict(case=r['case'], pair=r['pair'], variant=r['variant'], baseline=baseline,
                           mean_step_speedup=u['mean_step_ms']/r['mean_step_ms'],
                           median_step_speedup=u['median_step_ms']/r['median_step_ms'],
                           first_loss_abs_difference=abs(u['steps'][0]['loss']-r['steps'][0]['loss']),
                           formal_verdict='NOT_EVALUATED'))
    interactions = []
    for case, pair in sorted({(r['case'],r['pair']) for r in results if r['phase']=='defer'}):
        arms = {v: lookup.get((case,pair,v)) for v in ('U-eager','U-defer','Scap-eager','Scap-defer')}
        if (all(r and r['execution_status']=='COMPLETE_LOGS' and r['mean_step_ms'] for r in arms.values())
                and len({r['evaluation_protocol']['eval_iters'] for r in arms.values()}) == 1):
            t = {v:r['mean_step_ms'] for v,r in arms.items()}
            interactions.append(dict(case=case,pair=pair,
                interaction_ms=(t['Scap-defer']-t['U-defer'])-(t['Scap-eager']-t['U-eager'])))
    noise = []
    for case in sorted({r['case'] for r in results if r['phase']=='noise'}):
        runs = [r for r in results if r['case']==case and r['execution_status']=='COMPLETE_LOGS' and r['mean_step_ms']]
        values = [r['mean_step_ms'] for r in runs]
        deviations = [abs(x/y-1) for i,x in enumerate(values) for y in values[i+1:]]
        noise.append(dict(case=case, n_runs=len(runs), mean_step_ms=values,
                          descriptive_pairwise_relative_difference=statistics.median(deviations) if deviations else None,
                          note='Descriptive pairwise differences among repeated U runs; not a confidence interval or acceptance threshold.'))
    dump(base/'results.json', dict(runs=results, comparisons=comparisons, defer_interactions=interactions, noise=noise,
         notes=['Primary timing is mean native step ms; median is secondary.',
                'Native log-interval=1 adds the existing timer barrier each step; not a new boundary-only benchmark.',
                'No full gradient/update/checkpoint parity or automatic performance PASS.',
                'Token throughput is derived from the fixed synthetic fixture.']))
    print(f'Wrote {base / "results.json"}: {len(results)} runs, {len(comparisons)} comparisons')
    for result in results:
        if result['errors']:
            print(f"  {result['execution_status']} {result['case']} pair{result['pair']} {result['variant']}: "
                  + '; '.join(result['errors']), file=sys.stderr)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='_worker':
        return worker(sys.argv[2], int(sys.argv[3]))
    if len(sys.argv)>1 and sys.argv[1]=='summarize':
        summarize(sys.argv[2]); return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--megatron-root', default=os.getenv('MEGATRON_ROOT', '/workspace/repos/Megatron-LM'))
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--phase', choices=['smoke','screen','confirm','noise','defer'], default='screen')
    p.add_argument('--cases', default=CORE)
    p.add_argument('--matrix', help='Optional JSON array of additional frozen recipes; lengths/caps in tokens')
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--policies', help='Optional U,Smax,Scap subset; ignored for noise/defer')
    p.add_argument('--overlap', choices=['on','off'], default='on')
    p.add_argument('--recompute', choices=['full','none'], default='full')
    p.add_argument('--bucket-size', type=int, default=64000000)
    p.add_argument('--eval-iters', type=int, choices=(0, 1), default=1,
                   help='Default 1: create nonempty mock eval sets; evaluate only after training. '
                        'Use 0 with the disabled-eval loader fix.')
    p.add_argument('--repeats', type=int)
    p.add_argument('--iters', type=int)
    p.add_argument('--window-start', type=int, default=6)
    p.add_argument('--window-end', type=int, default=20)
    p.add_argument('--seed', type=int)
    p.add_argument('--run-id')
    p.add_argument('--resume', action='store_true',
                   help='Reuse --run-id with the same matrix options; retry whole incomplete pairs, keep complete pairs')
    p.add_argument('--master-addr', default='127.0.0.1')
    p.add_argument('--master-port', type=int, default=29750)
    p.add_argument('--node1', help='SSH host/user@host for node1; coordinator runs on node0')
    p.add_argument('--ssh-port', type=int, default=10710)
    p.add_argument('--preload', default='', help='Optional existing .so, e.g. fake_dns.so; never forced')
    p.add_argument('--ready-timeout', type=int, default=180)
    p.add_argument('--first-step-timeout', type=int, default=1800)
    p.add_argument('--step-timeout', type=int, default=600)
    p.add_argument('--teardown-timeout', type=int, default=120)
    p.add_argument('--run-timeout', type=int, default=7200)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.resume and not args.run_id:
        p.error('--resume requires the existing --run-id')
    base, plan = make_plan(args)
    if args.dry_run:
        if args.resume:
            resumed = prepare_resume(base, plan, fingerprint(args.megatron_root))
            print(json.dumps(resumed, indent=2))
            return 0
        print(json.dumps(dict(output=str(base), arm_count=len(plan), runs=[dict(request=r,
              commands=[command(r,n) for n in range(r['case']['world']//8)]) for r in plan]), indent=2))
        return 0
    if sys.platform != 'linux':
        p.error('Actual training must run on Linux GPU servers; use --dry-run locally')
    if any(r['case']['world']==16 for r in plan) and (not args.node1 or args.master_addr in ('127.0.0.1','localhost')):
        p.error('D16 needs --node1 and node0 reachable --master-addr')
    if any(v<=0 for v in (args.ready_timeout,args.first_step_timeout,args.step_timeout,args.teardown_timeout,args.run_timeout)):
        p.error('Timeouts must be positive')
    if args.resume:
        if not base.is_dir():
            p.error(f'Cannot resume: campaign directory does not exist: {base}')
    else:
        base.mkdir(parents=True, exist_ok=False)
    import fcntl
    with (base/'.coordinator.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another coordinator is already using this run-id') from exc
        source = fingerprint(args.megatron_root)
        skip = set()
        if args.resume:
            resumed = prepare_resume(base, plan, source)
            plan = resumed['plan']
            history = base/'resume_history'
            history.mkdir(exist_ok=True)
            dump(history/(resumed['attempt']+'.json'), resumed)
            for group in resumed['groups']:
                print(f"{group['action']}: {group['case']} pair{group['pair']}", flush=True)
                if group['action'] == 'SKIP_COMPLETE_PAIR':
                    skip.add((group['case'], group['pair']))
        else:
            dump(base/'source_manifest.json', source)
            (base/'working_tree.patch').write_bytes(subprocess.check_output(
                ['git','diff','HEAD','--binary'], cwd=args.megatron_root, timeout=30))
            dump(base/'plan.json', plan)
        dump(base/'active_plan.json', plan)
        try:
            for i, req in enumerate(plan,1):
                if (req['case']['case'], req['pair']) in skip:
                    continue
                req['source']=source
                print(f"[{i}/{len(plan)}] {req['case']['case']} pair{req['pair']} {req['variant']}", flush=True)
                execute(req,args)
                summarize(base)
        finally:
            summarize(base)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
