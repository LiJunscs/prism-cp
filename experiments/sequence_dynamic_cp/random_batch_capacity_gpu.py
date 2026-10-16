"""Frozen eight-arm full-update capacity experiment; no searches or retries."""
import argparse, importlib.util, json, os, subprocess, sys
from pathlib import Path
from results_io import load, save, summarize_records
from native import replay
from planner import Constraints, from_rank_ids, validate
from mixed_plan import coalesce_rounds
from joint_cp.replay import compile_replay, validate_replay
from two_dimensional_inputs import validate_manifest
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PYTHON=os.environ.get('PYTHON', sys.executable)
OUTPUT=None
HOSTS=[]
SSH_PORT=22
MASTER_ADDR=None
ARMS=[('L224_S4',24),('L224_S4',14),('L224_S4',28),('L224_S4',None),('irregular_S8',24),('irregular_S8',None),('L192_S4',24),('L192_S4',None)]
def arm_id(case,k): return case+('_joint' if k is None else f'_N_K{k}')
def prepare():
    if OUTPUT.exists(): raise ValueError('Fresh output required; no overwrite')
    source=HERE/'results/joint_cp_two_20261015'
    template=load(source/'inputs/72_samples_2p25Mi.json')
    base=load(HERE/'results/native_n_remeasure_20261016/plans/72_samples_2p25Mi.json')
    cases={r['case']:r for r in load(HERE/'results/random_batch_capacity_20261016_r1/analysis.json')['cases']}
    jobs=[]
    for case in dict(ARMS):
        lengths=cases[case]['lengths']; manifest=dict(template,case_id=case,input_tokens=sum(lengths),loss_tokens=sum(n//2 for n in lengths),samples=[dict(sample_id=i,sample_uid=f'{case}:{i:06d}',length=n,original_length=n,loss_prefix_tokens=n//2) for i,n in enumerate(lengths)])
        validate_manifest(manifest);save(OUTPUT/'inputs'/f'{case}.json',manifest)
    for index,(case,k) in enumerate(ARMS):
        aid=arm_id(case,k);manifest=load(OUTPUT/'inputs'/f'{case}.json');lengths=dict(enumerate(s['length'] for s in manifest['samples']))
        if k is None:
            plan=load(HERE/f'results/joint_advantage_search_20261016_r1/{case}_plans.json')['J_p1_packed_redistribute']
            frozen={'plan':plan,'source_key':'J_p1_packed_redistribute'}
            rows=compile_replay(frozen,manifest,updates=5,candidate_id=aid)
            for row in rows: validate_replay(row,16,'GGG*GGG*')
            save(OUTPUT/'jobs'/aid/'executor.json',load(HERE/'configs/joint_executor.json'))
        else:
            ranks=replay(list(lengths.items()),16,k*1024,beta_tokens=0.)
            plan=from_rank_ids(ranks);validate(plan,lengths,Constraints(16,k*1024,alignment=32))
            rounds=coalesce_rounds(plan,lengths,16,k*1024)
            frozen=dict(base,candidate_id=aid,case_id=case,capacity_tokens=k*1024,capacity_semantics='native_generation_and_execution_same_K',rank_ids=ranks,expected_lengths=list(lengths.values()),logical_sample_ids=list(lengths),execution_rounds=rounds,source_microbatches=len(ranks),merged_microbatches=len(rounds),input_tokens=manifest['input_tokens'],loss_tokens=manifest['loss_tokens'])
            rows=[dict(frozen,step=s) for s in range(1,6)]
        save(OUTPUT/'plans'/f'{aid}.json',frozen);save(OUTPUT/'jobs'/aid/'replay.json',rows)
        jobs.append(dict(id=aid,case=case,K_Ki=k,updates=5,warmup=[1,2],measure=[3,4,5],microbatches=rows[0]['merged_microbatches'],port=29810+index))
    save(OUTPUT/'matrix.json',dict(arms=jobs,maximum_launches=8,maximum_updates=40,retries=0,world=16,nodes=HOSTS,python=PYTHON,model=load(HERE/'configs/joint_executor.json'),pattern='GGG*GGG*',layers=8,bf16=True,TP=1,PP=1,recompute=False,variable_N_reuse={'L224_S4':'L224_S4_N_K14','irregular_S8':'irregular_S8_N_K24','L192_S4':'L192_S4_N_K24'},fixed_N_K_Ki=24,training='same input/content/labels/masks/seed; dense MLP/head/loss/F+B/finalize/norm clip/nonzero Adam/parameter synchronization; beta0 native next_hdp_group_packing_aware+DUPC without joint adapter; contiguous GDN including MLP; joint retains MLP redistribution',memory_guard='min(0.9*total,total-8GiB); unchanged training guard; existing once allocator empty cache before first finalize',constraints='no GPU preflight/numerics/logical/profiling/search/hash/inventory/retry; pause matrix on OOM, infrastructure or training-contract failure'))
    summarize()
def run_name(aid):
    return aid+'_resume1' if aid=='L224_S4_N_K24' and (OUTPUT/'runs'/f'{aid}_resume1').exists() else aid
def summarize():
    matrix=load(OUTPUT/'matrix.json');results={};gpu_launches=0
    for arm in matrix['arms']:
        aid=arm['id'];folder=OUTPUT/'runs'/run_name(aid)/'records';status_path=OUTPUT/'launch_status'/f'{run_name(aid)}.json'
        if folder.exists():
            row=summarize_records(folder,load(OUTPUT/'inputs'/f"{arm['case']}.json"),5,range(3,6),arm['microbatches'])
            ranks=row['steps_by_rank'];row['complete_updates']=len(set.intersection(*(set(v) for v in ranks.values()))) if ranks else 0
        else:
            row=dict(status='failed_before_complete_update' if (OUTPUT/'runs'/run_name(aid)).exists() else 'not_started',complete_updates=0)
        if (OUTPUT/'runs'/run_name(aid)).exists(): gpu_launches+=1
        if status_path.exists():row['launcher']=load(status_path)
        results[aid]=row
    if not matrix.get('archived', False):
        save(OUTPUT/'comparison.json',dict(arms=results,variable_N_reuse=matrix['variable_N_reuse']))
    # Historical supervision is immutable; new matrix ledgers update after each arm.
    if not load(OUTPUT/'matrix.json').get('archived', False):
        save(OUTPUT/'supervision.json',dict(actual_gpu_launches=gpu_launches,
            complete_updates=sum(r['complete_updates'] for r in results.values()),
            arms={k:dict(status=v['status'],complete_updates=v['complete_updates']) for k,v in results.items()}))
    return results
def worker(aid):
    if load(OUTPUT/'matrix.json').get('archived', False):
        raise ValueError('Historical evidence is read-only; prepare a fresh output root')
    arm=next(a for a in load(OUTPUT/'matrix.json')['arms'] if a['id']==aid)
    cmd=[PYTHON,str(HERE/'gpu_smoke.py'),'--output',str(OUTPUT/'runs'/run_name(aid)),'--replay',str(OUTPUT/'jobs'/aid/'replay.json'),'--two-dimensional-input',str(OUTPUT/'inputs'/f"{arm['case']}.json"),'--packet-role','S' if arm['K_Ki'] is None else 'N','--mixed-cp','--eval-iters','0','--pattern','GGG*GGG*','--num-layers','8','--nnodes','2','--node-rank',os.environ['NODE_RANK'],'--master-addr',os.environ['MASTER_ADDR'],'--port',os.environ['MASTER_PORT'],'--iterations','5','--warmup','2','--timeout','1800','--numerics','off','--logical','off','--skip-gpu-preflight','--reuse-source-evidence',str(OUTPUT/'matrix.json')]
    cmd+=['--two-dimensional-config',str(OUTPUT/'jobs'/aid/'executor.json')] if arm['K_Ki'] is None else ['--native-n','--native-fallback','--cap',str(arm['K_Ki']*1024)]
    return subprocess.call(cmd,cwd=ROOT,env=os.environ|dict(PYTHON=PYTHON,TORCHRUN=str(Path(PYTHON).with_name('torchrun'))))
def run(aid):
    if load(OUTPUT/'matrix.json').get('archived', False):
        raise ValueError('Historical evidence is read-only; prepare a fresh output root')
    ssh_output=OUTPUT/'ssh'/run_name(aid)
    if (OUTPUT/'runs'/run_name(aid)).exists(): raise ValueError('No overwrite or GPU retry')
    if ssh_output.exists(): raise ValueError('Use a fresh output for a new attempt')
    arm=next(a for a in load(OUTPUT/'matrix.json')['arms'] if a['id']==aid)
    spec=importlib.util.spec_from_file_location('launcher',ROOT/'scripts/torchrun_launcher.py');launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    command=['env']
    if os.environ.get('MEGATRON_ROOT'):
        command.append('MEGATRON_ROOT='+os.environ['MEGATRON_ROOT'])
    command += [PYTHON,str(Path(__file__).resolve()),'worker',aid,'--output',str(OUTPUT)]
    code=launcher.launch(HOSTS,command,ROOT,MASTER_ADDR or HOSTS[0],arm['port'],ssh_output,SSH_PORT,1920)
    save(OUTPUT/'launch_status'/f'{run_name(aid)}.json',dict(returncode=code,retries=0));rows=summarize();print(json.dumps(rows.get(aid,{})),flush=True)
    return code
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','worker','summarize'])
    p.add_argument('arm',nargs='?',choices=[arm_id(c,k) for c,k in ARMS])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hosts',nargs=2,help='Two 8-GPU SSH hosts; required only by run')
    p.add_argument('--ssh-port',type=int,default=22)
    p.add_argument('--master-addr',help='Rendezvous address reachable by both nodes')
    a=p.parse_args();OUTPUT=a.output.resolve();HOSTS=a.hosts or [];SSH_PORT=a.ssh_port;MASTER_ADDR=a.master_addr
    if a.action in ('run','worker') and not a.arm:p.error('An explicit arm is required')
    if a.action=='run' and not HOSTS:p.error('Supply the target cluster hosts explicitly')
    if a.action=='prepare':prepare()
    elif a.action=='summarize':summarize()
    elif a.action=='worker':raise SystemExit(worker(a.arm))
    else:raise SystemExit(run(a.arm))
