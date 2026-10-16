#!/usr/bin/env python3
"""Launch one torchrun-owning command per node through SSH, without a scheduler.

The remote command receives NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT. It must
start its local torchrun processes (native_n_remeasure's worker does this). Remote
process groups belong to a unique invocation; peer failures cancel only those.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import uuid


def ssh_prefix(host, port):
    if not host or host.startswith('-') or any(c.isspace() for c in host):
        raise ValueError('Use a hostname or user@hostname without whitespace')
    return ['ssh','-p',str(port),'-o','BatchMode=yes','-o','ConnectTimeout=15',
            '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',host]


def remote_script(command, workdir, env, pidfile):
    task=shlex.join(['env',*(f'{k}={v}' for k,v in env.items()),*command])
    # No remote shell expansion of user arguments. setsid gives us a distinct
    # process group, and the trap/recovery command only targets that group.
    return '\n'.join([
        'set -eu',f'cd -- {shlex.quote(str(workdir))}',
        f'pidfile={shlex.quote(pidfile)}',
        'child=',
        'cleanup() { if [ -n "$child" ]; then kill -TERM -- "-$child" 2>/dev/null || true; fi; rm -f -- "$pidfile"; }',
        'trap cleanup EXIT HUP INT TERM',
        f'setsid bash -c {shlex.quote("exec "+task)} &',
        'child=$!', 'printf "%s\\n" "$child" > "$pidfile"',
        'set +e', 'wait "$child"', 'status=$?', 'set -e',
        'cleanup', 'child=', 'exit "$status"',
    ])


def launch(hosts, command, workdir, master_addr, master_port, output, ssh_port=22,
           timeout=21600, dry_run=False):
    if not command or not hosts or len(set(hosts))!=len(hosts):
        raise ValueError('Supply a command and distinct node hosts')
    if not 1<=ssh_port<=65535 or not 1<=master_port<=65535 or timeout<=0:
        raise ValueError('Invalid port or timeout')
    token=uuid.uuid4().hex
    jobs=[]
    for rank,host in enumerate(hosts):
        env=dict(NNODES=len(hosts),NODE_RANK=rank,MASTER_ADDR=master_addr,MASTER_PORT=master_port,
                 PYTHONDONTWRITEBYTECODE=1,PRISM_LAUNCHER_GROUP=1)
        pidfile=f'/tmp/prism-torchrun-{token}-{rank}.pid'
        script=remote_script(command,workdir,env,pidfile)
        args=ssh_prefix(host,ssh_port)+['bash -lc '+shlex.quote(script)]
        jobs.append(dict(host=host,rank=rank,args=args,pidfile=pidfile,environment=env))
    if dry_run:
        print(json.dumps(dict(workdir=str(workdir),command=command,nodes=jobs),indent=2))
        return 0
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'launch.json').write_text(json.dumps(dict(command=command,workdir=str(workdir),
        master_addr=master_addr,master_port=master_port,ssh_port=ssh_port,nodes=jobs),indent=2)+'\n')
    active=[]
    status=1
    started=time.monotonic()
    def interrupted(signum,frame):
        raise KeyboardInterrupt
    previous_term=signal.signal(signal.SIGTERM,interrupted)
    try:
        for job in jobs:
            log=(output/f"ssh_node{job['rank']}.log").open('w')
            process=subprocess.Popen(job['args'],stdout=log,stderr=subprocess.STDOUT)
            active.append((job,process,log))
        while True:
            codes=[p.poll() for _,p,_ in active]
            if any(code not in (None,0) for code in codes):
                status=next(code for code in codes if code not in (None,0))
                break
            if all(code==0 for code in codes):
                status=0
                break
            if time.monotonic()-started > timeout:
                status=124
                break
            time.sleep(.2)
    except KeyboardInterrupt:
        status=130
    finally:
        signal.signal(signal.SIGTERM,previous_term)
        if status:
            for job,process,_ in active:
                # Unique pidfile cannot refer to another launcher invocation.
                cleanup=(f'f={shlex.quote(job["pidfile"])}; '
                    'if [ -f "$f" ]; then read -r p < "$f"; '
                    'case "$p" in ""|*[!0-9]*) exit 2;; esac; '
                    'kill -TERM -- "-$p" 2>/dev/null || true; '
                    'sleep 2; kill -KILL -- "-$p" 2>/dev/null || true; rm -f -- "$f"; fi')
                try:
                    subprocess.run(ssh_prefix(job['host'],ssh_port)+['bash -lc '+shlex.quote(cleanup)],
                                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)
                except (subprocess.TimeoutExpired,OSError):
                    pass
                if process.poll() is None:
                    process.terminate()
        records=[]
        for job,process,log in active:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
            log.close()
            records.append(dict(host=job['host'],node_rank=job['rank'],returncode=process.returncode))
        (output/'result.json').write_text(json.dumps(dict(returncode=status,nodes=records,
            elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hosts',nargs='+',required=True)
    p.add_argument('--ssh-port',type=int,default=22)
    p.add_argument('--master-addr',required=True)
    p.add_argument('--master-port',type=int,default=29580)
    p.add_argument('--workdir',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=21600)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    raise SystemExit(launch(args.hosts,command,args.workdir,args.master_addr,args.master_port,args.output,
                           args.ssh_port,args.timeout,args.dry_run))


if __name__=='__main__':
    main()
