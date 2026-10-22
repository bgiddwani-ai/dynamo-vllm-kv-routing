"""Bound inference tests by GPU temperature; stop only this release on a trip.

The worker wrapper also kills its own process group if NVML stops responding.
These checks reduce risk; they cannot recover an already hung GPU driver/host.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def emit(event, **data):
    print(json.dumps(dict(event=event, time=time.time(), **data)), flush=True)


def readings(uuids):
    result = subprocess.run([
        'nvidia-smi', '-i', ','.join(uuids),
        '--query-gpu=uuid,temperature.gpu,clocks_throttle_reasons.sw_thermal_slowdown,'
        'clocks_throttle_reasons.hw_thermal_slowdown', '--format=csv,noheader,nounits',
    ], capture_output=True, text=True, check=True, timeout=3)
    rows = [dict(uuid=r[0].strip(), temperature_c=int(r[1]),
                 sw_thermal=r[2].strip(), hw_thermal=r[3].strip())
            for r in csv.reader(io.StringIO(result.stdout))]
    if len(rows) != len(uuids) or {r['uuid'] for r in rows} != set(uuids):
        raise RuntimeError('GPU telemetry does not match the assigned UUIDs')
    return rows


def check(rows, threshold):
    for row in rows:
        if row['temperature_c'] >= threshold:
            raise RuntimeError(f"{row['uuid']}: {row['temperature_c']} C >= {threshold} C")
        if row['sw_thermal'] != 'Not Active' or row['hw_thermal'] != 'Not Active':
            raise RuntimeError(f'Thermal throttling or unknown throttle state: {row}')


def record_trip(path, reason):
    emit('safety_stop', reason=reason)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(time=time.time(), reason=reason), indent=2))
    except OSError as exc:
        emit('latch_write_failed', error=str(exc))


def terminate_group(proc, grace=1):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Kill any remaining engine children, including after the launcher exits.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


def worker(command):
    latch = Path(os.environ.get('NIC_SAFETY_LATCH', '/cache/safety-stop.json'))
    if latch.exists():
        emit('safety_latched', path=str(latch))
        return 78
    uuids = os.environ['NIC_ALLOWED_GPUS'].split(',')
    stop_temp = int(os.environ.get('NIC_MAX_GPU_TEMP_C', '75'))
    start_temp = int(os.environ.get('NIC_MAX_START_TEMP_C', '60'))
    proc = None
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stopped.append(signum))
    try:
        check(readings(uuids), start_temp)
        proc = subprocess.Popen(command, start_new_session=True)
        while proc.poll() is None and not stopped:
            rows = readings(uuids)
            check(rows, stop_temp)
            emit('gpu_health', gpus=rows)
            time.sleep(1)
        if stopped:
            terminate_group(proc, grace=5)
            return 128 + stopped[0]
        return proc.returncode
    except Exception as exc:
        record_trip(latch, str(exc))
        if proc is not None:
            terminate_group(proc)
        return 75


def stop_release(namespace, release):
    def kube(*args):
        return subprocess.run(['kubectl', '-n', namespace, *args], text=True,
                              capture_output=True, check=True, timeout=10).stdout
    graph = json.loads(kube('get', 'dynamographdeployment', release, '-o', 'json'))
    patch = [dict(op='replace', path=f'/spec/components/{i}/replicas', value=0)
             for i, component in enumerate(graph['spec']['components'])
             if component['type'] == 'worker']
    if not patch:
        raise RuntimeError('No worker components found; refusing unrelated changes')
    kube('patch', 'dynamographdeployment', release, '--type=json', '-p', json.dumps(patch))
    kube('delete', 'pods', '-l', f'nic.nvidia.com/release={release},nic.nvidia.com/role=worker',
         '--grace-period=1', '--wait=false')
    emit('release_workers_stopped', namespace=namespace, release=release)


def watch(args):
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stopped.append(signum))
    if args.latch.exists():
        raise RuntimeError(f'Safety latch exists: {args.latch}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open('a', buffering=1) as out:
            while not stopped:
                if args.mode == 'watch':
                    rows = readings(args.uuids.split(','))
                    check(rows, args.stop_temp)
                    data = dict(time=time.time(), gpus=rows)
                    out.write(json.dumps(data) + '\n')
                    tmp = args.heartbeat.with_suffix('.tmp')
                    tmp.write_text(json.dumps(data))
                    tmp.replace(args.heartbeat)
                else:
                    data = json.loads(args.heartbeat.read_text())
                    if time.time() - data['time'] > 10:
                        raise RuntimeError('Node GPU-health heartbeat is stale by >10 seconds')
                    if args.latch.exists():
                        raise RuntimeError('Primary safety monitor tripped')
                    peer = args.heartbeat.with_name('external-heartbeat.json')
                    tmp = peer.with_suffix('.tmp')
                    tmp.write_text(json.dumps(dict(time=time.time(), observed=data['time'])))
                    tmp.replace(peer)
                time.sleep(1)
    except Exception as exc:
        # Do not let a stalled shared-filesystem write delay the stop request.
        emit('safety_stop', reason=str(exc))
        try:
            stop_release(args.namespace, args.release)
        finally:
            record_trip(args.latch, str(exc))
        return 75
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        command = sys.argv[2:]
        if command and command[0] == '--':
            command = command[1:]
        if not command:
            raise SystemExit('A worker command is required')
        raise SystemExit(worker(command))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['watch', 'external'])
    p.add_argument('--namespace', required=True)
    p.add_argument('--release', required=True)
    p.add_argument('--uuids', default='')
    p.add_argument('--stop-temp', type=int, default=75)
    p.add_argument('--heartbeat', type=Path, required=True)
    p.add_argument('--latch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    raise SystemExit(watch(p.parse_args()))
