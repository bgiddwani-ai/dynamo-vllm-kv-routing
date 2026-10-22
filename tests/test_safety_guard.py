"""Exercise stop behavior with fake NVML and Kubernetes, without a real GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

GUARD = Path(__file__).resolve().parents[1] / 'scripts/safety_guard.py'


class SafetyIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ['PATH'],
                        NIC_ALLOWED_GPUS='GPU-test', NIC_MAX_START_TEMP_C='60',
                        NIC_MAX_GPU_TEMP_C='75', NIC_SAFETY_LATCH=str(self.root/'latch.json'),
                        NIC_TEST_ROOT=str(self.root))
        self.program('nvidia-smi', '''
import json,os,pathlib,sys,time
p=pathlib.Path(os.environ['NIC_TEST_ROOT'])
c=p/'counter'; i=int(c.read_text()) if c.exists() else 0
c.write_text(str(i+1))
s=json.loads((p/'sequence.json').read_text()); r=s[min(i,len(s)-1)]
if r.get('error'): sys.exit(2)
if r.get('hang'): time.sleep(5)
print('GPU-test, %s, %s, Not Active' % (r['temp'],r.get('sw','Not Active')))
''')
        self.program('kubectl', '''
import json,os,pathlib,sys
p=pathlib.Path(os.environ['NIC_TEST_ROOT'])
with (p/'kube-calls.jsonl').open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
if 'get' in sys.argv:
 print(json.dumps({'spec':{'components':[{'type':'frontend','replicas':1},{'type':'worker','replicas':1},{'type':'worker','replicas':1}]}}))
''')

    def program(self, name, source):
        p = self.root/name
        p.write_text('#!/usr/bin/env python3\n' + source)
        p.chmod(0o755)

    def run_guard(self, args, sequence):
        (self.root/'sequence.json').write_text(json.dumps(sequence))
        return subprocess.run([sys.executable, str(GUARD), *args], env=self.env,
                              capture_output=True, text=True, timeout=15)

    def test_healthy_child_completes(self):
        r = self.run_guard(['worker', '--', sys.executable, '-c', 'print("ok")'], [{'temp':30}])
        self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
        self.assertFalse((self.root/'latch.json').exists())

    def test_hot_start_never_launches_child(self):
        marker = self.root/'started'
        r = self.run_guard(['worker', '--', sys.executable, '-c',
                            f'from pathlib import Path; Path({str(marker)!r}).touch()'], [{'temp':65}])
        self.assertEqual(r.returncode, 75, r.stdout+r.stderr)
        self.assertFalse(marker.exists())
        # A later cool reading must not clear the trip automatically.
        r = self.run_guard(['worker', '--', sys.executable, '-c', 'print("unsafe restart")'], [{'temp':30}])
        self.assertEqual(r.returncode, 78)

    def assert_running_child_stopped(self, unsafe):
        pidfile = self.root/'child.pid'
        command = ('import os,time; from pathlib import Path; '
                   f'Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)')
        r = self.run_guard(['worker', '--', sys.executable, '-c', command],
                           [{'temp':30}, {'temp':30}, unsafe])
        self.assertEqual(r.returncode, 75, r.stdout+r.stderr)
        self.assertTrue((self.root/'latch.json').exists())
        self.assertTrue(pidfile.exists())
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)

    def test_overtemperature_stops_running_engine(self):
        self.assert_running_child_stopped({'temp':76})

    def test_throttle_stops_even_at_low_temperature(self):
        self.assert_running_child_stopped({'temp':40, 'sw':'Active'})

    def test_nvml_error_stops_running_engine(self):
        self.assert_running_child_stopped({'temp':30, 'error':True})

    def test_nvml_hang_stops_running_engine(self):
        self.assert_running_child_stopped({'temp':30, 'hang':True})

    def monitor_args(self, mode):
        return [mode, '--namespace', 'test-ns', '--release', 'only-our-release',
                '--uuids', 'GPU-test', '--heartbeat', str(self.root/'heartbeat.json'),
                '--latch', str(self.root/'latch.json'), '--output', str(self.root/'health.jsonl')]

    def assert_only_our_workers_stopped(self):
        calls = [json.loads(l) for l in (self.root/'kube-calls.jsonl').read_text().splitlines()]
        patch = next(c for c in calls if 'patch' in c)
        changes = json.loads(patch[-1])
        self.assertEqual([c['path'] for c in changes],
                         ['/spec/components/1/replicas','/spec/components/2/replicas'])
        self.assertTrue(all(c['value']==0 for c in changes))
        delete = next(c for c in calls if 'delete' in c)
        self.assertIn('nic.nvidia.com/release=only-our-release,nic.nvidia.com/role=worker', delete)
        self.assertTrue(all(c[:2]==['-n','test-ns'] for c in calls))

    def test_host_trip_scales_only_our_workers_down(self):
        r = self.run_guard(self.monitor_args('watch'), [{'temp':76}])
        self.assertEqual(r.returncode, 75, r.stdout+r.stderr)
        self.assert_only_our_workers_stopped()

    def test_external_monitor_stops_on_stale_heartbeat(self):
        (self.root/'heartbeat.json').write_text(json.dumps({'time':time.time()-20}))
        r = self.run_guard(self.monitor_args('external'), [{'temp':30}])
        self.assertEqual(r.returncode, 75, r.stdout+r.stderr)
        self.assert_only_our_workers_stopped()


if __name__ == '__main__':
    unittest.main()
