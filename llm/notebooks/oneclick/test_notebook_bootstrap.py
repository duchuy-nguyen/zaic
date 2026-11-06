"""Regression checks for Kaggle Python 3.13's missing inherited torch.

No pip installs, GPU requests or model downloads are performed by these tests.
"""
import ast
import json
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


NOTEBOOK = Path(__file__).resolve().parents[1] / 'LifeMate_OneClick.ipynb'


class BootstrapTests(unittest.TestCase):
    def test_t4_default_uses_explicit_9b_teacher_and_keeps_4b_student(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        config = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'TEACHER_ID =' in ''.join(c['source']))
        values = {}
        exec(compile(config, '<config-fixture>', 'exec'), values)
        self.assertEqual(values['TEACHER_ID'], 'Qwen/Qwen3.5-9B')
        self.assertFalse(values['TEACHER_PREQUANTIZED'])
        self.assertEqual(values['TEACHER_REVISION'], 'c202236235762e1c871ad0ccb60c8ee5ba337b9a')
        self.assertEqual(values['REUSE_ASR_FROM'], '')
        all_code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        self.assertIn("student_id='Qwen/Qwen3.5-4B'", all_code)
        self.assertLess(all_code.index("run_stage('student_precheck')"),all_code.index("run_stage('asr_train')"))
        self.assertLess(all_code.index("run_stage('student_precheck')"),all_code.index("run_stage('teacher')"))
        self.assertLess(all_code.index("run_stage('student_smoke')"),all_code.index("run_stage('student_base')"))

    def test_stage_failure_surfaces_original_worker_error(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'def run_stage' in ''.join(c['source']))
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run_stage')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'stage_asr_train.json').write_text(json.dumps({'error': 'ValueError: fixture training failure'}))
            process = SimpleNamespace(stdout=['Traceback (fixture)\n', 'ValueError: fixture training failure\n'], wait=lambda: 1)
            commands = []; archives=[]
            def popen(command, **kwargs): commands.append(command); return process
            namespace = dict(SESSION_PAUSED=False, TRAIN_ENV={}, os=os, RUN=root, json=json,
                save_resume_archive=lambda:archives.append(True),
                cfg={'seed': 1}, SOURCE=root, SESSION_DEADLINE=0, PYTHON='fixture-python', config_path=root / 'config.json',
                subprocess=SimpleNamespace(Popen=popen, PIPE=-1, STDOUT=-2))
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<runner-fixture>', 'exec'), namespace)
            with self.assertRaisesRegex(RuntimeError, 'ValueError: fixture training failure') as error:
                namespace['run_stage']('asr_train')
            self.assertIn('Worker log tail', str(error.exception))
            self.assertIn('exit 1', str(error.exception))
            self.assertEqual(commands[0][1], '-I')
            self.assertEqual(archives,[True])

    def test_isolated_install_precedes_torch_probe_and_workers_use_clean_env(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        source = next(s for s in sources if 'TORCH_PINS =' in s)
        tree = ast.parse(source)
        start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'TRAIN_ENV' for t in node.targets))
        for missing_pip in (False, True):
          with self.subTest(missing_pip=missing_pip), tempfile.TemporaryDirectory() as temporary:
            calls = []
            root = Path(temporary)
            if missing_pip:
                # Reproduce the partially created venv from failed ensurepip.
                (root / 'env/bin').mkdir(parents=True)
                (root / 'env/bin/python').touch()
            def fake_path(value):
                self.assertEqual(value, '/tmp/lifemate_training_env_v2')
                return root / 'env'
            def run(command, **kwargs):
                calls.append((command, kwargs))
                self.assertNotIn('PYTHONPATH', kwargs['env'])
                self.assertNotIn('PYTHONHOME', kwargs['env'])
                self.assertEqual(kwargs['env']['PYTHONNOUSERSITE'], '1')
                if 'venv' in command:
                    self.assertIn('--without-pip', command)
                    destination = Path(command[-1]); (destination / 'bin').mkdir(parents=True, exist_ok=True)
                    (destination / 'bin/python').touch()
                    (destination / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
                if command[-3:] == ['-m', 'pip', '--version'] and kwargs.get('capture_output'):
                    return SimpleNamespace(returncode=int(missing_pip))
                return SimpleNamespace(returncode=0)
            namespace = dict(os=SimpleNamespace(environ={'PYTHONPATH': '/bad/kernel/hooks',
                'PYTHONHOME': '/bad/kernel', 'CUDA_VISIBLE_DEVICES': '0,1'}),
                sys=SimpleNamespace(executable='kernel-python'), Path=fake_path,
                subprocess=SimpleNamespace(run=run), shutil=SimpleNamespace(which=lambda name: 'nvidia-smi'), hashlib=hashlib)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'# official bootstrap fixture\n')) as fetch:
                exec(compile(ast.Module(body=tree.body[start:], type_ignores=[]), '<bootstrap-fixture>', 'exec'), namespace)
                self.assertEqual(fetch.call_count, int(missing_pip))
            self.assertNotIn('--system-site-packages', calls[1][0])
            self.assertEqual(calls[1][0][:4], ['kernel-python', '-I', '-m', 'venv'])
            torch_call = next(c[0] for c in calls if 'torch==2.9.1' in c[0])
            self.assertIn('https://download.pytorch.org/whl/cu126', torch_call)
            pin_call = next(c[0] for c in calls if 'transformers==5.20.2' in c[0])
            self.assertIn('-c', pin_call)
            if missing_pip:
                bootstrap_call = next(c[0] for c in calls if 'pip==25.3' in c[0])
                self.assertLess(calls.index(next(c for c in calls if c[0] == bootstrap_call)),
                                calls.index(next(c for c in calls if c[0] == torch_call)))
            constraints = (root / 'env/torch-constraints.txt').read_text()
            self.assertEqual(constraints.splitlines(), ['torch==2.9.1+cu126',
                'torchvision==0.24.1+cu126', 'torchaudio==2.9.1+cu126'])
            self.assertIn('import sys, torch, torchvision', calls[-1][0][-1])
            self.assertIn('device="cuda"', calls[-1][0][-1])
            self.assertIn('dict(TRAIN_ENV,', next(s for s in sources if 'def run_stage' in s))

    def test_real_isolated_venv_ignores_injected_sitecustomize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); injection = root / 'injection'; injection.mkdir()
            (injection / 'sitecustomize.py').write_text('raise RuntimeError("injected sitecustomize ran")\n')
            inherited = dict(os.environ, PYTHONPATH=str(injection))
            clean = {k: v for k, v in inherited.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
            clean['PYTHONNOUSERSITE'] = '1'
            venv = root / 'venv'
            subprocess.run([sys.executable, '-I', '-m', 'venv', '--without-pip', str(venv)],
                           check=True, env=clean, capture_output=True, text=True)
            python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            probe = subprocess.run([str(python), '-I', '-c',
                'import sys; assert sys.prefix != sys.base_prefix; '
                'assert "sitecustomize" not in sys.modules; print("isolated")'],
                check=True, env=clean, capture_output=True, text=True)
            self.assertEqual(probe.stdout.strip(), 'isolated')
            self.assertNotIn('sitecustomize', probe.stderr)


if __name__ == '__main__':
    unittest.main()
