"""Parse tracked Python source without importing data pipelines or launching models."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
files = subprocess.check_output(['git', 'ls-files', '-z', '--', '*.py'], cwd=ROOT).decode().split('\0')
count = 0
for name in filter(None, files):
    path = ROOT / name
    if path.is_file():
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=name)
        count += 1
for name in ('scripts/ci/check_source.py', 'scripts/ci/check_data.py'):
    if name not in files:
        ast.parse((ROOT / name).read_text(encoding='utf-8'), filename=name)
        count += 1
print(f'Python syntax passed: {count} files')
