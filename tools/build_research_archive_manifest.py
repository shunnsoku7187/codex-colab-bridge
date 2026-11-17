"""Inventory tracked research files and check newly staged publication assets."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def paths(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').splitlines()


staged = paths('diff', '--cached', '--name-only', '--diff-filter=ACM')
secret = re.compile(rb'github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|https://[^\s/]+:[^\s/@]{8,}@github\.com')
for name in staged:
    data = (ROOT / name).read_bytes()
    if len(data) > 90 * 1024 * 1024:
        raise ValueError(f'File too large for ordinary Git: {name}')
    if (ROOT / name).suffix in {'.py', '.mjs', '.md', '.json', '.tex', '.sh', '.ps1'} and secret.search(data):
        raise ValueError(f'Possible credential in {name}; value omitted')

entries = []
for name in paths('ls-files'):
    if not name.startswith(('docs/', 'results/', 'jobs/', 'scripts/', 'src/', 'tools/', 'output/', 'logs/')):
        continue
    if name == 'docs/research_archive_manifest.json':
        continue
    path = ROOT / name
    if path.is_file():
        data = path.read_bytes()
        entries.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
out = ROOT / 'docs/research_archive_manifest.json'
out.write_text(json.dumps({'inventory_date': '2026-11-18', 'entry_point': 'docs/RESEARCH_HANDOFF.md',
                          'files': entries}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Inventoried {len(entries)} files; checked {len(staged)} staged files for size and credential patterns.')
