"""Read-only remote inventory; run with Python on caviar9, emit JSON only."""
import hashlib
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

home = Path.home()
work = home / 'codex-gpu-work'
images = work / 'data/mvtec_ad_materialized_v2'
categories = {}
if images.exists():
    for category in sorted(images.iterdir()):
        if category.is_dir():
            categories[category.name] = {
                'image_files': sum(1 for p in category.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'})
            }
weights = []
cache = home / '.cache/huggingface/hub/models--timm--wide_resnet50_2.racm_in1k'
for p in sorted(cache.glob('snapshots/*/model.safetensors')):
    digest = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    weights.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest.hexdigest()})
arrays = []
for dirname in ('results', 'artifacts'):
    root = work / 'colab-github-bridge' / dirname
    if root.exists():
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.npz', '.npy', '.pt', '.pth', '.pkl', '.h5', '.hdf5', '.safetensors'}:
                arrays.append({'path': str(p), 'bytes': p.stat().st_size})
print(json.dumps({'checked_at': datetime.now(timezone.utc).isoformat(), 'host': socket.getfqdn(),
                  'image_root': str(images), 'categories': categories, 'wide_resnet_weights': weights,
                  'saved_arrays_in_results_and_artifacts': arrays,
                  'scope': 'Known work repository results/artifacts and WideResNet Hugging Face cache; not a whole-disk search. Image count includes masks if stored inside category directories.'}, indent=2))
