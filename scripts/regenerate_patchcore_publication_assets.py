"""Run the publication comparison with persistent features, banks and scores."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from scripts import mvtec_patchcore_fixed_bank_profile_switch as experiment

ROOT = Path(os.environ.get('PATCHCORE_EXPORT_ROOT', '/home/shunya/codex-gpu-work/data/patchcore_publication_assets_001'))
MANIFEST = Path('results/patchcore_publication_assets_001_manifest.json')
records = {}
array_hashes = {}
evaluations = []
active_evaluation = None


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.save(f, array, allow_pickle=False)
    temp.replace(path)
    records[str(path.relative_to(ROOT))] = {'bytes': path.stat().st_size, 'sha256': digest(path),
                                          'shape': list(array.shape), 'dtype': str(array.dtype)}


def checkpoint():
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temp = MANIFEST.with_suffix('.tmp')
    temp.write_text(json.dumps({'root': str(ROOT), 'files': records, 'evaluations': evaluations,
                               'note': 'Regenerated arrays; original run did not retain array hashes.'}, indent=2))
    temp.replace(MANIFEST)


OriginalCache = experiment.FeatureCache


class PersistentCache(OriginalCache):
    def get(self, category, cfg):
        key = (category, cfg['backbone'], cfg['out_indices'], cfg['patch_grid'])
        fresh = key not in self.cache
        value = super().get(category, cfg)
        if fresh:
            tag = hashlib.sha256(repr(key).encode()).hexdigest()[:20]
            for name, array in zip(('train', 'test', 'test_labels', 'train_labels'), value[:4]):
                save_array(ROOT / 'features' / category / tag / (name + '.npy'), array)
            meta = ROOT / 'features' / category / tag / 'config.json'
            meta.write_text(json.dumps({'category': category, 'config': cfg, 'meta': value[4]}, indent=2))
            checkpoint()
            print(f'EXPORTED FEATURES {category} {tag}', flush=True)
        return value


original_bank = experiment.build_bank


def persistent_bank(features, categories, size, args, device):
    inputs = []
    for category in categories:
        array = features[category]
        ident = id(array)
        if ident not in array_hashes:
            array_hashes[ident] = hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()
        inputs.append((category, array_hashes[ident]))
    spec = {'inputs': inputs, 'bank_size': size, 'seed': args.seed, 'candidate_pool': args.coreset_candidate_pool}
    tag = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    path = ROOT / 'banks' / (tag + '.npy')
    if path.exists():
        bank = np.load(path, allow_pickle=False)
        if str(path.relative_to(ROOT)) not in records:
            save_array(path, bank)
    else:
        bank = original_bank(features, categories, size, args, device)
        save_array(path, bank)
    path.with_suffix('.json').write_text(json.dumps(spec, indent=2))
    checkpoint()
    print(f'BANK {categories} size={size} sha256={records[str(path.relative_to(ROOT))]["sha256"]}', flush=True)
    return bank


original_scores = experiment.patchcore_scores_gpu
original_evaluate = experiment.evaluate_category


def persistent_evaluate(category, cfg, bank, cache, args, device):
    global active_evaluation
    active_evaluation = {'category': category, 'config': cfg,
                         'bank_array_sha256': hashlib.sha256(memoryview(np.ascontiguousarray(bank)).cast('B')).hexdigest()}
    result = original_evaluate(category, cfg, bank, cache, args, device)
    active_evaluation['result'] = result
    evaluations.append(active_evaluation)
    active_evaluation = None
    checkpoint()
    return result


def persistent_scores(features, bank, chunk_size, device):
    scores = original_scores(features, bank, chunk_size, device)
    tag = hashlib.sha256(memoryview(np.ascontiguousarray(scores)).cast('B')).hexdigest()
    path = ROOT / 'patch_scores' / (tag + '.npy')
    if str(path.relative_to(ROOT)) not in records:
        save_array(path, scores)
    active_evaluation['patch_scores'] = str(path.relative_to(ROOT))
    checkpoint()
    return scores


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(ROOT).free
    if free < 40 * 1024**3:
        raise RuntimeError(f'Need at least 40 GiB free for persistent export; available={free}')
    provenance = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                  'disk_free_at_start': free, 'numpy': np.__version__}
    (ROOT / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    experiment.FeatureCache = PersistentCache
    experiment.build_bank = persistent_bank
    experiment.patchcore_scores_gpu = persistent_scores
    experiment.evaluate_category = persistent_evaluate
    experiment.main()
    checkpoint()
    print(f'EXPORT COMPLETE {len(records)} arrays at {ROOT}', flush=True)
