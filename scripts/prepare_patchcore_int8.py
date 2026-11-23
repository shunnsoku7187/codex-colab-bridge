"""Export real NN fixtures and evaluate train-calibrated int8 selected profiles."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/shunya/codex-gpu-work/data/patchcore_publication_assets_001')
OUT = Path('results/patchcore_int8_preparation_001')


def raw_hash(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()


def quantize(a, scale):
    return np.clip(np.rint(a / scale), -127, 127).astype(np.int8)


def main():
    import torch
    from scripts.mvtec_ad_parquet_anomaly_probe import curve_rows, image_scores_from_patch_scores, find_materialized_samples
    from scripts.mvtec_patchcore_lightweight_sweep import best_under_false_pass, normal_train_and_test

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    summary = json.loads(Path('results/patchcore_publication_assets_001_summary.json').read_text())
    manifest = json.loads(Path('results/patchcore_publication_assets_001_manifest.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for category, cfg in summary['selected_profiles'].items():
        print('START', category, flush=True)
        matches = []
        for p in (ROOT / 'features' / category).glob('*/config.json'):
            meta = json.loads(p.read_text())['config']
            if (meta['backbone'], meta['out_indices'], meta['patch_grid']) == (cfg['backbone'], cfg['out_indices'], cfg['patch_grid']):
                matches.append(p.parent)
        if len(matches) != 1:
            raise RuntimeError(f'Ambiguous feature cache: {category} {matches}')
        folder = matches[0]
        train = np.load(folder / 'train.npy', mmap_mode='r')
        test = np.load(folder / 'test.npy', mmap_mode='r')
        labels = np.load(folder / 'test_labels.npy')
        train_hash = raw_hash(train)
        candidates = []
        for sidecar in (ROOT / 'banks').glob('*.json'):
            spec = json.loads(sidecar.read_text())
            if spec['inputs'] == [[category, train_hash]] and spec['bank_size'] == 3000:
                candidates.append(sidecar.with_suffix('.npy'))
        if len(candidates) != 1:
            raise RuntimeError(f'Ambiguous bank: {category} {candidates}')
        bank_path = candidates[0]
        bank = np.load(bank_path)
        bank_hash = raw_hash(bank)
        evaluations = [e for e in manifest['evaluations'] if e['category'] == category
                       and e['config'] == cfg and e['bank_array_sha256'] == bank_hash]
        if not evaluations:
            raise RuntimeError(f'No original evaluation for {category}')
        original = evaluations[0]
        float_patch = np.load(ROOT / original['patch_scores'])
        float_scores = image_scores_from_patch_scores(float_patch, cfg['topk_fraction'])['topk_score']
        # One scalar shared by query and bank preserves Euclidean geometry before rounding.
        scale = max(float(np.max(np.abs(train))), 1e-12) / 127.0
        qb = quantize(bank, scale)
        flat = test.reshape(-1, test.shape[-1])
        distances = np.empty(len(flat), dtype=np.float64)
        bt = torch.as_tensor(qb.astype(np.float64), device='cuda')
        clipped = 0
        for start in range(0, len(flat), 512):
            chunk = np.asarray(flat[start:start+512])
            clipped += int(np.count_nonzero(np.abs(chunk / scale) > 127))
            qt = torch.as_tensor(quantize(chunk, scale).astype(np.float64), device='cuda')
            # FP64 followed by integer rounding represents the integer squared-L2 oracle.
            d2 = torch.cdist(qt, bt).square().round().min(dim=1).values
            distances[start:start+len(chunk)] = d2.sqrt().cpu().numpy() * scale
        quant_patch = distances.reshape(test.shape[:2])
        quant_scores = image_scores_from_patch_scores(quant_patch, cfg['topk_fraction'])['topk_score']
        threshold = float(original['result']['threshold'])
        def rates(scores):
            return {'good_pass': float(np.mean(scores[labels == 0] < threshold)),
                    'defect_pass': float(np.mean(scores[labels == 1] < threshold))}
        # Verify the decision convention agrees with the saved original before reporting deltas.
        float_best = best_under_false_pass(curve_rows(labels, float_scores, 120), 0.03)
        quant_best = best_under_false_pass(curve_rows(labels, quant_scores, 120), 0.03)
        chosen = [int(np.flatnonzero(labels == value)[0]) for value in (0, 1)]
        _, samples = normal_train_and_test(find_materialized_samples(ROOT.parent/'mvtec_ad_materialized_v2', category))
        if len(samples) != len(labels) or any(s.label != int(label) for s,label in zip(samples, labels)):
            raise RuntimeError('Source image order/labels differ from saved features')
        queries = []
        patches = []
        for index in chosen:
            patch = int(np.argmax(float_patch[index]))
            patches.append(patch)
            queries.append(quantize(test[index, patch], scale))
        queries = np.stack(queries)
        expected = np.array([np.sum((qb.astype(np.int64)-q.astype(np.int64))**2, axis=1).min()
                             for q in queries], dtype=np.int64)
        np.savez_compressed(OUT / f'{category}_rtl.npz', bank=qb, queries=queries, expected=expected,
                            image_indices=chosen, patch_indices=patches, labels=labels[chosen])
        np.savez_compressed(OUT / f'{category}_scores.npz', labels=labels, float_scores=float_scores,
                            int8_scores=quant_scores)
        row = {'category': category, 'profile': cfg, 'scale': scale, 'zero_point': 0,
               'bank_source': str(bank_path), 'bank_raw_sha256': bank_hash,
               'feature_source': str(folder), 'clipped_test_values': clipped,
               'fixture_images': [{'path':str(samples[i].image_path), 'sha256':hashlib.sha256(samples[i].image_path.read_bytes()).hexdigest(),
                                   'image_index':i,'patch_index':patch,'label':int(labels[i])} for i,patch in zip(chosen,patches)],
               'test_values': int(test.size), 'frozen_threshold': threshold,
               'float_frozen': rates(float_scores), 'int8_frozen': rates(quant_scores),
               'float_reselected': float_best, 'int8_reselected': quant_best,
               'max_image_score_error': float(np.max(np.abs(float_scores-quant_scores))),
               'rtl_expected': expected.tolist(), 'fixture_sha256': hashlib.sha256((OUT/f'{category}_rtl.npz').read_bytes()).hexdigest()}
        rows.append(row)
        (OUT / 'profiles_and_results.json').write_text(json.dumps(rows, indent=2))
        print('SAVED', category, 'scale', scale, 'frozen', row['int8_frozen'], flush=True)
    print('INT8_PREPARATION_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
