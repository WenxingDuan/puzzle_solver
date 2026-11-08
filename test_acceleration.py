"""GPU equivalence checks. Add --photos and --baseline-ref COMMIT for real-photo regression."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
import types
from unittest.mock import patch

import cv2
import numpy as np

import vision


def formula_checks(gpu):
    rng = np.random.default_rng(42)
    worst = 0
    for edge in (False, True):
        r = rng.random((90, 110), dtype=np.float32)
        valid = np.ones_like(r); valid[30:37, 45:52] = 0
        stone = np.ones_like(r); stone[:12, :35] = 0
        if edge:
            stone[42, 65] = 0
            valid[40:47, 60:70] = 0
        shapes = [(20, 10)] if edge else [(15, 17), (18, 14), (21, 23)]
        templates = []
        for h, w in shapes:
            patch = r[40:40+h, 60:60+w].copy()
            mask = np.ones((h, w), np.float32)
            if not edge: mask[:3, :4] = 0
            templates.append((patch, mask))
        maps = gpu.score_maps(r, valid, stone, templates).get()
        peaks = gpu.peaks(r, valid, stone, templates)
        for i, (patch, mask) in enumerate(templates):
            cpu = vision.masked_scores(r, valid, patch, mask)
            coverage = np.rint(cv2.matchTemplate(stone, mask, cv2.TM_CCORR)) / mask.sum()
            cpu[coverage < .995] = -1
            cuda = maps[i, :cpu.shape[0], :cpu.shape[1]]
            error = float(np.max(np.abs(cpu - cuda))); worst = max(worst, error)
            assert np.array_equal(cpu == -1, cuda == -1), 'GPU changed valid positions'
            assert error < 1e-4, error
            assert cv2.minMaxLoc(cpu)[3] == peaks[i][1] == (60, 40), peaks[i]
            assert np.all(maps[i, cpu.shape[0]:] == -1) and np.all(maps[i, :, cpu.shape[1]:] == -1)
    # An empty/flat reference must not create a false high-scoring match.
    for value in (0, 1):
        r = np.full((60, 80), value, np.float32)
        patch = rng.random((15, 17), dtype=np.float32); mask = np.ones_like(patch)
        scores = gpu.score_maps(r, np.ones_like(r), np.ones_like(r), [(patch, mask)]).get()
        assert np.all(np.isfinite(scores)) and scores.max() < .01
    print(f'PASS full score maps, mixed sizes, excluded pixels and exact edge thresholds; max error {worst:.8g}', flush=True)
    return worst


def baseline_module(revision):
    if not revision:
        return vision
    source = subprocess.check_output(['git', 'show', f'{revision}:vision.py'])
    module = types.ModuleType('cpu_baseline')
    exec(compile(source, f'{revision}:vision.py', 'exec'), module.__dict__)
    for name in ('decode', 'segment', 'preprocess', 'reference', 'inside_stone', 'rootsift', 'feature_matches', 'rotated_pair'):
        original = ast.parse(source).body
        original = next(node for node in original if isinstance(node, ast.FunctionDef) and node.name == name)
        current = ast.parse(inspect.getsource(getattr(vision, name))).body[0]
        assert ast.dump(original) == ast.dump(current), f'Original {name} changed'
    print('PASS preprocessing, SIFT, transformations and final silhouette checks match the original source', flush=True)
    return module


def fallback_checks(gpu):
    rng = np.random.default_rng(7)
    binary = np.uint8(rng.random((90, 110)) > .7) * 255
    full = np.full_like(binary, 255)
    prepared = {'binary': binary[40:60, 60:76].copy(), 'mask': full[:20, :16].copy()}
    ref = (None, binary, full, None, None, full)
    cpu = vision.template_matches(prepared, ref, backend='cpu')
    assert cpu and cpu[0]['backend'] == 'cpu'
    for error in (ImportError('test missing CUDA'), RuntimeError('test GPU failure')):
        with patch.object(gpu, 'peaks', side_effect=error):
            assert vision.template_matches(prepared, ref, backend='auto') == cpu
            try:
                vision.template_matches(prepared, ref, backend='cuda')
            except type(error):
                pass
            else:
                raise AssertionError('Forced CUDA silently fell back to CPU')
    print('PASS automatic CPU fallback and strict CUDA errors', flush=True)


def fingerprint(templates):
    return [(patch.shape, hashlib.sha256(patch.tobytes() + mask.tobytes()).hexdigest()) for patch, mask in templates]


def photo_checks(gpu, revision):
    root = Path(__file__).resolve().parent
    fixtures = root / '.runtime/fixtures'
    names = ['mat-close.jpg', 'mat-wide.jpg', 'second-piece.jpg', 'hand.jpg', 'glass.jpg']
    assert all((fixtures / name).is_file() for name in names), 'Local .runtime/fixtures photos are required'
    ref_data = (root / 'assets/rosetta-reference.png').read_bytes()
    baseline = baseline_module(revision)
    baseline.reference(ref_data); vision.reference(ref_data)
    im = vision.decode(ref_data)
    red = np.uint8((im[:, :, 2].astype(int) > im[:, :, 1].astype(int) + 60) &
                  (im[:, :, 2].astype(int) > im[:, :, 0].astype(int) + 60)) * 255
    boxes = sorted(cv2.boundingRect(c) for c in vision.contours(red) if cv2.contourArea(c) > 500)
    assert len(boxes) == 2
    original_scores, original_peaks = baseline.masked_scores, gpu.peaks
    records = []
    for name in names:
        data = (fixtures / name).read_bytes()
        cpu_inputs, cuda_inputs = [], []

        def observe_cpu(r, valid, patch, mask):
            cpu_inputs.extend(fingerprint([(patch, mask)]))
            return original_scores(r, valid, patch, mask)

        def observe_cuda(r, valid, stone, templates):
            cuda_inputs.extend(fingerprint(templates))
            return original_peaks(r, valid, stone, templates)

        baseline.masked_scores, gpu.peaks = observe_cpu, observe_cuda
        try:
            start = time.perf_counter()
            before = baseline.preprocess(data)
            cpu, _ = baseline.locate(before, ref_data) if revision else baseline.locate(before, ref_data, backend='cpu')
            cpu_seconds = time.perf_counter() - start
            start = time.perf_counter()
            after = vision.preprocess(data)
            cuda, _ = vision.locate(after, ref_data, backend='cuda')
            cuda_seconds = time.perf_counter() - start
        finally:
            baseline.masked_scores, gpu.peaks = original_scores, original_peaks
        assert np.array_equal(before['mask'], after['mask']) and np.array_equal(before['binary'], after['binary']), name
        assert cpu_inputs == cuda_inputs and len(cpu_inputs) == 576, (name, len(cpu_inputs), len(cuda_inputs))
        assert cpu and cuda and cuda[0]['backend'] == 'cuda', name
        error = float(np.max(np.abs(np.array(cpu[0]['polygon']) - np.array(cuda[0]['polygon']))))
        assert error < 1e-4 and abs(cpu[0]['score'] - cuda[0]['score']) <= .001, (name, cpu[0], cuda[0])
        center = np.mean(cuda[0]['polygon'], axis=0)
        x, y, w, h = boxes[1 if name == 'second-piece.jpg' else 0]
        assert x <= center[0] <= x+w and y <= center[1] <= y+h, (name, center)
        assert all(vision.inside_stone(after['mask'], candidate['polygon'], vision.reference(ref_data)[5]) for candidate in cuda)
        record = {'photo': name, 'cpu_seconds': round(cpu_seconds, 4), 'cuda_seconds': round(cuda_seconds, 4),
                  'speedup': round(cpu_seconds/cuda_seconds, 2), 'templates': len(cpu_inputs),
                  'top_polygon_max_error': error, 'cpu_top': cpu[0], 'cuda_top': cuda[0]}
        records.append(record)
        print(f'PASS {name}: 576 identical templates; same first position; CPU {cpu_seconds:.3f}s, CUDA {cuda_seconds:.3f}s', flush=True)
    centers = [np.mean(record['cuda_top']['polygon'], axis=0) for record in records if record['photo'] != 'second-piece.jpg']
    assert np.max(np.linalg.norm(centers - np.mean(centers, axis=0), axis=1)) < 15
    print('PASS all four views of the same piece agree with the user-confirmed location', flush=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photos', action='store_true')
    parser.add_argument('--baseline-ref', help='Compare against vision.py from this Git commit')
    args = parser.parse_args()
    import gpu_match
    fallback_checks(gpu_match)
    report = {'baseline_ref': args.baseline_ref, 'gpu': gpu_match.cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'formula_max_error': formula_checks(gpu_match)}
    if args.photos: report['photos'] = photo_checks(gpu_match, args.baseline_ref)
    folder = Path(__file__).resolve().parent / '.runtime'
    folder.mkdir(exist_ok=True)
    (folder / 'acceleration-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('All acceleration checks passed.', flush=True)
