"""CUDA FFT evaluation of the same weighted NCC used by vision.masked_scores."""
from functools import lru_cache
import os

if os.name == 'nt':
    # NVRTC reductions avoid CUB's external MSVC requirement on Windows.
    os.environ.setdefault('CUPY_ACCELERATORS', '')
import cupy as cp
import numpy as np


_normalize = cp.ElementwiseKernel(
    'float32 total, float32 sr, float32 srr, float32 sq, float32 sqq, '
    'float32 cross, float32 coverage, float32 mass, int32 rows, int32 cols, int32 y, int32 x',
    'float32 score',
    '''
    float count = rintf(total);
    float covered = rintf(coverage);
    float n = fmaxf(count, 1.0f);
    float vr = fmaxf(srr - sr * sr / n, 0.001f);
    float vq = fmaxf(sqq - sq * sq / n, 0.001f);
    score = (cross - sr * sq / n) / sqrtf(vr * vq);
    if (y >= rows || x >= cols || count < 0.65f * mass ||
        covered / mass < 0.995f || !isfinite(score)) score = -1.0f;
    ''', 'puzzle_weighted_ncc', options=('--fmad=false',))


@lru_cache(maxsize=2)
def _reference_spectra(r_bytes, valid_bytes, stone_bytes, shape, fft_shape):
    r = np.frombuffer(r_bytes, np.float32).reshape(shape)
    valid = np.frombuffer(valid_bytes, np.float32).reshape(shape)
    stone = np.frombuffer(stone_bytes, np.float32).reshape(shape)
    fields = cp.asarray(np.stack([valid, r * valid, r * r * valid, stone]))
    return cp.fft.rfft2(fields, s=fft_shape, axes=(-2, -1))


def score_maps(r, valid, stone, templates):
    """Return one full, valid-position score map per (patch, mask), as float32."""
    cp.cuda.Device(0).use()
    if not templates:
        raise ValueError('No templates supplied')
    height, width = r.shape
    max_h = max(p.shape[0] for p, _ in templates)
    max_w = max(p.shape[1] for p, _ in templates)
    fft_shape = (1 << (height + max_h - 2).bit_length(),
                 1 << (width + max_w - 2).bit_length())
    spectra = _reference_spectra(r.tobytes(), valid.tobytes(), stone.tobytes(), r.shape, fft_shape)
    kernels = np.zeros((len(templates), 3, max_h, max_w), np.float32)
    masses, rows, cols = [], [], []
    for i, (patch, mask) in enumerate(templates):
        h, w = patch.shape
        kernels[i, 0, :h, :w] = mask
        kernels[i, 1, :h, :w] = patch * mask
        kernels[i, 2, :h, :w] = patch * patch * mask
        masses.append(mask.sum())
        rows.append(height - h + 1)
        cols.append(width - w + 1)
    query_fft = cp.fft.rfft2(cp.asarray(kernels), s=fft_shape, axes=(-2, -1)).conj()
    correlations = cp.empty((7, len(templates), height, width), cp.float32)
    # Positive lags of FFT correlation equal cv2.matchTemplate(TM_CCORR).
    for i, (source, query) in enumerate([(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 1), (3, 0)]):
        full = cp.fft.irfft2(spectra[source] * query_fft[:, query], s=fft_shape, axes=(-2, -1))
        correlations[i] = full[:, :height, :width]
    return _normalize(*correlations,
        cp.asarray(masses, dtype=cp.float32)[:, None, None],
        cp.asarray(rows, dtype=cp.int32)[:, None, None],
        cp.asarray(cols, dtype=cp.int32)[:, None, None],
        cp.arange(height, dtype=cp.int32)[None, :, None],
        cp.arange(width, dtype=cp.int32)[None, None, :])


def peaks(r, valid, stone, templates):
    """Evaluate all templates; transfer only each maximum and its location to CPU."""
    result = []
    # ponytail: batches of 16 bound FFT workspace memory; tune only for smaller GPUs if needed.
    for start in range(0, len(templates), 16):
        maps = score_maps(r, valid, stone, templates[start:start + 16])
        flat = maps.reshape(maps.shape[0], -1)
        indices = cp.argmax(flat, axis=1)
        values = cp.take_along_axis(flat, indices[:, None], axis=1)[:, 0]
        cpu_values, cpu_indices = values.get(), indices.get()
        for value, index in zip(cpu_values, cpu_indices):
            y, x = divmod(int(index), r.shape[1])
            result.append((float(value), (x, y)))
    return result
