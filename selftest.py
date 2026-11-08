"""Run with python selftest.py. Local photo fixtures stay under the private .runtime directory."""
from pathlib import Path
import json
import sys
import cv2
import numpy as np
import vision


def overlap(a, b):
    x = max(a[0], b[0]); y = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2]); bottom = min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - x) * max(0, bottom - y)
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


def main():
    sprite = np.zeros((160, 160, 4), np.uint8)
    cv2.fillPoly(sprite, [np.int32([[50, 10], [140, 20], [150, 130], [20, 150], [20, 70]])], (20, 20, 20, 255))
    cv2.putText(sprite, 'ABC', (45, 90), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255, 255), 1)
    sprite[70:80, 75:85, 3] = 0
    hidden = sprite[:, :, 3] == 0
    sprite[hidden, :3] = 0
    ref_a = vision.reference(vision.png(sprite))
    sprite[hidden, :3] = np.random.default_rng(42).integers(0, 256, (np.count_nonzero(hidden), 3), dtype=np.uint8)
    ref_b = vision.reference(vision.png(sprite))
    assert np.array_equal(ref_a[2], ref_b[2]) and np.array_equal(ref_a[1], ref_b[1])
    assert not np.any(ref_b[5][hidden]) and not np.any(ref_b[0][:, :, 3][hidden])
    print('PASS transparent corners and holes are excluded regardless of hidden RGB values')

    stone = np.zeros((120, 120), np.uint8); stone[30:90, 30:90] = 255; stone[30:50, 70:90] = 0
    query = np.full((40, 40), 255, np.uint8); query[:20, 20:] = 0
    polygon = [[50, 30], [90, 30], [90, 70], [50, 70]]
    assert vision.inside_stone(query, polygon, stone), 'Empty rectangular corners rejected an actual edge piece'
    assert not vision.inside_stone(np.full_like(query, 255), polygon, stone), 'A piece outside the stone was accepted'
    assert not vision.inside_stone(query, [[0, 0], [40, 0], [40, 40], [0, 40]], stone)
    print('PASS real edge-piece shapes are accepted while outside-stone candidates are rejected')

    rng = np.random.default_rng(81)
    reference = np.uint8(rng.random((900, 700)) > .78) * 255
    patch = reference[320:368, 250:298].copy()
    mask = np.full_like(patch, 255); mask[:5] = 0; mask[:, :5] = 0
    noisy = patch.copy(); noisy[mask == 0] = 255 - noisy[mask == 0]
    r = reference.astype(np.float32) / 255
    valid = np.ones_like(r); valid[330:340, 260:270] = 0
    a = vision.masked_scores(r, valid, patch.astype(np.float32) / 255, mask.astype(np.float32) / 255)
    b = vision.masked_scores(r, valid, noisy.astype(np.float32) / 255, mask.astype(np.float32) / 255)
    altered = r.copy(); altered[valid == 0] = 1 - altered[valid == 0]
    c = vision.masked_scores(altered, valid, patch.astype(np.float32) / 255, mask.astype(np.float32) / 255)
    assert np.allclose(a, b) and np.allclose(a, c), 'Excluded background or annotation changed the match'
    _, score, _, point = cv2.minMaxLoc(a)
    assert point == (250, 320) and score > .99, (point, score)
    print('PASS masked matching finds a known patch and ignores background and reference annotations')

    fixtures = Path(__file__).resolve().parent / '.runtime/fixtures'
    expected = {'mat-close.jpg': (280, 690, 610, 720), 'mat-wide.jpg': (390, 705, 420, 520),
                'second-piece.jpg': (550, 735, 270, 390), 'hand.jpg': (405, 620, 350, 435),
                'glass.jpg': (475, 890, 325, 410)}
    for name, bounds in expected.items():
        path = fixtures / name
        if not path.exists():
            print('SKIP local photo fixture', name)
            continue
        prepared = vision.preprocess(path.read_bytes())
        assert overlap(prepared['bounds'], bounds) > .78, (name, prepared['bounds'])
        assert .04 < prepared['white_fraction'] < .45, (name, prepared['white_fraction'])
        assert np.all(prepared['binary'][prepared['mask'] == 0] == 0), name
        assert 0 in prepared['cutout'][:, :, 3], 'Cutout is not transparent'
        stricter = vision.preprocess(path.read_bytes(), 20)
        assert stricter['white_fraction'] <= prepared['white_fraction'], name
        print('PASS foreground, binary text, transparency and strength:', name)
    if '--match-photos' in sys.argv or '--verify-existing' in sys.argv:
        data = (Path(__file__).resolve().parent / 'assets/rosetta-reference.png').read_bytes()
        im = vision.decode(data)
        red = np.uint8((im[:, :, 2].astype(int) > im[:, :, 1].astype(int) + 60) &
                      (im[:, :, 2].astype(int) > im[:, :, 0].astype(int) + 60)) * 255
        rectangles = sorted([cv2.boundingRect(c) for c in vision.contours(red) if cv2.contourArea(c) > 500])
        assert len(rectangles) == 2, rectangles
        existing = json.loads((fixtures.parent / 'real-matches.json').read_text(encoding='utf-8')) if '--verify-existing' in sys.argv else None
        centers = []
        for i, name in enumerate(expected):
            matches = existing[i]['matches'] if existing else vision.locate(vision.preprocess((fixtures / name).read_bytes()), data)[0]
            assert matches, name
            center = np.mean(matches[0]['polygon'], axis=0)
            x, y, w, h = rectangles[1 if name == 'second-piece.jpg' else 0]
            assert x <= center[0] <= x + w and y <= center[1] <= y + h, (name, center, (x, y, w, h))
            if name != 'second-piece.jpg': centers.append(center)
            print('PASS first candidate inside user-confirmed reference box:', name)
        assert np.max(np.linalg.norm(np.array(centers) - np.mean(centers, axis=0), axis=1)) < 15
        print('PASS all four views of the same piece agree')
    print('All checks passed.')


if __name__ == '__main__':
    main()
