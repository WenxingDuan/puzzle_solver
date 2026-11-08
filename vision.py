"""Foreground extraction, lighting normalization, and masked puzzle matching."""
from functools import lru_cache
from io import BytesIO
import math

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

cv2.setNumThreads(2)


def decode(data, max_side=2000, alpha=False):
    try:
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > 40000000:
                raise ValueError('图片超过 4000 万像素，请缩小后重试。')
            source = ImageOps.exif_transpose(source).convert('RGBA' if alpha else 'RGB')
            source.thumbnail((max_side, max_side))
            return cv2.cvtColor(np.array(source), cv2.COLOR_RGBA2BGRA if alpha else cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('无法读取图片，请使用 JPG、PNG 或 WebP。') from error


def png(image):
    ok, result = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('图片处理失败。')
    return result.tobytes()


def contours(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def candidate_contours(mask, center):
    h, w = mask.shape
    result = []
    for contour in contours(mask):
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        distance = math.hypot((x + width / 2) / w - center[0], (y + height / 2) / h - center[1])
        if (.004 * w * h < area < .35 * w * h and area / (width * height) > .35
                and x > .025 * w and x + width < .975 * w and y > 2 and y + height < h - 2
                and distance < .32):
            result.append((area * math.exp(-12 * distance * distance), contour))
    return result


@lru_cache(maxsize=6)
def segment(data, box=None):
    original = decode(data)
    h0, w0 = original.shape[:2]
    image = cv2.resize(original, (round(w0 * 720 / max(h0, w0)), round(h0 * 720 / max(h0, w0))))
    h, w = image.shape[:2]
    if box:
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        rectangle = (max(1, round(x1 * w)), max(1, round(y1 * h)),
                     max(2, round((x2 - x1) * w)), max(2, round((y2 - y1) * h)))
        rectangle = (rectangle[0], rectangle[1], min(rectangle[2], w - rectangle[0] - 1),
                     min(rectangle[3], h - rectangle[1] - 1))
        if min(rectangle[2:]) < 12:
            raise ValueError('框选范围过小，请把整块拼图圈进去。')
        gc = np.zeros((h, w), np.uint8)
        cv2.grabCut(image, gc, rectangle, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
        rough = np.uint8(np.isin(gc, [1, 3])) * 255
        found = contours(rough)
        if not found:
            raise ValueError('框内没有找到拼图，请稍微放大框选范围。')
        contour = min(found, key=lambda c: -cv2.contourArea(c) * math.exp(-12 * (
            ((cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2) / w - center[0]) ** 2
            + ((cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] / 2) / h - center[1]) ** 2)))
    else:
        # ponytail: centered-object heuristic; use the user's loose rectangle when clutter wins.
        center = (.5, .56)
        lab = cv2.cvtColor(cv2.GaussianBlur(image, (7, 7), 0), cv2.COLOR_BGR2LAB)
        cv2.setRNGSeed(17)
        _, labels, _ = cv2.kmeans(lab.reshape(-1, 3).astype(np.float32), 6, None,
                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .4),
                                  3, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(h, w)
        candidates = []
        for label in range(6):
            rough = np.uint8(labels == label) * 255
            rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            candidates.extend(candidate_contours(rough, center))
        if not candidates:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            deviation = np.sqrt(np.maximum(cv2.blur(gray * gray, (5, 5)) - cv2.blur(gray, (5, 5)) ** 2, 0))
            rough = np.uint8(deviation > 16) * 255
            rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            candidates = candidate_contours(rough, center)
        if not candidates:
            raise ValueError('没有找到完整拼图。请在照片上拖动框选整块拼图，再试一次。')
        contour = max(candidates, key=lambda item: item[0])[1]
        rough = np.zeros((h, w), np.uint8)
        cv2.drawContours(rough, [contour], -1, 255, -1)
        gc = np.zeros((h, w), np.uint8)
        gc[cv2.dilate(rough, np.ones((31, 31), np.uint8)) > 0] = cv2.GC_PR_BGD
        gc[rough > 0] = cv2.GC_PR_FGD
        gc[cv2.erode(rough, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_FGD
        cv2.grabCut(image, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
        rough = np.uint8(np.isin(gc, [1, 3])) * 255
        contour = max(contours(rough), key=cv2.contourArea)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    # This puzzle is neutral grey/white; reject colored skin and reflected yellow furniture.
    color = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)
    mask[color[:, :, 1] > 60] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contour = max(contours(mask), key=cv2.contourArea)
    mask[:] = 0
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    contour = max(contours(mask), key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    outlined = original.copy()
    cv2.drawContours(outlined, [contour], -1, (45, 190, 50), max(2, round(max(h0, w0) / 500)))
    crop = original[y:y + height, x:x + width].copy()
    mask = mask[y:y + height, x:x + width].copy()
    ratio = min(1, 512 / max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, crop.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    return crop, mask, outlined, (x, y, width, height)


def preprocess(data, strength=8, box=None):
    crop, mask, outlined, bounds = segment(data, box)
    # Blur only foreground samples; a dark table must not brighten the piece's edge.
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    weight = mask.astype(np.float32) / 255
    sigma = max(6, max(gray.shape) / 22)
    baseline = cv2.GaussianBlur(gray * weight, (0, 0), sigma) / np.maximum(
        cv2.GaussianBlur(weight, (0, 0), sigma), .001)
    contrast = gray - baseline
    valid = cv2.erode(mask, np.ones((11, 11), np.uint8))
    values = np.clip(contrast[valid > 0] + 80, 0, 255).astype(np.uint8)
    if values.size < 100:
        raise ValueError('拼图有效区域过小，请靠近拍摄。')
    threshold = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] - 80
    binary = np.uint8(contrast > max(2, threshold + strength)) * 255
    binary[valid == 0] = 0
    binary = cv2.medianBlur(binary, 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < max(3, np.count_nonzero(valid) * .00025):
            binary[labels == label] = 0
    binary[valid == 0] = 0
    fraction = np.count_nonzero(binary) / max(1, np.count_nonzero(valid))
    cutout = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    cutout[:, :, 3] = mask
    preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGRA)
    preview[:, :, 3] = mask
    return {'cutout': cutout, 'binary_preview': preview, 'binary': binary, 'mask': valid,
            'outlined': outlined, 'bounds': bounds, 'white_fraction': fraction,
            'threshold': round(float(threshold + strength), 1)}


@lru_cache(maxsize=2)
def reference(data):
    source = decode(data, alpha=True)
    opacity = source[:, :, 3:4].astype(np.float32) / 255
    image = np.uint8(np.round(source[:, :, :3] * opacity + 255 * (1 - opacity)))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    annotation = np.uint8((hsv[:, :, 1] > 100) & ((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170))) * 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.inpaint(gray, annotation, 3, cv2.INPAINT_TELEA)
    image = cv2.inpaint(image, annotation, 3, cv2.INPAINT_TELEA)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    stone = np.zeros_like(gray)
    found = contours(np.uint8(binary == 0) * 255)
    if not found:
        raise ValueError('参考图中没有找到深色石碑区域。')
    cv2.drawContours(stone, [max(found, key=cv2.contourArea)], -1, 255, -1)
    stone[source[:, :, 3] < 128] = 0
    valid = stone.copy()
    valid[cv2.dilate(annotation, np.ones((5, 5), np.uint8)) > 0] = 0
    detector = cv2.SIFT_create(nfeatures=40000, contrastThreshold=.01, edgeThreshold=15)
    keys, descriptors = detector.detectAndCompute(gray, valid)
    preview = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    preview[:, :, 3] = stone
    return preview, binary, valid, keys, descriptors, stone


def inside_stone(piece_mask, polygon, stone):
    """Check the projected piece itself; its rectangular corners may be empty."""
    points = np.asarray(polygon, np.float32)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)) or cv2.contourArea(points) < 1:
        return False
    if np.any(points < 0) or np.any(points[:, 0] > stone.shape[1]) or np.any(points[:, 1] > stone.shape[0]):
        return False
    height, width = piece_mask.shape
    transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [width, 0], [width, height], [0, height]]), points)
    projected = cv2.warpPerspective(piece_mask, transform, stone.shape[1::-1], flags=cv2.INTER_NEAREST) > 0
    count = np.count_nonzero(projected)
    return bool(count and np.count_nonzero(projected & (stone > 0)) / count >= .995)


def rootsift(descriptors):
    return np.sqrt(descriptors / (descriptors.sum(axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def feature_matches(prepared, ref):
    image, _, _, reference_keys, reference_descriptors, stone = ref
    if reference_descriptors is None:
        return []
    detector = cv2.SIFT_create(contrastThreshold=.01, edgeThreshold=15)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=96))
    matcher.add([rootsift(reference_descriptors)])
    matcher.train()
    mask = prepared['mask']
    b = np.where(mask > 0, prepared['binary'], 0).astype(np.uint8)
    found = []
    for size in [70, 100, 140, 200, 300]:
        scale = size / max(b.shape)
        query = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        valid = cv2.resize(mask, query.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        keys, descriptors = detector.detectAndCompute(query, cv2.erode(valid, np.ones((3, 3), np.uint8)))
        if descriptors is None:
            continue
        neighbors = matcher.knnMatch(rootsift(descriptors), k=2)
        for ratio in [.8, .9, .95]:
            good = [a for a, z in neighbors if a.distance < ratio * z.distance]
            if len(good) < 4:
                continue
            src = np.float32([keys[m.queryIdx].pt for m in good])
            dst = np.float32([reference_keys[m.trainIdx].pt for m in good])
            transform, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                ransacReprojThreshold=3, maxIters=4000)
            if transform is None:
                continue
            inside = inliers.ravel().astype(bool)
            unique = len(np.unique(np.round(src[inside] / 2).astype(int), axis=0))
            span = max(query.shape) * np.linalg.norm(transform[0, :2])
            if unique < 4 or not .02 * max(image.shape[:2]) < span < .18 * max(image.shape[:2]):
                continue
            if cv2.contourArea(cv2.convexHull(src[inside])) < .035 * query.size:
                continue
            polygon = cv2.transform(np.float32([[[0, 0], [query.shape[1], 0],
                [query.shape[1], query.shape[0]], [0, query.shape[0]]]]), transform)[0]
            if np.any(polygon < 0) or np.any(polygon[:, 0] >= image.shape[1]) or np.any(polygon[:, 1] >= image.shape[0]):
                continue
            if not inside_stone(mask, polygon, stone):
                continue
            found.append({'polygon': polygon.tolist(), 'score': round(unique / len(good), 3),
                          'inliers': unique, 'method': 'features', 'reliable': unique >= 5})
    return sorted(found, key=lambda item: (item['inliers'], item['score']), reverse=True)


def masked_scores(r, valid_ref, patch, mask):
    """Weighted NCC: exclude background and reference annotations on both sides."""
    total = cv2.matchTemplate(valid_ref, mask, cv2.TM_CCORR)
    n = np.maximum(total, 1)
    sum_r = cv2.matchTemplate(r * valid_ref, mask, cv2.TM_CCORR)
    sum_rr = cv2.matchTemplate(r * r * valid_ref, mask, cv2.TM_CCORR)
    sum_q = cv2.matchTemplate(valid_ref, patch * mask, cv2.TM_CCORR)
    sum_qq = cv2.matchTemplate(valid_ref, patch * patch * mask, cv2.TM_CCORR)
    cross = cv2.matchTemplate(r * valid_ref, patch * mask, cv2.TM_CCORR)
    variance = np.maximum(sum_rr - sum_r * sum_r / n, .001) * np.maximum(sum_qq - sum_q * sum_q / n, .001)
    result = (cross - sum_r * sum_q / n) / np.sqrt(variance)
    result[(total < .65 * mask.sum()) | ~np.isfinite(result)] = -1
    return result


def rotated_pair(binary, mask, angle, aspect):
    h, w = binary.shape
    width = max(1, round(w * aspect))
    b = cv2.resize(binary, (width, h), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (width, h), interpolation=cv2.INTER_NEAREST)
    transform = cv2.getRotationMatrix2D((width / 2, h / 2), angle, 1)
    cosine, sine = abs(transform[0, 0]), abs(transform[0, 1])
    new_width, new_height = round(width * cosine + h * sine), round(h * cosine + width * sine)
    transform[0, 2] += (new_width - width) / 2
    transform[1, 2] += (new_height - h) / 2
    b = cv2.warpAffine(b, transform, (new_width, new_height))
    m = cv2.warpAffine(m, transform, (new_width, new_height), flags=cv2.INTER_NEAREST)
    corners = cv2.transform(np.float32([[[0, 0], [width, 0], [width, h], [0, h]]]), transform)[0]
    return b, m, corners


def template_matches(prepared, ref):
    _, binary, valid_ref, _, _, stone = ref
    factor = min(1, 900 / max(binary.shape))
    r = cv2.resize(binary, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    r = cv2.GaussianBlur(r, (3, 3), .6)
    silhouette = cv2.resize(stone, r.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255
    valid_ref = cv2.resize(valid_ref, r.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255
    candidates = []
    mask = prepared['mask']
    b = np.where(mask > 0, prepared['binary'], 0).astype(np.uint8)
    expected = math.sqrt(np.count_nonzero(silhouette) / 800 * b.size / max(1, np.count_nonzero(mask)))
    # ponytail: +/-10 degrees and 10% stretch cover these phone photos; use homography for stronger perspective.
    for angle, aspect in [(base + delta, aspect) for base in range(0, 360, 90)
                          for delta in (-10, 0, 10) for aspect in (.9, 1, 1.1)]:
        rb, rm, corners = rotated_pair(b, mask, angle, aspect)
        for size in np.linspace(max(25, expected * .8), min(140, expected * 2), 16):
            scale = size / max(rb.shape)
            dims = (max(2, round(rb.shape[1] * scale)), max(2, round(rb.shape[0] * scale)))
            if dims[0] >= r.shape[1] or dims[1] >= r.shape[0]:
                continue
            patch = cv2.resize(rb, dims, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            m = cv2.resize(rm, dims, interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255
            n = m.sum()
            if n < 100:
                continue
            if np.std(patch[m > 0]) < .01:
                continue
            scores = masked_scores(r, valid_ref, patch, m)
            coverage = cv2.matchTemplate(silhouette, m, cv2.TM_CCORR) / n
            scores[coverage < .995] = -1
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            if score > .15:
                polygon = (corners * np.float32([dims[0] / rb.shape[1], dims[1] / rb.shape[0]]) + [x, y]) / factor
                candidates.append({'polygon': polygon.tolist(), 'score': round(float(score), 3),
                                   'method': 'template', 'reliable': False, 'inliers': 0})
    return sorted(candidates, key=lambda item: item['score'], reverse=True)


def locate(prepared, ref_data):
    if not .008 < prepared['white_fraction'] < .65:
        return [], '黑白结果的文字过少或白色过多，请调整分割强度，确认去背景预览后重试。'
    ref = reference(ref_data)
    candidates = feature_matches(prepared, ref)
    if not candidates:
        candidates = template_matches(prepared, ref)
    selected = []
    for candidate in candidates:
        if not inside_stone(prepared['mask'], candidate['polygon'], ref[5]):
            continue
        center = np.mean(candidate['polygon'], axis=0)
        if any(np.linalg.norm(center - np.mean(old['polygon'], axis=0)) < 35 for old in selected):
            continue
        selected.append(candidate)
        if len(selected) == 3:
            break
    if selected and selected[0]['reliable']:
        message = '找到有多处文字对应的位置，请核对下方叠加图。'
    elif selected and selected[0]['score'] >= .5 and (len(selected) == 1 or selected[0]['score'] - selected[1]['score'] >= .06):
        message = '已找到字形较相近的候选区域，请对照放大图核实。'
    else:
        message = '匹配依据不足，下方仅列出候选位置。可调整黑白强度或改善拍摄后重试。'
    return selected, message
