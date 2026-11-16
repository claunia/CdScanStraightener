#!/usr/bin/env python3
"""Compares straightened crops against hand-made references.

For every PNG present in both folders, finds the quarter turn and offset that place the
reference inside (or around) the machine output, and prints how many millimetres the
machine output keeps beyond the reference on each side of the reference (positive: too
much kept, negative: cut too far). A non-zero quarter turn means the orientation differs.

    python3 tools/compare-crops.py <machine-dir> <reference-dir> [--dpi 600] [--jobs N]

Development aid only; needs numpy, scipy and Pillow.
"""
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numpy.fft import rfft2, irfft2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load(path, k):
    im = Image.open(path).convert('L')
    w, h = im.size
    return np.asarray(im.resize((max(1, w // k), max(1, h // k)), Image.BOX), dtype=np.float32)


def ncc(scene, tmpl):
    """Best normalized cross-correlation of tmpl fully inside scene: (score, (y, x))."""
    H, W = scene.shape[0] + tmpl.shape[0], scene.shape[1] + tmpl.shape[1]
    t = tmpl - tmpl.mean()
    fs = rfft2(scene, (H, W))
    fo = np.conj(rfft2(np.ones_like(t), (H, W)))
    c = irfft2(fs * np.conj(rfft2(t, (H, W))), (H, W))
    s1 = irfft2(fs * fo, (H, W))
    s2 = irfft2(rfft2(scene * scene, (H, W)) * fo, (H, W))
    var = np.maximum(s2 - s1 * s1 / t.size, 1e-3)
    sc = c / np.sqrt(var) / np.sqrt((t * t).sum())
    sc[scene.shape[0] - tmpl.shape[0] + 1:, :] = -1
    sc[:, scene.shape[1] - tmpl.shape[1] + 1:] = -1
    i = np.unravel_index(np.argmax(sc), sc.shape)
    return float(sc[i]), i


def padded(img, by):
    out = np.full((img.shape[0] + 2 * by, img.shape[1] + 2 * by), img.mean(), np.float32)
    out[by:by + img.shape[0], by:by + img.shape[1]] = img
    return out


def locate(machine, ref, margin):
    """Offset (y, x) of ref's top-left in machine, using ref's interior as the template."""
    tmpl = ref[margin:-margin, margin:-margin]
    pad = margin * 2
    score, (y, x) = ncc(padded(machine, pad), tmpl)
    return score, y - pad - margin, x - pad - margin


def compare(args):
    name, mdir, rdir, dpi = args
    coarse_k, fine_k = 16, 3
    mc, rc = load(os.path.join(mdir, name), coarse_k), load(os.path.join(rdir, name), coarse_k)
    best = max(((locate(np.rot90(mc, q), rc, 8)[0], q) for q in range(4)))
    q = best[1]
    mf = np.rot90(load(os.path.join(mdir, name), fine_k), q)
    rf = load(os.path.join(rdir, name), fine_k)
    score, top, left = locate(mf, rf, 60)
    bottom = mf.shape[0] - (top + rf.shape[0])
    right = mf.shape[1] - (left + rf.shape[1])
    mm = lambda px: px * fine_k / dpi * 25.4
    return name, score, q, mm(top), mm(bottom), mm(left), mm(right)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('machine')
    p.add_argument('reference')
    p.add_argument('--dpi', type=float, default=600)
    p.add_argument('--jobs', type=int, default=os.cpu_count())
    a = p.parse_args()
    names = sorted(set(f for f in os.listdir(a.machine) if f.lower().endswith('.png')) &
                   set(f for f in os.listdir(a.reference) if f.lower().endswith('.png')))
    worst, total, turned = 0.0, 0.0, 0
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, score, q, t, b, l, r in ex.map(compare, [(n, a.machine, a.reference, a.dpi) for n in names]):
            flag = '' if q == 0 else f'  WRONG ORIENTATION (off by {q * 90}°)'
            turned += q != 0
            print(f'{name}  ncc {score:.3f}  T {t:+5.1f}  B {b:+5.1f}  L {l:+5.1f}  R {r:+5.1f} mm{flag}')
            for v in (t, b, l, r):
                worst = max(worst, abs(v))
                total += abs(v)
    if names:
        print(f'{len(names)} files: mean |side| {total / (4 * len(names)):.2f} mm, worst {worst:.1f} mm, '
              f'{turned} wrongly oriented')


if __name__ == '__main__':
    sys.exit(main())
