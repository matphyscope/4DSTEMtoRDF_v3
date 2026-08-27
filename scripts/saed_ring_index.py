#!/usr/bin/env python3
"""
saed_ring_index.py
==================
Measure polycrystalline ring d-spacings from a *rendered* SAED / nanobeam
diffraction image (PNG/JPG/TIF) and index them against candidate phases.

Why this exists: the 4D-STEM dm4 only reaches q~1.1 1/A (long camera length),
too low for RDF. A separate selected-area pattern taken at a SHORTER camera
length reaches much higher q and shows sharp powder rings — ideal for indexing.
This reads the exported image, uses the blue scale bar for calibration, radially
integrates, finds the rings, and fits a cubic lattice constant for each
candidate (LiF / Li2O / Li2S are FCC; others listed for manual compare).

Usage
-----
    python scripts/saed_ring_index.py path/to/pattern.png
    # options (edit CONFIG below or pass flags):
    python scripts/saed_ring_index.py pattern.png --scale-q 0.5 --center 960,960

`--scale-q` is the 1/A that the scale bar represents: a "5 1/nm" bar = 0.5 1/A,
a "10 1/nm" bar = 1.0 1/A.  If the blue bar isn't auto-found, pass --bar-px N
(its length in pixels) or measure it in any image viewer.

Output: prints the ring d-spacings + best phase indexing, and writes
`<image>_rings.png` (radial profile + detected rings) and `<image>_rings.csv`.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

# candidate cubic phases (FCC): lattice constant a (A). Rings at d = a/sqrt(N),
# N = h^2+k^2+l^2 with h,k,l all-odd or all-even (rock salt / antifluorite).
CUBIC = {"LiF": 4.0263, "Li2O": 4.619, "Li2S": 5.716}
NALLOW = [3, 4, 8, 11, 12, 16, 19, 20, 24, 27, 32]
# non-cubic candidates: just list their strong powder d (A) for manual compare.
OTHER_D = {"Li3N": [3.872, 3.153, 2.451, 1.936, 1.821],
           "Li2CO3": [4.157, 3.033, 2.813, 2.492, 2.145, 1.873]}


def load_gray(path):
    from PIL import Image
    im = Image.open(path)
    rgb = np.asarray(im.convert("RGB"), float)
    gray = rgb.mean(2)
    return gray, rgb


def find_scale_bar_px(rgb):
    """Length (px) of the blue scale bar: blue high, red/green low."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    blue = (B > 120) & (B > R + 40) & (B > G + 20)
    if blue.sum() < 20:
        return None
    ys, xs = np.where(blue)
    # the bar is the widest horizontal run of blue (ignore the text glyphs)
    best = 0
    for y in range(ys.min(), ys.max() + 1):
        row = xs[ys == y]
        if row.size:
            best = max(best, row.max() - row.min())
    return float(best) if best > 20 else None


def find_center(gray):
    """Beam center = centroid of the saturated direct-beam core."""
    thr = np.percentile(gray, 99.7)
    ys, xs = np.where(gray >= thr)
    if xs.size < 5:
        h, w = gray.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def radial_median(gray, center, rmax, nbin):
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - center[0], yy - center[1])
    edges = np.linspace(0, rmax, nbin + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    g = gray.ravel()
    prof = np.full(nbin, np.nan)
    for b in range(nbin):
        sel = g[idx == b]
        if sel.size:
            prof[b] = np.median(sel)
    rc = 0.5 * (edges[:-1] + edges[1:])
    return rc, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--scale-q", type=float, default=0.5,
                    help="1/A represented by the scale bar (5 1/nm=0.5, 10 1/nm=1.0)")
    ap.add_argument("--bar-px", type=float, default=None, help="scale-bar length in px (override auto)")
    ap.add_argument("--center", type=str, default=None, help="cx,cy in px (override auto)")
    ap.add_argument("--nbin", type=int, default=600)
    ap.add_argument("--rmin-q", type=float, default=0.15, help="ignore rings below this q (beam)")
    ap.add_argument("--prom", type=float, default=0.06, help="peak prominence (frac of range)")
    args = ap.parse_args()

    gray, rgb = load_gray(args.image)
    h, w = gray.shape

    bar = args.bar_px or find_scale_bar_px(rgb)
    if not bar:
        print("!! scale bar not auto-found — pass --bar-px <length in px>"); sys.exit(2)
    qpp = args.scale_q / bar                       # 1/A per image pixel
    if args.center:
        cx, cy = (float(v) for v in args.center.split(","))
    else:
        cx, cy = find_center(gray)
    rmax = min(cx, cy, w - cx, h - cy)
    print(f"image {w}x{h} | scale bar {bar:.0f}px = {args.scale_q} 1/A -> {qpp:.5f} 1/A/px")
    print(f"center=({cx:.0f},{cy:.0f}) | q_max at edge ~ {rmax*qpp:.2f} 1/A (d_min {1/(rmax*qpp):.2f} A)")

    rc, prof = radial_median(gray, (cx, cy), rmax, args.nbin)
    q = rc * qpp
    good = np.isfinite(prof)
    q, prof = q[good], prof[good]
    # smooth background (broad) subtract to expose rings
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    base = uniform_filter1d(prof, max(5, args.nbin // 15))
    res = prof - base
    m = q >= args.rmin_q
    rng = np.ptp(res[m]) if m.any() else 1.0
    pk, _ = find_peaks(res * m, prominence=args.prom * rng, distance=4)
    rings_q = [float(q[i]) for i in pk]
    rings_d = [1.0 / x for x in rings_q]
    print(f"\ndetected rings ({len(rings_q)}):  q(1/A) | d(A)")
    for qi, di in zip(rings_q, rings_d):
        print(f"   {qi:.3f}   {di:.3f}")

    # index cubic candidates: fit a from each ring -> a=d*sqrt(N)
    print("\ncubic indexing (a = d*sqrt(N), N=allowed FCC):")
    for name, a_ref in CUBIC.items():
        aa = []
        rows = []
        for di in rings_d:
            N = min(NALLOW, key=lambda N: abs(di * np.sqrt(N) - a_ref))
            a_imp = di * np.sqrt(N)
            err = 100 * (a_imp - a_ref) / a_ref
            rows.append((di, N, a_imp, err))
            if abs(err) < 3:
                aa.append(a_imp)
        hit = f"a_fit={np.mean(aa):.3f}A ({len(aa)} rings <3%)" if aa else "no clean match"
        print(f"  {name:6s} (a={a_ref:.3f}): {hit}")
        for di, N, a_imp, err in rows:
            flag = " <==" if abs(err) < 3 else ""
            print(f"        d={di:.2f} -> N={N:2d}  a={a_imp:.3f}  {err:+5.1f}%{flag}")
    print("\nnon-cubic candidate strong lines (A), compare by eye:")
    for name, ds in OTHER_D.items():
        print(f"  {name}: {ds}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    g = (np.clip(gray, 0, None) / gray.max()) ** 0.4
    ax[0].imshow(g, cmap="gray"); ax[0].plot(cx, cy, "c+", ms=9)
    for qi in rings_q:
        ax[0].add_patch(plt.Circle((cx, cy), qi / qpp, fill=False, ec="yellow", ls="--", lw=0.6, alpha=0.7))
    ax[0].set_title("SAED + detected rings"); ax[0].axis("off")
    ax[1].plot(q, prof, "k-", lw=0.8, label="I(q) median")
    ax[1].plot(q, base, "g--", lw=0.8, label="background")
    for qi, di in zip(rings_q, rings_d):
        ax[1].axvline(qi, color="r", ls=":", lw=0.8)
        ax[1].text(qi, ax[1].get_ylim()[1] * 0.9, f"{di:.2f}", rotation=90, fontsize=6, color="r", ha="center")
    ax[1].set_xlabel("q (1/A)"); ax[1].set_ylabel("I (median)"); ax[1].legend(fontsize=8)
    ax[1].set_title("radial profile + rings (d in A)")
    plt.tight_layout()
    out = os.path.splitext(args.image)[0]
    fig.savefig(out + "_rings.png", dpi=150, bbox_inches="tight")
    import csv
    with open(out + "_rings.csv", "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["q_invA", "d_A"])
        wtr.writerows([[f"{qi:.4f}", f"{di:.3f}"] for qi, di in zip(rings_q, rings_d)])
    print(f"\nsaved: {out}_rings.png , {out}_rings.csv")


if __name__ == "__main__":
    main()
