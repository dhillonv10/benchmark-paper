"""Publication figure system shared with the notebook: palette, rcParams, and an automated layout gate that
fails if any two pieces of text overlap, if text sits on a marker / line / bar, or if any font is below 6 pt.
Extracted verbatim from build/part1.py so the perspective figure meets the same standard as the notebook's figures."""
import os
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------- publication figure system (one place; every figure in this paper uses it)
FIGDIR = os.environ.get('PERSPECTIVE_FIGDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')); os.makedirs(FIGDIR, exist_ok=True)
import matplotlib as mpl
from matplotlib.text import Text as _MplText
from matplotlib.path import Path as _MplPath
from matplotlib.transforms import Bbox as _Bbox
from matplotlib.patches import Rectangle as _Rect
from matplotlib.collections import PathCollection as _PathColl
# Colours: a validated categorical palette (all-pairs colour-vision-deficiency separation checked), a one-hue
# sequential ramp for ordered/nested model families, and a reserved status colour used only for "this is wrong".
PAL = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', violet='#6250d6',
           seq=('#9ec5f4', '#5598e7', '#2a78d6', '#184f95'),       # light -> dark: more information added
           bad='#d03b3b',                                          # status: critical (always carries a label too)
           ink='#0b0b0b', ink2='#52514e', grid='#d8d7d2', grey='#8c8c8c', grey2='#bdbcb7', surface='#ffffff')
SINGLE, MID, FULL = 3.5, 5.2, 7.2                                  # figure widths in inches (single / 1.5 / double column)
FONT_STACK = [os.environ['ALLOFM_FONT']] if os.environ.get('ALLOFM_FONT') else ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams.update({
    'figure.facecolor': PAL['surface'], 'axes.facecolor': PAL['surface'], 'savefig.facecolor': PAL['surface'],
    'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK,
    'font.size': 7.0, 'axes.labelsize': 7.4, 'axes.titlesize': 7.4, 'xtick.labelsize': 6.6, 'ytick.labelsize': 6.6,
    'legend.fontsize': 6.5, 'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.borderpad': .2,
    'axes.labelcolor': PAL['ink'], 'text.color': PAL['ink'], 'xtick.color': PAL['ink2'], 'ytick.color': PAL['ink2'],
    'axes.edgecolor': PAL['ink2'], 'axes.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.6, 'ytick.major.size': 2.6, 'lines.linewidth': 1.5, 'lines.markersize': 3.4,
    'grid.color': PAL['grid'], 'grid.linewidth': .5, 'axes.grid': False, 'figure.dpi': 140, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .02, 'pdf.fonttype': 42, 'ps.fonttype': 42})

# ------------------------------------------------------------------ layout gate: nothing ships if text collides
MIN_FONT_PT = 6.0
STRICT_FIGS = os.environ.get('ALLOFM_STRICT_FIGS', '0') == '1'    # build sets this to 1; reviewers' runs only warn

def _drawn_texts(fig):
    """Draw once and record every Text artist that was actually drawn (hidden / off-view ticks excluded).
    Hooks Text.draw itself: the renderer only receives a handle to single-line text, so hooking the
    renderer would silently skip every multi-line label."""
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); seen, orig = {}, _MplText.draw
    def rec(self, renderer):
        if self.get_visible() and str(self.get_text()).strip(): seen[id(self)] = self
        return orig(self, renderer)
    _MplText.draw = rec
    try: fig.draw(rend)
    finally: _MplText.draw = orig
    return rend, list(seen.values())

def _poly(t, rend):
    """Text outline in display pixels as a (possibly rotated) rectangle; the text-only extent, never the arrow."""
    bb = _MplText.get_window_extent(t, rend); rot = float(t.get_rotation() or 0)
    if rot % 90 == 0 or bb.width == 0: 
        return np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]]), bb
    t.set_rotation(0); b0 = _MplText.get_window_extent(t, rend); t.set_rotation(rot)
    cx, cy, a = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2, np.deg2rad(rot)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = np.array([[-b0.width / 2, -b0.height / 2], [b0.width / 2, -b0.height / 2],
                    [b0.width / 2, b0.height / 2], [-b0.width / 2, b0.height / 2]])
    return pts @ R.T + np.array([cx, cy]), bb

def _sat(P, Q, tol):
    """Separating-axis test for two convex polygons; True only if they overlap by more than `tol` pixels."""
    for poly in (P, Q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]; n = np.array([-e[1], e[0]]); nn = np.linalg.norm(n)
            if nn == 0: continue
            n = n / nn; p, q = P @ n, Q @ n
            if p.max() <= q.min() + tol or q.max() <= p.min() + tol: return False
    return True

def layout_report(fig, tol=1.0):
    """Every way in which text can be wrong: text on text, text on a marker / line / bar, text that is too small."""
    rend, texts = _drawn_texts(fig); geo = [(t, *_poly(t, rend)) for t in texts]; bad = []
    label = lambda t: repr(t.get_text().replace('\n', ' ')[:34])
    for i in range(len(geo)):                                              # 1. text on text
        for j in range(i + 1, len(geo)):
            if _sat(geo[i][1], geo[j][1], tol): bad.append(('text/text', label(geo[i][0]), label(geo[j][0])))
    pts, segs, rects = [], [], []                                          # 2. text on data ink (clipped to what is visible)
    for ax in fig.axes:
        if not ax.axison: continue
        vis = ax.bbox
        for ln in ax.lines:
            if not ln.get_visible(): continue
            xy = ln.get_xydata()
            if len(xy) == 0: continue
            d = ln.get_transform().transform(xy)
            if ln.get_marker() not in (None, 'None', '', ' '):
                r = (ln.get_markersize() or 0) * fig.dpi / 144 + 1
                pts += [(p, r, 'marker') for p in d if vis.x0 - r <= p[0] <= vis.x1 + r and vis.y0 - r <= p[1] <= vis.y1 + r]
            if ln.get_linestyle() not in ('None', '', ' ', None) and len(d) >= 2:
                path = _MplPath(d)
                if ln.get_clip_on(): path = path.clip_to_bbox(vis)
                if len(path.vertices) >= 2: segs.append((path, 'line'))
        for coll in ax.collections:
            if isinstance(coll, _PathColl) and coll.get_visible() and len(coll.get_offsets()):
                d = coll.get_offset_transform().transform(coll.get_offsets()); sz = coll.get_sizes()
                r = (np.sqrt(sz.max()) if len(sz) else 6) * fig.dpi / 144 + 1
                pts += [(p, r, 'marker') for p in d if vis.x0 - r <= p[0] <= vis.x1 + r and vis.y0 - r <= p[1] <= vis.y1 + r]
        for p in ax.patches:
            if isinstance(p, _Rect) and p.get_visible():
                b = p.get_window_extent()
                if p.get_clip_on(): b = _Bbox.intersection(b, vis)
                if b is not None and b.width > 1.5 and b.height > 1.5: rects.append((b, 'bar'))
    for t, poly, bb in geo:
        if getattr(t, '_ok_overlap', False): continue
        for p, r, kind in pts:
            if bb.x0 - r < p[0] < bb.x1 + r and bb.y0 - r < p[1] < bb.y1 + r: bad.append((f'text/{kind}', label(t), '')); break
        tb = _Bbox([[bb.x0 + 1, bb.y0 + 1], [bb.x1 - 1, bb.y1 - 1]])
        for path, kind in segs:
            if path.intersects_bbox(tb, filled=False): bad.append((f'text/{kind}', label(t), '')); break
        for b, kind in rects:
            if min(bb.x1, b.x1) - max(bb.x0, b.x0) > 1.5 and min(bb.y1, b.y1) - max(bb.y0, b.y0) > 1.5:
                bad.append((f'text/{kind}', label(t), '')); break
    small = sorted({(round(t.get_fontsize(), 1), label(t)) for t, _, _ in geo if t.get_fontsize() < MIN_FONT_PT - 1e-6})
    return dict(n_text=len(geo), collisions=bad, too_small=small, min_pt=min(t.get_fontsize() for t, _, _ in geo))

FIGS = {}
def panel(ax, letter, dx=None, dy=None):
    """Bold panel letter above the top-left corner of the axes (dx, dy kept for old call sites; ignored)."""
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(0, 3), textcoords='offset points',
                ha='left', va='bottom', fontsize=9, fontweight='bold', color=PAL['ink'], annotation_clip=False)

def ygrid(ax, axis='y'):
    ax.set_axisbelow(True); ax.grid(axis=axis, color=PAL['grid'], lw=.5)

def finish(fig, num, slug, caption):
    """Gate, then save one figure as vector PDF + 600-dpi PNG, register it, and show it."""
    rep = layout_report(fig)
    status = 'PASS' if not rep['collisions'] and not rep['too_small'] else 'FAIL'
    print(f"[layout {status}] Fig {num}: {rep['n_text']} text items, {len(rep['collisions'])} collisions, "
          f"{len(rep['too_small'])} below {MIN_FONT_PT:g} pt (smallest {rep['min_pt']:.1f} pt)")
    for k, a, b in rep['collisions']: print(f'    {k}: {a} {"vs " + b if b else ""}')
    for pt, s in rep['too_small']: print(f'    too small ({pt} pt): {s}')
    if status == 'FAIL' and STRICT_FIGS: raise RuntimeError(f'Fig {num} failed the layout gate')
    base = os.path.join(FIGDIR, f'Fig{num:02d}_{slug}')
    for ext in ('pdf', 'png'): fig.savefig(f'{base}.{ext}')
    FIGS[f'Fig {num}'] = dict(slug=slug, caption=caption, files=[f'{base}.pdf', f'{base}.png'], layout=status)
    plt.show(); print(f'[saved] {base}.pdf  +  {base}.png')

def dlabel(ax, x, y, text, color, dx=4, dy=0, ha='left', va='center', size=6.5, weight='bold'):
    """Direct series label (used instead of, or alongside, a legend: identity is never colour-alone)."""
    ax.annotate(text, (x, y), textcoords='offset points', xytext=(dx, dy), ha=ha, va=va,
                fontsize=size, color=color, fontweight=weight, annotation_clip=False)
print(f'figure system ready -> {os.path.abspath(FIGDIR)}  | font stack {FONT_STACK[0]} | strict={STRICT_FIGS}')
