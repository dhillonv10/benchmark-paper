CELLS = []
def md(s): CELLS.append(('markdown', s.strip('\n')))
def code(s): CELLS.append(('code', s.strip('\n')))

# =====================================================================================================  FRONT MATTER
md(r"""
# ALLO-FM and GvHD-Bench
### A clinical foundation model for graft-versus-host disease: plausibility argument, one-institution data plan, and a few-shot benchmark

*A position-and-methods paper, written as an executable notebook and modelled on EHRSHOT (one institution, a released pretrained model kept frozen, a few-shot protocol).* Every number in a grey output cell is computed here. Literature facts carry **✓** (checked against a source during this project) or **†** (from domain knowledge; verify against the primary document before submission). Every simulation is marked **[SIM]** and shows what follows *if* its stated assumptions hold; simulations are not evidence about patients. Figures are written to `figures/` as vector PDF and 600-dpi PNG; the specification is written to `spec/`. **No costs are estimated anywhere in this paper, and no plan asks for more than 1,500 patients.**

---

## Abstract

**Background.** Graft-versus-host disease (GvHD) is the principal cause of non-relapse morbidity and mortality after allogeneic haematopoietic cell transplantation (allo-HCT). Every step of transplant care rests on a prediction — which donor, which conditioning, which prophylaxis, when to intervene pre-emptively, which second-line agent, when to taper immunosuppression — and each is currently made with a different small model, or with none. Published GvHD prediction is single-endpoint risk modelling on registry fields (§1.1); we found no few-shot or foundation-model benchmark for it.

**What this paper is.** Not a trained model. It is (i) a **plausibility argument** with five falsifiable premises, (ii) an explicit statement of **what such a model can and cannot be expected to predict**, (iii) a **data specification** precise enough that two centres would produce the same variable, (iv) **GvHD-Bench v1.0**, a 40-task benchmark with frozen splits, leakage-safe metrics, required baselines and a **few-shot protocol after EHRSHOT** — *and a validation of the benchmark itself* — and (v) a **plan sized for one transplant centre**: about 200 transplants a year, **at most 1,500 patients** with an EHR timeline, about 420 of them with banked specimens.

**Capability claims (§3) — hypotheses to be tested.** Discrimination has a ceiling set by how much of the causal chain is observable at the decision point. Pre-transplant prediction is close to its ceiling already (achievable ≈ 0.62–0.70), so a foundation model should not be sold on it. The proposed headroom is **early post-transplant** (day +7 to +14), where §3 computes 0.74–0.80 against ≈ 0.70–0.75 for today's biomarker-plus-clinical models. **These are hypotheses, not findings, and the one real-data check does not support them:** on the only public longitudinal cohort (`perspective/`; one centre, 584 adults), routine day-7 and day-14 laboratory timelines added nothing to baseline covariates, and a label-free 5-dimensional summary of the near-onset window gained +0.017 AUROC (95 % CI −0.003 to +0.036), indistinguishable from a shuffled-outcome null. The §3 ranges are the targets a benchmark must test. For **transcriptomic profiles at diagnosis** the answer splits: strong for confirming and differentiating the diagnosis in tissue (0.78–0.88), plausible but unproven for stratifying the course once GvHD has begun, and **not** supported for pre-onset prognosis — the pilot found no reproducible pre-onset blood signal.

**Model and compute (§5–§7).** The backbone is the **clinical timeline**; transcriptomes, repertoire, microbiome and pathology enter through **adapters that are optional at inference**. A centre's 1,500 timelines are ≈ 11 million event tokens, which makes the compute-optimal model trained from scratch only **≈ 0.57 M parameters** (about 5 seconds on one A100). That is a baseline, not the foundation model. The foundation-model claim rests on a **released pretrained timeline model, kept frozen or lightly adapted (LoRA), with a head of 5–25 parameters** — the EHRSHOT recipe. The whole enumerated programme (seeds, ablations, embeddings, LoRA runs, 750 head fits) is **≈ 19 minutes on one A100**. Compute is not the constraint. Patients and labels are.

**GvHD-Bench and its own operating characteristics (§8).** 40 tasks in 7 families, each with a landmark *issue day* (draw day plus assay turnaround), competing events and the clinical baseline it must beat. At one institution **12 tasks are runnable, 9 estimable only and 4 out of reach** (the other 15 are not single-AUROC comparisons or are research-only). We test the benchmark like a model. A percentile bootstrap over centres **under-covers** (88 % at 8 centres, 93 % at 20; the centre-level *t*-interval holds 94–95 %). A gate that requires a point gain of +0.03 **passes a gain of exactly +0.03 only half the time**; the adopted network gate (lower bound ≥ +0.01, ≥ 20 centres) passes it 85 % of the time and a gain of +0.01 about 2 %. The **single-centre gate (Level 1C)** — a temporal split and a paired lower bound ≥ +0.01 on 600 test patients — passes a true gain of +0.05 about 89 % of the time, +0.03 about a third of the time, and +0.01 about 2 %. The harness recovers planted effects of known size, returns chance on permuted outcomes, flags an informative-ordering shortcut, and catches a planted leak that inflates AUROC by 0.2. In the few-shot protocol (§8.15) a head of 5–10 effective parameters is within 0.02 of a 0.75 ceiling at ≈ 50–100 patients per class; a 25-parameter head needs ≈ 260.

**What one institution can and cannot do (§9).** One AUROC is known to ±0.056 at 400 patients and ±0.029 at 1,500. Showing **+0.03 over a strong correlated baseline needs ≈ 840 patients**, +0.05 about 300, and +0.02 about 1,900 — beyond one centre. Cohort size sets head size: **1,500 patients support a head of ≈ 24 parameters, the 900-patient development set ≈ 14, the 420 with specimens ≈ 6.** **Transcriptome at diagnosis:** a pre-specified tissue score can be validated on 60–120 biopsies (120 collected in three years); a 10-parameter adapter head for the tissue differential diagnosis needs ≈ 240 patients against ≈ 210 eligible (marginal); steroid refractoriness, the day-7 blood transcriptome and any whole-transcriptome head are not feasible at one centre. **Longitudinal:** at single-draw reliability 0.5, six draws per window recover the oracle AUROC to within 0.02; for a 10-day prodrome, draws must be 2–3.5 days apart; weekly draws to day +56 in the 420 specimen patients give ≈ 140 cases with two or more pre-onset specimens. **Not within one centre:** any claim about another centre (Level 2 needs ≈ 20 centres), subgroup audits (≈ 1,000 patients per subgroup), and gains below ≈ +0.03. The **data plan** is three nested packages — S0 (EHR timelines, no new specimens), S1 (+ specimens banked on routine draws over three years, 120 biopsies) and S2 (+ a weekly trajectory cohort) — with specimen counts, storage and calendar time (§9.9).

**Conclusion.** A GvHD foundation model is plausible if its backbone is the clinical timeline and biology enters through adapters, if the pretraining is borrowed from a released model and only a small head is fitted at the centre, and if its value is judged against registry-grade baselines with a gate whose operating characteristics are known. One institution can make a complete, publishable **single-centre** claim (Level 1C); §9 says precisely what it cannot, and what a network would add.
""")

md(r"""
### Contributions
1. **A plausibility argument in five premises** (§2), each with its evidence, its falsifier, and an executed simulation showing *when* pretraining helps and when it cannot.
2. **A capability statement** (§3): a computed discrimination ceiling per decision point, a task-by-task claim table, and a direct answer on transcriptomics at diagnosis, with the sample size each open question needs.
3. **A data specification for allo-HCT** (§4): 129 elements in 25 domains with units, timing relative to day 0, coding standards, availability tiers and leakage flags; a protocolised sampling schedule; and executable outcome-label functions with unit tests.
4. **An architecture and objective stack** (§5–§6): event tokenisation with knowledge-time masking, a donor–recipient pair encoder, optional modality adapters, and seven objective families with matched controls.
5. **Training requirements for one institution** (§7): token budgets per tier, a model ladder from a tuned simple model to a frozen released model with a small head, and GPU-time — computed from the specification, with no cost estimates.
6. **GvHD-Bench v1.0** (§8): 40 tasks modelled on EHRSHOT's few-shot design, exported as a standalone specification with frozen artefacts and hashes, **the benchmark validated as a measuring instrument** (§8.12: interval coverage, false-pass and power of the gate, planted-effect recovery, a permuted-outcome control, an informative-presence shortcut test), drift and clinical translation (§8.13), and a **few-shot protocol** (§8.15).
7. **A one-institution data calculus** (§9): precision, paired power and the largest head a cohort supports; accrual at 100–300 transplants a year; **labelled transcriptomes at diagnosis** (closed-form learning curve, validated against simulation); **longitudinal sampling design**; validation, subgroup and label-noise limits; the **task-by-task feasibility** at one centre; and a **three-package data plan** (S0/S1/S2) that never exceeds 1,500 patients.
8. **A staged path with gates** (§11), grounded in the pilot results this repository already contains (§10).

### Paper map
| § | Section | Type |
|---|---|---|
| 1 | Introduction: the decision timeline of allo-HCT; **1.1 template (EHRSHOT) and prior art** | narrative + Fig 1 |
| 2 | Plausibility argument: five premises | evidence + executed [SIM] + Fig 2 |
| 3 | **What can be predicted** — ceilings, claims, transcriptomics at diagnosis | executed + Fig 3 |
| 4 | **Data specification** — domains, elements, schedule, sources, labels, leakage | specification + executed label tests |
| 5 | Architecture and tokenisation | design + Fig 4, Fig 5 |
| 6 | Training objectives | formal definitions |
| 7 | **Training requirements for one institution** — tokens, model ladder, GPU-time | executed + Fig 6 |
| 8 | **GvHD-Bench v1.0**, incl. **8.12 the benchmark's own operating characteristics**, **8.13 clinical translation**, **8.15 few-shot protocol** | benchmark spec + executed metric validation + executed harness + self-validation + Fig 7–10 |
| 9 | **What one institution can do**: precision and power, head size, accrual, **transcriptomes at diagnosis, longitudinal design**, centres (context), validation and subgroups, **task feasibility, the three-package data plan** | executed calculators, each validated + Fig 11–17 |
| 10 | Pilot evidence (the public transcriptomic benchmark) | results carried forward + Fig 18 |
| 11 | Engineering path, harness, milestones, governance | plan |
| 12 | Limitations · 13 Conclusion · References · Appendices | — |
""")

code(r'''
# ----------------------------------------------------------------------------- setup (CPU only; numpy / pandas / scipy / scikit-learn / matplotlib)
import os, json, math, time, warnings, itertools, collections
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from IPython.display import display, Markdown
warnings.simplefilter('ignore')
SEED = 20261026
np.random.seed(SEED)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25, 'font.size': 9})
QUICK = os.environ.get('ALLOFM_QUICK', '0') == '1'      # QUICK=1 cuts simulation replicates ~4x (numbers get noisier)
SPEC_DIR = 'spec'; os.makedirs(SPEC_DIR, exist_ok=True)
# The scale this paper plans for: ONE large transplant centre. Everything the plan asks for is sized to fit inside these numbers (planning values, †).
TRANSPLANTS_PER_YEAR, EHR_YEARS, SPECIMEN_YEARS, USABLE_FRACTION = 200, 7.5, 3, .70
N_CAP = int(round(TRANSPLANTS_PER_YEAR * EHR_YEARS))                                   # 1,500 patients with an EHR timeline (the ceiling cohort)
N_SPEC = int(round(TRANSPLANTS_PER_YEAR * SPECIMEN_YEARS * USABLE_FRACTION))            # 420 of them with prospectively banked specimens
COL = dict(raw='#4C72B0', pre='#55A868', rand='#8C8C8C', oracle='#C44E52', clin='#937860', bio='#DD8452', leak='#8172B2')

def md_table(df, index=False, max_w=None):
    """Render a DataFrame as a GitHub-markdown table (keeps the notebook free of extra dependencies)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    def cell(v):
        s = '' if (isinstance(v, float) and np.isnan(v)) else str(v)
        s = s.replace('|', '\\|').replace('\n', ' ')
        return s if not max_w or len(s) <= max_w else s[:max_w - 1] + '…'
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    out += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in d.itertuples(index=False)]
    return Markdown('\n'.join(out))

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float('nan')
print('QUICK =', QUICK, '| spec files ->', os.path.abspath(SPEC_DIR))
''')

code(r'''
# ----------------------------------------------------------------------------- statistical toolkit (defined once; validated against simulation in section 9.1)
import functools
def hm_se(A, n1, n0):
    """Hanley-McNeil (1982) standard error of one AUROC."""
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    return math.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n0 - 1) * (Q2 - A * A)) / (n1 * n0))

def n_for_ci(A, prev, half_width=.07):
    """Patients needed so that the 95% CI half-width of ONE AUROC is <= half_width."""
    for n in range(20, 400001, 10):
        n1 = max(int(round(n * prev)), 2)
        if 1.96 * hm_se(A, n1, n - n1) <= half_width: return n
    return None

_ZC = np.random.default_rng(11).standard_normal((2, 600_000))          # fixed common draws: the calculators are deterministic
@functools.lru_cache(maxsize=None)
def _pv_components(A1, A2, rho):
    """Asymptotic DeLong structural variances of the AUROC DIFFERENCE for two equal-variance binormal scores whose
    within-class correlation is rho: variance of the placement-value difference among positives (s10) and negatives (s01)."""
    d1, d2 = math.sqrt(2) * norm.ppf(A1), math.sqrt(2) * norm.ppf(A2)
    z1 = _ZC[0]; z2 = rho * _ZC[0] + math.sqrt(1 - rho * rho) * _ZC[1]
    v10 = norm.cdf(d2 + z2) - norm.cdf(d1 + z1)                       # positive i: P(a negative scores below it)
    v01 = norm.cdf(z1 - d1) - norm.cdf(z2 - d2)                       # negative j: P(a positive scores above it), model 2 minus model 1
    return float(np.var(v10)), float(np.var(v01))
def pv_components(A1, A2, rho): return _pv_components(round(float(A1), 4), round(float(A2), 4), round(float(rho), 4))

def se_delta(n, prev, A1, A2, rho=.85):
    """SE of AUROC(model 2) - AUROC(model 1) on the SAME n patients (paired)."""
    s10, s01 = pv_components(A1, A2, rho); return math.sqrt(s10 / (n * prev) + s01 / (n * (1 - prev)))
def power_delta(n, prev, A1, A2, rho=.85, alpha=.05):
    d = A2 - A1; z = norm.ppf(1 - alpha / 2); se = se_delta(n, prev, A1, A2, rho)
    return float(norm.cdf(d / se - z) + norm.cdf(-d / se - z))
def n_for_delta(A1, delta, prev, rho=.85, power=.8, alpha=.05):
    """Patients for `power` to detect an AUROC gain `delta` over a baseline AUROC A1 (paired DeLong, closed form)."""
    s10, s01 = pv_components(A1, A1 + delta, rho)
    return int(math.ceil((norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2 * (s10 / prev + s01 / (1 - prev)) / delta ** 2))
''')

# =====================================================================================================  1. INTRODUCTION
code(r'''
# ----------------------------------------------------------------------------- publication figure system (one place; every figure in this paper uses it)
FIGDIR = 'figures'; os.makedirs(FIGDIR, exist_ok=True)
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
''')

md(r'''
---
# 1 · Introduction: allo-HCT as a sequence of predictions

Allo-HCT is one of the most protocolised processes in medicine. A shared clock (day 0 = graft infusion) anchors every event. Conditioning occupies days −10 to −1. Prophylaxis runs on a fixed schedule (for example PtCy on days +3/+4, then tacrolimus from day +5). Neutrophil engraftment falls around days +14 to +28. Acute GvHD typically appears between days +14 and +100, though late-onset aGvHD is recognised after day +100. Chronic GvHD typically begins from month 3 onward. Outcomes are graded with consensus instruments: MAGIC for aGvHD, NIH 2014 for cGvHD, and the 2018 EBMT–NIH–CIBMTR definitions for steroid refractoriness. National registries (CIBMTR, EBMT, JSHCT) have collected the same core variables on structured forms for decades.

Grade II–IV aGvHD affects roughly 30–50 % of recipients and grade III–IV roughly 10–15 %, depending on donor type and prophylaxis. Steroid-refractory aGvHD carries high mortality. Chronic GvHD affects roughly 30–50 % of long-term survivors and is the leading cause of late morbidity (†, ranges across eras and donor types).

The figure below is drawn in code and shows **where the decisions are**. Each decision is a prediction task in GvHD-Bench (§8), and each task can use only the data that exist *before* its decision point.
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 1: the allo-HCT timeline, its risk windows and its decision points
PHASES = [  # (start day, end day, row label, colour)
    (-10, -1,  'Conditioning',           PAL['seq'][1]),
    (-5,  -1,  'Serotherapy (ATG)',      PAL['seq'][0]),
    (3,    5,  'PtCy (day +3, +4)',      PAL['violet']),
    (5,  180,  'CNI ± MMF, then taper',  PAL['grey2']),
    (14,  28,  'Engraftment',            PAL['aqua']),
    (14, 100,  'Acute GvHD window',      PAL['orange']),
    (90, 730,  'Chronic GvHD window',    PAL['blue'])]
DECISIONS = [  # (day, decision, task ids)
    (-45, 'Select donor',               'A1–A3, A6'), (-21, 'Conditioning, serotherapy', 'A1, A5'),
    (-3,  'GvHD prophylaxis',           'A7'),        (7,   'Day-7 risk, pre-emptive tx', 'B1, B5'),
    (14,  'Early-warning surveillance', 'B2–B4'),     (24,  'Is this GvHD?',              'C1–C3'),
    (33,  'Steroid response, 2nd line', 'C4–C6, D1'), (100, 'cGvHD risk, IS taper',      'E1, E4'),
    (365, 'Late effects, BOS',          'E2, E3')]
def sx(d): return np.sign(d) * np.log1p(abs(d))           # signed-log day axis: dense early, compressed late

fig = plt.figure(figsize=(FULL, 4.4))
ax = fig.add_axes([.195, .335, .78, .625]); key = fig.add_axes([.02, .015, .96, .20]); key.axis('off')
for i, (a, b, lab, c) in enumerate(PHASES):
    ax.barh(i, sx(b) - sx(a), left=sx(a), height=.56, color=c, lw=0, zorder=3)
ax.axvline(0, color=PAL['ink'], lw=1.0, zorder=2)
def badge(axx, x, y, n):
    """Numbered marker; the number is meant to sit on the marker, so it is exempt from the overlap gate."""
    axx.plot(x, y, 'o', ms=11, color=PAL['ink'], zorder=4, clip_on=False)
    t = axx.text(x, y, str(n), color='white', fontsize=6.4, fontweight='bold', ha='center', va='center', zorder=5)
    t._ok_overlap = True
LANES = (-1.2, -2.4)                                       # alternate lanes so neighbouring badges never touch
for k, (d, lab, tasks) in enumerate(DECISIONS):
    y = LANES[k % 2]
    ax.plot([sx(d)] * 2, [y, len(PHASES) - .45], color=PAL['grey2'], lw=.5, ls=':', zorder=1)
    badge(ax, sx(d), y, k + 1)
ax.set_yticks([-1.8] + list(range(len(PHASES)))); ax.set_yticklabels(['Decision points'] + [p[2] for p in PHASES])
ax.set_ylim(len(PHASES) - .45, -3.2); ax.set_xlim(sx(-62), sx(1000))
ticks = [-30, -10, 0, 7, 14, 28, 100, 365, 730]
ax.set_xticks([sx(t) for t in ticks]); ax.set_xticklabels([f'{t:+d}' if t else '0' for t in ticks])
ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.set_axisbelow(True); ax.grid(axis='y', color=PAL['grid'], lw=.4)
ax.set_xlabel('day relative to graft infusion (day 0; signed-log scale)')
key.set_xlim(0, 3); key.set_ylim(0, 3)
for k, (d, lab, tasks) in enumerate(DECISIONS):                # the key: three columns of three
    col, row = k // 3, k % 3; y = 2.45 - row * .95; x = col + .04
    badge(key, x, y, k + 1)
    key.text(x + .07, y, f'{lab}  [{tasks}]', fontsize=6.6, va='center', ha='left')
finish(fig, 1, 'decision_timeline',
       'The allo-HCT timeline. Bars: the protocolised exposures and the risk windows they open. Numbered markers: '
       'the clinical decisions, each of which is a family of GvHD-Bench tasks (task ids in brackets; section 8). '
       'The time axis is signed-log, so the first 100 days, where nearly every decision sits, are legible beside '
       'two years of follow-up. A model standing at any decision may use only data whose results were known then.')

''')

md(r'''
**Why a *clinical* foundation model, and why not the one we tried first.** Foundation models trained on timestamped structured health records have moved the field forward. CLMBR (next-code pretraining, †), MOTOR (time-to-event pretraining on 55 M records and 9 B events; up to 95 % better label efficiency on 19 tasks, ✓), Foresight (†) and Delphi-2M (generative disease trajectories, Nature 2025, †) all treat a patient as a *sequence of events in time*, and they learn a representation that transfers to many downstream risk tasks with few labels. Allo-HCT fits that template better than most areas of medicine, because its timeline is short, dense, protocolised and anchored.

Our earlier attempt took the other route: a *transcriptomic* foundation model, with expression as the backbone and GvHD as the label. Its benchmark is in this repository (`GvHD_FM_benchmark_blueprint.ipynb`, summarised in §10). It produced a clear negative on public data. Frozen Geneformer did not beat normalised genes, a pathway prior did not replicate, and blood T-cell signals did not transfer between studies, although tissue-biopsy signals did. This paper keeps those lessons and changes the architecture. **The clinical timeline is the backbone. Transcriptomes, TCRs, microbiome and pathology become adapters** that add information when it is available and that can be ablated to show whether they help.

**What this paper is.** It is a specification: plausibility, data, objectives and benchmark, with executed sample-size calculus, executed metric code, executed label functions, the pilot ledger, and simulations marked **[SIM]**. **What it is not:** a trained model, or any claim that ALLO-FM improves outcomes.
''')

md(r'''
### 1.1 Template and prior art

**What others have done on GvHD.** GvHD prediction has been published as single-endpoint risk models on registry fields, usually at a fixed pre-transplant landmark:

| Study | Data | What it did | Result as reported |
|---|---|---|---|
| Arai et al., *Blood Adv* 2019 ✓ | 26,695 Japanese adult allo-HSCT recipients, 1992–2016 | alternating decision tree on 15 of > 40 registry variables; 70 / 30 split | grade II–IV aGvHD 29.0 % (low-risk group) vs 58.7 % (high-risk), HR 2.57 |
| Jo et al., *Commun Med* 2023 ✓ | 18,763 Japanese adults, 2008–2018 | convolutional network combined with a natural-language-processing technique and an explanation algorithm | grade II–IV 42.0 %, III–IV 15.6 % in the cohort (discrimination not checked †) |
| MatchGraft.AI, ASH 2023 abstract ✓ (not peer-reviewed) | 1,035 German and North American donor–recipient sets | absolute aGvHD risk per donor–recipient pair, for donor selection; 80 / 20 split | AUC 0.70, integrated calibration index 0.028 |
| Chandra et al., *BMC Med Res Methodol* 2024 ✓ | CIBMTR, T-cell prolymphocytic leukaemia | classical classifiers for aGvHD occurrence and grade | best test balanced accuracy 0.58; the authors attribute it to rarity and small *n* |
| arXiv:2511.11623 ✓ | EHR data, **liver** transplantation | multi-modal deep learning for early GvHD under heavy class imbalance | details not checked † |

The pattern is consistent: one endpoint, registry-level fields, a fixed landmark, no competing-risk-aware time-dependent metric, no held-out centres, and no benchmark that anyone else can re-run. A scoping review of machine learning in GvHD ✓ ends on the same need for multicentre collaboration. **We found no few-shot or foundation-model benchmark for GvHD** (a search done during this project, not a systematic review †). The gap is the one this paper fills — with a specification and a benchmark rather than a model.

**The template we follow is EHRSHOT** (Wornow, Thapa, Steinberg, Fries and Shah; NeurIPS Datasets & Benchmarks 2023; arXiv:2307.02028 ✓). It is the closest published precedent for a benchmark of this kind, and it is deliberately modest in scale: **one institution**, de-identified structured records of **6,739 patients**, **15 few-shot classification tasks**, a released model (**CLMBR-T-base**, 141 M parameters, pretrained on 2.57 M patients' records) that is **kept frozen** with a small logistic-regression head, and a **few-shot protocol** — train on *k* positive and *k* negative patients for *k* from 1 to 128, tune hyper-parameters on equally sized validation draws, score on the *entire* held-out split, repeat over draws. Access is by a data-use agreement. A benchmark of that shape is what one transplant centre can actually produce, which is why this paper adopts it.

| EHRSHOT section ✓ | What it contains | Where this paper has the GvHD counterpart |
|---|---|---|
| Introduction and motivation | few labels, many tasks; foundation models claim label efficiency | §1–§2: the plausibility argument, and [SIM] S1 on when pretraining helps |
| Dataset | one institution, 6,739 patients, coded structured records | §4 data specification (129 elements, 25 domains); §9.8 the one-institution cohort, **at most 1,500 patients** |
| Tasks | 15 few-shot classification tasks | §8: 40 tasks in 7 families, each with an issue-day landmark and competing events; 12 runnable at one centre |
| Models evaluated | released foundation model, frozen, plus a count-based baseline | §7.2 model ladder: L0 tuned simple model → L1 tiny timeline model → L2 released model, frozen, small head → L3 LoRA |
| Few-shot protocol | *k* cases and *k* controls, tune on equal validation draws, test on the full split | §8.15: the same protocol, with the head size capped by what the label count supports |
| Metrics | AUROC and AUPRC | IPCW time-dependent AUROC, Aalen–Johansen calibration, net benefit, leakage audit (§8.6) |
| Results | few-shot curves per task | §8.15 and §9.8 (simulated, marked **[SIM]**); §10 (pilot, real public data) |
| Access, ethics, limitations | data-use agreement; one hospital | §11 governance; §12 limitations; the centre's own claim level (§8.10, Level 1C) |

**What GvHD adds to that template**, each of which the paper then tests rather than asserts:
1. **Time-to-event labels with competing risks** (death and relapse) and censoring, where EHRSHOT's labels are classification targets at a fixed prediction time, with no censoring and no competing events.
2. **Knowledge-time discipline.** Every task is issued on a *day* (the draw day plus assay turnaround), and a leakage audit checks that no input was known later (§4.7, §8.7).
3. **The benchmark is validated as a measuring instrument** (§8.12): interval coverage, false-pass and power of the gate, planted-effect recovery and controls.
4. **Specimens and omics as optional adapters** on a clinical-timeline backbone, with the transcriptome-at-diagnosis question sized explicitly (§9.4).
5. **A claim ladder that matches the data.** One institution can make a *single-centre* claim (Level 1C, temporal split) and no more; claims across centres need a network (Level 2).
''')


# =====================================================================================================  2. PLAUSIBILITY
md(r'''
---
# 2 · The plausibility argument

A foundation model for GvHD is worth building only if **all five** premises below hold. Each premise has the evidence for it, the observation that would refute it, and where in this paper it is tested.

| # | Premise | Evidence for it | What would falsify it | Tested in |
|---|---|---|---|---|
| **P1** | **Predictive signal exists before and early after transplant** | Established risk factors. HLA mismatch (each allele mismatch raises aGvHD risk), mismatched HLA-DPB1 T-cell-epitope groups (non-permissive) and HLA-B leader mismatch†. Female donor → male recipient (H-Y minor antigens)†. PBSC vs BM (more cGvHD)†. Older donor age†. Myeloablative / TBI conditioning†. Prophylaxis: BMT CTN 1703 randomised 431 patients; 1-year GRFS 52.7 % (PtCy/Tac/MMF) vs 34.9 % (Tac/MTX) ✓. Plasma ST2 and REG3α at day 7 (MAGIC algorithm, 1,287 patients ✓) predict NRM. Lower faecal microbiota diversity at engraftment → higher mortality (1,362 patients, 8,767 stool samples, 4 centres ✓) | Clinical + biomarker models at chance in leave-centre-out validation | §8 Family A–B baselines; [SIM] harness |
| **P2** | **The signal is spread across modalities and time**, so no single measurement is enough | Each published predictor reaches moderate discrimination on its own (registry-only ML models for aGvHD typically AUROC ≈ 0.6–0.65†; MAP biomarkers mainly stratify NRM and severe GvHD rather than any GvHD†). The biology involves several steps: conditioning injury → DAMPs/PAMPs → antigen presentation → donor T-cell priming and homing → epithelial target damage → dysbiosis feedback. Each step is measured by a different modality | One modality (e.g. day-7 MAP) captures all the information: adding clinical timeline and biology yields ΔAUROC ≈ 0 | modality-knockout ablation (§8.6) |
| **P3** | **The structure is shared across centres and tasks**, so one pretrained model can serve all of them | EHR foundation models gain the most at low label counts (MOTOR ✓, EHRSHOT ✓). Our pilot: tissue-biopsy GvHD signal transfers across labs and organs (LOSO AUROC ≈ 0.80), while blood T-cell signal does not (≈ 0.59) | Leave-centre-out gains disappear; pretrained ≈ random-weight twin | [SIM] S1 below; §10 pilot |
| **P4** | **Enough data exist**: for pretraining (already done elsewhere, by released models) and for evaluation at one institution (labelled patients of our own) | Pretraining is not this project's job: released timeline models were pretrained on millions of records (CLMBR-T-base: 141 M parameters, 2.57 M patients ✓; MOTOR: 55 M ✓) and are used frozen or lightly adapted (§7.2). What one centre must supply is *evaluation* data: **up to 1,500 transplant recipients with an EHR timeline** (≈ 200 transplants a year over 7.5 years †) and **≈ 420 with prospectively banked specimens** (§9.8). Registries (CIBMTR: > 470,000 recipients from > 500 centres ✓; EBMT †; MAGIC; the MSK hospitalome with > 10,000 stool samples ✓) are where the *network* stage of the plan would draw its centres (§11), not what a single centre needs | No suitable released model can be obtained under a data-use agreement, **or** the centre cannot reach the labelled count that the primary task needs (§9.8) | §4.5 sources; §9.8 |
| **P5** | **A gain can be detected** with honest statistics | Paired-test power and development-n are computable (§9). Gates are fixed in advance (§8.7) | Required n for ΔAUROC = 0.03 exceeds every accessible cohort | §9 |

**Where the earlier attempt broke.** The transcriptome-first design failed **P3** (blood signal not shared between studies) and **P4** (public labelled n of 17–101 per study, mostly microarrays the model could not read). A clinical-timeline backbone moves the labelled-data problem from GEO (~10² samples per study) to the centre's own record (up to 1,500 patients, each with thousands of timestamped events), borrows the pretraining from a released model, and treats omics as optional adapters. **That reframing is the main design decision of this paper.**

### 2.1 [SIM] S1: when does pretraining help? Shared structure versus label count
Here we make P3 quantitative. Ten source centres and one new target centre each observe a 600-dimensional feature vector (a stand-in for a timeline or omics profile). It is generated from six latent biological factors through a centre-specific loading matrix, $W_c=\sqrt{\rho}\,W_0+\sqrt{1-\rho}\,U_c$, plus centre-specific nuisance structure that no other centre shares. Here **ρ is the fraction of feature structure shared across centres**. The outcome depends only on the latent factors, in the same way at every centre. We compare four arms:
* **raw**: a ridge classifier on the 600 features, with the penalty chosen by inner 3-fold CV on the training labels (the "normalised genes" baseline);
* **pretrained**: a 16-d representation learned *without labels* on pooled source-centre data (PCA, the simplest self-supervised encoder), with the same ridge head on top;
* **random-weight twin**: a random 16-d projection, matching the control used in the pilot;
* **oracle**: the true latent factors (an upper bound).

The labelled target patients are drawn from a pool of 400, and all arms are scored on the 800 remaining target-centre patients, against the number of labelled target patients *k*.
''')

code(r'''
# ----------------------------------------------------------------------------- [SIM] S1: value of label-free pretraining as a function of cross-centre shared structure (rho) and label count k
def s1_world(rho, n_src_total=4000, n_centres=10, p=600, k_lat=6, sig=.45, seed=0):
    """Ten source centres and one new target centre. Six latent biological factors drive the outcome identically
    everywhere; each centre maps them to features through W_c = sqrt(rho)*W0 + sqrt(1-rho)*U_c, so `rho` is the
    fraction of feature structure shared across centres. Each centre also has its own nuisance structure, and
    p (600) >> the number of labels we will allow, which is the regime a foundation model is meant to help in."""
    r = np.random.default_rng(seed)
    W0 = r.standard_normal((k_lat, p)); beta = r.standard_normal(k_lat); beta *= 2.0 / np.linalg.norm(beta)
    def centre(n, rr):
        U = rr.standard_normal((k_lat, p)); W = math.sqrt(rho) * W0 + math.sqrt(1 - rho) * U
        z = rr.standard_normal((n, k_lat))
        nuisance = rr.standard_normal((n, 8)) @ rr.standard_normal((8, p)) * .9     # centre-specific batch structure
        x = z @ W * sig + nuisance + rr.standard_normal((n, p))
        y = (rr.random(n) < 1 / (1 + np.exp(-(z @ beta - .3)))).astype(int)
        return x, y, z
    src = [centre(n_src_total // n_centres, np.random.default_rng(seed * 100 + c)) for c in range(n_centres)]
    return np.vstack([s[0] for s in src]), centre(1200, np.random.default_rng(seed * 100 + 99))

ALPHA_GRID = (30., 300., 3000.)
def head(X, y):
    """Linear ridge head; the penalty is chosen by inner 3-fold CV on the TRAINING labels only (never the test set).
    A closed-form ridge keeps the simulation fast enough to re-run; the conclusions are unchanged with
    ridge-logistic (checked), because at these sample sizes the penalty, not the link, is what matters."""
    mk = lambda a: make_pipeline(StandardScaler(), RidgeClassifier(alpha=a))
    best, best_a = -1.0, ALPHA_GRID[1]
    if min(np.bincount(y, minlength=2)) >= 3:
        for a in ALPHA_GRID:
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
                oof[te] = mk(a).fit(X[tr], y[tr]).decision_function(X[te])
            sc = roc_auc_score(y, oof)
            if sc > best: best, best_a = sc, a
    return mk(best_a).fit(X, y)

def s1_run(rho, ks, seed):
    Xs, (xt, yt, zt) = s1_world(rho, seed=seed)
    enc = PCA(16, random_state=seed).fit(Xs)                                   # label-free "pretraining" on source centres only
    Rp = np.random.default_rng(seed + 7).standard_normal((xt.shape[1], 16))
    test = slice(400, None); out = []
    r = np.random.default_rng(seed + 1)
    for k in ks:
        for d in range(2 if QUICK else 4):
            idx = r.choice(400, k, replace=False)                      # labelled target patients come from the first 400 only
            if len(set(yt[idx])) < 2: continue
            for arm, F in (('raw', xt), ('pretrained', enc.transform(xt)), ('random twin', xt @ Rp), ('oracle', zt)):
                m = head(F[idx], yt[idx]); out.append(dict(rho=rho, k=k, seed=seed, arm=arm, auroc=auroc(yt[test], m.decision_function(F[test]))))
    return out

t0 = time.time()
KS = [25, 50, 100, 200, 400]; RHOS = [0.0, 0.5, 0.8, 1.0]; SEEDS = range(2 if QUICK else 4)
S1 = pd.DataFrame([row for rho in RHOS for s in SEEDS for row in s1_run(rho, KS, s)])
s1 = S1.groupby(['rho', 'arm', 'k']).auroc.mean().unstack('k').round(3)
print(f'S1 finished in {time.time() - t0:.0f}s'); display(s1)
ARMS = [('raw', PAL['blue'], '-', 'raw features + tuned ridge'),
        ('pretrained', PAL['aqua'], '-', 'label-free pretrained encoder'),
        ('random twin', PAL['grey'], '-', 'random-weight twin'),
        ('oracle', PAL['ink'], ':', 'oracle (true latent factors)')]
fig = plt.figure(figsize=(FULL, 2.8))
gl = fig.add_gridspec(1, 4, left=.075, right=.705, bottom=.285, top=.865, wspace=.16)
gr = fig.add_gridspec(1, 1, left=.805, right=.985, bottom=.285, top=.865)
axs = [fig.add_subplot(gl[0, i]) for i in range(4)]
for i, (rho, ax) in enumerate(zip(RHOS, axs)):
    ygrid(ax)
    for arm, c, ls, _ in ARMS:
        ax.plot(KS, s1.loc[(rho, arm)].values, marker='o', ms=3.0, color=c, ls=ls,
                lw=1.5 if arm != 'oracle' else 1.1, zorder=3 if arm == 'pretrained' else 2)
    ax.set_xscale('log'); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.minorticks_off(); ax.set_ylim(.52, .92)
    ax.set_title(f'$\\rho$ = {rho:g}', pad=3); ax.set_xlabel('labelled patients $k$')
    if i: ax.set_yticklabels([])
    else: ax.set_ylabel('AUROC at the new centre')
panel(axs[0], 'a')
ax = fig.add_subplot(gr[0, 0]); ygrid(ax)
for k, c, m in ((25, PAL['aqua'], 'o'), (400, PAL['grey'], 's')):
    ax.plot(RHOS, [s1.loc[(r, 'pretrained'), k] - s1.loc[(r, 'raw'), k] for r in RHOS], marker=m, ms=3.2,
            color=c, lw=1.5, label=f'$k$ = {k}')
ax.axhline(0, color=PAL['ink2'], lw=.6); ax.set_xticks([0, .5, 1]); ax.set_xlim(-.06, 1.06)
ax.set_xlabel('shared structure $\\rho$'); ax.set_ylabel('AUROC gain from pretraining')
ax.legend(loc='lower right'); panel(ax, 'b')
fig.legend([plt.Line2D([], [], color=c, ls=ls, marker='o', ms=3, lw=1.4) for _, c, ls, _ in ARMS],
           [lab for _, _, _, lab in ARMS], ncol=4, loc='lower center', bbox_to_anchor=(.5, -.01),
           handlelength=1.8, columnspacing=1.8)
finish(fig, 2, 'pretraining_vs_shared_structure',
       '[SIM] Label-free pretraining pays in proportion to the structure shared across centres. (a) AUROC at a new '
       'centre against the number of labelled patients there, at four levels of shared structure rho. (b) The same '
       'result as the gain over the tuned raw-feature baseline: positive at few labels and high rho, absent at '
       'rho = 0, and shrinking everywhere as labels accumulate. Simulated data; effect sizes follow from the stated model.')
''')

code(r'''
g = lambda rho, arm, k: s1.loc[(rho, arm), k]
gain_small = {rho: g(rho, 'pretrained', 25) - g(rho, 'raw', 25) for rho in RHOS}
gain_big = {rho: g(rho, 'pretrained', 400) - g(rho, 'raw', 400) for rho in RHOS}
display(Markdown(f"""**Result [SIM].** With fully shared structure (ρ = 1) the pretrained encoder is ahead of the raw-feature baseline by **{gain_small[1.0]:+.3f} AUROC at k = 25** and by {gain_big[1.0]:+.3f} at k = 400, so the advantage shrinks as labels accumulate. This is the label-efficiency pattern reported for EHR foundation models. At ρ = 0.8 the gain is {gain_small[0.8]:+.3f} at k = 25. At ρ = 0.5 it is {gain_small[0.5]:+.3f}. **At ρ = 0 (nothing shared across centres) pretraining is {gain_small[0.0]:+.3f}, no better than the random-weight twin.** That is the pattern the pilot found for blood T cells (§10).

**What this licenses.** Pretraining can only exploit structure that is shared. A GvHD foundation model is therefore plausible only for modalities and centres where ρ is high. ρ can be *measured before any model is built*: decode centre from features, or compare principal subspaces between sites (§8.6, *shared-structure audit*). Structured clinical variables recorded on common registry forms are high-ρ by design. Blood transcriptomes from heterogeneous public studies were low-ρ in our pilot. This is the quantitative reason for making the clinical timeline the backbone."""))
''')
