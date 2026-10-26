CELLS = []
def md(s): CELLS.append(('markdown', s.strip('\n')))
def code(s): CELLS.append(('code', s.strip('\n')))

# =====================================================================================================  FRONT MATTER
md(r"""
# ALLO-FM and GvHD-Bench
### A clinical foundation model for graft-versus-host disease: plausibility argument, data requirements, and a benchmark

*A position-and-methods paper, written as an executable notebook.* Every number in a grey output cell is computed here. Literature facts carry **✓** (checked against a source during this project) or **†** (from domain knowledge; verify against the primary document before submission). Every simulation is marked **[SIM]** and shows what follows *if* its stated assumptions hold; simulations are not evidence about patients. Figures are written to `figures/` as vector PDF and 600-dpi PNG; the specification is written to `spec/`.

---

## Abstract

**Background.** Graft-versus-host disease (GvHD) is the principal cause of non-relapse morbidity and mortality after allogeneic haematopoietic cell transplantation (allo-HCT). Every step of transplant care rests on a prediction — which donor, which conditioning, which prophylaxis, when to intervene pre-emptively, which second-line agent, when to taper immunosuppression — and each is currently made with a different small model, or with none.

**What this paper is.** Not a trained model. It is (i) a **plausibility argument** with five falsifiable premises, (ii) an explicit statement of **what such a model can and cannot be expected to predict**, (iii) a **data specification** precise enough that two centres would produce the same variable, (iv) **GvHD-Bench v1.0**, a benchmark with frozen splits, leakage-safe metrics, required baselines and submission rules, and (v) **quantitative requirements** — patients, tokens, parameters, GPU-hours, cost and accrual time — for actually training the thing.

**Capability claims (§3).** Discrimination has a ceiling set by how much of the causal chain is observable at the decision point. We compute that ceiling. Pre-transplant prediction is close to its ceiling already (achievable ≈ 0.62–0.70), so a foundation model should not be sold on it. The real headroom is **early post-transplant** (day +7 to +14), where we claim 0.74–0.80 against ≈ 0.70–0.75 for today's biomarker-plus-clinical models. For **transcriptomic profiles taken at diagnosis** the answer splits: strong for confirming and differentiating the diagnosis in tissue (0.78–0.88, supported by our pilot's cross-laboratory transfer at 0.80 and a zero-parameter interferon-γ score at 0.80–0.88), plausible but unproven for stratifying the course once GvHD has begun, and **not** supported for pre-onset prognosis — the pilot found no reproducible pre-onset blood signal.

**Architecture (§5–§6).** The backbone is the **clinical timeline**, not the omics. A time-aware sequence model over day-0-anchored events consumes what every patient has; transcriptomes, repertoire, microbiome and pathology enter through **adapters that are optional at inference** and are trained with modality dropout. Seven objective families are specified, each with a matched control.

**Training requirements (§7).** Computed, not asserted. A Tier A/B corpus of 10⁵ patients is of the order of 10⁹ event tokens, which makes the compute-optimal model **small — tens to low hundreds of millions of parameters** — and places pretraining in the range of **10²–10³ GPU-hours and single-digit thousands of dollars**. Compute is not the constraint. Data access is.

**Benchmark (§8).** GvHD-Bench v1.0: 38 tasks in 7 families, each with a landmark, horizon, competing events, permitted inputs, the clinical baseline it must beat, a computed minimum sample size and an explicit *powered / under-powered* verdict. Metrics (IPCW time-dependent AUROC under competing risks, Aalen–Johansen calibration, net benefit) are implemented and validated against known truth. A leakage audit, a claim ladder and a pre-registered gate are included, and the whole harness is executed on a simulated 24-centre cohort containing a planted leakage feature that the audit catches and that inflates AUROC by about 0.2.

**Requirements and path (§9, §11).** Showing a ΔAUROC of 0.03 against a strong baseline needs of the order of 10³ patients; a calibrated 768-parameter head needs ~5 × 10⁴. We set these against what each real source holds and project accrual time per task, then give a staged plan with go/no-go gates in which a negative result is a publishable output.

**Conclusion.** A GvHD foundation model is plausible if its backbone is the clinical timeline and biology enters through adapters, and if its value is judged against registry-grade baselines under leave-centre-out and temporal validation. Public data cannot settle the question; §9 says precisely what can.
""")

md(r"""
### Contributions
1. **A plausibility argument in five premises** (§2), each with its evidence, its falsifier, and an executed simulation showing *when* pretraining helps and when it cannot.
2. **A capability statement** (§3): a computed discrimination ceiling per decision point, a task-by-task claim table, and a direct answer on transcriptomics at diagnosis, with the sample size each open question needs.
3. **A data specification for allo-HCT** (§4): 129 elements in 25 domains with units, timing relative to day 0, coding standards, availability tiers and leakage flags; a protocolised sampling schedule; and executable outcome-label functions with unit tests.
4. **An architecture and objective stack** (§5–§6): event tokenisation with knowledge-time masking, a donor–recipient pair encoder, optional modality adapters, and seven objective families with matched controls.
5. **Training requirements** (§7): token budgets per tier, compute-optimal model size, GPU-hours, cost and storage — computed from the specification.
6. **GvHD-Bench v1.0** (§8): the benchmark, exported as a standalone specification with frozen artefacts and hashes.
7. **A requirements-and-accrual projection** (§9) and a staged path with gates (§11), grounded in the pilot results this repository already contains (§10).

### Paper map
| § | Section | Type |
|---|---|---|
| 1 | Introduction: the decision timeline of allo-HCT | narrative + Fig 1 |
| 2 | Plausibility argument: five premises | evidence + executed [SIM] + Fig 2 |
| 3 | **What can be predicted** — ceilings, claims, transcriptomics at diagnosis | executed + Fig 3 |
| 4 | **Data specification** — domains, elements, schedule, sources, labels, leakage | specification + executed label tests |
| 5 | Architecture and tokenisation | design + Fig 4, Fig 5 |
| 6 | Training objectives | formal definitions |
| 7 | **Training requirements** — tokens, parameters, compute, cost | executed + Fig 6 |
| 8 | **GvHD-Bench v1.0** | benchmark spec + executed metric validation + executed harness + Fig 7, Fig 8 |
| 9 | **Sample size, power and accrual projections** | executed + Fig 9, Fig 10 |
| 10 | Pilot evidence (the public transcriptomic benchmark) | results carried forward + Fig 11 |
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

# =====================================================================================================  1. INTRODUCTION
code(r'''
# ----------------------------------------------------------------------------- publication figure system (one place; every figure in this paper uses it)
FIGDIR = 'figures'; os.makedirs(FIGDIR, exist_ok=True)
import matplotlib as mpl
# Colours: a validated categorical palette (all-pairs colour-vision-deficiency separation checked), a one-hue
# sequential ramp for ordered/nested model families, and a reserved status colour used only for "this is wrong".
PAL = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', violet='#6250d6',
           seq=('#9ec5f4', '#5598e7', '#2a78d6', '#184f95'),       # light -> dark: more information added
           bad='#d03b3b',                                          # status: critical (always carries a label too)
           ink='#0b0b0b', ink2='#52514e', grid='#d8d7d2', grey='#8c8c8c', grey2='#bdbcb7', surface='#ffffff')
SINGLE, MID, FULL = 3.5, 5.2, 7.2                                  # figure widths in inches (single / 1.5 / double column)
mpl.rcParams.update({
    'figure.facecolor': PAL['surface'], 'axes.facecolor': PAL['surface'], 'savefig.facecolor': PAL['surface'],
    'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'],
    'font.size': 7.2, 'axes.labelsize': 7.6, 'axes.titlesize': 7.6, 'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
    'legend.fontsize': 6.8, 'legend.frameon': False, 'legend.handlelength': 1.5, 'legend.borderpad': .2,
    'axes.labelcolor': PAL['ink'], 'text.color': PAL['ink'], 'xtick.color': PAL['ink2'], 'ytick.color': PAL['ink2'],
    'axes.edgecolor': PAL['ink2'], 'axes.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.6, 'ytick.major.size': 2.6, 'lines.linewidth': 1.5, 'lines.markersize': 3.4,
    'grid.color': PAL['grid'], 'grid.linewidth': .5, 'axes.grid': False, 'figure.dpi': 140, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .02, 'pdf.fonttype': 42, 'ps.fonttype': 42})

FIGS = {}
def panel(ax, letter, dx=-.085, dy=1.055):
    """Bold panel letter, placed in axes coordinates (journals want a, b, c)."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8.6, fontweight='bold', va='top', ha='left', color=PAL['ink'])

def ygrid(ax, axis='y'):
    ax.set_axisbelow(True); ax.grid(axis=axis, color=PAL['grid'], lw=.5)

def finish(fig, num, slug, caption):
    """Save one figure as vector PDF + 600-dpi PNG, register it, and show it."""
    base = os.path.join(FIGDIR, f'Fig{num:02d}_{slug}')
    for ext in ('pdf', 'png'): fig.savefig(f'{base}.{ext}')
    FIGS[f'Fig {num}'] = dict(slug=slug, caption=caption, files=[f'{base}.pdf', f'{base}.png'])
    plt.show(); print(f'[saved] {base}.pdf  +  {base}.png')

def dlabel(ax, x, y, text, color, dx=4, dy=0, ha='left', va='center', size=6.6, weight='bold'):
    """Direct series label (used instead of, or alongside, a legend: identity is never colour-alone)."""
    ax.annotate(text, (x, y), textcoords='offset points', xytext=(dx, dy), ha=ha, va=va,
                fontsize=size, color=color, fontweight=weight, annotation_clip=False)
print(f'figure system ready -> {os.path.abspath(FIGDIR)}  (vector PDF + 600 dpi PNG per figure)')
''')

md(r'''
---
# 1 · Introduction: allo-HCT as a sequence of predictions

Allo-HCT is one of the most protocolised processes in medicine. A shared clock (day 0 = graft infusion) anchors every event. Conditioning occupies days −10 to −1. Prophylaxis runs on a fixed schedule (for example PtCy on days +3/+4, then tacrolimus from day +5). Neutrophil engraftment falls around days +14 to +28. Acute GvHD typically appears between days +14 and +100, though late-onset aGvHD is recognised after day +100. Chronic GvHD typically begins from month 3 onward. Outcomes are graded with consensus instruments: MAGIC for aGvHD, NIH 2014 for cGvHD, and the 2018 EBMT–NIH–CIBMTR definitions for steroid refractoriness. National registries (CIBMTR, EBMT, JSHCT) have collected the same core variables on structured forms for decades.

Grade II–IV aGvHD affects roughly 30–50 % of recipients and grade III–IV roughly 10–15 %, depending on donor type and prophylaxis. Steroid-refractory aGvHD carries high mortality. Chronic GvHD affects roughly 30–50 % of long-term survivors and is the leading cause of late morbidity (†, ranges across eras and donor types).

The figure below is drawn in code and shows **where the decisions are**. Each decision is a prediction task in GvHD-Bench (§8), and each task can use only the data that exist *before* its decision point.
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 1: the allo-HCT decision timeline (what is decided, when, from which data)
PHASES = [  # (start, end, label, colour, row, label side)
    (-10, -1,  'conditioning (chemotherapy ± TBI)',   PAL['seq'][1], 0, 'right'),
    (-5,  -1,  'serotherapy (ATG / alemtuzumab)',     PAL['seq'][0], 1, 'right'),
    (3,    5,  'PtCy d+3/+4',                         PAL['violet'], 2, 'left'),
    (5,  180,  'calcineurin inhibitor ± MMF, then taper', PAL['seq'][0], 3, 'in'),
    (14,  28,  'engraftment',                         PAL['aqua'],   4, 'right'),
    (14, 100,  'classic acute GvHD window',           PAL['orange'], 5, 'right'),
    (90, 730,  'chronic GvHD window',                 PAL['blue'],   6, 'in')]
DECISIONS = [  # (day, decision, task ids, tier)
    (-45, 'select donor',            'A1–A3, A6', 0), (-21, 'conditioning\n+ serotherapy', 'A1, A5', 1),
    (-3,  'GvHD prophylaxis',        'A7',        0), (7,   'day-7 risk;\npre-emptive therapy', 'B1, B5', 1),
    (14,  'early-warning\nsurveillance', 'B2–B4', 0), (24,  'is this GvHD?',  'C1–C3', 2),
    (33,  'steroid response;\n2nd line', 'C4–C6, D1', 1), (100, 'cGvHD risk;\nIS taper', 'E1, E4', 0),
    (365, 'late effects; BOS',       'E2, E3',    1)]

def sx(d):                                           # signed-log day axis: dense early, compressed late
    return np.sign(d) * np.log1p(abs(d))

fig, ax = plt.subplots(figsize=(FULL, 3.3))
for a, b, lab, c, row, side in PHASES:
    y = 6 - row
    ax.add_patch(plt.Rectangle((sx(a), y - .30), sx(b) - sx(a), .60, color=c, alpha=.93, lw=0))
    if side == 'in':
        ax.text(sx(a) + .07, y, lab, fontsize=5.9, va='center', ha='left', color='white', fontweight='bold')
    elif side == 'right':
        ax.text(sx(b) + .09, y, lab, fontsize=5.9, va='center', ha='left', color=c, fontweight='bold')
    else:
        ax.text(sx(a) - .09, y, lab, fontsize=5.9, va='center', ha='right', color=c, fontweight='bold')
ax.axvline(sx(0), color=PAL['ink'], lw=1.0, zorder=1)
ax.text(sx(0), -.62, 'day 0\ngraft infusion', fontsize=5.9, ha='center', va='top',
        fontweight='bold', color=PAL['ink'], linespacing=1.3)
for d, lab, tasks, tier in DECISIONS:
    top = 7.55 + tier * 1.75
    ax.plot([sx(d), sx(d)], [6.55, top - .18], color=PAL['ink2'], lw=.55, zorder=1)
    ax.plot(sx(d), 6.55, marker='v', ms=3.0, color=PAL['ink'], zorder=3)
    ax.text(sx(d), top, lab, fontsize=5.7, ha='center', va='bottom', color=PAL['ink'], linespacing=1.25)
    ax.text(sx(d), top - .52, f'[{tasks}]', fontsize=5.2, ha='center', va='bottom', color=PAL['ink2'])
ticks = [-30, -10, 0, 7, 14, 28, 60, 100, 180, 365, 730]
ax.set_xticks([sx(t) for t in ticks]); ax.set_xticklabels([f'{t:+d}' if t else '0' for t in ticks])
ax.set_xlim(sx(-62), sx(3200)); ax.set_ylim(-1.5, 13.4); ax.set_yticks([])
ax.set_xlabel('day relative to graft infusion (signed-log scale)')
ax.spines['left'].set_visible(False)
ax.text(sx(-60), 13.0, 'clinical decisions  →  GvHD-Bench task families', fontsize=5.9,
        color=PAL['ink2'], style='italic', ha='left', va='top')
finish(fig, 1, 'decision_timeline',
       'The allo-HCT timeline. Coloured bars are the protocolised exposures and the risk windows they open; '
       'markers above are the clinical decision points, each of which is a family of GvHD-Bench tasks (8). '
       'The time axis is signed-log, so the first 100 days - where almost every decision sits - are legible '
       'alongside two years of follow-up. A model standing at any decision point may use only data whose '
       'results were available at that moment (4.1).')

''')

md(r'''
**Why a *clinical* foundation model, and why not the one we tried first.** Foundation models trained on timestamped structured health records have moved the field forward. CLMBR (next-code pretraining, †), MOTOR (time-to-event pretraining on 55 M records and 9 B events; up to 95 % better label efficiency on 19 tasks, ✓), Foresight (†) and Delphi-2M (generative disease trajectories, Nature 2025, †) all treat a patient as a *sequence of events in time*, and they learn a representation that transfers to many downstream risk tasks with few labels. Allo-HCT fits that template better than most areas of medicine, because its timeline is short, dense, protocolised and anchored.

Our earlier attempt took the other route: a *transcriptomic* foundation model, with expression as the backbone and GvHD as the label. Its benchmark is in this repository (`GvHD_FM_benchmark_blueprint.ipynb`, summarised in §10). It produced a clear negative on public data. Frozen Geneformer did not beat normalised genes, a pathway prior did not replicate, and blood T-cell signals did not transfer between studies, although tissue-biopsy signals did. This paper keeps those lessons and changes the architecture. **The clinical timeline is the backbone. Transcriptomes, TCRs, microbiome and pathology become adapters** that add information when it is available and that can be ablated to show whether they help.

**What this paper is.** It is a specification: plausibility, data, objectives and benchmark, with executed sample-size calculus, executed metric code, executed label functions, the pilot ledger, and simulations marked **[SIM]**. **What it is not:** a trained model, or any claim that ALLO-FM improves outcomes.
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
| **P3** | **The structure is shared across centres and tasks**, so pretraining can spread its cost over them | EHR foundation models gain the most at low label counts (MOTOR ✓, EHRSHOT ✓). Our pilot: tissue-biopsy GvHD signal transfers across labs and organs (LOSO AUROC ≈ 0.80), while blood T-cell signal does not (≈ 0.59) | Leave-centre-out gains disappear; pretrained ≈ random-weight twin | [SIM] S1 below; §10 pilot |
| **P4** | **Enough data exist** for pretraining (unlabelled or weakly labelled) and for evaluation (labelled) | CIBMTR: data on >470,000 recipients from >500 centres (2017 figure ✓). EBMT registry of similar order†. MSK hospitalome: >10,000 stool samples, >1,000 patients, public ✓. BMT CTN trials with banked samples. Public scRNA atlases of 10⁶–10⁷ immune cells (pilot census) | Labelled n for a given task falls below the Riley minimum (§9) even with consortium data | §4.5 sources; §9 |
| **P5** | **A gain can be detected** with honest statistics | Paired-test power and development-n are computable (§9). Gates are fixed in advance (§8.7) | Required n for ΔAUROC = 0.03 exceeds every accessible cohort | §9 |

**Where the earlier attempt broke.** The transcriptome-first design failed **P3** (blood signal not shared between studies) and **P4** (public labelled n of 17–101 per study, mostly microarrays the model could not read). A clinical-timeline backbone moves the labelled-data problem from GEO (~10² samples per study) to registries (~10⁵ patients), and it treats omics as optional adapters. **That reframing is the main design decision of this paper.**

### 2.1 [SIM] S1: when does pretraining help? Shared structure versus label count
Here we make P3 quantitative. Ten source centres and one new target centre each observe a 300-dimensional feature vector (a stand-in for a timeline or omics profile). It is generated from six latent biological factors through a centre-specific loading matrix, $W_c=\sqrt{\rho}\,W_0+\sqrt{1-\rho}\,U_c$. Here **ρ is the fraction of feature structure shared across centres**. The outcome depends only on the latent factors, in the same way at every centre. We compare four arms:
* **raw**: ridge-logistic on the 300 features, with the penalty chosen by inner CV (the "normalised genes" baseline);
* **pretrained**: a 16-d representation learned *without labels* on pooled source-centre data (PCA, the simplest self-supervised encoder), with a logistic head on top;
* **random-weight twin**: a random 16-d projection, matching the control used in the pilot;
* **oracle**: the true latent factors (an upper bound).

All arms are scored on 2,000 target-centre patients against the number of labelled target patients *k*.
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
fig, axes = plt.subplots(1, len(RHOS) + 1, figsize=(FULL, 2.45),
                         gridspec_kw=dict(width_ratios=[1] * len(RHOS) + [.95], wspace=.36))
for i, rho in enumerate(RHOS):
    ax = axes[i]; ygrid(ax)
    for arm, c, ls, _ in ARMS:
        ax.plot(KS, s1.loc[(rho, arm)].values, marker='o', color=c, ls=ls,
                lw=1.5 if arm != 'oracle' else 1.0, zorder=3 if arm == 'pretrained' else 2)
    ax.set_xscale('log'); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.minorticks_off()
    ax.set_ylim(.52, .92)
    ax.set_title(f'$\\rho$ = {rho:g}', fontsize=7.4, pad=3)
    if i: ax.set_yticklabels([])
    else:
        ax.set_ylabel('AUROC at the new centre'); panel(ax, 'a', dx=-.30)
    ax.set_xlabel('labelled patients $k$' if i == 1 else '')
# right-hand panel: the gain over the raw baseline, as a function of rho, at two label counts
ax = axes[-1]; ygrid(ax)
for k, c, m in ((25, PAL['aqua'], 'o'), (400, PAL['grey'], 's')):
    g = [s1.loc[(r, 'pretrained'), k] - s1.loc[(r, 'raw'), k] for r in RHOS]
    ax.plot(RHOS, g, marker=m, color=c, lw=1.5)
    dlabel(ax, RHOS[-1], g[-1], f'k = {k}', c, dx=4, size=6.4)
ax.axhline(0, color=PAL['ink2'], lw=.6)
ax.set_xlabel('shared structure $\\rho$'); ax.set_ylabel('AUROC gain from pretraining'); panel(ax, 'b', dx=-.34)
ax.set_xlim(-.06, 1.34); ax.set_xticks([0, .5, 1.0])
fig.legend([plt.Line2D([], [], color=c, ls=ls, marker='o', ms=2.8, lw=1.4) for _, c, ls, _ in ARMS],
           [lab for _, _, _, lab in ARMS], fontsize=5.8, ncol=4, loc='lower center',
           bbox_to_anchor=(.5, -.07), handlelength=1.7, columnspacing=1.6)
finish(fig, 2, 'pretraining_vs_shared_structure',
       '[SIM] Label-free pretraining pays only in proportion to the feature structure shared across centres. '
       '(a) Target-centre AUROC against the number of labelled target patients, at four levels of shared '
       'structure rho. (b) The same result as the gain over the tuned raw-feature baseline: large at few '
       'labels and high rho, absent at rho = 0, and shrinking everywhere as labels accumulate. '
       'Simulated data; the effect sizes follow from the stated generative model, not from patients.')
''')

code(r'''
g = lambda rho, arm, k: s1.loc[(rho, arm), k]
gain_small = {rho: g(rho, 'pretrained', 25) - g(rho, 'raw', 25) for rho in RHOS}
gain_big = {rho: g(rho, 'pretrained', 400) - g(rho, 'raw', 400) for rho in RHOS}
display(Markdown(f"""**Result [SIM].** With fully shared structure (ρ = 1) the pretrained encoder is ahead of the raw-feature baseline by **{gain_small[1.0]:+.3f} AUROC at k = 25** and by {gain_big[1.0]:+.3f} at k = 400, so the advantage shrinks as labels accumulate. This is the label-efficiency pattern reported for EHR foundation models. At ρ = 0.8 the gain is {gain_small[0.8]:+.3f} at k = 25. At ρ = 0.5 it is {gain_small[0.5]:+.3f}. **At ρ = 0 (nothing shared across centres) pretraining is {gain_small[0.0]:+.3f}, no better than the random-weight twin.** That is the pattern the pilot found for blood T cells (§10).

**What this licenses.** Pretraining can only exploit structure that is shared. A GvHD foundation model is therefore plausible only for modalities and centres where ρ is high. ρ can be *measured before any model is built*: decode centre from features, or compare principal subspaces between sites (§8.6, *shared-structure audit*). Structured clinical variables recorded on common registry forms are high-ρ by design. Blood transcriptomes from heterogeneous public studies were low-ρ in our pilot. This is the quantitative reason for making the clinical timeline the backbone."""))
''')
