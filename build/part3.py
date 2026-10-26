from part1 import CELLS, md, code

md(r'''
---
# 5 · ALLO-FM: architecture

## 5.1 The shape of the model


**Backbone = the timeline, not the omics.** This is the single most important design choice, and it is the one that differs from our first attempt (§10). The sequence model consumes what every transplant patient has (labs, drugs, vitals, stages, registry facts) and treats the deep assays as *extra tokens that may be absent*. A patient with Tier A data only still gets a prediction; a patient with Tier C/D data gets a prediction informed by more tokens. Modality dropout during training (§6.3) is what makes this work rather than collapsing onto whichever modality is most often present.

## 5.2 Tokenisation, in detail

* **Code embedding.** Each element in the dictionary maps to a vocabulary entry (LOINC / RxNorm / SNOMED / the HCT extension). Vocabulary size is of order 10⁴.
* **Value encoding.** Numeric values are encoded two ways at once: a *decile bin within the element's own training distribution* (robust to unit drift between centres), and a standardised scalar passed through a small MLP. Categorical values use their own embedding. A "not done" value is its own token, so informative missingness is representable (D25).
* **Time encoding.** `φ(Δt)` is a random-Fourier / sinusoidal encoding of **time since day 0** and **time since the previous event**, on a signed-log scale so that day +7 and day +8 are far apart while day +400 and day +410 are close. Absolute calendar time is given only as the era flag (D25), never as a raw date, so the model cannot memorise cohorts.
* **Drug exposure.** A drug is not one token: it is a *rate function*. Each administration contributes a token, and the encoder also receives a derived "exposure so far" channel (cumulative mg/kg, trough level, time in therapeutic range). This matters because the GvHD-relevant quantity for a calcineurin inhibitor is exposure, not prescription.
* **Clamp.** Every head receives a landmark L and a mask that zeroes every token with `t_known > L`. The mask is applied inside the attention, and the harness tests it (§8.8).

## 5.3 The pair encoder

Alloreactivity is a property of a **donor–recipient pair**, so the pair gets its own encoder rather than a list of flat features.

* HLA alleles (recipient and donor, 2–4 field) are embedded; the pair representation is built from the *pairing*, with separate channels for GvH-direction and HvG-direction mismatch, plus derived immunogenetics (DPB1 TCE permissiveness, HLA-B leader, HED, eplet/PIRCHE load, KIR B-content, KIR-ligand missing-self).
* With germline genotypes (D20), predicted minor-antigen disparity load enters as a vector.
* The pair encoder is pretrained by a *matching* objective (O-P1, §6.2): given a recipient and a set of candidate donors, score how similar the pairing is, so that the representation groups pairs by disparity structure rather than by centre. This is also the component that makes the donor-ranking task (A6) possible.

## 5.4 What the heads are

The model is not a GvHD classifier. It is a **multi-state model** on the transplant timeline: from the alive-and-GvHD-free state a patient can move to aGvHD (and within it to grade III–IV, steroid-refractory, resolved), to cGvHD, to relapse, or to death. Heads emit discrete-time cause-specific hazards on a grid (daily to day +100, then weekly to day +365), from which every task's quantity is a function: cumulative incidence with competing risks, time-dependent AUROC, calibration, and expected days alive and GvHD-free. One multi-state head rather than many binary heads is also what makes the tasks mutually consistent — a model cannot claim a 40 % chance of grade III–IV aGvHD by day 100 and a 90 % chance of being GvHD-free at day 100.
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 4: the ALLO-FM architecture (every box is sized from its own text, so nothing can overflow)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
FW = FULL
fig = plt.figure(figsize=(FW, 4.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
_rend = fig.canvas.get_renderer()
def _wid(txt, size, weight='normal'):
    t = ax.text(0, 0, txt, fontsize=size, fontweight=weight)
    w = _MplText.get_window_extent(t, _rend).width / fig.dpi; t.remove(); return w
TFS, BFS, PADX, PADY, GAPT = 6.8, 6.3, .10, .09, .07
TPITCH, BPITCH = TFS * 1.40 / 72, BFS * 1.55 / 72            # fixed line pitch: even rhythm whatever the glyphs in a line
ASC_T, ASC_B = TFS * .74 / 72, BFS * .74 / 72
def make_box(title, body, fc, ec, tc=None, bc=None, center=False):
    tl, bl = title.split('\n'), body.split('\n')
    wt = max(_wid(l, TFS, 'bold') for l in tl); wb = max(_wid(l, BFS) for l in bl if l.strip())
    block = len(tl) * TPITCH + GAPT + len(bl) * BPITCH
    return dict(tl=tl, bl=bl, fc=fc, ec=ec, tc=tc or PAL['ink'], bc=bc or PAL['ink2'], center=center,
                w=max(wt, wb) + 2 * PADX, h=block + 2 * PADY, block=block)
BLU, VIO, WHT = '#eef4fd', '#f4f1fb', '#ffffff'
col1 = [make_box('STATIC PAIR   D01–D06, D20',
                 'recipient · disease · DRI · HCT-CI\ndonor age, sex, parity · CMV · ABO\nHLA (R and D) · DPB1-TCE · B-leader\nKIR · HED · minor antigens', BLU, PAL['seq'][2]),
        make_box('EVENT TIMELINE   D08–D13, D24',
                 'conditioning · PtCy · CNI troughs\nMMF · antibiotics · CBC · CRP\nbilirubin · creatinine · vitals\nstool volume · rash % BSA\nsteroid, 2nd-line exposure', BLU, PAL['seq'][2]),
        make_box('MODALITY ADAPTERS   D07, D14–D19',
                 'Tier C: flow, plasma proteins,\nmicrobiome, transcriptome\nTier D: TCRβ repertoire,\nwhole-slide images, single-cell', VIO, PAL['violet'])]
col2 = [make_box('PAIR ENCODER',
                 'allele embeddings → separate\nGvH and HvG mismatch channels;\npretrained by donor matching\n(O-C3)', WHT, PAL['ink2']),
        make_box('EVENT TOKENISER',
                 'emb(code) + emb(value decile)\n+ φ(Δt, signed-log) + emb(source)\n"not done" is its own token', WHT, PAL['ink2']),
        make_box('PER-MODALITY ENCODERS',
                 'one summary token per assay,\nplaced at its t_known; masked-\nmodality objectives (O-M1, O-M2);\ndropout p ≈ 0.5, so none is\nrequired at inference', WHT, PAL['ink2'])]
for r_ in range(3):                                          # rows share a height so the arrows are level
    h = max(col1[r_]['h'], col2[r_]['h']); col1[r_]['h'] = col2[r_]['h'] = h
w1, w2 = max(b['w'] for b in col1), max(b['w'] for b in col2)
for b in col1: b['w'] = w1
for b in col2: b['w'] = w2
VG = .17; H = sum(b['h'] for b in col1) + 2 * VG
bb_ = make_box('TIMELINE\nTRANSFORMER', 'causal, time-aware,\ncentre-adversarial\n\nlandmark mask\nt_known ≤ L\nenforced inside\nthe attention',
               PAL['seq'][3], PAL['seq'][3], tc='white', bc='white', center=True)
bb_['w'] = bb_['w'] + .05; bb_['h'] = H
col4 = [make_box('HEADS at landmark L', 'multi-state, cause-specific hazards\nfor aGvHD II–IV and III–IV,\nsteroid-refractory, cGvHD,\nrelapse, NRM and death', '#eaf7f1', PAL['aqua']),
        make_box('DERIVED QUANTITIES', 'organ-specific stages (O-S2)\nday +28 response (O-S1)\ncounterfactual contrast (O-S3)\ncumulative incidence,\ncalibration, net benefit', '#eaf7f1', PAL['aqua'])]
w4 = max(b['w'] for b in col4)
for b in col4: b['w'] = w4
M, GAP_MIN = .07, .26
FW = max(FULL, 2 * M + w1 + w2 + bb_['w'] + w4 + 3 * GAP_MIN)      # a wider fallback font widens the figure instead of overflowing it
gap = (FW - 2 * M - w1 - w2 - bb_['w'] - w4) / 3
LEG_H = .50; FH = H + LEG_H + .20
fig.set_size_inches(FW, FH); ax.set_xlim(0, FW); ax.set_ylim(0, FH)
x1 = M; x2 = x1 + w1 + gap; x3 = x2 + w2 + gap; x4 = x3 + bb_['w'] + gap; ytop = FH - .10
def draw(b, x, y):                                           # y = top edge
    ax.add_patch(FancyBboxPatch((x, y - b['h']), b['w'], b['h'], boxstyle='round,pad=0,rounding_size=.05',
                                fc=b['fc'], ec=b['ec'], lw=.9))
    y0 = y - (b['h'] - b['block']) / 2 if b['center'] else y - PADY               # centred boxes centre their text block vertically
    xt, ha = (x + b['w'] / 2, 'center') if b['center'] else (x + PADX, 'left')
    for i, l in enumerate(b['tl']):
        ax.text(xt, y0 - ASC_T - i * TPITCH, l, fontsize=TFS, fontweight='bold', va='baseline', ha=ha, color=b['tc'])
    yb = y0 - len(b['tl']) * TPITCH - GAPT
    for k, l in enumerate(b['bl']):
        if l.strip(): ax.text(xt, yb - ASC_B - k * BPITCH, l, fontsize=BFS, va='baseline', ha=ha, color=b['bc'])
def arrow(xa, ya, xb, yb):
    ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle='-|>', mutation_scale=8, color=PAL['ink2'], lw=.9, shrinkA=0, shrinkB=0))
y = ytop; mids = []
for b1, b2 in zip(col1, col2):
    draw(b1, x1, y); draw(b2, x2, y); mids.append(y - b1['h'] / 2)
    arrow(x1 + w1, mids[-1], x2, mids[-1]); arrow(x2 + w2, mids[-1], x3, mids[-1]); y -= b1['h'] + VG
draw(bb_, x3, ytop)
h4 = sum(b['h'] for b in col4) + VG; y = ytop - (H - h4) / 2
for b in col4:
    draw(b, x4, y); arrow(x3 + bb_['w'], y - b['h'] / 2, x4, y - b['h'] / 2); y -= b['h'] + VG
for xk, fc, ec, lab in ((M, BLU, PAL['seq'][2], 'every patient (Tier A/B)'), (M + 2.55, VIO, PAL['violet'], 'optional at inference (Tier C/D)')):
    ax.add_patch(FancyBboxPatch((xk, .22), .16, .16, boxstyle='round,pad=0,rounding_size=.03', fc=fc, ec=ec, lw=.9))
    ax.text(xk + .24, .30, lab, fontsize=6.6, va='center', ha='left')
finish(fig, 4, 'architecture',
       'ALLO-FM. The backbone is the day-0-anchored clinical event timeline, which every transplant patient has. Biology '
       'enters through modality adapters that each contribute one summary token at the time its result became known; they '
       'are trained with modality dropout and are therefore optional at inference. Every head reads only tokens whose '
       'results were available at its landmark L; the mask is enforced inside the attention and tested in section 8.11.')
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 3: time encoding and the landmark mask, made concrete on one synthetic patient
def phi(dt, k=6):
    """signed-log time features: sin/cos at several scales + the raw signed-log value (what the backbone receives)."""
    s = np.sign(dt) * np.log1p(np.abs(dt))
    return np.concatenate([[s]] + [[np.sin(s * 2 ** j), np.cos(s * 2 ** j)] for j in range(k)])

EV = [(-10, 'busulfan, day −10', 'drug'), (-9, 'busulfan, day −9', 'drug'), (-7, 'fludarabine', 'drug'), (-1, 'TBI 2 Gy', 'proc'),
      (0, 'graft infusion (CD34 5.1e6/kg)', 'proc'), (3, 'PtCy 50 mg/kg, day +3', 'drug'), (4, 'PtCy 50 mg/kg, day +4', 'drug'),
      (5, 'tacrolimus start', 'drug'), (7, 'ST2 / REG3α, result day +9', 'assay'), (8, 'CRP 84 mg/L', 'lab'),
      (14, 'ANC engraftment', 'derived'), (16, 'stool 700 mL/d', 'stage'), (18, 'rash 12% BSA', 'stage'),
      (20, 'GI biopsy, signed out day +23', 'assay'), (21, 'methylprednisolone 2 mg/kg', 'drug*')]
L = 21.0
rows = []
for t, lab, kind in EV:
    tk = t + (2 if 'ST2' in lab else 3 if 'biopsy' in lab else 0)
    rows.append(dict(event=lab, kind=kind, t_valid=t, t_known=tk, visible_at_L=(tk <= L), banned_proxy=kind.endswith('*')))
ev = pd.DataFrame(rows)
ev['usable for a day+21 prediction'] = np.where(ev.banned_proxy, 'NO (treatment proxy)', np.where(ev.visible_at_L, 'yes', 'NO (t_known > L)'))
fig = plt.figure(figsize=(FULL, 3.6))
axa = fig.add_axes([.285, .165, .395, .765]); axb = fig.add_axes([.715, .275, .20, .40])
ax = axa; ax.set_axisbelow(True); ax.grid(axis='x', color=PAL['grid'], lw=.4)
for i, r in ev.iterrows():
    ok = r['usable for a day+21 prediction'] == 'yes'
    c = PAL['aqua'] if ok else (PAL['bad'] if r.banned_proxy else PAL['orange'])
    if r.t_known != r.t_valid:                                       # the lag between "it happened" and "we knew"
        ax.plot([r.t_valid, r.t_known], [i, i], color=c, lw=1.2, zorder=3)
        ax.plot(r.t_known, i, 'o', ms=4.4, mfc='white', mec=c, mew=1.1, zorder=4)
    ax.plot(r.t_valid, i, 'o', color=c, ms=4.6, zorder=5)
ax.set_yticks(range(len(ev))); ax.set_yticklabels(ev.event); ax.tick_params(axis='y', length=0)
ax.axvline(L, color=PAL['ink'], ls='--', lw=.9, zorder=2)
ax.text(L + .5, -1.0, 'landmark L = day +21', fontsize=6.4, ha='left', va='center', fontweight='bold')
ax.set_xticks([-10, 0, 7, 14, 21]); ax.set_xticklabels(['−10', '0', '+7', '+14', '+21'])
ax.set_xlim(-12, 33); ax.set_ylim(-1.7, len(ev) - .4); ax.set_xlabel('day relative to graft infusion'); ax.spines['left'].set_visible(False)
ax.legend([plt.Line2D([], [], color=c, marker='o', ls='none', ms=4.6) for c in (PAL['aqua'], PAL['orange'], PAL['bad'])] +
          [plt.Line2D([], [], color=PAL['ink2'], marker='o', ls='-', ms=4.4, mfc='white', mew=1.1)],
          ['legal input at L', 'result known only after L', 'banned: treatment proxy', 'when the result is known'],
          loc='upper left', labelspacing=.35, handletextpad=.3)
panel(ax, 'a')
TT = np.array([-10, -1, 0, 1, 3, 7, 14, 21, 60, 100, 365])
P = np.array([phi(t) for t in TT]); nrm = np.linalg.norm(P, axis=1); sim = P @ P.T / (nrm[:, None] * nrm[None, :])
im = axb.imshow(sim, cmap='Blues', vmin=-.2, vmax=1)
axb.set_xticks(range(len(TT))); axb.set_xticklabels(TT, rotation=90); axb.set_yticks(range(len(TT))); axb.set_yticklabels(TT)
axb.set_xlabel('day'); axb.set_ylabel('day'); axb.grid(False); panel(axb, 'b')
cax = axb.inset_axes([1.07, 0, .07, 1]); cb = fig.colorbar(im, cax=cax); cb.set_label('similarity of $\\varphi(\\Delta t)$'); cb.ax.tick_params(labelsize=6.4)
finish(fig, 5, 'knowledge_time_and_time_encoding',
       'Knowledge time and time encoding. (a) One patient timeline. Filled marker: when an event happened; open marker: when its '
       'result became known. An event is a legal input at landmark L only if its result was known by then, which is not the same '
       'as having happened by then: the day-20 biopsy signs out on day +23 and is inadmissible, and the day-21 steroid order is '
       'banned outright as a proxy for the outcome. (b) The signed-log time encoding of the model keeps early days distinguishable '
       '(day +7 and +14 are different decisions) and compresses late follow-up.')
display(md_table(ev[['event', 'kind', 't_valid', 't_known', 'usable for a day+21 prediction']]))
''')

code(r'''
n_usable = int((ev['usable for a day+21 prediction'] == 'yes').sum())
display(Markdown(f"""**Result.** Of {len(ev)} events on this patient's timeline, **{n_usable} are legal inputs** for a day +21 prediction. The day-7 biomarker draw is legal (result known day +9), the day-20 biopsy is **not** (signed out day +23, after the landmark), and the day-21 steroid order is banned outright because it is a treatment proxy for the outcome being predicted. A pipeline that filtered on `t_valid ≤ L` alone would have used the biopsy and inflated its own performance. The right panel shows why the signed-log time encoding is the right choice for this disease: day +7 and day +14 must be distinguishable (different biology, different decisions), whereas day +300 and day +365 need not be."""))
''')

md(r'''
---
# 6 · Training objectives

Three stages: **(1) self-supervised pretraining** on timelines and modalities, **(2) alignment** across modalities and centres, **(3) supervised adaptation** to the multi-state clinical outcome. An objective earns a place only if it passes three filters, taken from the published audits of single-cell foundation models and confirmed by our own pilot: (i) it is computable from data that exist at Tier A/B scale; (ii) its loss is not trivially minimised by a linear model or by a shortcut such as missingness; (iii) it comes with a **matched control** that shows what it adds beyond a dumb twin.

## 6.1 Notation
A patient's record is an event sequence $E=\{e_i\}$ with $e_i=(c_i, v_i, u_i, t_i, \tau_i)$ — code, value, unit, `t_valid`, `t_known` — plus static pair features $s$. Let $E_{\le L}=\{e_i:\tau_i\le L\}$ be the legal history at landmark $L$, and $z_L=f_\theta(E_{\le L}, s)$ the representation. Modality $m$ contributes a summary token $g^{(m)}_\phi(x^{(m)})$.

## 6.2 Objective catalogue

| # | Objective | Loss | Needs | Intrinsic metric vs its control | What it should buy | Failure mode |
|---|---|---|---|---|---|---|
| **O-T1** | **Next-event prediction** (the CLMBR-style backbone objective) | $-\sum_i\log p_\theta(c_{i+1}\mid E_{\le\tau_i})$ | Tier A/B timelines | top-k code accuracy vs a **frequency prior** and a **1-step Markov** model | generic timeline representation | collapses to predicting the daily CBC; fix by down-weighting high-frequency codes |
| **O-T2** | **Time-to-next-event / piecewise-exponential pretraining** (MOTOR-style) | $\sum_{k}\big[\delta_k\log\lambda_\theta^{(k)}(t)-\int\lambda_\theta^{(k)}\big]$ over thousands of code-specific "tasks" | Tier A/B timelines with dates | time-dependent C vs **Kaplan–Meier-only** and **Cox on 20 registry covariates** | the objective closest to the clinical question; best label efficiency | censoring is informative (transfer of care) → include a censoring model |
| **O-T3** | **Masked value modelling** on the timeline | $\mathbb E\sum_{i\in M}\ell(v_i, \hat v_i)$ (Huber for numeric, CE for categorical) | Tier A/B | held-out MSE vs **last-value-carried-forward** and **patient-mean**; must beat LVCF | imputation and a smoother representation | LVCF is very strong for labs; a model that cannot beat it has learned nothing |
| **O-T4** | **Exposure–response (PK/PD) auxiliary** | predict next tacrolimus trough / busulfan AUC from dose, weight, interacting drugs, CYP3A5 | D10, D20 | $R^2$ vs a **population PK model** | makes the model represent drug exposure, not just orders | overfits centre-specific protocols |
| **O-M1** | **Per-modality masked modelling** (flow panel, Olink, 16S, expression) | modality-specific masked reconstruction | Tier C/D | reconstruction vs **PCA-ridge** and **mean predictor** | adapters that are useful before any label | the pilot's warning: a well-fit objective need not transfer |
| **O-M2** | **Repertoire objective** | masked CDR3 / clone-abundance modelling; optional InfoNCE between a clone's sequence and its cell state | D16 | held-out clone retrieval vs **V/J+length** and **k-mer** baselines | donor-reactive clone representation | pilot result: no model class beat k-mers on an alloreactivity proxy |
| **O-C1** | **Cross-modality alignment** | InfoNCE between the timeline representation at $\tau$ and modality tokens measured at $\tau$ | paired samples | retrieval accuracy vs **random pairing**; alignment/uniformity | lets a Tier A patient inherit structure learned from Tier C/D patients | aligns to batch rather than biology if positives are mis-specified |
| **O-C2** | **Centre-adversarial invariance** | $\mathcal L_{\text{task}}-\lambda\,\mathcal L_{\text{centre}}$ (gradient reversal), or an explicit CORAL/MMD penalty | centre IDs | **centre decodability** from $z$ (lower is better) with biology preserved | the premise P3 lever: makes representations transferable | removes real case-mix differences along with batch; needs the two-axis score (§8.4) |
| **O-C3** | **Pair-matching** (donor–recipient) | InfoNCE over (recipient, true donor) vs sampled candidate donors | D06, D20 | matching accuracy vs **mismatch-count-only** | the representation donor ranking (A6) needs | learns registry conventions instead of immunogenetics |
| **O-S1** | **Multi-state supervised head** | cause-specific discrete-time NLL over {aGvHD II-IV, III-IV, SR, cGvHD, relapse, NRM, death}, with competing risks | labelled registry | time-dependent AUROC and calibration vs **clinical baselines** (§8.3) | the deployable model | the label definitions must be the §4.6 functions |
| **O-S2** | **Organ-specific and severity heads** | multi-label hazards for skin / upper GI / lower GI / liver (and cGvHD organ scores) | daily staging (D13, D23) | macro-AUROC vs **overall-grade-only** model | clinically actionable detail (which organ) | staging noise between centres |
| **O-S3** | **Counterfactual / treatment-arm head** | T-learner or DR-learner for prophylaxis platform; validated on randomised arms (BMT CTN 1703) | D10, D24, trial IDs | policy value and calibration-in-the-large **within the randomised subset** | the clinically useful question ("which prophylaxis for this pair?") | confounded in registry data; only randomised anchors can validate it |

## 6.3 Training recipe (and the ablations that must accompany it)

1. **Stage 1 — timeline pretraining.** O-T1 + O-T2 + O-T3 (+O-T4 where levels exist) on all available timelines, including patients with no GvHD and (for O-T1/O-T3) non-transplant haematology admissions if available for extra sequence data. Hold out whole centres and whole eras.
2. **Stage 2 — adapters and alignment.** Train each modality adapter with O-M1/O-M2 on whatever subset has that modality; then O-C1 to align adapters to the timeline, with **modality dropout** (each modality independently dropped with p ≈ 0.5) so the model never requires any of them. O-C2 is applied with λ tuned against the two-axis score, not against downstream AUROC (otherwise it is tuned on the test question).
3. **Stage 3 — adaptation.** O-S1 (+O-S2, +O-S3 where randomised data exist), with the backbone frozen *and* fine-tuned, both reported. Fine-tuning with few labels is seed-sensitive, so every fine-tuned number is reported over ≥ 5 seeds with its spread.
4. **Mandatory ablations, reported in the same table as the headline result:** random-weight twin of the whole backbone; no-pretraining (stage 3 only); each stage-1 objective removed in turn; each modality knocked out; pretraining-corpus scaling (10 %, 30 %, 100 %); and the shared-structure audit from §2.1.

**The lesson from the pilot, stated as a rule.** A pretrained representation must beat (a) its own random-weight twin, (b) the best simple baseline on the same inputs, and (c) the clinical baseline for the task. In the pilot, one Geneformer configuration cleared (a) and nothing cleared (b). **Clearing (a) alone is not a result.**
''')
