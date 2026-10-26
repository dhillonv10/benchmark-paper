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
# ----------------------------------------------------------------------------- Figure 4: the ALLO-FM architecture (drawn, so it can be read at print size)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(FULL, 3.9)); ax.set_xlim(0, 122); ax.set_ylim(0, 66); ax.axis('off')
def box(x, y, w, h, title, body, fc, tc=None, fs=5.3, tfs=6.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5,rounding_size=1.0',
                                fc=fc, ec=PAL['ink2'], lw=.7))
    ax.text(x + 1.4, y + h - 1.5, title, fontsize=tfs, fontweight='bold', va='top', color=tc or PAL['ink'])
    ax.text(x + 1.4, y + h - 5.0, body, fontsize=fs, va='top', color=tc or PAL['ink2'], linespacing=1.42)
def arrow(x1, y1, x2, y2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7,
                                 color=PAL['ink2'], lw=.8, connectionstyle=f'arc3,rad={rad}'))
# ---- inputs (left column)
box(0, 46, 36, 15, 'STATIC PAIR  D01–D06, D20',
    'recipient · disease · DRI · HCT-CI\n'
    'donor age, sex, parity · CMV · ABO\n'
    'HLA GL-strings (R and D) · DPB1-TCE\n'
    'B-leader · KIR · HED · minor antigens', '#eef4fd')
box(0, 26, 36, 16, 'EVENT TIMELINE  D08–D13, D24',
    'conditioning doses by day · PtCy\n'
    'CNI troughs · MMF · antibiotics\n'
    'CBC · bilirubin · CRP · creatinine\n'
    'vitals · stool volume · rash %BSA\n'
    'steroid and second-line exposure', '#eef4fd')
box(0, 1, 36, 22, 'MODALITY ADAPTERS  D07, D14–D19',
    'graft composition, flow          Tier C\n'
    'T-cell reconstitution, flow      Tier C\n'
    'plasma proteins ST2, REG3α       Tier C\n'
    'stool microbiome                 Tier C\n'
    'transcriptome, tissue / blood  Tier C/D\n'
    'TCRβ repertoire                  Tier D\n'
    'biopsy whole-slide image         Tier D', '#f4f1fb')
# ---- encoders (middle column)
box(41, 46, 31, 15, 'PAIR ENCODER',
    'allele embeddings → separate\n'
    'GvH-direction and HvG-direction\n'
    'mismatch channels; pretrained by\n'
    'donor matching (O-C3)', '#ffffff')
box(41, 28, 31, 12, 'EVENT TOKENISER',
    'emb(code) + emb(value decile)\n'
    '+ φ(Δt, signed-log) + emb(source)\n'
    '"not done" is its own token', '#ffffff')
box(41, 1, 31, 22, 'PER-MODALITY ENCODERS',
    'each contributes ONE summary\n'
    'token, placed at its t_known\n\n'
    'masked-modality objectives\n'
    '(O-M1 / O-M2)\n\n'
    'modality dropout p ≈ 0.5:\nnone is required at inference', '#ffffff')
# ---- backbone
ax.add_patch(FancyBboxPatch((77, 14), 16, 39, boxstyle='round,pad=0.5,rounding_size=1.0',
                            fc=PAL['seq'][3], ec=PAL['seq'][3], lw=.7))
ax.text(85, 50.5, 'TIMELINE\nTRANSFORMER', fontsize=6.4, fontweight='bold', ha='center', va='top',
        color='white', linespacing=1.3)
ax.text(85, 40, 'causal\ntime-aware\ncentre-adversarial', fontsize=5.3, ha='center', va='top',
        color='white', linespacing=1.5)
ax.text(85, 29, 'landmark mask\nt_known ≤ L\nenforced inside\nthe attention', fontsize=5.3, ha='center',
        va='top', color='white', linespacing=1.5, fontweight='bold')
# ---- heads
box(97, 33, 25, 21, 'HEADS at landmark L',
    'multi-state cause-specific\nhazards over:\n'
    'aGvHD II–IV · III–IV\nsteroid-refractory\ncGvHD · relapse · NRM\ndeath', '#eaf7f1')
box(97, 4, 25, 22, 'DERIVED QUANTITIES',
    'organ-specific stages (O-S2)\n\n'
    'day +28 response (O-S1)\n\n'
    'counterfactual arm contrast\n(O-S3)\n\n'
    'cumulative incidence,\ncalibration, net benefit', '#eaf7f1', fs=5.1)
for y in (53, 34, 12): arrow(36.6, y, 40.4, y)
arrow(72.6, 53, 76.4, 47, rad=.10); arrow(72.6, 34, 76.4, 33); arrow(72.6, 12, 76.4, 20, rad=-.10)
arrow(93.6, 43, 96.4, 43); arrow(93.6, 22, 96.4, 16, rad=-.10)
ax.text(18, 63.6, 'what every patient has   (Tier A/B)', fontsize=5.9, ha='center', color=PAL['ink2'], style='italic')
ax.text(0, 24.2, 'what some patients have   (Tier C/D — optional at inference)', fontsize=5.9, ha='left',
        color=PAL['ink2'], style='italic')
finish(fig, 4, 'architecture',
       'ALLO-FM. The backbone is the day-0-anchored clinical event timeline, which every transplant patient '
       'has. Biology enters through modality adapters that each contribute a single summary token at the time '
       'its result became known, are trained with modality dropout, and are therefore optional at inference. '
       'Every head reads only tokens whose results were available at its landmark L; the mask is enforced '
       'inside the attention and is tested in 8.11.')
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 3: time encoding and the landmark mask, made concrete on one synthetic patient
def phi(dt, k=6):
    """signed-log time features: sin/cos at several scales + the raw signed-log value (what the backbone receives)."""
    s = np.sign(dt) * np.log1p(np.abs(dt))
    return np.concatenate([[s]] + [[np.sin(s * 2 ** j), np.cos(s * 2 ** j)] for j in range(k)])

EV = [(-10, 'busulfan dose', 'drug'), (-9, 'busulfan dose', 'drug'), (-7, 'fludarabine', 'drug'), (-1, 'TBI 2 Gy', 'proc'),
      (0, 'graft infusion (CD34 5.1e6/kg)', 'proc'), (3, 'PtCy 50 mg/kg', 'drug'), (4, 'PtCy 50 mg/kg', 'drug'),
      (5, 'tacrolimus start', 'drug'), (7, 'ST2/REG3a (t_known=+9)', 'assay'), (8, 'CRP 84 mg/L', 'lab'),
      (14, 'ANC engraftment', 'derived'), (16, 'stool 700 mL/d', 'stage'), (18, 'rash 12% BSA', 'stage'),
      (20, 'GI biopsy taken (t_known=+23)', 'assay'), (21, 'methylprednisolone 2 mg/kg', 'drug*')]
L = 21.0
rows = []
for t, lab, kind in EV:
    tk = t + (2 if 'ST2' in lab else 3 if 'biopsy' in lab else 0)
    rows.append(dict(event=lab, kind=kind, t_valid=t, t_known=tk, visible_at_L=(tk <= L), banned_proxy=kind.endswith('*')))
ev = pd.DataFrame(rows)
ev['usable for a day+21 prediction'] = np.where(ev.banned_proxy, 'NO (treatment proxy)', np.where(ev.visible_at_L, 'yes', 'NO (t_known > L)'))
fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.0), gridspec_kw=dict(width_ratios=[1.55, 1], wspace=.30))
ax = axes[0]
for i, r in ev.iterrows():
    ok = r['usable for a day+21 prediction'] == 'yes'
    c = PAL['aqua'] if ok else (PAL['bad'] if r.banned_proxy else PAL['orange'])
    if r.t_known != r.t_valid:
        ax.annotate('', xy=(np.log1p(r.t_known), i), xytext=(sx(r.t_valid), i),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=1.0, mutation_scale=5, shrinkA=2, shrinkB=0))
        ax.plot(np.log1p(r.t_known), i, 'o', color=c, ms=2.6, mfc='white', mew=.9)
    ax.plot(sx(r.t_valid), i, 'o', color=c, ms=3.6)
    ax.annotate(r.event, (sx(r.t_valid), i), textcoords='offset points', xytext=(-6, 0),
                ha='right', va='center', fontsize=5.4, color=PAL['ink'])
ax.axvline(np.log1p(L), color=PAL['ink'], ls='--', lw=.9)
ax.text(np.log1p(L), -1.5, 'landmark\nL = day +21', fontsize=5.8, ha='center', va='top', fontweight='bold')
ax.set_yticks([]); ax.set_xticks([sx(t) for t in (-10, 0, 7, 21, 60)])
ax.set_xticklabels(['−10', '0', '+7', '+21', '+60'])
ax.set_xlim(sx(-900), np.log1p(90)); ax.set_ylim(-3.2, len(ev) + 1.6)
ax.set_xlabel('day (signed-log)'); ax.spines['left'].set_visible(False); panel(ax, 'a', dx=-.03)
for j, (c, lab) in enumerate(((PAL['aqua'], 'legal input at L'),
                              (PAL['orange'], 'result not known until after L'),
                              (PAL['bad'], 'banned: treatment proxy'))):
    ax.plot(sx(-800), len(ev) + 1.0 - j * .75, 'o', color=c, ms=3.6)
    ax.annotate(lab, (sx(-800), len(ev) + 1.0 - j * .75), textcoords='offset points', xytext=(6, 0),
                fontsize=5.5, va='center', ha='left', color=c, fontweight='bold')
ax = axes[1]
TT = np.array([-10, -1, 0, 1, 3, 7, 14, 21, 60, 100, 365])
P = np.array([phi(t) for t in TT]); nrm = np.linalg.norm(P, axis=1)
sim = P @ P.T / (nrm[:, None] * nrm[None, :])
im = ax.imshow(sim, cmap='Blues', vmin=-.2, vmax=1)
ax.set_xticks(range(len(TT))); ax.set_xticklabels(TT, fontsize=5.8, rotation=90)
ax.set_yticks(range(len(TT))); ax.set_yticklabels(TT, fontsize=5.8); ax.grid(False)
ax.set_xlabel('day'); ax.set_ylabel('day'); panel(ax, 'b', dx=-.26)
cb = plt.colorbar(im, ax=ax, fraction=.046, pad=.03); cb.ax.tick_params(labelsize=5.8)
cb.set_label('similarity of $\\varphi(\\Delta t)$', fontsize=6.2)
finish(fig, 5, 'knowledge_time_and_time_encoding',
       'Knowledge time and time encoding. (a) One patient\'s timeline. An event is a legal input at landmark L '
       'only if its result was known by then, which is not the same as having happened by then: the day-20 '
       'biopsy signs out on day +23 and is inadmissible, and the day-21 steroid order is banned outright as a '
       'proxy for the outcome. (b) The signed-log time encoding keeps early days distinguishable (day +7 vs '
       '+14 are different decisions) while compressing late follow-up, where a week does not matter.')
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
