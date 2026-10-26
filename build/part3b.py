from part1 import CELLS, md, code

md(r'''
---
# 7 · Training requirements: how much data, how big a model, how much compute

This section turns "pretrain a foundation model on allo-HCT" into numbers. Everything below is computed in the notebook from the data specification of §4 plus a small set of stated engineering assumptions. The headline result is counter-intuitive and consequential: **the compute is trivial; the data access is everything.**

## 7.1 How many tokens does one patient produce?

A patient is a sequence of events (§5.2). The token budget follows from the sampling frequencies written into the dictionary: a daily CBC for 100 days is ~700 numeric tokens by itself, whereas the whole registry core is a few hundred. We count per tier, over a 365-day window, with the frequencies below.
''')

code(r'''
# ----------------------------------------------------------------------------- token budget per patient, by data tier (computed from the sampling frequencies of §4)
# (stream, tier, tokens per observation, observations per patient over 365 days, note)
STREAMS = [
 ('Registry core (demographics, disease, donor, HLA, graft, conditioning, prophylaxis platform)', 'A', 1, 260, 'one token per coded field, incl. derived immunogenetics'),
 ('Outcome / follow-up records (grades, response, relapse, death, visits)',                      'A', 1, 120, 'daily aGvHD staging while admitted, then per visit'),
 ('Drug administrations (all agents, every dose)',                                               'B', 2, 900, 'code + dose token; ~2.5 drug events/day mean'),
 ('Drug level measurements (CNI troughs, busulfan PK, MMF)',                                     'B', 2, 90,  'daily early, then weekly'),
 ('Laboratory results (CBC diff, chemistry, liver, inflammation)',                               'B', 2, 1800, '~18 analytes x 100 inpatient days, then weekly'),
 ('Vital signs and fluid balance (daily summaries)',                                             'B', 2, 450, '6 channels x ~75 summarised days'),
 ('Microbiology and virology results',                                                           'B', 2, 120, 'cultures, CMV/EBV PCR, stool PCR'),
 ('Clinical notes (summary tokens from a text adapter, not raw text)',                           'B', 8, 60,  'one pooled embedding per note type per week'),
 ('Flow cytometry panels (graft + reconstitution)',                                              'C', 1, 8,   'one summary token per panel per timepoint'),
 ('Plasma protein biomarkers (MAGIC panel and extensions)',                                      'C', 1, 13,  'one token per draw'),
 ('Stool microbiome profiles',                                                                   'C', 1, 12,  'one token per sample'),
 ('Transcriptomes (tissue and/or blood)',                                                        'C', 1, 6,   'one token per library'),
 ('TCR repertoire samples',                                                                      'D', 1, 6,   'one token per repertoire'),
 ('Whole-slide images (biopsy)',                                                                  'D', 1, 3,   'one token per slide from a pathology adapter'),
]
st = pd.DataFrame(STREAMS, columns=['stream', 'tier', 'tokens/obs', 'obs/patient', 'note'])
st['tokens/patient'] = st['tokens/obs'] * st['obs/patient']
TIER_ORDER = ['A', 'B', 'C', 'D']
cum, tok_by_tier = 0, {}
for t in TIER_ORDER:
    cum += int(st.loc[st.tier == t, 'tokens/patient'].sum()); tok_by_tier[t] = cum
budget = pd.DataFrame([dict(**{'tiers included': ' + '.join(TIER_ORDER[:i + 1]),
                               'what it means': ['registry only', 'registry + full EHR timeline',
                                                 '+ protocolised biospecimen assays', '+ deep research phenotyping'][i],
                               'tokens per patient': f'{tok_by_tier[t]:,}'})
                       for i, t in enumerate(TIER_ORDER)])
print(st.groupby('tier')['tokens/patient'].sum().to_string())
display(md_table(st[['stream', 'tier', 'tokens/obs', 'obs/patient', 'tokens/patient', 'note']], max_w=95))
display(md_table(budget))
''')

md(r'''
## 7.2 What one institution can train: a small model, or a released one fine-tuned

A single transplant centre does roughly 150–250 allogeneic transplants a year, so this paper plans on a **ceiling of 1,500 patients** (≈ 7.5 years of one large centre; §9.9). At that size the right question is not "how big a model can we pretrain?" but "**what do we adapt, and how small must the part we train be?**" We lay out a ladder, cheapest first; each rung must beat the one below it on the same patients (that comparison *is* the benchmark's Level 1C, §8.10):

| Rung | Model | What is trained at the centre | Needs |
|---|---|---|---|
| **L0** | tuned simple model (penalised logistic / gradient-boosted trees on 20–40 registry variables) | 20–40 coefficients or trees | CPU |
| **L1** | **tiny timeline transformer trained from scratch** on the centre's own event timelines | ≈ 0.5 M parameters (compute-optimal for the centre's corpus, below) | one GPU for minutes |
| **L2** | **a released pretrained EHR foundation model, frozen, with a small head** (EHRSHOT's protocol: pretrained weights frozen, a logistic-regression head fitted on k labelled patients) | **a head of ≤ 5–25 parameters**, the size the cohort can support (Table 9.1) | embeddings for 1,500 patients: minutes; the weights may need a data-use agreement and a code-vocabulary mapping ✓ |
| **L3** | the same model fine-tuned with low-rank adapters (LoRA) | ≈ 10⁶ adapter parameters, strongly shrunk | one GPU for an hour |
| (L4) | registry-scale pretraining (tens of millions of parameters, 10⁵ patients) | — | a network, not a centre; **not part of this plan** |

The released rung is not hypothetical: the EHRSHOT release includes **CLMBR-T-base, a 141 M-parameter transformer pretrained on 2.57 M patients' structured records, whose weights are frozen and used with a logistic-regression head fitted on 1–128 labelled patients per class** ✓ (access requires a data-use agreement). MOTOR (143 M parameters, time-to-event pretraining) is a second published option ✓. We do not claim either transfers to allo-HCT; whether it does is the first thing the benchmark measures.

**Compute model, stated once.** Compute-optimal size ≈ 20 unique tokens per parameter †; training FLOPs per token $=6N+12\,L\,d\,T$ (attention over $T=2{,}048$ tokens, width $d=64L$) †; 4 passes over a small corpus †; an A100-80GB peaks at 312 TFLOP/s in bf16 and a small model sustains about 30 % of it (we show 15–45 %) †. We report **GPU-time only**.
''')

code(r'''
# ----------------------------------------------------------------------------- 7.2: corpus -> model size -> FLOPs (6N + attention) -> GPU-hours, at the scale of ONE institution
A100_PEAK, CTX, EPOCHS = 312e12, 2048, 4                           # bf16 peak FLOP/s; context length; passes over the corpus
MFU = dict(pessimistic=.15, planned=.30, optimistic=.45)            # fraction of peak that a small model sustains
def shape_for(N):
    """Depth and width at aspect ratio 64 (d = 64 L, N = 12 L d^2 = 49,152 L^3)."""
    L = max(2, round((N / 49152) ** (1 / 3))); return L, 64 * L
def train_flops(N, D, epochs=EPOCHS, ctx=CTX):
    L, d = shape_for(N); return (6 * N + 12 * L * d * ctx) * D * epochs
def a100_hours(flops, mfu=MFU['planned']): return flops / (A100_PEAK * mfu) / 3600
def fmt_n(v):
    for s, u in ((1e9, 'B'), (1e6, 'M'), (1e3, 'k')):
        if v >= s: return f'{v / s:.3g}{u}'
    return f'{v:.3g}'
def fmt_h(v): return '<0.01 h' if v < .01 else (f'{v:.2g} h' if v < 10 else f'{v:,.0f} h')
def fmt_t(h):
    """GPU time in the unit a person thinks in."""
    return f'{h * 3600:.0f} s' if h < .02 else (f'{h * 60:.0f} min' if h < 2 else f'{h:.1f} h')

rows = []
for n_pat in (150, 500, 1000, N_CAP):
    for tiers, tpp in (('A', tok_by_tier['A']), ('A+B', tok_by_tier['B'])):
        D = n_pat * tpp; N = D / 20; fl = train_flops(N, D); L, d = shape_for(N)
        rows.append(dict(patients=n_pat, tiers=tiers, tokens=D, params=N, L=L, d=d, flops=fl, hours=a100_hours(fl), hours_pess=a100_hours(fl, MFU['pessimistic']), hours_opt=a100_hours(fl, MFU['optimistic'])))
cmp = pd.DataFrame(rows)
display(md_table(pd.DataFrame({'patients': [f'{v:,}' for v in cmp.patients], 'tiers': cmp.tiers.values, 'corpus tokens': [fmt_n(v) for v in cmp.tokens],
    'compute-optimal parameters': [fmt_n(v) for v in cmp.params], 'shape (L x d)': [f'{l} x {d}' for l, d in zip(cmp.L, cmp.d)], 'training FLOPs': [f'{v:.1e}' for v in cmp.flops],
    'A100 time (15% / 30% / 45% utilisation)': [f'{fmt_t(a)} / {fmt_t(b)} / {fmt_t(c)}' for a, b, c in zip(cmp.hours_pess, cmp.hours, cmp.hours_opt)]})))

# ---------- the ladder at the ceiling cohort (1,500 patients, Tier A+B)
TARGET = dict(patients=N_CAP, tiers='A+B', tokens_per_patient=tok_by_tier['B'])
D_t = TARGET['patients'] * TARGET['tokens_per_patient']
N_tiny = D_t / 20; C_tiny = train_flops(N_tiny, D_t); H_tiny = a100_hours(C_tiny)                       # L1: from scratch
N_REL = 141e6; C_ft = train_flops(N_REL, D_t, epochs=3); H_ft = a100_hours(C_ft)                        # L3: full fine-tune of a released 141 M model (upper bound)
C_emb = 2 * N_REL * D_t; H_emb = a100_hours(C_emb)                                                      # L2: one forward pass to extract frozen embeddings
L_rel, d_rel = shape_for(N_REL); N_LORA = 2 * 8 * d_rel * 4 * L_rel                                    # LoRA rank 8 on the four attention matrices of every layer
def programme(D):
    """The whole one-institution programme, run by run, for a corpus of D unique tokens: (component, number of runs, FLOPs per run)."""
    Nt = D / 20
    return [('L1 tiny model from scratch, 3 seeds', 3, train_flops(Nt, D)),
            ('L1 ablations (§8.8): 12 variants x 3 seeds', 36, train_flops(Nt, D)),
            ('L2 frozen released model: embeddings for the whole cohort, 3 vintages of the cohort', 3, 2 * N_REL * D),
            ('L3 LoRA fine-tune of the released model, 3 seeds', 3, .67 * train_flops(N_REL, D, epochs=3)),
            ('k-shot head fits: 25 tasks x 6 values of k x 5 draws (CPU, negligible)', 750, 1e9)]
PROG = programme(D_t)
prog = pd.DataFrame(PROG, columns=['component', 'runs', 'FLOPs per run']); prog['total FLOPs'] = prog.runs * prog['FLOPs per run']; prog['A100 time'] = prog['total FLOPs'].map(a100_hours)
PROG_FLOPS = float(prog['total FLOPs'].sum()); PROG_H = float(prog['A100 time'].sum()); PROG_H_RANGE = (a100_hours(PROG_FLOPS, MFU['optimistic']), a100_hours(PROG_FLOPS, MFU['pessimistic']))
LLM_FLOPS = 6 * 7e9 * 2e12; OOM_VS_LLM = math.log10(LLM_FLOPS / PROG_FLOPS)                           # a 7B-parameter language model on 2T tokens †
display(md_table(pd.DataFrame({'component': prog.component, 'runs': prog.runs.map(lambda v: f'{v:,}'), 'A100 time': prog['A100 time'].map(fmt_t)})))
lad = pd.DataFrame([
    ('L0 tuned simple model', '20–40 coefficients', 'CPU, seconds'),
    ('L1 tiny timeline model from scratch', f'{fmt_n(N_tiny)} parameters ({shape_for(N_tiny)[0]} layers x width {shape_for(N_tiny)[1]})', f'{fmt_t(H_tiny)} on one A100 ({fmt_t(a100_hours(C_tiny) * 10)} on a laptop GPU)'),
    ('L2 released 141 M model, frozen, small head', 'a head of at most 5–25 parameters (Table 9.1)', f'embeddings for {N_CAP:,} patients: {fmt_t(H_emb)} on one A100; head fits on CPU'),
    ('L3 released model + LoRA (rank 8)', f'{fmt_n(N_LORA)} adapter parameters', f'{fmt_t(.67 * H_ft)} per run (full fine-tune upper bound {fmt_t(H_ft)})')], columns=['rung', 'what is trained', 'compute'])
display(md_table(lad))
print(f"whole one-institution programme: {PROG_FLOPS:.1e} FLOPs = {fmt_t(PROG_H)} on one A100 at 30% utilisation (range {fmt_t(PROG_H_RANGE[0])} - {fmt_t(PROG_H_RANGE[1])}); {OOM_VS_LLM:.1f} orders of magnitude below a 7B language model")
sens = pd.DataFrame([dict(**{'tokens per patient': f'x{m:g}  ({TARGET["tokens_per_patient"] * m:,.0f})'}, **{'corpus tokens': fmt_n(D_t * m), 'tiny-model parameters': fmt_n(D_t * m / 20),
                             'L1 run': fmt_t(a100_hours(train_flops(D_t * m / 20, D_t * m))), 'programme': fmt_t(a100_hours(sum(r_ * f_ for _, r_, f_ in programme(D_t * m))))}) for m in (.25, .5, 1, 2, 4)])
display(md_table(sens))
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 6: the training-requirement landscape
fig = plt.figure(figsize=(FULL, 3.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, .85], left=.075, right=.985, bottom=.20, top=.88, wspace=.40)
ax = fig.add_subplot(gs[0, 0]); ygrid(ax)                                       # (a) tokens per patient, cumulative by tier
vals = [tok_by_tier[t] for t in TIER_ORDER]
ax.bar(range(4), vals, color=PAL['seq'], width=.66, zorder=3)
for i, v in enumerate(vals): ax.text(i, v * 1.12, f'{v:,}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
ax.set_yscale('log'); ax.set_ylim(100, 4e4); ax.set_xticks(range(4))
ax.set_xticklabels(['A\nregistry', '+B\nEHR', '+C\nassays', '+D\nresearch'])
ax.set_ylabel('event tokens per patient'); ax.set_xlabel('data tier (cumulative)'); panel(ax, 'a')
ax = fig.add_subplot(gs[0, 1]); ygrid(ax)                                       # (b) what a centre's own corpus supports, against a released model
pp = np.array([150, 300, 500, 1000, 1500])
for tiers, c in (('A', PAL['seq'][0]), ('A+B', PAL['seq'][3])):
    ax.plot(pp, [tok_by_tier[tiers if tiers == 'A' else 'B'] * n / 20 / 1e6 for n in pp], marker='o', ms=3.4, color=c, lw=1.6, label=f'compute-optimal, Tier {tiers}', zorder=3)
ax.axhline(141, color=PAL['aqua'], lw=1.3, ls='--', label='released model (141 M)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(3e-3, 400); ax.set_xlim(120, 2000); ax.set_xticks([150, 300, 500, 1000, 1500]); ax.set_xticklabels(['150', '300', '500', '1,000', '1,500']); ax.minorticks_off()
ax.set_xlabel('patients at one institution'); ax.set_ylabel('model size (M parameters)')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=1, handlelength=1.4, labelspacing=.3); panel(ax, 'b')
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                       # (c) compute
REF = [('L1\nrun', H_tiny, PAL['aqua']), ('L3\nrun', .67 * H_ft, PAL['seq'][2]), ('all\nruns', PROG_H, PAL['seq'][3]), ('7B\nLLM', a100_hours(LLM_FLOPS), PAL['grey'])]
ax.bar(range(len(REF)), [r[1] for r in REF], color=[r[2] for r in REF], width=.6, zorder=3)
for i, (lab, v, c) in enumerate(REF): ax.text(i, v * 1.7, fmt_t(v) if v < 100 else f'{v:,.0f} h', ha='center', va='bottom', fontsize=6.4, fontweight='bold')
ax.set_yscale('log'); ax.set_ylim(1e-3, 3e7); ax.set_xticks(range(len(REF))); ax.set_xticklabels([r[0] for r in REF])
ax.set_ylabel('A100-hours (log scale)'); panel(ax, 'c')
finish(fig, 6, 'training_requirements',
       'What one institution has to train. (a) Event tokens per patient, cumulative across data tiers: the EHR timeline (Tier B) supplies nearly all of them, and the '
       'deep assays add almost none. (b) The compute-optimal size of a model trained from scratch on a centre\'s own corpus (up to 1,500 patients) is a fraction of a million '
       'parameters, two to three orders below a released 141 M-parameter model (dashed), which is why the plan adapts a released model with a small head rather than '
       'pretraining. (c) GPU time on one A100 at 30% utilisation: a tiny model from scratch, one LoRA fine-tune of the released model, the whole enumerated programme, and '
       'a general 7B language model for scale. Compute is not the binding constraint.')
''')

code(r'''
stor = pd.DataFrame([
 ('A  registry core',        '~50 kB',     f'{50e3 * N_CAP / 1e6:.0f} MB',          'structured coded fields only'),
 ('B  EHR timeline',         '~2–10 MB',   f'{2e6 * N_CAP / 1e9:.0f}–{10e6 * N_CAP / 1e9:.0f} GB', 'labs, drugs, vitals, note embeddings'),
 ('C  biospecimen assays',   '~0.1–1 GB',  f'{.1 * N_SPEC:.0f}–{1 * N_SPEC:.0f} GB (on {N_SPEC} patients)',  'bulk RNA-seq, flow FCS, 16S, plasma panels'),
 ('D  deep phenotyping',     '~5–50 GB',   'a nested research subset of 50–100 patients: 0.3–5 TB', 'scRNA + VDJ, whole-slide images'),
], columns=['tier', 'storage per patient', f'for the plan ({N_CAP:,} patients)', 'note'])
display(md_table(stor))
display(Markdown(f"""**Reading §7.** At the ceiling cohort — **{N_CAP:,} patients with Tier A+B data, ≈ {D_t / 1e6:.0f} million event tokens** — the compute-optimal model trained from scratch is only **{fmt_n(N_tiny)} parameters**, trainable in **{fmt_t(H_tiny)} on one A100** (about {fmt_t(a100_hours(C_tiny) * 10)} on a laptop GPU). That is a legitimate baseline (L1) and a useful one, but it will not carry the foundation-model claim on its own: a model that small has seen little. The foundation-model claim therefore rests on **L2 and L3 — adapting a released model that was pretrained on millions of patients** — and on keeping the part trained at the centre small: extracting frozen embeddings for the whole cohort takes {fmt_t(H_emb)}, and the head fitted on them has at most 5–25 parameters (Table 9.1), because that is all {N_CAP:,} patients can support. Enumerating the *whole* programme (three seeds, twelve ablations, three cohort vintages, three LoRA fine-tunes and 750 head fits) gives **{PROG_FLOPS:.1e} FLOPs ≈ {fmt_t(PROG_H)} on one A100** ({fmt_t(PROG_H_RANGE[0])}–{fmt_t(PROG_H_RANGE[1])} across the utilisation range), **{OOM_VS_LLM:.1f} orders of magnitude** below a 7B-parameter language model. Even a 4× error in the token budget, in either direction, leaves it on one GPU.

**Three conclusions that should shape how this project is proposed.**
1. **Do not ask for a cluster, and do not pretrain.** The binding constraints are the patients (≤ {N_CAP:,}), the labels, and the data-use agreement for a released model — not compute. The real engineering effort is the harness (§11.0), not the training run.
2. **The part trained at the centre must be small**: a tiny model from scratch (L1) or a ≤ 25-parameter head on a released model's embeddings (L2). A head of several hundred parameters (a raw embedding) needs far more patients than one centre has (Figure 11c) and is not proposed.
3. **Tier C/D data buys information, not tokens.** The deep assays contribute a handful of tokens per patient (7,580 → 7,628), so their value must be demonstrated per patient by modality knockout (§8.8) and tier ablation (task G4), never assumed from volume."""))
''')
