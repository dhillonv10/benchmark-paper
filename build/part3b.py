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
## 7.2 Corpus size, compute-optimal model size, and the bill

Given tokens per patient, the corpus is patients × tokens. The compute model is deliberately explicit, because "it is cheap" is a claim that has to survive its own assumptions:

* **Model size.** The compute-optimal ratio from transformer scaling work is near **20 unique tokens per parameter** †; we use it to set $N=D/20$, and also report a 10× smaller, over-trained model (200 tokens per parameter), which is what one deploys.
* **Training FLOPs per token** $=6N+12\,L\,d\,T$ (forward and backward; the second term is attention over a context of $T=2{,}048$ tokens, with width $d=64L$) †. Attention is not negligible at these model sizes, so a bare $6ND$ would understate the cost by about half.
* **Passes over the data.** A registry corpus is small, so we train **4 epochs**; repeating data up to about four times is close to lossless in the language-model literature †.
* **Throughput.** An A100-80GB peaks at $312$ TFLOP/s in bf16 †. Small models do not saturate it: we plan at **30 % utilisation** and report a pessimistic 15 % and an optimistic 45 %.
* **Price.** **\$2.00 per A100-hour** on-demand †.
''')

code(r'''
# ----------------------------------------------------------------------------- corpus -> model size -> FLOPs (6N + attention) -> GPU-hours -> dollars, with a utilisation range
A100_PEAK, USD_PER_H, CTX, EPOCHS = 312e12, 2.00, 2048, 4          # bf16 peak FLOP/s; $/A100-hour; context length; passes over the corpus
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
def fmt_usd(v):
    if v < .5: return '<$1'
    if v < 1e4: return f'${v:,.0f}'
    if v < 1e6: return f'${v / 1e3:,.0f}k'
    return f'${v / 1e6:.1f}M' if v < 1e7 else f'${v / 1e6:,.0f}M'
def fmt_h(v): return '<0.01 h' if v < .01 else (f'{v:.2g} h' if v < 10 else f'{v:,.0f} h')

PATIENT_COUNTS = [(1_000, 'one centre, 5 years'), (10_000, 'one large centre, 25 y / small consortium'),
                  (100_000, 'multi-registry extract (realistic CIBMTR/EBMT ask)'), (470_000, 'CIBMTR full (2017 figure)')]
rows = []
for n_pat, label in PATIENT_COUNTS:
    for tiers, tpp in (('A', tok_by_tier['A']), ('A+B', tok_by_tier['B']), ('A+B+C', tok_by_tier['C'])):
        D = n_pat * tpp
        for ratio, rname in ((20, 'compute-optimal (20 tok/param)'), (200, 'over-trained (200 tok/param)')):
            N = D / ratio; fl = train_flops(N, D); L, d = shape_for(N)
            rows.append(dict(patients=n_pat, cohort=label, tiers=tiers, tokens=D, regime=rname, params=N, L=L, d=d, flops=fl,
                             hours=a100_hours(fl), hours_pess=a100_hours(fl, MFU['pessimistic']), hours_opt=a100_hours(fl, MFU['optimistic'])))
cmp = pd.DataFrame(rows); cmp['usd'] = cmp.hours * USD_PER_H
show = cmp[cmp.regime.str.startswith('compute-optimal')]
display(md_table(pd.DataFrame({'patients': [f'{v:,}' for v in show.patients], 'cohort': show.cohort.values, 'tiers': show.tiers.values,
    'corpus tokens': [fmt_n(v) for v in show.tokens], 'parameters': [fmt_n(v) for v in show.params], 'shape (L x d)': [f'{l} x {d}' for l, d in zip(show.L, show.d)],
    'training FLOPs': [f'{v:.1e}' for v in show.flops],
    'A100-hours (15% / 30% / 45% MFU)': [f'{fmt_h(a)} / {fmt_h(b)} / {fmt_h(c)}' for a, b, c in zip(show.hours_pess, show.hours, show.hours_opt)],
    'cost at 30%': [fmt_usd(v) for v in show.usd]})))

# ---------- the realistic target configuration, then the whole programme enumerated run by run
TARGET = dict(patients=100_000, tiers='A+B', tokens_per_patient=tok_by_tier['B'])
D_t = TARGET['patients'] * TARGET['tokens_per_patient']; N_opt, N_small = D_t / 20, D_t / 200
C_main = train_flops(N_opt, D_t); gpu_opt = a100_hours(C_main); gpu_small = a100_hours(train_flops(N_small, D_t))
def programme(D):
    """The whole programme, run by run, for a corpus of D unique tokens: (component, number of runs, FLOPs per run)."""
    N0 = D / 20; C0 = train_flops(N0, D)
    return [('main pretraining run, 3 seeds', 3, C0),
            ('scaling study: 5 model sizes (N/8 ... 2N) x 4 data fractions (1/8 ... 1)', 20,
             sum(train_flops(N0 * m, D * f) for m in (1 / 8, 1 / 4, 1 / 2, 1, 2) for f in (1 / 8, 1 / 4, 1 / 2, 1)) / 20),
            ('ablation suite (§8.8): 16 variants x 3 seeds at half scale', 48, train_flops(N0 / 2, D / 2)),
            ('modality adapters: 5 adapters x 3 seeds', 15, .05 * C0),
            ('downstream fine-tuning: 40 tasks x 24 leave-centre-out folds x 3 seeds', 2880, .002 * C0)]
PROG = programme(D_t)
prog = pd.DataFrame(PROG, columns=['component', 'runs', 'FLOPs per run'])
prog['total FLOPs'] = prog.runs * prog['FLOPs per run']; prog['A100-hours'] = prog['total FLOPs'].map(a100_hours)
prog['share'] = prog['total FLOPs'] / prog['total FLOPs'].sum()
PROG_FLOPS = float(prog['total FLOPs'].sum()); PROG_H = float(prog['A100-hours'].sum())
PROG_H_RANGE = (a100_hours(PROG_FLOPS, MFU['optimistic']), a100_hours(PROG_FLOPS, MFU['pessimistic']))
LLM_FLOPS = 6 * 7e9 * 2e12                                                    # a 7B-parameter language model on 2T tokens †
OOM_VS_LLM = math.log10(LLM_FLOPS / PROG_FLOPS)
print(f"ALLO-FM v1 target: {TARGET['patients']:,} patients x {TARGET['tokens_per_patient']:,} tokens = {D_t / 1e9:.2f}B tokens")
print(f"  compute-optimal : {N_opt / 1e6:.0f}M params ({shape_for(N_opt)[0]} layers x width {shape_for(N_opt)[1]}), {C_main:.1e} FLOPs, {fmt_h(gpu_opt)} on one A100 at 30% MFU = {fmt_usd(gpu_opt * USD_PER_H)}")
print(f"  smaller, over-trained: {N_small / 1e6:.1f}M params, {fmt_h(gpu_small)}")
display(md_table(pd.DataFrame({'component': prog.component, 'runs': prog.runs.map(lambda v: f'{v:,}'),
    'A100-hours': prog['A100-hours'].map(fmt_h), 'share': prog.share.map(lambda v: f'{100 * v:.0f} %')})))
print(f"whole programme: {PROG_FLOPS:.1e} FLOPs = {fmt_h(PROG_H)} at 30% MFU (range {fmt_h(PROG_H_RANGE[0])} - {fmt_h(PROG_H_RANGE[1])}) = {fmt_usd(PROG_H * USD_PER_H)}; "
      f"{OOM_VS_LLM:.1f} orders of magnitude below a 7B language model ({LLM_FLOPS:.1e} FLOPs)")
# sensitivity: the token budget per patient is the softest input (tokenisation choices change it by a factor of a few)
sens = pd.DataFrame([dict(**{'tokens per patient': f'x{m:g}  ({TARGET["tokens_per_patient"] * m:,.0f})'},
                          **{'corpus tokens': fmt_n(D_t * m), 'compute-optimal parameters': fmt_n(D_t * m / 20),
                             'one run, A100-hours': fmt_h(a100_hours(train_flops(D_t * m / 20, D_t * m))),
                             'programme, A100-hours': fmt_h(a100_hours(sum(r_ * f_ for _, r_, f_ in programme(D_t * m))))}) for m in (.25, .5, 1, 2, 4)])
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
ax = fig.add_subplot(gs[0, 1]); ygrid(ax)                                       # (b) compute-optimal size
for tiers, c in (('A', PAL['seq'][0]), ('A+B', PAL['seq'][3])):
    q = cmp[(cmp.tiers == tiers) & (cmp.regime.str.startswith('compute'))].sort_values('patients')
    ax.plot(q.patients, q.params / 1e6, marker='o', ms=3.4, color=c, lw=1.6, label=f'Tier {tiers}', zorder=3)
band = ax.axhspan(30, 200, color=PAL['aqua'], alpha=.16, lw=0, label='ALLO-FM v1: 30–200 M')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-2, 2e4); ax.set_xlim(7e2, 7e5)
ax.set_xlabel('patients in the pretraining corpus'); ax.set_ylabel('compute-optimal size (M parameters)')
ax.legend(loc='upper left', labelspacing=.3); panel(ax, 'b')
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                       # (c) the bill
REF = [('one run', gpu_opt, PAL['aqua']), ('programme', PROG_H, PAL['seq'][2]), ('7B LLM', a100_hours(LLM_FLOPS), PAL['grey'])]
ax.bar(range(3), [r[1] for r in REF], color=[r[2] for r in REF], width=.6, zorder=3)
for i, (lab, v, c) in enumerate(REF):
    ax.text(i, v * 1.6, f'{fmt_h(v)}\n{fmt_usd(v * USD_PER_H)}', ha='center', va='bottom', fontsize=6.4, fontweight='bold')
ax.set_yscale('log'); ax.set_ylim(1, 1e8); ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in REF])
ax.set_ylabel('A100-hours (log scale)'); panel(ax, 'c')
finish(fig, 6, 'training_requirements',
       'Training requirements, computed from the data specification. (a) Event tokens per patient, cumulative across data '
       'tiers: the EHR timeline (Tier B) supplies nearly all of them, and the deep assays add almost none while costing the '
       'most to obtain. (b) Compute-optimal model size at 20 tokens per parameter: a realistic corpus implies tens to '
       'low hundreds of millions of parameters, not billions (Tier A+B+C lies on the A+B line). (c) Pretraining cost at '
       '$2 per A100-hour, 30% utilisation and 4 epochs: one run (100,000 patients, Tier A+B), the whole enumerated programme (§7.2 table), and a general 7B language model. '
       'Compute is not the binding constraint; assay and collection cost is (§9.8).')
''')

code(r'''
stor = pd.DataFrame([
 ('A  registry core',        '~50 kB',     '~5 GB',                 'structured coded fields only'),
 ('B  EHR timeline',         '~2–10 MB',   '~0.2–1 TB',             'labs, drugs, vitals, note embeddings'),
 ('C  biospecimen assays',   '~0.1–1 GB',  '~10–100 TB',            'bulk RNA-seq, flow FCS, 16S, proteomics; realistically on 10⁴ patients → 1–10 TB'),
 ('D  deep phenotyping',     '~5–50 GB',   'not feasible at 10⁵',   'scRNA + VDJ, whole-slide images, spatial; realistically 10²–10³ patients'),
], columns=['tier', 'storage per patient', 'for 100,000 patients', 'note'])
display(md_table(stor))
display(Markdown(f"""**Reading §7.** For the realistic target — **{TARGET['patients']:,} patients with Tier A+B data, ≈ {D_t/1e9:.2f} billion event tokens** — the compute-optimal model is **{N_opt/1e6:.0f} M parameters** ({shape_for(N_opt)[0]} layers, width {shape_for(N_opt)[1]}), trainable in **{fmt_h(gpu_opt)} on one A100** ({C_main:.1e} FLOPs, attention included, 4 epochs, 30 % utilisation; {fmt_h(a100_hours(C_main, MFU['optimistic']))}–{fmt_h(a100_hours(C_main, MFU['pessimistic']))} across the utilisation range). A smaller **{N_small/1e6:.1f} M-parameter** model trained for ten times as many token passes costs {fmt_h(gpu_small)} and is far cheaper to serve. Enumerating the *whole* programme — three seeds, a 20-run scaling study, a 48-run ablation suite, five adapters and 2,880 downstream fine-tunes — gives **{PROG_FLOPS:.1e} FLOPs ≈ {fmt_h(PROG_H)} on one A100 ≈ {fmt_usd(PROG_H * USD_PER_H)}**, which is **{OOM_VS_LLM:.1f} orders of magnitude** below a 7B-parameter language model. The sensitivity table shows that even a 4× error in the token budget, in either direction, leaves the programme at the scale of GPU-hours to GPU-days on a single device, not a cluster.

**Three conclusions that should shape how this project is proposed.**
1. **Do not ask for a GPU cluster; ask for data access.** The binding constraints are the registry study proposal, the data-use agreement and the biobank collaboration — not compute. The compute for the whole programme is a few GPU-days; the assays and specimen collection of §9.8 cost three to four orders of magnitude more. A plan whose critical path is hardware procurement has misidentified its own bottleneck. The real engineering cost is the harness (§11.0), not the training run.
2. **The right model is small.** At ~10⁹ tokens, anything above a few hundred million parameters is past compute-optimal and will be data-limited; the published single-cell scaling study found performance plateauing well before current corpus sizes ✓, and MOTOR — the closest clinical analogue — is a 143 M-parameter model ✓. Anyone proposing a multi-billion-parameter GvHD model should be asked which tokens it is going to read.
3. **Tier C/D data buys information, not tokens.** Panel (a) makes this unusually clear: the deep assays contribute a handful of tokens per patient (7,580 → 7,628) while costing the most to acquire. Their value must therefore be demonstrated per-patient by modality knockout (§8.8) and tier ablation (task G4), never assumed from volume."""))
''')
