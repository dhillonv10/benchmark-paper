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

Given tokens per patient, the corpus is just patients × tokens. Three further assumptions, all standard and all stated:

* **Compute-optimal parameter count.** Transformer scaling work puts the compute-optimal ratio near **20 tokens per parameter** †. We report that ratio and also a 10× over-trained variant (200 tokens/parameter), because small models trained long are what one actually deploys.
* **Training FLOPs.** $C \approx 6ND$ for $N$ parameters and $D$ tokens †, multiplied by an epoch/search factor for the fact that nobody trains once.
* **Throughput.** An A100-80GB sustains of order $1\times10^{14}$ effective FLOP/s on a model of this size in bf16 (roughly 30 % of peak); an H100 about $3\times10^{14}$ †. We price at **\$2.00 per A100-hour** on-demand.
''')

code(r'''
# ----------------------------------------------------------------------------- corpus -> model size -> GPU-hours -> dollars
PATIENT_COUNTS = [(1_000, 'one centre, 5 years'), (10_000, 'one large centre, 25 y / small consortium'),
                  (100_000, 'multi-registry extract (realistic CIBMTR/EBMT ask)'), (470_000, 'CIBMTR full (2017 figure)')]
FLOPS_A100, COST_A100, EPOCH_FACTOR = 1e14, 2.00, 6.0     # effective FLOP/s; $/GPU-hour; epochs + hyper-parameter search
rows = []
for n_pat, label in PATIENT_COUNTS:
    for tiers, tpp in (('A', tok_by_tier['A']), ('A+B', tok_by_tier['B']), ('A+B+C', tok_by_tier['C'])):
        D = n_pat * tpp
        for ratio, rname in ((20, 'compute-optimal (20 tok/param)'), (200, 'over-trained (200 tok/param)')):
            N = D / ratio
            flops = 6 * N * D * EPOCH_FACTOR
            gpu_h = flops / FLOPS_A100 / 3600
            rows.append(dict(**{'patients': n_pat, 'cohort': label, 'tiers': tiers,
                                'corpus tokens': D, 'regime': rname, 'parameters': N,
                                'A100-hours': gpu_h, 'cost (USD)': gpu_h * COST_A100}))
cmp = pd.DataFrame(rows)
show = cmp[cmp.regime.str.startswith('compute-optimal')].copy()
show['corpus tokens'] = show['corpus tokens'].map(lambda v: f'{v/1e9:.2f}B' if v >= 1e9 else f'{v/1e6:.0f}M')
show['parameters'] = show['parameters'].map(lambda v: f'{v/1e6:.0f}M' if v >= 1e6 else f'{v/1e3:.0f}k')
show['A100-hours'] = show['A100-hours'].map(lambda v: f'{v:,.0f}')
show['cost (USD)'] = show['cost (USD)'].map(lambda v: f'${v:,.0f}')
display(md_table(show[['patients', 'cohort', 'tiers', 'corpus tokens', 'parameters', 'A100-hours', 'cost (USD)']]))

# the realistic target configuration, stated explicitly
TARGET = dict(patients=100_000, tiers='A+B', tokens_per_patient=tok_by_tier['B'])
D_t = TARGET['patients'] * TARGET['tokens_per_patient']
N_opt, N_small = D_t / 20, D_t / 200          # compute-optimal, and a deployment-sized model trained 10x longer
gpu_opt = 6 * N_opt * D_t * EPOCH_FACTOR / FLOPS_A100 / 3600
gpu_small = 6 * N_small * (D_t * 10) * EPOCH_FACTOR / FLOPS_A100 / 3600    # same compute, 10x the token passes
print(f"ALLO-FM v1 target: {TARGET['patients']:,} patients x {TARGET['tokens_per_patient']:,} tokens "
      f"= {D_t/1e9:.2f}B tokens")
print(f"  compute-optimal     : {N_opt/1e6:.0f}M params, {gpu_opt:,.0f} A100-hours, ${gpu_opt*COST_A100:,.0f}")
print(f"  smaller, trained 10x: {N_small/1e6:.1f}M params, {gpu_small:,.0f} A100-hours, ${gpu_small*COST_A100:,.0f} (same budget, cheaper to serve)")
print(f"  whole programme (x40 for adapters, scaling, ablations, seeds): {gpu_opt*40:,.0f} A100-hours, ${gpu_opt*40*COST_A100:,.0f}")
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
    ax.plot(q.patients, q.parameters / 1e6, marker='o', ms=3.4, color=c, lw=1.6, label=f'Tier {tiers}', zorder=3)
band = ax.axhspan(30, 200, color=PAL['aqua'], alpha=.16, lw=0, label='ALLO-FM v1: 30–200 M')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-2, 2e4); ax.set_xlim(7e2, 7e5)
ax.set_xlabel('patients in the pretraining corpus'); ax.set_ylabel('compute-optimal size (M parameters)')
ax.legend(loc='upper left', labelspacing=.3); panel(ax, 'b')
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                       # (c) the bill
REF = [('one run\nA+B, 100k', gpu_opt, PAL['aqua']), ('whole\nprogramme', gpu_opt * 40, PAL['seq'][2]),
       ('7B LLM\n(for scale)', 6 * 7e9 * 2e12 / FLOPS_A100 / 3600, PAL['grey'])]
ax.bar(range(3), [r[1] for r in REF], color=[r[2] for r in REF], width=.6, zorder=3)
for i, (lab, v, c) in enumerate(REF):
    txt = f'{v:,.0f} h\n${v * COST_A100:,.0f}' if v < 1e5 else f'{v:,.0f} h\n${v * COST_A100 / 1e6:.1f}M'
    ax.text(i, v * 1.6, txt, ha='center', va='bottom', fontsize=6.4, fontweight='bold')
ax.set_yscale('log'); ax.set_ylim(1, 1e8); ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in REF])
ax.set_ylabel('A100-hours (log scale)'); panel(ax, 'c')
finish(fig, 6, 'training_requirements',
       'Training requirements, computed from the data specification. (a) Event tokens per patient, cumulative across data '
       'tiers: the EHR timeline (Tier B) supplies nearly all of them, and the deep assays add almost none while costing the '
       'most to obtain. (b) Compute-optimal model size at 20 tokens per parameter: a realistic corpus implies tens to '
       'low hundreds of millions of parameters, not billions (Tier A+B+C lies on the A+B line). (c) Pretraining cost at '
       '$2 per A100-hour against a general 7B language model. Compute is not the binding constraint.')
''')

code(r'''
stor = pd.DataFrame([
 ('A  registry core',        '~50 kB',     '~5 GB',                 'structured coded fields only'),
 ('B  EHR timeline',         '~2–10 MB',   '~0.2–1 TB',             'labs, drugs, vitals, note embeddings'),
 ('C  biospecimen assays',   '~0.1–1 GB',  '~10–100 TB',            'bulk RNA-seq, flow FCS, 16S, proteomics; realistically on 10⁴ patients → 1–10 TB'),
 ('D  deep phenotyping',     '~5–50 GB',   'not feasible at 10⁵',   'scRNA + VDJ, whole-slide images, spatial; realistically 10²–10³ patients'),
], columns=['tier', 'storage per patient', 'for 100,000 patients', 'note'])
display(md_table(stor))
display(Markdown(f"""**Reading §7.** For the realistic target — **{TARGET['patients']:,} patients with Tier A+B data, ≈ {D_t/1e9:.2f} billion event tokens** — the compute-optimal model is **{N_opt/1e6:.0f} M parameters**, trainable in about **{gpu_opt:,.0f} A100-hours (≈ \${gpu_opt*COST_A100:,.0f})**. A smaller **{N_small/1e6:.1f} M-parameter** model trained for ten times as many token passes costs the same and is far cheaper to serve. Multiplying by forty for the modality adapters, the scaling experiment, the full ablation suite and five seeds still leaves the **entire pretraining programme at ≈ {gpu_opt*40:,.0f} A100-hours (≈ \${gpu_opt*40*COST_A100:,.0f})** — less than the cost of sequencing a handful of samples, and five orders of magnitude below a general-purpose language model.

**Three conclusions that should shape how this project is proposed.**
1. **Do not ask for a GPU cluster; ask for data access.** The binding constraints are the registry study proposal, the data-use agreement and the biobank collaboration — not compute. A plan whose critical path is hardware procurement has misidentified its own bottleneck. The real engineering cost is the harness (§11.0), not the training run.
2. **The right model is small.** At ~10⁹ tokens, anything above a few hundred million parameters is past compute-optimal and will be data-limited; the published single-cell scaling study found performance plateauing well before current corpus sizes ✓, and MOTOR — the closest clinical analogue — is a 143 M-parameter model ✓. Anyone proposing a multi-billion-parameter GvHD model should be asked which tokens it is going to read.
3. **Tier C/D data buys information, not tokens.** Panel (a) makes this unusually clear: the deep assays contribute a handful of tokens per patient (7,580 → 7,628) while costing the most to acquire. Their value must therefore be demonstrated per-patient by modality knockout (§8.8) and tier ablation (task G4), never assumed from volume."""))
''')
