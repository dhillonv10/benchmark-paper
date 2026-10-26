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
fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.5), gridspec_kw=dict(wspace=.38))
# (a) tokens per patient, cumulative by tier
ax = axes[0]; ygrid(ax)
vals = [tok_by_tier[t] for t in TIER_ORDER]
bars = ax.bar(range(4), vals, color=PAL['seq'], width=.68)
for i, v in enumerate(vals):
    ax.text(i, v * 1.06, f'{v:,}', ha='center', fontsize=6.2, color=PAL['ink'], fontweight='bold')
ax.set_yscale('log'); ax.set_xticks(range(4))
ax.set_xticklabels(['A\nregistry', 'A+B\n+ EHR', 'A+B+C\n+ assays', '+D\n+ research'], fontsize=6.2)
ax.set_ylabel('event tokens per patient'); ax.set_ylim(100, max(vals) * 3); panel(ax, 'a', dx=-.30)
# (b) corpus vs compute-optimal parameters
ax = axes[1]; ygrid(ax)
for tiers, c in (('A', PAL['seq'][0]), ('A+B', PAL['seq'][3])):
    q = cmp[(cmp.tiers == tiers) & (cmp.regime.str.startswith('compute'))].sort_values('patients')
    ax.plot(q.patients, q.parameters / 1e6, marker='o', color=c, lw=1.5)
    dlabel(ax, q.patients.iloc[-1], q.parameters.iloc[-1] / 1e6, tiers, c, dx=4, size=6.2)
ax.text(5.5e5, .035, 'A+B+C lies on the A+B line:\nthe deep assays add information,\nnot tokens',
        fontsize=5.3, color=PAL['ink2'], va='bottom', ha='right')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('patients in the pretraining corpus')
ax.set_ylabel('compute-optimal size (M parameters)')
ax.axhspan(30, 200, color=PAL['aqua'], alpha=.13, lw=0)
ax.text(1.05e3, 33, 'ALLO-FM v1:  30–200 M', fontsize=5.9, color=PAL['ink'], fontweight='bold', ha='left', va='bottom')
panel(ax, 'b', dx=-.32)
# (c) GPU-hours and cost for the target configuration, against reference workloads
ax = axes[2]; ygrid(ax)
REF = [('one run\nA+B, 100k pts', gpu_opt, PAL['aqua']),
       ('whole\nprogramme', gpu_opt * 40, PAL['seq'][2]),
       ('7B LLM\n(for scale)', 6 * 7e9 * 2e12 / FLOPS_A100 / 3600, PAL['grey'])]
ax.bar(range(len(REF)), [r[1] for r in REF], color=[r[2] for r in REF], width=.6)
for i, (lab, v, c) in enumerate(REF):
    txt = f'{v:,.0f} h\n${v*COST_A100:,.0f}' if v < 1e5 else f'{v:,.0f} h\n${v*COST_A100/1e6:.1f}M'
    ax.text(i, v * 1.6, txt, ha='center', fontsize=5.8, color=PAL['ink'], fontweight='bold')
ax.set_yscale('log'); ax.set_xticks(range(len(REF)))
ax.set_xticklabels([r[0] for r in REF], fontsize=6.0); ax.set_ylabel('A100-hours (log scale)')
ax.text(1, 3.0, 'adapters, scaling,\nablations, 5 seeds', fontsize=5.0, ha='center', color=PAL['ink2'])
ax.set_ylim(1, 1e8); panel(ax, 'c', dx=-.30)
finish(fig, 6, 'training_requirements',
       'Training requirements, computed from the data specification. (a) Event tokens per patient, cumulative '
       'across data tiers: the EHR timeline (Tier B) dominates, and the deep assays add almost nothing to the '
       'token count while costing the most to obtain. (b) Compute-optimal model size at 20 tokens per '
       'parameter: a realistic allo-HCT corpus implies a model of 50–150 M parameters, not billions. '
       '(c) Pretraining cost at $2 per A100-hour, against a general-purpose 7B model for scale. Compute is '
       'not the binding constraint on a GvHD foundation model.')
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
