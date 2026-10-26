from part1 import CELLS, md, code

md(r'''
---
# 9 · How much data is needed?

Three different questions, three different answers. All are computed here.
1. **Precision** — how wide is the confidence interval of one AUROC at sample size n? (Hanley–McNeil standard error, checked against simulation.)
2. **Power for a comparison** — to claim "ALLO-FM beats MAP + clinical" we need a *paired* test. How many patients for ΔAUROC = 0.02 / 0.03 / 0.05? (DeLong test on correlated scores.)
3. **Development sample size** — Riley et al.'s criteria for a model with P candidate parameters.
''')

code(r'''
# ----------------------------------------------------------------------------- (1) precision of one AUROC; (2) paired power (DeLong)
def hm_se(A, n1, n0):
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    return math.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n0 - 1) * (Q2 - A * A)) / (n1 * n0))
def sim_se(A, n1, n0, reps=900, seed=0):
    rng = np.random.default_rng(seed); d = math.sqrt(2) * norm.ppf(A)
    return float(np.std([roc_auc_score(np.r_[np.ones(n1), np.zeros(n0)], np.r_[rng.normal(d, 1, n1), rng.normal(0, 1, n0)]) for _ in range(reps)]))
print('Hanley–McNeil SE vs simulated SE:', [f'A={A}, n1={a}, n0={b}: {hm_se(A, a, b):.3f} vs {sim_se(A, a, b):.3f}'
                                            for A, a, b in [(.72, 35, 65), (.72, 140, 260)]])
ns = np.array([100, 200, 400, 800, 1600, 3200, 6400])
prec = pd.DataFrame({f'AUROC {A}, prev {pi}': [1.96 * hm_se(A, max(int(n * pi), 2), max(int(n * (1 - pi)), 2)) for n in ns]
                     for A in (.70, .75) for pi in (.15, .35)}, index=ns); prec.index.name = 'patients'

def delong_z(y, s1, s2):
    """DeLong z for the difference of two correlated AUROCs on the same patients (rank form)."""
    m, n = int((y == 1).sum()), int((y == 0).sum()); aucs, v01, v10 = [], [], []
    for s in (s1, s2):
        X, Y = s[y == 1], s[y == 0]; tx, ty, tz = rankdata(X), rankdata(Y), rankdata(np.r_[X, Y])
        aucs.append(tz[:m].sum() / (m * n) - (m + 1) / (2 * n)); v01.append((tz[:m] - tx) / n); v10.append(1 - (tz[m:] - ty) / m)
    S = np.cov(np.vstack(v01)) / m + np.cov(np.vstack(v10)) / n
    return (aucs[1] - aucs[0]) / math.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-12))
def delong_slow(y, s1, s2):
    pos = y == 1; m, n = pos.sum(), (~pos).sum()
    def comps(s):
        psi = (s[pos][:, None] > s[~pos][None, :]) + .5 * (s[pos][:, None] == s[~pos][None, :]); return psi.mean(), psi.mean(1), psi.mean(0)
    a1, x1, y1 = comps(s1); a2, x2, y2 = comps(s2)
    S = np.cov(np.vstack([x1, x2])) / m + np.cov(np.vstack([y1, y2])) / n
    return (a2 - a1) / math.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
_r = np.random.default_rng(0); _y = (_r.random(150) < .35).astype(int)
_a, _b = _r.normal(size=150) + _y, _r.normal(size=150) + .6 * _y
assert abs(delong_z(_y, _a, _b) - delong_slow(_y, _a, _b)) < 1e-9, 'fast and textbook DeLong disagree'

def paired_power(n, pi, A1, A2, rho, reps=120 if not QUICK else 40, seed=0):
    rng = np.random.default_rng(seed); n1 = max(int(round(n * pi)), 3); n0 = n - n1
    d1, d2 = math.sqrt(2) * norm.ppf(A1), math.sqrt(2) * norm.ppf(A2); cov = np.array([[1, rho], [rho, 1]]); hit = 0
    for _ in range(reps):
        sp = rng.multivariate_normal([d1, d2], cov, n1); sn = rng.multivariate_normal([0, 0], cov, n0)
        y = np.r_[np.ones(n1), np.zeros(n0)]
        hit += abs(delong_z(y, np.r_[sp[:, 0], sn[:, 0]], np.r_[sp[:, 1], sn[:, 1]])) > 1.96
    return hit / reps
grid = [200, 400, 800, 1600, 3200, 6400]
pw_rows = []
for d in (.02, .03, .05):
    for rho in (.6, .85):
        p_ = {n: paired_power(n, .35, .70, .70 + d, rho) for n in grid}
        n80_ = next((n for n in grid if p_[n] >= .8), f'>{grid[-1]}')
        pw_rows.append(dict(delta_AUROC=d, score_corr=rho, **{f'n={n}': p_[n] for n in grid}, n_for_80pct_power=n80_))
pw = pd.DataFrame(pw_rows)
PARAMS = [1, 5, 10, 20, 50, 100, 256, 768, 3072]
riley = pd.DataFrame([dict(params=P, **{f'AUROC {A}': riley_n(P, A, .35)['n'] for A in (.68, .72, .75)})
                      for P in PARAMS]).set_index('params')
fig = plt.figure(figsize=(FULL, 3.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1], left=.075, right=.985, bottom=.255, top=.875, wspace=.38)
ax = fig.add_subplot(gs[0, 0]); ygrid(ax)                                          # (a) precision of one AUROC
for c_, col in zip(prec.columns, (PAL['seq'][3], PAL['seq'][1], PAL['orange'], PAL['grey'])):
    ax.plot(prec.index, prec[c_], marker='o', ms=2.8, color=col, lw=1.3, label=c_.replace('AUROC ', 'AUROC ').replace('prev ', '$\\pi$ = '))
ax.axhline(.03, color=PAL['bad'], ls=':', lw=.9)
ax.annotate('target ±0.03', (prec.index[0] * 1.04, .03), xytext=(0, 3), textcoords='offset points', fontsize=6.4, color=PAL['bad'], va='bottom')
ax.set_xscale('log'); ax.set_xlabel('patients'); ax.set_ylabel('95% CI half-width of one AUROC'); ax.set_xlim(90, 9e3)
ax.legend(loc='upper right', labelspacing=.3); panel(ax, 'a')
ax = fig.add_subplot(gs[0, 1]); ygrid(ax)                                          # (b) power to show a gain over a correlated baseline
DCOL = {.02: PAL['seq'][0], .03: PAL['seq'][2], .05: PAL['violet']}
for _, r_ in pw.iterrows():
    ax.plot(grid, [r_[f'n={n}'] for n in grid], marker='o', ms=2.8, color=DCOL[r_.delta_AUROC], lw=1.3, ls='-' if r_.score_corr == .85 else '--')
ax.axhline(.8, color=PAL['bad'], ls=':', lw=.9)
ax.set_xscale('log'); ax.set_xticks(grid); ax.set_xticklabels([f'{g // 1000}k' if g >= 1000 else g for g in grid]); ax.minorticks_off()
ax.set_xlim(150, 8e3); ax.set_ylim(0, 1.05); ax.set_xlabel('patients (prevalence 0.35)'); ax.set_ylabel('power, paired DeLong test')
hs = [plt.Line2D([], [], color=DCOL[d], lw=1.5) for d in (.05, .03, .02)] + [plt.Line2D([], [], color=PAL['ink2'], lw=1.2), plt.Line2D([], [], color=PAL['ink2'], lw=1.2, ls='--')]
ax.legend(hs, ['gain 0.05', 'gain 0.03', 'gain 0.02', 'scores r = 0.85', 'scores r = 0.60'], loc='upper center',
          bbox_to_anchor=(.5, -.24), ncol=5, columnspacing=1.3, handlelength=1.6, labelspacing=.3); panel(ax, 'b')
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                          # (c) patients needed to develop a model
for c_, col in zip(riley.columns, (PAL['seq'][1], PAL['seq'][2], PAL['seq'][3])):
    ax.plot(riley.index, riley[c_], marker='o', ms=2.8, color=col, lw=1.3, label=c_.replace('AUROC ', 'target AUROC '))
for P_, lab in ((20, 'registry\nmodel'), (768, 'FM embedding\nhead')):                  # labelled above the axes, centred on their own line
    ax.axvline(P_, ymin=.32, color=PAL['ink2'], lw=.6, ls=':')
    ax.annotate(lab, (P_, 1.0), xycoords=('data', 'axes fraction'), xytext=(0, 3), textcoords='offset points',
                fontsize=6.4, color=PAL['ink2'], va='bottom', ha='center')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('candidate parameters $P$'); ax.set_ylabel('patients to develop the model')
ax.set_xlim(.8, 1.2e4); ax.set_ylim(25, 6e5); ax.legend(loc='lower right', labelspacing=.3); panel(ax, 'c')
finish(fig, 9, 'sample_size_and_power',
       'Three different sample-size questions. (a) Precision: the 95% confidence interval of a single AUROC against the number of '
       'patients. (b) Power: the paired DeLong test for an improvement of 0.02 to 0.05 AUROC over a correlated baseline, the regime '
       'any foundation-model claim lives in; solid lines are scores correlated at 0.85, dashed at 0.60. (c) Development: the Riley '
       'minimum sample size against the number of candidate parameters, which is why head size sets the data requirement.')
display(md_table(prec.round(3), index=True)); display(md_table(pw.round(2))); display(md_table(riley, index=True))
''')

code(r'''
# ----------------------------------------------------------------------------- what each data source can actually support
n80 = lambda d, rho: pw[(pw.delta_AUROC == d) & (pw.score_corr == rho)].n_for_80pct_power.iloc[0]
SOURCES = [('One large centre, 10 years (Tier A+B)', 2500), ('BMT CTN 1703 (randomised, Tier A+C)', 431),
           ('MAGIC day-7 biomarker cohort (Tier A+C)', 1287), ('MSK hospitalome (public, Tier A+B+microbiome)', 1000),
           ('CIBMTR extract, recent era, selected donor types (Tier A)', 25000), ('CIBMTR full (Tier A)', 470000),
           ('Typical public GEO GvHD series (Tier D)', 40)]
need = {'registry model (P=20)': riley_n(20, .72, .35)['n'], 'pathway/panel head (P=50)': riley_n(50, .72, .35)['n'],
        'FM embedding head (P=768)': riley_n(768, .72, .35)['n'],
        f'show Δ=0.03 vs baseline (corr .85)': n80(.03, .85), f'show Δ=0.05 vs baseline (corr .85)': n80(.05, .85)}
cap = pd.DataFrame([dict(source=s, patients=n, **{k: ('yes' if isinstance(v, (int, float)) and n >= v else 'NO') for k, v in need.items()}) for s, n in SOURCES])
print('requirements (patients):', {k: (f'{v:,}' if isinstance(v, (int, float)) else v) for k, v in need.items()})
display(md_table(cap))
display(Markdown(f"""**Reading.** (1) Even a *single* AUROC is imprecise below ~1,000 patients: at n = 400 and prevalence 0.35 the 95 % CI half-width is ±{prec.loc[400, 'AUROC 0.7, prev 0.35']:.3f}. (2) Showing that a foundation model beats MAP + clinical covariates by Δ = 0.03 — the realistic size of a genuine gain between correlated models — needs about **{n80(.03, .85) if isinstance(n80(.03,.85), str) else format(n80(.03,.85), ',')}** patients at score correlation 0.85, and more if the correlation is lower; Δ = 0.02 is out of reach of anything short of a registry. (3) Developing a 768-parameter head at AUROC 0.72 and prevalence 0.35 needs **{riley_n(768, .72, .35)['n']:,}** patients unpenalised; ridge shrinkage and pretraining are precisely the arguments for needing fewer, which is the claim the benchmark exists to test.

**The conclusion that drives the whole roadmap.** No public transcriptomic resource and no single trial can answer the central question. **A registry-scale Tier A/B cohort, with Tier C biomarkers on a subset, is the minimum viable evaluation substrate.** This is why ALLO-FM is specified as a clinical-timeline model with optional omics adapters, and not the other way round."""))
''')

md(r"""
## 9.3 How long would it take? Accrual projections per task

Required n becomes a plan only when divided by a rate. The rates below are deliberately conservative and are stated as assumptions: a large transplant centre performs of the order of **150–250 allogeneic transplants a year** †; a committed **20-centre consortium** with protocolised biospecimen collection therefore accrues of the order of **2,000–4,000 a year**; a **registry extract is retrospective**, so its "accrual" is the historical back-catalogue plus roughly 10,000 new US allogeneic transplants a year †. Prospective collection also loses patients to consent and to specimen failure; we assume 70 % usable.

The projection below answers, for each task that has a numeric target: given the n it needs, how long would each route take to get there? This is what converts "we need more data" into a proposal.
""")

code(r'''
# ----------------------------------------------------------------------------- accrual projection: years to reach the required n, by route
ROUTES = [  # name, patients per year contributing to a Tier-C-grade task, tier reachable, retrospective back-catalogue
    ('single centre, prospective',        200 * .70,  'A+B+C', 0),
    ('20-centre consortium, prospective', 3000 * .70, 'A+B+C', 0),
    ('registry extract (CIBMTR/EBMT)',    10000.0,    'A+B',   120000),
    ('existing biobank (MAGIC/BMT CTN)',  0.0,        'A+B+C', 4000),
]
need_tbl = reg_full.dropna(subset=['n to show +0.03']).copy()
need_tbl['n required'] = need_tbl[['n to develop (50-param)', 'n to show +0.03']].max(axis=1)
rows = []
for _, r in need_tbl.iterrows():
    tier_need = str(r['tier needed'])
    for name, rate, tier_ok, backlog in ROUTES:
        deep_needed = ('C' in tier_need) or ('D' in tier_need)
        route_has_deep = 'C' in tier_ok
        if deep_needed and not route_has_deep:
            yrs = np.nan                                     # this route cannot serve this task at all
        else:
            remaining = max(r['n required'] - backlog, 0)
            yrs = 0.0 if remaining == 0 else (remaining / rate if rate > 0 else np.inf)
        rows.append(dict(task=r.id, route=name, **{'n required': int(r['n required']), 'years': yrs}))
acc = pd.DataFrame(rows)
piv_acc = acc.pivot(index='task', columns='route', values='years')
piv_acc = piv_acc.reindex([c[0] for c in ROUTES], axis=1)
fmt = piv_acc.copy()
for c in fmt.columns:
    fmt[c] = [('—' if not np.isfinite(v) and np.isnan(v) else ('never' if v == np.inf else ('now' if v == 0 else f'{v:.1f} y')))
              for v in piv_acc[c]]
fmt.insert(0, 'n required', need_tbl.set_index('id').loc[fmt.index, 'n required'].astype(int).values)
display(md_table(fmt.reset_index()))

# ------------------------------------------------- Figure 10: what exists against what is needed, and how long it would take
FWIN, FHIN = FULL, 4.6
fig = plt.figure(figsize=(FWIN, FHIN))
axa = fig.add_axes([1.33 / FWIN, .95 / FHIN, 1.75 / FWIN, 3.15 / FHIN])
axb = fig.add_axes([4.15 / FWIN, .95 / FHIN, 2.95 / FWIN, 3.15 / FHIN])
SRC = [('CIBMTR full (2017)', 470000, PAL['aqua']), ('Registry extract', 120000, PAL['aqua']), ('MAGIC day-7 cohort', 1287, PAL['orange']),
       ('MSK hospitalome', 1000, PAL['orange']), ('Pilot, 9 studies', 838, PAL['grey2']), ('BMT CTN 1703', 431, PAL['orange']),
       ('GEO series (typical)', 40, PAL['grey2'])]
ax = axa; ygrid(ax, 'x')
ax.barh(range(len(SRC)), [v for _, v, _ in SRC], color=[c for _, _, c in SRC], height=.62, zorder=3)
ax.set_yticks(range(len(SRC))); ax.set_yticklabels([f'{n}\nn = {v:,}' for n, v, _ in SRC]); ax.tick_params(axis='y', length=0)
ax.set_xscale('log'); ax.set_xlim(10, 1e6); ax.set_ylim(len(SRC) - .5, -.5); ax.minorticks_off()
n20, n768 = riley_n(20, .72, .35)['n'], riley_n(768, .72, .35)['n']
ax.axvline(n20, color=PAL['ink'], ls='--', lw=.9, zorder=4); ax.axvline(n768, color=PAL['bad'], ls='--', lw=.9, zorder=4)
ax.set_xlabel('patients (log scale)')
ax.annotate('a', (0, 1), xycoords='axes fraction', xytext=(-1.25 * 72, 32), textcoords='offset points', fontsize=9, fontweight='bold')
fig.legend([plt.Rectangle((0, 0), 1, 1, color=PAL['aqua']), plt.Rectangle((0, 0), 1, 1, color=PAL['orange']), plt.Rectangle((0, 0), 1, 1, color=PAL['grey2']),
            plt.Line2D([], [], color=PAL['ink'], ls='--', lw=.9), plt.Line2D([], [], color=PAL['bad'], ls='--', lw=.9)],
           ['registry', 'consortium / biobank', 'public', f'needed to develop a 20-parameter model ({n20:,})',
            f'needed to develop a 768-parameter head ({n768:,})'], ncol=2, loc='lower left', bbox_to_anchor=(.02, .0),
           columnspacing=1.4, handlelength=1.4, labelspacing=.4)
# (b) years to reach the required n, task by task and route, as a matrix
ROUTE_NAMES = ['single\ncentre', '20-centre\nconsortium', 'registry\nextract', 'existing\nbiobank']
RT = ['single centre, prospective', '20-centre consortium, prospective', 'registry extract (CIBMTR/EBMT)', 'existing biobank (MAGIC/BMT CTN)']
tbl = piv_acc[RT].copy(); tbl['best'] = tbl.min(axis=1, skipna=True); tbl = tbl.sort_values('best').drop(columns='best')
nreq = need_tbl.set_index('id')['n required']
def cls(v):
    if np.isnan(v): return 3
    if v == 0: return 0
    return 0 if v <= 1 else 1 if v <= 5 else 2
def txt(v):
    if np.isnan(v): return '—'
    if v == 0: return 'now'
    return '>40 y' if v > 40 else (f'{v:.1f} y' if v < 10 else f'{v:.0f} y')
ax = axb
CELLC = [PAL['aqua'], PAL['seq'][0], PAL['orange'], '#ececea']; CELLA = [.85, .85, .55, 1.0]
nr, nc = len(tbl), 5
for i, (tid, row) in enumerate(tbl.iterrows()):
    ax.text(0.5, i, f'{int(nreq[tid]):,}', ha='center', va='center', fontsize=6.4, color=PAL['ink'])
    for j, rt in enumerate(RT):
        v = row[rt]; ax.add_patch(plt.Rectangle((j + 1.03, i - .46), .94, .92, color=CELLC[cls(v)], alpha=CELLA[cls(v)], lw=0))
        _t = ax.text(j + 1.5, i, txt(v), ha='center', va='center', fontsize=6.3, color=PAL['ink'] if cls(v) != 3 else PAL['ink2'])
        _t._ok_overlap = True      # the label belongs to this cell
ax.set_xlim(0, nc); ax.set_ylim(nr - .5, -.5)
ax.set_yticks(range(nr)); ax.set_yticklabels(tbl.index); ax.tick_params(axis='y', length=0)
ax.xaxis.tick_top(); ax.set_xticks(np.arange(nc) + .5); ax.set_xticklabels(['patients\nrequired'] + ROUTE_NAMES); ax.tick_params(axis='x', length=0)
for sp in ax.spines.values(): sp.set_visible(False)
ax.annotate('b', (0, 1), xycoords='axes fraction', xytext=(-.45 * 72, 32), textcoords='offset points', fontsize=9, fontweight='bold')
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c, alpha=a) for c, a in zip(CELLC, CELLA)],
           ['covered now or within 1 year', 'within 5 years', 'more than 5 years', 'route cannot serve the task'],
           ncol=2, loc='lower right', bbox_to_anchor=(.99, .0), columnspacing=1.2, handlelength=1.2, labelspacing=.4)
finish(fig, 10, 'data_landscape_and_accrual',
       'What exists against what is needed, and how long it would take. (a) Patients available from each real source on a log scale, '
       'against the Riley minimum for two head sizes: public transcriptomic resources are orders of magnitude short, registry extracts are '
       'not. (b) Projected time to reach the patients required for each task, by route. Tasks that need protocolised biospecimens cannot '
       'be served by a registry extract at all, which is why the staged plan buys the registry tasks first and the biobank tasks second.')
''')

code(r'''
reg_now = int((piv_acc['registry extract (CIBMTR/EBMT)'] == 0).sum())
cons_5y = int((piv_acc['20-centre consortium, prospective'] <= 5).sum())
sc_5y = int((piv_acc['single centre, prospective'] <= 5).sum())
display(Markdown(f"""**Result.** Of the {len(piv_acc)} tasks with a numeric target, **{reg_now} are already satisfied by the historical back-catalogue of a registry extract** — no waiting, no new collection, only a data-use agreement. A **20-centre consortium reaches {cons_5y} of them within five years**; a **single centre reaches {sc_5y}**, which is the quantitative case for a consortium rather than a local study.

**The strategic ordering falls out of the arithmetic.** Buy the registry tasks first (they are free in time and need only governance), use an existing biobank for the day-7 biomarker tasks (MAGIC-scale specimens already exist), and start prospective collection *now* for the tasks nothing can shortcut — the adjudicated differential-diagnosis set (C1, C2) and the treated-patient response cohorts (C4, C5) — because those are the ones whose clock has not started. That ordering is §11's staged plan, and this projection is its justification."""))
''')


md(r'''
---
# 10 · Pilot evidence: what the transcriptome-first attempt established

Before this specification we built and ran the opposite design: expression as the backbone, GvHD as the label, public data only. The full notebook is `GvHD_FM_benchmark_blueprint.ipynb` in this repository; its numbers are reproduced here as the empirical starting point of the present paper. Nine public studies, 838 samples after QC, three platforms, blood and tissue, patient-grouped splits throughout, with random-weight twins and confounder-only comparators.

| Finding (executed on public data) | Number | What it implies for ALLO-FM |
|---|---|---|
| Normalised genes + ridge head, leave-one-study-out over 7 studies | mean AUROC **0.684** | the baseline that any representation must beat |
| Split by sample type | tissue biopsy **0.804**, blood/sorted T cells **0.594** | ρ (shared structure) is high for tissue, low for blood → tissue adapters first (C7), blood transcriptome adapters later |
| Hallmark-50 pathway prior, under a gate fixed before the run | **0.589**, below genes (0.684) and below the shuffled-membership control (0.623 ± 0.033); all four gate criteria failed | a curated prior is not automatically a better small-n representation; the earlier 4-study advantage (0.695 vs 0.615) did not replicate |
| Frozen Geneformer 10M / 104M / 316M, 4 RNA-seq studies, LOSO | **0.498 / 0.678 / 0.615** vs genes **0.756**; 104M beat its random-weight twin by **+0.064** [+0.035, +0.104], 316M and 10M did not | pretraining contributed something at one scale, but **nothing beat the simple baseline**; scale did not help |
| Within-study, pretrained vs random-weight twin | better in **10 of 21** task × model pairs | at n ≈ 20–90 a high-dimensional random embedding is a lottery ticket; single-task comparisons are uninformative |
| Input compatibility | **5 of 9** studies (microarray or gene-restricted release) could not be embedded at all | a single-cell-counts foundation model cannot read most public GvHD data |
| Zero-parameter IFN-γ / allograft-rejection score, no fitting | skin and colon **0.80–0.88**; sorted PtCy T cells **below 0.5** | the model-free biology baseline is strong *in inflamed tissue*; it is the baseline for C7 |
| cfRNA, the only large public longitudinal set (92 patients) | pre-GvHD samples AUROC **0.32** [0.14, 0.48]; forecasting month 1 → month 2 barely beat the population mean | **no reproducible pre-onset signal** in public blood; prognosis needs protocolised collection (§4.4) |
| Label efficiency | tissue curve still rising at k = 256; blood flat at 0.52–0.59 whatever k | blood's problem was **domain shift, not sample count** — the ρ = 0 regime of Figure 2 |
| Data quality | 4 of 54 skin libraries were empty, including 2 of 9 controls | QC audits belong in the harness, not in the appendix |

**Three design decisions follow directly.**
1. **Change the backbone.** The modality with high ρ and registry-scale n is the clinical timeline. Omics becomes an adapter with modality dropout, so a Tier A patient still gets a prediction (§5.1).
2. **Stop relying on retrospective public specimens for prognosis.** The sampling-time confound and the absence of onset dates make pre-onset tasks unanswerable on GEO. Hence the protocolised schedule (§4.4) and `t_known` discipline (§4.7).
3. **Keep the controls.** Random-weight twins, confounder-only models, shuffled-prior controls and pre-stated gates did their job: they turned an exciting four-study result into a correctly-reported replication failure. They are mandatory in GvHD-Bench (§8.3, §8.6).

**What the pilot does not show.** It does not show that transcriptomes are useless for GvHD — tissue biopsies transferred well, and the IFN-γ programme is real. It shows that *blood transcriptome alone, from heterogeneous public studies, at n ≈ 10²* cannot support a foundation-model claim.
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 11: the pilot results that this paper is built on (numbers from the earlier notebook)
PILOT_LOSO = [('PBMC T cells', .790, 'blood'), ('CD8 subsets', .413, 'blood'), ('PtCy T cells', .577, 'blood'), ('sc pseudobulk', .595, 'blood'),
              ('colon (1)', .826, 'tissue'), ('colon (2)', .833, 'tissue'), ('skin', .754, 'tissue')]
FM_ARMS = [('plain\ngenes', .756, None), ('GF\n10M', .498, .547), ('GF\n104M', .678, .614), ('GF\n316M', .615, .652)]
ZP = [('skin', .84, 'tissue'), ('colon', .84, 'tissue'), ('PBMC T cells', .70, 'blood'), ('sorted PtCy T', .45, 'blood'), ('CD8 homing', .42, 'blood')]
TC, BC = PAL['aqua'], PAL['seq'][1]
fig = plt.figure(figsize=(FULL, 3.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.2, 1.0], left=.075, right=.985, bottom=.235, top=.885, wspace=.38)
def vlab(ax, i, v, ref=.5, x=None, dx=0):
    """Value label above its bar; inside the bar if it would otherwise cross the chance line."""
    inside = ref - .09 <= v < ref
    t = ax.text((i if x is None else x) + dx, v - .012 if inside else v + .014, f'{v:.2f}'[1:] if v < 1 else f'{v:.2f}', ha='center',
                va='top' if inside else 'bottom', fontsize=6.4, fontweight='bold')
    if inside: t._ok_overlap = True
ax = fig.add_subplot(gs[0, 0]); ygrid(ax)                                           # (a) leave-one-study-out transfer
ax.bar(range(7), [v for _, v, _ in PILOT_LOSO], color=[TC if g == 'tissue' else BC for _, _, g in PILOT_LOSO], width=.66, zorder=3)
for i, (_, v, _) in enumerate(PILOT_LOSO): vlab(ax, i, v)
ax.axhline(.5, color=PAL['ink2'], ls='--', lw=.6)
m_b = np.mean([v for _, v, g in PILOT_LOSO if g == 'blood']); m_t = np.mean([v for _, v, g in PILOT_LOSO if g == 'tissue'])
for x0, x1, m, g, c in ((-.4, 3.4, m_b, 'blood / sorted T cells', BC), (3.6, 6.4, m_t, 'tissue biopsy', TC)):
    ax.plot([x0, x1], [1.05, 1.05], color=c, lw=1.6, solid_capstyle='butt')
    ax.annotate(f'{g}\nmean {m:.2f}', ((x0 + x1) / 2, 1.05), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=6.4, fontweight='bold')
ax.set_xticks(range(7)); ax.set_xticklabels([n for n, _, _ in PILOT_LOSO], rotation=35, ha='right', rotation_mode='anchor')
ax.set_ylim(0, 1.0 + .30); ax.set_yticks([0, .2, .4, .6, .8, 1.0]); ax.spines['left'].set_bounds(0, 1.0)
ax.set_ylabel('leave-one-study-out AUROC'); panel(ax, 'a')
ax = fig.add_subplot(gs[0, 1]); ygrid(ax)                                           # (b) frozen foundation models against their random twins
x = np.arange(len(FM_ARMS)); w = .40
ax.bar([0], [FM_ARMS[0][1]], w * 1.35, color=PAL['seq'][2], zorder=3, label='frozen model')
ax.bar(x[1:] - w / 2, [a[1] for a in FM_ARMS[1:]], w, color=PAL['seq'][2], zorder=3)
ax.bar(x[1:] + w / 2, [a[2] for a in FM_ARMS[1:]], w, color=PAL['grey2'], zorder=3, label='random-weight twin')
ax.axhline(FM_ARMS[0][1], color=PAL['ink'], ls='--', lw=.8, label='plain normalised genes')
vlab(ax, 0, FM_ARMS[0][1], ref=9, x=0)
for xi, a in zip(x[1:], FM_ARMS[1:]):
    vlab(ax, 0, a[1], ref=9, x=xi - w / 2); vlab(ax, 0, a[2], ref=9, x=xi + w / 2)
ax.set_xticks(x); ax.set_xticklabels([a[0] for a in FM_ARMS]); ax.set_ylim(.35, 1.0); ax.set_yticks([.4, .5, .6, .7, .8])
ax.spines['left'].set_bounds(.35, .8); ax.set_ylabel('4-study LOSO AUROC'); ax.legend(loc='upper right', labelspacing=.3); panel(ax, 'b')
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                           # (c) the zero-parameter score
ax.bar(range(5), [v for _, v, _ in ZP], color=[TC if g == 'tissue' else BC for _, _, g in ZP], width=.64, zorder=3)
for i, (_, v, _) in enumerate(ZP): vlab(ax, i, v)
ax.axhline(.5, color=PAL['ink2'], ls='--', lw=.6)
ax.set_xticks(range(5)); ax.set_xticklabels([n for n, _, _ in ZP], rotation=35, ha='right', rotation_mode='anchor')
ax.set_ylim(0, 1.0); ax.set_ylabel('AUROC, no fitting\n(IFN-γ / allograft score)'); panel(ax, 'c')
ax.legend([plt.Rectangle((0, 0), 1, 1, color=TC), plt.Rectangle((0, 0), 1, 1, color=BC)], ['tissue', 'blood / T cells'],
          loc='upper right', handlelength=1.0, labelspacing=.3)
finish(fig, 11, 'pilot_results',
       'The pilot this paper is built on (9 public studies, 838 samples; numbers from the earlier notebook in this repository). '
       '(a) Leave-one-study-out transfer with plain normalised genes: tissue biopsies transfer across laboratories and organs, blood '
       'and sorted T cells do not. (b) Frozen Geneformer (GF) at three sizes against its own random-weight twins and against plain '
       'genes: one size beats its twin, none beats the simple baseline, and the largest model is not the best. (c) A zero-parameter '
       'score written down in advance, the mean expression of the interferon-gamma and allograft-rejection gene sets, matches '
       'fitted models in inflamed tissue and fails in sorted blood T cells.')
''')


md(r'''
---
# 11 · Roadmap, governance and ethics

''')

md(r"""
## 11.0 Do we need a harness? Yes — and this is what it is

A benchmark is a claim about reproducibility, so it has to ship as software. Without a harness, every one of the leakage modes in §4.7 is re-litigated by every group, and no two results are comparable. The harness is small — this is tabular and sequence data, not a distributed training problem — and its job is to make the dangerous things impossible rather than merely discouraged.

**Seven components, in dependency order.**

| Component | Responsibility | Why it must be code, not a document |
|---|---|---|
| **1 · Ingest & schema** | map a site's OMOP + HCT-extension tables to the one event schema of §4.1, with units, coding systems, and both `t_valid` and `t_known` | the knowledge-time distinction is the single highest-yield leakage control, and it is lost the moment a human is asked to remember it |
| **2 · Label library** | the executable functions of §4.6 (MAGIC grade, steroid-refractory phenotype, NIH cGvHD, GRFS) with their unit tests | two centres implementing "grade II–IV by day 100" from prose will not agree |
| **3 · Cohort & landmark builder** | apply inclusion/exclusion, landmark eligibility, competing-event handling; emit the mandatory attrition table | attrition is where silent selection enters |
| **4 · Split manager** | construct, freeze, hash and verify the leave-centre-out / temporal / leave-registry-out assignments | a split that can be regenerated differently is not a split |
| **5 · Feature gate** | per-task admissibility: `t_known ≤ L`, banned treatment proxies, declared tier; refuses to emit an inadmissible matrix | the gate must be upstream of the model, so that leaking requires editing the harness rather than forgetting a rule |
| **6 · Metric & audit library** | IPCW time-dependent AUROC, Aalen–Johansen calibration, net benefit, patient-cluster bootstrap, BH adjustment, the leakage audit, the leakage canary | all implemented and validated in §8.6/§8.7 of this notebook |
| **7 · Baseline zoo & runner** | reference implementations of every mandatory comparator; runs submissions in a container; records version, hashes, seeds, ablations | "we compared to a logistic regression" is not a baseline; a pinned implementation is |

**Federated by construction.** Because the event schema is identical across sites, the harness runs in two modes: *pooled*, on a central extract under one data-use agreement, and *federated*, where each site runs ingest, labels, cohort, splits and metrics locally and only gradients or summary statistics leave. The federated mode is not an afterthought for v2 — it is how a 20-centre consortium becomes possible without a central patient-level repository, and the only component that changes is the runner.

**Engineering estimate.** Components 1–7 are of the order of **15–25k lines of Python** over a conventional stack: Polars or DuckDB for the event store, PyTorch for the model, a tokeniser of a few hundred lines, scikit-learn and XGBoost for the baseline zoo, and lifelines or scikit-survival cross-checked against the implementations here. The model is the small part. The metric, audit and split machinery of this notebook is already the seed of components 4 and 6.

## 11.1 Architecture decisions, stated once

| Decision | Choice | Why, and what would change it |
|---|---|---|
| Backbone input | **the day-0-anchored clinical event timeline** | it is the only modality with registry-scale n and high cross-centre shared structure (§2.1, §10); if a prospective study showed blood transcriptomes transferring across centres at ρ ≳ 0.8, an omics-first backbone would become defensible |
| Sequence model | decoder-style transformer with explicit continuous-time encoding; **not** a fixed-grid RNN | visit times are irregular and clinically meaningful; a fixed grid discards exactly the information that distinguishes day +7 from day +14 |
| Size | **50–150 M parameters** | compute-optimal at the realistic corpus (§7.2); larger is data-limited and unfalsifiable at this n |
| Pretraining objectives | next-event + time-to-event + masked-value (O-T1/T2/T3) | the time-to-event objective is the closest to every downstream question, and is the one with published label-efficiency gains ✓ |
| Omics | **adapters with modality dropout**, one summary token each, optional at inference | a model that requires single-cell data for every patient is untrainable and undeployable this decade |
| Heads | one **multi-state** head over cause-specific hazards | keeps the tasks mutually consistent; prevents a model claiming 40 % aGvHD risk and 90 % GvHD-free survival at once |
| Centre invariance | gradient-reversal / CORAL penalty, tuned against the two-axis score, **not** against downstream AUROC | tuning invariance on the test question is circular; the two-axis score (biology kept, centre discarded) is measurable without labels |
| Calibration | isotonic or Platt recalibration **per deployment site**, re-fit on local data | calibration, not discrimination, is what transfers worst between centres, and it is the part that changes clinical decisions |
""")


md(r'''
## 11.2 Staged plan with go / no-go gates
| Stage | Work | Data needed | Go / no-go |
|---|---|---|---|
| **0 · Specification** | This document: data dictionary, label functions, task registry, metrics, gate, harness (all exported to `spec/`) | none | Done when the notebook runs end-to-end and the exported files validate |
| **1 · One-centre Tier A+B** | Build the OMOP + HCT-extension extract for one large centre (10³–10⁴ transplants); implement O-T1/O-T2/O-T3; run tasks A1–A5, B1–B6, C1–C5 with leave-era-out | single-centre IRB | Proceed only if the timeline model reaches **Level 1** against tuned trees on registry variables |
| **2 · Multi-centre Tier A** | CIBMTR (and/or EBMT) extract; pretrain at scale; leave-centre-out and leave-registry-out; run the scaling ablation | registry study proposal + DUA | Proceed only if **Level 2** is reached on the primary task B1, with the random-weight twin and no-pretraining arms both beaten |
| **3 · Tier C adapters** | MAGIC / BMT CTN biorepository: biomarker, flow and microbiome adapters; tier-ablation task G4 | consortium collaboration | Proceed if an adapter adds ≥ 0.02 AUROC over Tier A+B at the same n, out of centre |
| **4 · Tier D and mechanism** | Repertoire, single-cell, pathology adapters on nested sub-cohorts; tasks C7, F1–F3 | prospective protocol (§4.4) | Research output regardless; no deployment claim |
| **5 · Prospective silent mode** | Model runs without influencing care; calibration drift monitored; subgroup audit | prospective approval | Level 3 only if calibration and net benefit hold prospectively |
| **6 · Interventional trial** | Randomise the *decision* (e.g. model-guided pre-emptive therapy vs standard) | trial | Only this licenses "improves outcomes" |

**Publish the negative.** If stage 1 or 2 fails its gate, the result is a paper: *"a clinical foundation model does not beat tuned registry baselines for GvHD prediction, at this scale, under these splits."* That is the output our pilot produced in the transcriptomic setting, and it was worth reporting.

## 11.3 What makes this fail (pre-mortem)
1. **Registry labels are too coarse.** If only maximum aGvHD grade is recorded, landmark tasks B1–B4 collapse. *Mitigation*: the daily-staging requirement in D13/D23; use centres with day-level data for the early-warning task.
2. **Centre practice is the real signal.** Prophylaxis platform, biopsy habits and steroid thresholds differ, so a model can learn the centre. *Mitigation*: confounder-only comparator, leave-centre-out as default, centre-adversarial objective, and the equity audit G3.
3. **Pretraining adds nothing over gradient-boosted trees.** Tabular registry data is where trees are strongest. *Mitigation*: the gate measures exactly this; Level 1 is about *label efficiency*, which is where sequence pretraining has the best published case.
4. **Omics adapters never become available at scale.** *Mitigation*: Tier A/B must stand alone; adapters are optional by construction.
5. **Temporal drift.** Practice changed (letermovir, PtCy for matched donors, ruxolitinib, belumosudil, axatilimab). A model trained on the old era may be useless. *Mitigation*: temporal split is mandatory (G1), era is an explicit feature, and recalibration is part of the deployment plan.

## 11.4 Governance, ethics and reporting
* **Consent and access.** Registry and repository use runs through study proposals, DUAs and GA4GH DUO-coded consent. Nothing in this specification requires re-identifiable data to leave its custodian: the model can be trained federated-first, since the event schema is identical across sites.
* **Equity.** Donor availability differs by genetic ancestry, which means both the exposure (donor type) and the outcome distribution differ by group. A model that is accurate on average and poorly calibrated for under-served groups would worsen the existing disparity. Task G3 is therefore mandatory, not optional, and worst-subgroup calibration is a reported primary quantity.
* **Paediatrics.** Thresholds (stool volume per kg), scores (Lansky) and biology (thymic output) differ. Paediatric patients are a reported subgroup, never a pooled afterthought.
* **Reporting standards.** TRIPOD+AI for the prediction model, with a data card per source, a model card stating the pretraining corpus and required preprocessing, frozen split files, seeds, and a published negative-results table. CONSORT-AI / SPIRIT-AI apply at stage 6.
* **Clinical framing.** The model's outputs are risks, not diagnoses. GvHD diagnosis remains clinical and histological; the intended use of a Level-3 model is to *prioritise* investigation, pre-emptive therapy trials and monitoring intensity.

---
# 12 · Limitations

1. **Nothing here is trained.** This is a specification with executed statistics and simulations, not a model. Figures 2 and 4 are **[SIM]**: their effect sizes are ours, chosen to exercise the machinery, and they say nothing about patients.
2. **The clinical-timeline premise is itself untested for GvHD.** Label efficiency from EHR foundation models is published for other endpoints (✓ for MOTOR's reported numbers), not for GvHD. Stage 1 exists to test it, and may refute it.
3. **The data dictionary is a target, not an inventory.** No single existing cohort holds all 25 domains. The tier structure is the honest statement of what is available where.
4. **Some definitions are from domain knowledge (†)** and must be checked line-by-line against the primary consensus documents before submission: the MAGIC grade table, the 2018 steroid-refractoriness criteria, NIH 2014 organ scoring, refined DRI, HCT-CI weights, the DPB1 TCE algorithm, and the EBMT SOS criteria. The *code* in §4.6 is the authoritative version of what this benchmark means by each label, and it is testable; if it disagrees with the consensus table, the consensus table wins and the code must change.
5. **Prevalences in the registry are planning assumptions**, so every "minimum n" moves with the real prevalence of the cohort.
6. **Competing-risk discrimination is still contested.** IPCW time-dependent AUROC depends on a correct censoring model; informative censoring (transfer of care, which correlates with outcome) will bias it. We report Brier/IBS and calibration alongside, and the censoring model is part of the audit.
7. **Simulation cannot validate the premises.** Figure 2 shows *that* pretraining depends on shared structure; only real multi-centre data can show *whether* GvHD data has it.

---
# 13 · Conclusion

A foundation model for GvHD is plausible, but not in the form we first tried. The transcriptome-first attempt failed on public data for understandable reasons: too few labelled patients per study, no shared structure across studies for blood, no onset dates, and input formats the model could not read. The clinical timeline of allo-HCT is the opposite: anchored at day 0, densely sampled, protocolised, recorded with the same forms in every centre, and available for hundreds of thousands of patients. That is where a foundation model has something to learn from.

What such a model needs is not a new architecture. It needs a **data specification** precise enough that two centres produce the same variable (§4), **objectives** that each come with a control (§6), and a **benchmark** whose tasks are the real decisions, whose baselines are what clinicians use today, whose splits are leave-centre-out and temporal, and whose gates were written before the run (§8). The sample-size calculus (§9) says plainly that only consortium-scale data can settle the question, and the pilot (§10) says plainly what happens when you try to settle it with less.

If the gates are passed, the result is a model that can be taken to a prospective evaluation. If they are not, the result is a negative finding reported at a useful level of detail. Both outcomes are publishable; only one of them is currently assumed.
''')

code(r'''
# ----------------------------------------------------------------------------- export GvHD-Bench as a standalone specification document
def _bench_md():
    """Assemble spec/GVHD_BENCH_v1.0.md from the same objects the notebook computed (never hand-typed)."""
    L = ['# GvHD-Bench v1.0 — benchmark specification', '',
         f'Generated by `ALLO_FM_GvHD_foundation_model.ipynb` on {time.strftime("%Y-%m-%d", time.gmtime())}. '
         'This file is self-contained: it is the normative definition of the benchmark. '
         'Section numbers refer to the notebook.', '',
         '## 1 Scope and governance', '',
         '- Unit of analysis: the patient (donor-recipient pair where both contribute data).',
         '- Clock: days relative to first graft infusion (day 0).',
         '- Admissibility: an event is readable at landmark L only if `t_known <= L`.',
         '- Label authority: the executable functions in notebook 4.6; a consensus document always wins over the code.',
         '- Versioning: patch = label/metric bug fix; minor = new task or baseline; major = changed split, cohort or gate.',
         '  Results are comparable only within a major version.', '',
         '## 2 Cohort', '',
         '**Included**: first allogeneic HCT, any indication, donor type, graft source or conditioning intensity;',
         'recorded day 0; at least one contact after day +7.', '',
         '**Excluded**: autologous/syngeneic; second and later transplants (new pair_id, analysed separately);',
         'consent withdrawn; records with missing or inconsistent day 0 or outcome dates (counted, never silently dropped).', '',
         '**Landmark eligibility (per task)**: alive, in follow-up, and outcome-free at L. Patients who died, relapsed',
         'or were censored before L are excluded from that task and appear in its attrition table.', '',
         '**Competing events**: death and relapse are events, not censoring, for every GvHD endpoint.', '',
         '## 3 Tasks', '',
         f'{len(reg_full)} tasks in {reg_full.family.nunique()} families. '
         f"Powered at accessible n: {(reg_full.verdict == 'POWERED').sum()}; "
         f"under-powered: {(reg_full.verdict == 'UNDER-POWERED').sum()}; "
         f"not single-AUROC tasks: {(reg_full.verdict == '-').sum()}.", '',
         'Full machine-readable definitions: `task_registry.csv` (all fields) and `task_cards.csv`',
         '(readiness, accessible n, requirements, verdict). Columns are documented in notebook 8.3-8.4.', '']
    for fam in reg_full.family.unique():
        L.append(f'### Family {fam}'); L.append('')
        for _, r in reg_full[reg_full.family == fam].iterrows():
            L += [f'**{r.id} — {r.task}**', '',
                  f'- landmark: {r.landmark} · horizon: {r.horizon}',
                  f'- inputs allowed: tiers {r["inputs allowed (tiers)"]} · outcome: {r["outcome (label fn)"]}',
                  f'- competing events: {r["competing events"]}',
                  f'- primary metric: {r["primary metric"]}',
                  f'- must beat: {r["clinical baseline to beat"]}',
                  f'- split protocol: {r["split protocol"]}',
                  f'- readiness: {r.readiness} · tier needed: {r["tier needed"]}',
                  f'- assumed prevalence: {r["assumed prevalence"]} · target AUROC: {r["target AUROC"]}'
                  f' · achievable (notebook 3.2): {r["achievable AUROC"] or "n/a"}',
                  f'- n accessible (est.): {r["n accessible (est.)"]:,} · n to develop (50-param): '
                  f'{r["n to develop (50-param)"]} · n to show +0.03: {r["n to show +0.03"]}',
                  f'- **verdict: {r.verdict}**', '']
    L += ['## 4 Metrics', '',
          '- Discrimination: IPCW cumulative/dynamic time-dependent AUROC at the horizon, competing events excluded',
          '  from the control set. Implementation and validation: notebook 8.6.',
          '- Calibration: observed Aalen-Johansen cumulative incidence by predicted-risk decile; slope and intercept.',
          '- Utility: decision-curve net benefit over thresholds 0.10-0.40.',
          '- Uncertainty: patient-cluster bootstrap; paired DeLong for model-vs-baseline; Benjamini-Hochberg within a family.',
          '- Reported per held-out centre, never as a pooled number alone.', '',
          '## 5 Required baselines (every task, same table as the model)', '']
    for b in GATE['baselines_to_beat']: L.append(f'- {b}')
    L += ['- confounder-only model (centre, era, assay batch, pre-analytics, sampling day)',
          '- treatment-proxy-only model, run deliberately as a leakage canary', '',
          '## 6 Splits', '',
          '- leave-centre-out (default), temporal (mandatory for deployment claims), leave-registry-out (hardest).',
          '- Patients never span folds; donors never span folds; centres are never split.',
          '- Assignments frozen, hashed and published; the harness refuses a run whose split hash does not match.',
          '- Three-way data split: development / validation / evaluation, with evaluation labels withheld.', '',
          '## 7 Submission rules', '',
          '- At most five scored submissions per team per major version, each with a written changelog.',
          '- Every submission declares data tier, pretraining corpus, parameter count, frozen vs fine-tuned.',
          '- Fine-tuned entries report >= 5 seeds and the spread.',
          '- The mandatory ablations (random-weight twin, no-pretraining, objective removal, modality knockout,',
          '  tier ablation, corpus scaling) are submitted with the model; an entry without them cannot make a Level 2 claim.',
          '- Code and a model card are deposited; the model card states the preprocessing a deployment site must reproduce.', '',
          '## 8 Claim ladder', '']
    for lvl, txt in (('0', 'nothing'),
                     ('1', 'more label-efficient than a tuned simple model (equal at full labels, better at k <= 100, beats its random-weight twin)'),
                     ('2', 'better out-of-centre discrimination (>= +0.03 vs best simple model AND vs the clinical baseline; centre-bootstrap CI excludes 0; improves in >= 70% of held-out centres; survives BH; beats random-weight twin and no-pretraining)'),
                     ('3', 'clinically useful (Level 2 plus calibration slope 0.8-1.25, positive net benefit 10-40%, leave-centre-out AND temporal validation, n >= the Riley minimum, prospective silent-mode evaluation)'),
                     ('4', 'improves outcomes — requires a prospective trial of the model-guided decision; nothing less licenses the word')):
        L.append(f'- **Level {lvl}**: {txt}')
    L += ['', '## 9 Pre-registered gate for the v1 attempt', '', '```json', json.dumps(GATE, indent=2), '```', '',
          '## 10 Required artefacts for a published result', '',
          '- benchmark version and artefact hashes; attrition table; per-task primary metric with CI; per-centre values;',
          '- calibration slope and intercept; net benefit 10-40%; all mandatory baselines; the ablation suite;',
          '- the leakage audit output; the seed list; and the negative-results table of what did not beat its baseline.', '']
    return '\n'.join(L)

bench_path = os.path.join(SPEC_DIR, 'GVHD_BENCH_v1.0.md')
open(bench_path, 'w').write(_bench_md())
print(f'wrote {bench_path}  ({os.path.getsize(bench_path)/1024:.0f} kB, {len(open(bench_path).read().splitlines())} lines)')
print(open(bench_path).read()[:700] + '\n   [...]')
''')

code(r'''
# ----------------------------------------------------------------------------- figure manifest (every figure, its caption and its files)
figman = pd.DataFrame([dict(figure=k, slug=v['slug'],
                            files=' · '.join(os.path.basename(f) for f in v['files']),
                            caption=v['caption']) for k, v in FIGS.items()])
figman.to_csv(os.path.join(FIGDIR, 'FIGURES.csv'), index=False)
with open(os.path.join(FIGDIR, 'FIGURES.md'), 'w') as fh:
    fh.write('# Figures\n\nVector PDF and 600-dpi PNG of each figure in the paper.\n\n')
    for k, v in FIGS.items():
        fh.write(f"## {k} — {v['slug']}\n\n{v['caption']}\n\nFiles: " +
                 ', '.join(f'`{os.path.basename(f)}`' for f in v['files']) + '\n\n')
sizes = {k: sum(os.path.getsize(f) for f in v['files']) // 1024 for k, v in FIGS.items()}
print(f'{len(FIGS)} figures written to {os.path.abspath(FIGDIR)} (PDF + PNG each)')
display(md_table(figman[['figure', 'slug', 'files']]))
''')


code(r'''
# ----------------------------------------------------------------------------- export the specification bundle and verify it
manifest = {}
for f, desc in [('data_dictionary.csv', f'{len(dd)} data elements in {dd.domain.nunique()} domains'),
                ('sampling_schedule.csv', f'{len(sched)} specimen types x {len(SCHED_DAYS)} visits'),
                ('task_registry.csv', f'{len(reg_full)} benchmark tasks, full fields'),
                ('task_cards.csv', 'readiness, accessible n, requirements, powered verdict'),
                ('GVHD_BENCH_v1.0.md', 'standalone benchmark specification'),
                ('gate.json', 'pre-stated gate for the ALLO-FM v1 attempt')]:
    p = os.path.join(SPEC_DIR, f); manifest[f] = dict(description=desc, bytes=os.path.getsize(p),
        sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16])
json.dump(manifest, open(os.path.join(SPEC_DIR, 'MANIFEST.json'), 'w'), indent=2)
display(md_table(pd.DataFrame(manifest).T.reset_index().rename(columns={'index': 'file'})))

# ----------------------------------------------------------------------------- consistency check: every number quoted in the abstract/summary must match what was computed
checks = [
 ('data elements in the dictionary', len(dd), 129, 0),
 ('domains in the dictionary', dd.domain.nunique(), 25, 0),
 ('elements required for ALLO-FM v1', int((dd.priority == 'R').sum()), 74, 0),
 ('benchmark tasks', len(reg_full), 38, 0),
 ('figures written (PDF + PNG each)', len(FIGS), 11, 0),
 ('tokens per patient, Tier A+B', tok_by_tier['B'], 7580, 100),
 ('task families', reg.family.nunique(), 7, 0),
 ('Riley n, P=768, AUROC .72, prev .35', riley_n(768, .72, .35)['n'], 47662, 2000),
]
directional = [
 ('[SIM] S1: pretraining does NOT help when nothing is shared (rho=0, k=25)',
  s1.loc[(0.0, 'pretrained'), 25] - s1.loc[(0.0, 'raw'), 25], 'gain <= +0.02',
  s1.loc[(0.0, 'pretrained'), 25] - s1.loc[(0.0, 'raw'), 25] <= .02),
 ('[SIM] S1: pretraining helps at few labels when structure is shared (rho=1, k=25)',
  s1.loc[(1.0, 'pretrained'), 25] - s1.loc[(1.0, 'raw'), 25], 'gain >= +0.03',
  s1.loc[(1.0, 'pretrained'), 25] - s1.loc[(1.0, 'raw'), 25] >= .03),
 ('[SIM] S1: the advantage shrinks as labels accumulate (rho=1, k=400)',
  s1.loc[(1.0, 'pretrained'), 400] - s1.loc[(1.0, 'raw'), 400], 'gain at k=400 < gain at k=25',
  (s1.loc[(1.0, 'pretrained'), 400] - s1.loc[(1.0, 'raw'), 400]) < (s1.loc[(1.0, 'pretrained'), 25] - s1.loc[(1.0, 'raw'), 25])),
 ('[SIM] harness: the planted leak is clearly visible',
  summ.loc['LEAKING pipeline (adds steroid order d+5)', 'mean_ipcw_auroc'] - summ.loc['full (clinical + day-7 + biomarkers)', 'mean_ipcw_auroc'],
  'inflation >= +0.05', (summ.loc['LEAKING pipeline (adds steroid order d+5)', 'mean_ipcw_auroc']
                         - summ.loc['full (clinical + day-7 + biomarkers)', 'mean_ipcw_auroc']) >= .05),
 ('[SIM] harness: confounder-only model is near chance',
  summ.loc['confounder-only (centre + era)', 'mean_ipcw_auroc'], 'within 0.10 of 0.5',
  abs(summ.loc['confounder-only (centre + era)', 'mean_ipcw_auroc'] - .5) <= .10),
]
cc = pd.DataFrame([dict(quantity=q, computed=round(float(v), 3), expected=e, ok=abs(float(v) - e) <= tol) for q, v, e, tol in checks]
                  + [dict(quantity=q, computed=round(float(v), 3), expected=e, ok=bool(o)) for q, v, e, o in directional])
display(md_table(cc))
assert cc.ok.all(), cc[~cc.ok]
print(f'all {len(cc)} consistency checks passed')
print(f"MAGIC label vignettes: {len(tt)} passed | steroid/GRFS vignettes: {len(lt)} passed | metric validations: {len(val)} passed")
''')

md(r'''
---
## References

✓ = checked against a source during this project. † = from domain knowledge; verify against the primary document before submission.

**Clinical foundation models and sequence models over health records**
1. Steinberg E, Fries J, Wornow M, Shah N, et al. **MOTOR: a time-to-event foundation model for structured medical records.** ICLR 2024; arXiv:2301.03150. Pretrained on up to 55 M patient records / 9 B events; 19 tasks across 3 databases; reported +4.6 % time-dependent C and up to 95 % better label efficiency; 143 M-parameter model released. ✓
2. Steinberg E, et al. **Language models are an effective representation learning technique for electronic health record data** (CLMBR). *J Biomed Inform* 2021. †
3. Wornow M, et al. **EHRSHOT: an EHR benchmark for few-shot evaluation of foundation models.** NeurIPS Datasets & Benchmarks 2023; arXiv:2307.02028. ✓
4. Shmatko A, et al. **Delphi-2M: generative modelling of disease trajectories.** *Nature* 2025. †
5. Kraljevic Z, et al. **Foresight — generative pretrained transformer for modelling of patient timelines.** *Lancet Digit Health* 2024. †
6. Harutyunyan H, et al. **Multitask learning and benchmarking with clinical time series data.** *Sci Data* 6:96 (2019). ✓

**Audits of omics foundation models (the reason for this paper's control structure)**
7. Kedzierska KZ, Crawford L, Amini AP, Lu AX. **Zero-shot evaluation reveals limitations of single-cell foundation models.** *Genome Biol* 2025. doi:10.1186/s13059-025-03574-x ✓
8. Boiarsky R, et al. **Deeper evaluation of a single-cell foundation model.** *Nat Mach Intell* 2024. ✓
9. Ahlmann-Eltze C, Huber W, Anders S. **Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines.** *Nat Methods* 22:1657–1661 (2025). ✓
10. **Evaluating the role of pretraining dataset size and diversity on single-cell foundation model performance.** bioRxiv 2024.12.13.628448; *Nat Methods* 2026. ✓
11. Theodoris CV, et al. **Transfer learning enables predictions in network biology** (Geneformer). *Nature* 618:616–624 (2023). ✓
12. Cui H, et al. **scGPT.** *Nat Methods* 2024. ✓ · Hao M, et al. **scFoundation.** *Nat Methods* 2024. ✓ · Rosen Y, et al. **Universal Cell Embeddings.** bioRxiv 2023. ✓ · Gélard A, et al. **BulkRNABert.** PMLR 259 (ML4H 2025). ✓
13. Luecken MD, et al. **Benchmarking atlas-level data integration in single-cell genomics** (scIB). *Nat Methods* 19:41–50 (2022). ✓

**GvHD: grading, biomarkers, prophylaxis and biology**
14. Harris AC, Young R, Devine S, et al. **International, multicenter standardization of acute graft-versus-host disease clinical data collection** (MAGIC criteria). *Biol Blood Marrow Transplant* 22:4–10 (2016). ✓ (criteria exist; the staging table in §4.6 is implemented from the standard formulation †)
15. Levine JE, et al. **A prognostic score for acute GvHD based on biomarkers** (Ann Arbor). *Lancet Haematol* 2:e21–e29 (2015). ✓
16. **Day-7 two-biomarker (ST2 + REG3α) MAGIC algorithm, 1,287 patients.** *JCI Insight* 2017; article 89798. ✓
17. Bolaños-Meade J, et al. **Post-transplantation cyclophosphamide-based GvHD prophylaxis** (BMT CTN 1703; n = 431; 1-y GRFS 52.7 % vs 34.9 %). *N Engl J Med* 388:2338–2348 (2023). ✓
18. Jagasia MH, et al. **NIH consensus development project on criteria for clinical trials in chronic GvHD: I. Diagnosis and staging** (NIH 2014). *Biol Blood Marrow Transplant* 21:389–401 (2015). †
19. Schoemans HM, et al. **EBMT–NIH–CIBMTR task force position statement on standardized terminology and guidance for GvHD assessment.** *Bone Marrow Transplant* 53:1401–1415 (2018). †
20. Holtan SG, et al. **Composite end point of graft-versus-host disease-free, relapse-free survival** (GRFS). *Blood* 125:1333–1338 (2015). †
21. Sorror ML, et al. **Hematopoietic cell transplantation (HCT)-specific comorbidity index.** *Blood* 106:2912–2919 (2005). † · Armand P, et al. **Validation and refinement of the Disease Risk Index.** *Blood* 123:3664–3671 (2014). †
22. Fleischhauer K, et al. **Effect of T-cell-epitope matching at HLA-DPB1.** *Lancet Oncol* 13:366–374 (2012). † · Petersdorf EW, et al. **HLA-B leader and outcome.** *Blood* / *NEJM* series. † · Petersdorf EW, et al. **HLA-C expression level.** † 
23. Peled JU, et al. **Microbiota as predictor of mortality in allogeneic hematopoietic-cell transplantation.** *N Engl J Med* 382:822–834 (2020). 1,362 patients, 8,767 stool samples, 4 centres. ✓
24. Liao C, et al. **Compilation of longitudinal microbiota data and hospitalome from hematopoietic cell transplantation patients.** *Sci Data* 8:71 (2021). >10,000 stool samples from >1,000 patients; public. ✓
25. Zeiser R, Blazar BR. **Acute graft-versus-host disease — biologic process, prevention, and therapy.** *N Engl J Med* 377:2167–2179 (2017). † (the three-step model used in §4.3)
26. Frick M, et al. **Role of donor clonal hematopoiesis in allogeneic HCT.** *J Clin Oncol* 2019. † · Gibson CJ, et al. † 
27. Admiraal R, et al. **Population pharmacokinetic modeling of anti-thymocyte globulin.** *Lancet Haematol* 2015. †

**Registries, repositories and methods**
28. **CIBMTR Research Database.** >470,000 HSC transplant recipients from >500 centres in 48 countries (2017 NIH funding notice); 2025 annual report: 18,000+ collected and derived variables, 148 downloadable final datasets. ✓
29. **EBMT Registry** and Transplant Complications Working Party. †
30. Collins GS, et al. **TRIPOD+AI statement.** *BMJ* 385:e078378 (2024). ✓ · Wolff RF, et al. **PROBAST.** *Ann Intern Med* 2019. †
31. Riley RD, et al. **Calculating the sample size required for developing a clinical prediction model.** *BMJ* 368:m441 (2020); `pmsampsize`. ✓ (criteria 2–3 as implemented here: †)
32. Hanley JA, McNeil BJ. *Radiology* 143:29–36 (1982). † · DeLong ER, et al. *Biometrics* 44:837–845 (1988). † · Sun X, Xu W. *IEEE Signal Process Lett* 21:1389–1393 (2014). †
33. Blanche P, Dartigues J-F, Jacqmin-Gadda H. **Estimating and comparing time-dependent areas under ROC curves for censored event times with competing risks.** *Stat Med* 32:5381–5397 (2013). †
34. Vickers AJ, Elkin EB. **Decision curve analysis.** *Med Decis Making* 26:565–574 (2006). †
35. Austin PC, Fine JP. **Practical recommendations for reporting Fine-Gray model analyses for competing risk data.** *Stat Med* 2017. †
36. **AIRR Community (MiAIRR) standards** for repertoire data. † · **OMOP CDM** and the Oncology extension (OHDSI). † · **GA4GH Data Use Ontology.** †

---
## Appendix A — Reproducibility

* **Runtime.** CPU only. Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib. End-to-end run ≈ 2–5 minutes; set the environment variable `ALLOFM_QUICK=1` to cut simulation replicates (numbers get noisier and the consistency tolerances still apply).
* **Seeds.** Global seed in the setup cell; every simulation takes an explicit seed. The [SIM] cohort generator is `simulate_cohort(...)` and is deterministic given its seed.
* **Exported specification** (written to `spec/`, hashes in `spec/MANIFEST.json`): `data_dictionary.csv`, `sampling_schedule.csv`, `task_registry.csv`, `gate.json`.
* **Figure layout gate.** Every figure is measured before it is saved: the build fails if any two pieces of rendered text overlap, if text sits on a data marker, line or bar, or if any font is below 6 pt (set `ALLOFM_STRICT_FIGS=1` to make a failure stop the run; otherwise it is printed as a warning). The whole suite passes under both Liberation Sans and the wider DejaVu Sans that Colab falls back to (`ALLOFM_FONT="DejaVu Sans"`).
* **Tests executed in-notebook.** MAGIC grading vignettes; steroid-response and GRFS vignettes; fast-vs-textbook DeLong agreement; Hanley–McNeil versus simulated standard error; IPCW td-AUROC against the uncensored AUROC; IPCW recovery under censoring; Aalen–Johansen against empirical incidence; net-benefit bounds for perfect and random models; Riley criterion-1 arithmetic; the leakage audit on a cohort with a planted leak; and a final consistency check of every number quoted in the narrative.
* **What is *not* in this notebook.** Any trained foundation model, any real patient data, and any claim about clinical performance. The pilot results in §10 come from the executed notebook `GvHD_FM_benchmark_blueprint.ipynb` in this repository.

## Appendix B — How to use this specification
1. **To build a cohort:** take `spec/data_dictionary.csv` as the extraction checklist; implement `t_valid` / `t_known` for every element; implement the §4.6 label functions; run the §8.5 audit before any modelling.
2. **To run the benchmark:** take `spec/task_registry.csv`; for each task fix the landmark, horizon and competing events as given; use the §8.4 metrics; report every mandatory baseline from §8.3 in the same table.
3. **To make a claim:** check it against the ladder in §8.7 and against `spec/gate.json`, whose hash should be reported with the results.
''')
