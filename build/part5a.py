from part1 import CELLS, md, code

md(r'''
---
# 9 · How much data is needed?

This section answers the question in eight parts. Everything is computed in the notebook; every assumption is stated where it enters, and the closed-form calculators are validated against simulation before they are used.

| § | Question | Method |
|---|---|---|
| 9.1 | How precise is one AUROC? How many patients to *show* a gain? How many to *develop* a model? | Hanley–McNeil; closed-form paired DeLong power (validated by simulation); Riley et al. |
| 9.2 | What can each real data source support? | the above, against source sizes |
| 9.3 | How long would it take to accrue? | per-task accrual by route |
| 9.4 | **How many labelled transcriptomes at diagnosis?** | closed-form learning curve of a linear head, validated by simulation; effective dimension; cross-centre transfer |
| 9.5 | **What longitudinal sampling: how many draws, how often, how many specimens?** | reliability (Spearman–Brown), prodrome-window sampling, the §4.4 schedule evaluated against onset times |
| 9.6 | **How many centres?** | between-centre variance, design effect, centres needed for the gate |
| 9.7 | **How many patients to validate a claim, per subgroup, with imperfect labels?** | Fisher information for calibration; subgroup precision; label-noise attenuation |
| 9.8 | **The data bill of materials** | minimum-viable / credible / definitive packages, binding constraints, cost and time |

## 9.1 Precision, power and development size
Three different questions, three different answers.
1. **Precision** — how wide is the confidence interval of one AUROC at sample size n? (Hanley–McNeil standard error, checked against simulation.)
2. **Power for a comparison** — to claim "ALLO-FM beats MAP + clinical" we need a *paired* test. How many patients for ΔAUROC = 0.02 / 0.03 / 0.05? We use the **closed-form asymptotic DeLong variance** of the AUROC difference for two correlated scores (the toolkit above), and *validate it against a Monte-Carlo DeLong test* on simulated data before relying on it. The closed form gives a continuous answer; a simulation on a coarse grid of n can only say "the first grid point that reaches 80 %", which overstates the requirement.
3. **Development sample size** — Riley et al.'s criteria for a model with P candidate parameters.
''')

code(r'''
# ----------------------------------------------------------------------------- (1) precision of one AUROC; (2) paired power: closed form, validated against a DeLong simulation
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

SIM_REPS = 120 if not QUICK else 40
def paired_power(n, pi, A1, A2, rho, reps=SIM_REPS, seed=0):
    rng = np.random.default_rng(seed); n1 = max(int(round(n * pi)), 3); n0 = n - n1
    d1, d2 = math.sqrt(2) * norm.ppf(A1), math.sqrt(2) * norm.ppf(A2); cov = np.array([[1, rho], [rho, 1]]); hit = 0
    for _ in range(reps):
        sp = rng.multivariate_normal([d1, d2], cov, n1); sn = rng.multivariate_normal([0, 0], cov, n0)
        y = np.r_[np.ones(n1), np.zeros(n0)]
        hit += abs(delong_z(y, np.r_[sp[:, 0], sn[:, 0]], np.r_[sp[:, 1], sn[:, 1]])) > 1.96
    return hit / reps
grid = [200, 400, 800, 1600, 3200, 6400]
DELTAS, RHOS_ = (.02, .03, .05), (.6, .85)
pw_rows, sim_pts = [], {}
for d in DELTAS:
    for rho in RHOS_:
        sim_pts[(d, rho)] = {n: paired_power(n, .35, .70, .70 + d, rho) for n in grid}
        pw_rows.append(dict(delta_AUROC=d, score_corr=rho, **{f'n={n}': round(power_delta(n, .35, .70, .70 + d, rho), 2) for n in grid},
                            n_for_80pct_power=n_for_delta(.70, d, .35, rho)))
pw = pd.DataFrame(pw_rows)
# validation: closed form against the Monte-Carlo DeLong test at every (delta, rho, n)
dev = np.array([abs(power_delta(n, .35, .70, .70 + d, rho) - sim_pts[(d, rho)][n]) for d in DELTAS for rho in RHOS_ for n in grid])
MC_TOL = 3.5 * math.sqrt(.25 / SIM_REPS)
print(f'closed-form paired power vs Monte-Carlo DeLong: {len(dev)} comparisons, mean |diff| {dev.mean():.3f}, max {dev.max():.3f} '
      f'(Monte-Carlo noise at {SIM_REPS} replicates: SD <= {math.sqrt(.25 / SIM_REPS):.3f})')
assert dev.max() < MC_TOL, 'closed-form paired power disagrees with the simulation'
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
fine = np.geomspace(150, 8000, 60)
for d in DELTAS:
    for rho in RHOS_:
        ax.plot(fine, [power_delta(n, .35, .70, .70 + d, rho) for n in fine], color=DCOL[d], lw=1.3, ls='-' if rho == .85 else '--')
        ax.plot(grid, [sim_pts[(d, rho)][n] for n in grid], 'o', ms=2.6, mfc='white', mec=DCOL[d], mew=.8, ls='None')
ax.axhline(.8, color=PAL['bad'], ls=':', lw=.9)
ax.set_xscale('log'); ax.set_xticks(grid); ax.set_xticklabels([f'{g // 1000}k' if g >= 1000 else g for g in grid]); ax.minorticks_off()
ax.set_xlim(150, 8e3); ax.set_ylim(0, 1.05); ax.set_xlabel('patients (prevalence 0.35)'); ax.set_ylabel('power, paired DeLong test')
hs = [plt.Line2D([], [], color=DCOL[d], lw=1.5) for d in (.05, .03, .02)] + [plt.Line2D([], [], color=PAL['ink2'], lw=1.2), plt.Line2D([], [], color=PAL['ink2'], lw=1.2, ls='--'),
      plt.Line2D([], [], color=PAL['ink2'], marker='o', ms=3, mfc='white', ls='None')]
ax.legend(hs, ['gain 0.05', 'gain 0.03', 'gain 0.02', 'r = 0.85', 'r = 0.60', 'simulated'], loc='upper center',
          bbox_to_anchor=(.5, -.24), ncol=6, columnspacing=1.0, handlelength=1.4, labelspacing=.3, handletextpad=.4); panel(ax, 'b')
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                          # (c) patients needed to develop a model
for c_, col in zip(riley.columns, (PAL['seq'][1], PAL['seq'][2], PAL['seq'][3])):
    ax.plot(riley.index, riley[c_], marker='o', ms=2.8, color=col, lw=1.3, label=c_.replace('AUROC ', 'target AUROC '))
for P_, lab in ((20, 'registry\nmodel'), (768, 'FM embedding\nhead')):                  # labelled above the axes, centred on their own line
    ax.axvline(P_, ymin=.32, color=PAL['ink2'], lw=.6, ls=':')
    ax.annotate(lab, (P_, 1.0), xycoords=('data', 'axes fraction'), xytext=(0, 3), textcoords='offset points',
                fontsize=6.4, color=PAL['ink2'], va='bottom', ha='center')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('candidate parameters $P$'); ax.set_ylabel('patients to develop the model')
ax.set_xlim(.8, 1.2e4); ax.set_ylim(25, 6e5); ax.legend(loc='lower right', labelspacing=.3); panel(ax, 'c')
finish(fig, 11, 'sample_size_and_power',
       'Three different sample-size questions. (a) Precision: the 95% confidence interval of a single AUROC against the number of '
       'patients. (b) Power: the paired DeLong test for an improvement of 0.02 to 0.05 AUROC over a correlated baseline, the regime '
       'any foundation-model claim lives in; lines are the closed form, circles are a Monte-Carlo DeLong test; solid lines are scores '
       'correlated at 0.85, dashed at 0.60. (c) Development: the Riley minimum sample size against the number of candidate '
       'parameters, which is why head size sets the data requirement.')
display(md_table(prec.round(3), index=True)); display(md_table(pw)); display(md_table(riley, index=True))
''')

md(r'''
## 9.2 What can each real data source support?
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
cap = pd.DataFrame([dict(source=s, patients=n, **{k: ('yes' if isinstance(v, (int, float, np.integer, np.floating)) and n >= v else 'NO') for k, v in need.items()}) for s, n in SOURCES])
print('requirements (patients):', {k: (f'{v:,}' if isinstance(v, (int, float)) else v) for k, v in need.items()})
display(md_table(cap))
display(Markdown(f"""**Reading.** (1) Even a *single* AUROC is imprecise below ~1,000 patients: at n = 400 and prevalence 0.35 the 95 % CI half-width is ±{prec.loc[400, 'AUROC 0.7, prev 0.35']:.3f}. (2) Showing that a foundation model beats MAP + clinical covariates by Δ = 0.03 — the realistic size of a genuine gain between correlated models — needs about **{n80(.03, .85):,}** patients at score correlation 0.85 (and **{n80(.03, .6):,}** at 0.60, when the two models disagree more); Δ = 0.02 needs **{n80(.02, .85):,}** even at correlation 0.85 (**{n80(.02, .6):,}** at 0.60), which only a registry or a large consortium can supply. (3) Developing a 768-parameter head at AUROC 0.72 and prevalence 0.35 needs **{riley_n(768, .72, .35)['n']:,}** patients unpenalised; ridge shrinkage and pretraining are precisely the arguments for needing fewer, which is the claim the benchmark exists to test.

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

# ------------------------------------------------- Figure 12: what exists against what is needed, and how long it would take
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
finish(fig, 12, 'data_landscape_and_accrual',
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
