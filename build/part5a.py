from part1 import CELLS, md, code

md(r'''
---
# 9 · How much data is needed — at one institution?

This paper plans for **one large transplant centre**: about **200 allogeneic transplants a year**, a ceiling of **1,500 patients with an EHR timeline** (7.5 years) and **about 420 of them with prospectively banked specimens** (3 years at 70 % usable) †. Nothing the plan asks for exceeds that. The section answers, in order, what a cohort of that size can *learn*, *show* and *estimate*; what it cannot; and what to collect. Everything is computed in the notebook, every assumption is stated where it enters, and each closed-form calculator is validated against simulation before it is used.

| § | Question | Method |
|---|---|---|
| 9.1 | How precise is one AUROC? How many patients to *show* a gain? How large a *head* can n patients support? | Hanley–McNeil; closed-form paired DeLong power (validated by simulation); Riley et al. |
| 9.2 | What can each real data source support? | the above, against source sizes |
| 9.3 | How many years of accrual does each task need at 100, 200 or 300 transplants a year? | per-task requirement ÷ eligible rate |
| 9.4 | **How many labelled transcriptomes at diagnosis?** | closed-form learning curve of a linear head, validated by simulation; effective dimension |
| 9.5 | **How many draws, how often, how many specimens?** | reliability (Spearman–Brown), prodrome-window sampling, the §4.4 schedule evaluated against onset times |
| 9.6 | How many centres *if a network forms*? | between-centre variance, closed-form gate power (context, **not the ask**) |
| 9.7 | How many patients to validate, to audit a subgroup, with imperfect labels? | Fisher information for calibration; subgroup precision; label-noise attenuation |
| 9.8 | **What can one institution run?** | task-by-task feasibility at 1,500 patients; transcriptome and longitudinal claims; the single-centre gate |
| 9.9 | **The one-institution data plan** | three nested packages, ≤ 1,500 patients, specimens, storage, time |

## 9.1 Precision, power and development size
Three different questions, three different answers.
1. **Precision** — how wide is the confidence interval of one AUROC at sample size n? (Hanley–McNeil standard error, checked against simulation.)
2. **Power for a comparison** — to claim "ALLO-FM beats MAP + clinical" we need a *paired* test. How many patients for ΔAUROC = 0.02 / 0.03 / 0.05? We use the **closed-form asymptotic DeLong variance** of the AUROC difference for two correlated scores (the toolkit above), and *validate it against a Monte-Carlo DeLong test* on simulated data before relying on it.
3. **Development sample size** — Riley et al.'s criteria for a model with P candidate parameters, inverted to answer the question one centre actually has: **how many parameters can my n patients support?**
''')

code(r'''
# ----------------------------------------------------------------------------- (1) precision of one AUROC; (2) paired power: closed form, validated against a DeLong simulation; (3) head size a cohort supports
def sim_se(A, n1, n0, reps=900, seed=0):
    rng = np.random.default_rng(seed); d = math.sqrt(2) * norm.ppf(A)
    return float(np.std([roc_auc_score(np.r_[np.ones(n1), np.zeros(n0)], np.r_[rng.normal(d, 1, n1), rng.normal(0, 1, n0)]) for _ in range(reps)]))
print('Hanley–McNeil SE vs simulated SE:', [f'A={A}, n1={a}, n0={b}: {hm_se(A, a, b):.3f} vs {sim_se(A, a, b):.3f}' for A, a, b in [(.72, 35, 65), (.72, 140, 260)]])
ns = np.array([50, 100, 200, 400, 800, N_CAP])
prec = pd.DataFrame({f'AUROC {A}, prev {pi}': [1.96 * hm_se(A, max(int(n * pi), 2), max(int(n * (1 - pi)), 2)) for n in ns]
                     for A in (.70, .75) for pi in (.15, .35)}, index=ns); prec.index.name = 'patients'
capn = lambda v: f'{v:,}' if v <= N_CAP else f'> {N_CAP:,}'              # nothing in this paper's plan exceeds the ceiling cohort

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
grid = [100, 200, 400, 800, N_CAP]
DELTAS, RHOS_ = (.02, .03, .05), (.6, .85)
pw_rows, sim_pts = [], {}
for d in DELTAS:
    for rho in RHOS_:
        sim_pts[(d, rho)] = {n: paired_power(n, .35, .70, .70 + d, rho) for n in grid}
        pw_rows.append(dict(delta_AUROC=d, score_corr=rho, **{f'n={n}': round(power_delta(n, .35, .70, .70 + d, rho), 2) for n in grid}, n_for_80pct_power=n_for_delta(.70, d, .35, rho)))
pw = pd.DataFrame(pw_rows)
dev = np.array([abs(power_delta(n, .35, .70, .70 + d, rho) - sim_pts[(d, rho)][n]) for d in DELTAS for rho in RHOS_ for n in grid])
MC_TOL = 3.5 * math.sqrt(.25 / SIM_REPS)
print(f'closed-form paired power vs Monte-Carlo DeLong: {len(dev)} comparisons, mean |diff| {dev.mean():.3f}, max {dev.max():.3f} (Monte-Carlo noise at {SIM_REPS} replicates: SD <= {math.sqrt(.25 / SIM_REPS):.3f})')
assert dev.max() < MC_TOL, 'closed-form paired power disagrees with the simulation'
pw_show = pw.copy(); pw_show['n_for_80pct_power'] = pw_show['n_for_80pct_power'].map(capn)
PARAMS = [1, 3, 5, 10, 15, 20, 30, 50]
riley = pd.DataFrame([dict(params=P, **{f'AUROC {A}': riley_n(P, A, .35)['n'] for A in (.68, .72, .75)}) for P in PARAMS]).set_index('params')
maxP = pd.DataFrame([dict(patients=n, **{f'AUROC {A}': max_params(n, A, .35) for A in (.68, .72, .75)}) for n in (150, 420, 900, 1200, N_CAP)]).set_index('patients')
print('LARGEST HEAD (candidate parameters) that n patients can support — Riley criteria inverted, prevalence 0.35 (Table 9.1):'); display(md_table(maxP, index=True))
fig = plt.figure(figsize=(FULL, 3.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1], left=.075, right=.985, bottom=.255, top=.875, wspace=.38)
ax = fig.add_subplot(gs[0, 0]); ygrid(ax)                                          # (a) precision of one AUROC
for c_, col in zip(prec.columns, (PAL['seq'][3], PAL['seq'][1], PAL['orange'], PAL['grey'])):
    ax.plot(prec.index, prec[c_], marker='o', ms=2.8, color=col, lw=1.3, label=c_.replace('prev ', '$\\pi$ = '))
ax.axhline(.05, color=PAL['bad'], ls=':', lw=.9)
ax.annotate('±0.05', (prec.index[0] * 1.04, .05), xytext=(0, 3), textcoords='offset points', fontsize=6.4, color=PAL['bad'], va='bottom')
ax.set_xscale('log'); ax.set_xlabel('patients'); ax.set_ylabel('95% CI half-width of one AUROC'); ax.set_xlim(40, 2200)
ax.set_xticks([50, 100, 200, 500, 1500]); ax.set_xticklabels(['50', '100', '200', '500', '1,500']); ax.minorticks_off()
ax.legend(loc='upper right', labelspacing=.3); panel(ax, 'a')
ax = fig.add_subplot(gs[0, 1]); ygrid(ax)                                          # (b) power to show a gain over a correlated baseline, up to the ceiling cohort
DCOL = {.02: PAL['seq'][0], .03: PAL['seq'][2], .05: PAL['violet']}
fine = np.geomspace(60, N_CAP, 60)
for d in DELTAS:
    for rho in RHOS_:
        ax.plot(fine, [power_delta(n, .35, .70, .70 + d, rho) for n in fine], color=DCOL[d], lw=1.3, ls='-' if rho == .85 else '--')
        ax.plot(grid, [sim_pts[(d, rho)][n] for n in grid], 'o', ms=2.6, mfc='white', mec=DCOL[d], mew=.8, ls='None')
ax.axhline(.8, color=PAL['bad'], ls=':', lw=.9)
ax.set_xscale('log'); ax.set_xticks(grid); ax.set_xticklabels([f'{g:,}' for g in grid]); ax.minorticks_off()
ax.set_xlim(70, 1700); ax.set_ylim(0, 1.05); ax.set_xlabel('patients (prevalence 0.35)'); ax.set_ylabel('power, paired DeLong test')
hs = [plt.Line2D([], [], color=DCOL[d], lw=1.5) for d in (.05, .03, .02)] + [plt.Line2D([], [], color=PAL['ink2'], lw=1.2), plt.Line2D([], [], color=PAL['ink2'], lw=1.2, ls='--'),
      plt.Line2D([], [], color=PAL['ink2'], marker='o', ms=3, mfc='white', ls='None')]
ax.legend(hs, ['gain 0.05', 'gain 0.03', 'gain 0.02', 'r = 0.85', 'r = 0.60', 'simulated'], loc='upper center',
          bbox_to_anchor=(.5, -.24), ncol=3, columnspacing=1.0, handlelength=1.4, labelspacing=.3, handletextpad=.4); panel(ax, 'b')
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                          # (c) patients needed to develop a model, against the ceiling cohort
for c_, col in zip(riley.columns, (PAL['seq'][1], PAL['seq'][2], PAL['seq'][3])):
    ax.plot(riley.index, riley[c_], marker='o', ms=2.8, color=col, lw=1.3, label=c_.replace('AUROC ', 'target AUROC '))
ax.axhline(N_CAP, color=PAL['bad'], ls=':', lw=.9, label=f'ceiling cohort ({N_CAP:,})')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([1, 3, 10, 30, 50]); ax.set_xticklabels(['1', '3', '10', '30', '50']); ax.minorticks_off()
ax.set_yticks([300, 1000, 3000, 6000]); ax.set_yticklabels(['300', '1,000', '3,000', '6,000'])
ax.set_xlabel('candidate parameters $P$ of the head'); ax.set_ylabel('patients to develop the model'); ax.set_xlim(.8, 70); ax.set_ylim(200, 8000)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.24), ncol=2, columnspacing=1.0, handlelength=1.4, labelspacing=.3, handletextpad=.4); panel(ax, 'c')
finish(fig, 11, 'sample_size_and_power',
       'What one institution\'s cohort can resolve. (a) Precision: the 95% confidence interval of a single AUROC against the number of patients; at the ceiling cohort of 1,500 it is about '
       '±0.03. (b) Power: the paired DeLong test for an improvement of 0.02 to 0.05 AUROC over a correlated baseline, up to the ceiling cohort; lines are the closed form, circles are a '
       'Monte-Carlo DeLong test; solid lines are scores correlated at 0.85, dashed at 0.60. A gain of 0.03 is within reach at score correlation 0.85; 0.02 is not. (c) Development: the Riley '
       'minimum sample size against the number of candidate parameters of the head, with the ceiling cohort dotted: it supports a head of roughly 15 to 25 parameters, which is why this plan '
       'adapts a model and fits a small head rather than training a large one.')
display(md_table(prec.round(3), index=True)); display(md_table(pw_show)); display(md_table(riley, index=True))
''')

md(r'''
## 9.2 What can each real data source support?
''')

code(r'''
# ----------------------------------------------------------------------------- what each data source can actually support
n80 = lambda d, rho: pw[(pw.delta_AUROC == d) & (pw.score_corr == rho)].n_for_80pct_power.iloc[0]
SOURCES = [(f'One large centre, 7.5 years (the plan; Tier A+B)', N_CAP), (f'  of whom with banked specimens (the plan; Tier A+C)', N_SPEC),
           ('BMT CTN 1703 (randomised, Tier A+C)', 431), ('MAGIC day-7 biomarker cohort (Tier A+C)', 1287),
           ('MSK hospitalome (public, Tier A+B+microbiome)', 1000), ('Typical public GEO GvHD series (Tier D)', 40),
           ('CIBMTR extract, recent era (Tier A; needs a data-use agreement)', 25000), ('CIBMTR full (Tier A; context)', 470000)]
need = {'head P = 5': riley_n(5, .72, .35)['n'], 'head P = 10': riley_n(10, .72, .35)['n'], 'head P = 25': riley_n(25, .72, .35)['n'],
        'show +0.05 (corr .85)': n80(.05, .85), 'show +0.03 (corr .85)': n80(.03, .85), 'show +0.02 (corr .85)': n80(.02, .85)}
cap = pd.DataFrame([dict(source=s, patients=f'{n:,}', **{k: ('yes' if n >= v else 'NO') for k, v in need.items()}) for s, n in SOURCES])
print('requirements (patients):', {k: f'{v:,}' for k, v in need.items()})
display(md_table(cap))
display(Markdown(f"""**Reading.** (1) Even a *single* AUROC is imprecise below ~1,000 patients: at n = 400 and prevalence 0.35 the 95 % CI half-width is ±{prec.loc[400, 'AUROC 0.7, prev 0.35']:.3f}, at the ceiling cohort ±{prec.loc[N_CAP, 'AUROC 0.7, prev 0.35']:.3f}. (2) Showing that a model beats MAP plus clinical covariates by Δ = 0.03 needs about **{n80(.03, .85):,}** patients at score correlation 0.85 — **inside the ceiling cohort** — and by Δ = 0.05 about **{n80(.05, .85):,}**, which even the {N_SPEC} patients with banked specimens nearly reach; Δ = 0.02 needs **{n80(.02, .85):,}**, beyond one centre. If the two models disagree more (correlation 0.60) the same gains need {capn(n80(.03, .6))} and {capn(n80(.05, .6))}. (3) What limits one centre is the **size of the head**: a head with 10 parameters at AUROC 0.72 and prevalence 0.35 needs **{riley_n(10, .72, .35)['n']:,}** patients, 25 parameters {riley_n(25, .72, .35)['n']:,}, and 50 parameters {capn(riley_n(50, .72, .35)['n'])}. A head of several hundred parameters (a raw embedding) is therefore not on the table, and shrinkage, frozen pretrained weights and pre-specified scores are precisely the way to stay inside it.

**The conclusion that drives the plan.** One large centre can learn a head of roughly 10–25 parameters from its 1,500 EHR timelines, can show a gain of 0.03 or more over a correlated baseline, and can run a head of about 5 parameters on its {N_SPEC} patients with specimens. It cannot show 0.02, cannot fit a high-dimensional head, and cannot audit subgroups (§9.7). **The model must therefore be small or adapted, and every claim must be a claim about this centre** — which is exactly what Level 1C of the ladder (§8.10) says."""))
''')

md(r"""
## 9.3 How long would one institution need? Years of accrual per task

Required n becomes a plan only when divided by a rate. For every task with a numeric target the cell below takes the requirement of a **one-institution design** — a head of 10 parameters (Riley) *and* enough patients to show a gain of +0.05 over the clinical baseline (paired DeLong) — and divides it by the rate at which a centre accrues eligible patients: transplants a year × the fraction eligible for the task (§8.4) × 70 % usable if the task needs prospectively banked specimens. The answer is the number of years of EHR history (Tier A/B tasks) or of prospective collection (Tier C/D tasks) a centre of 100, 200 or 300 transplants a year needs. This converts "we need more data" into a plan a department can read.
""")

code(r'''
# ----------------------------------------------------------------------------- one-institution accrual: years to reach the n each task needs
VOLS = (100, 200, 300)
si_tasks = []
for _, r in reg_full[reg_full.id.isin(ELIG)].iterrows():
    A_, pv_ = r['target AUROC'], r['assumed prevalence']; prosp = CARDS[r.id][0] != 'R1'
    need_si = max(riley_n(10, A_, pv_)['n'], n_for_delta(A_ - .05, .05, pv_))
    si_tasks.append(dict(id=r.id, need=need_si, prospective=prosp, **{f'{v}/yr': need_si / (v * ELIG[r.id] * (USABLE_FRACTION if prosp else 1.0)) for v in VOLS}))
si_acc = pd.DataFrame(si_tasks).set_index('id')
fmt_y = lambda y: f'{y:.1f} y' if y <= 10 else '> 10 y'
show_acc = si_acc.copy()
for v in VOLS: show_acc[f'{v}/yr'] = show_acc[f'{v}/yr'].map(fmt_y)
show_acc['need'] = show_acc['need'].map(lambda v: f'{v:,}'); show_acc['source'] = ['prospective specimens' if p_ else 'EHR history' for p_ in si_acc.prospective]
print('YEARS OF ACCRUAL a centre needs per task (10-parameter head and +0.05 over the clinical baseline), by annual transplant volume:')
display(md_table(show_acc.drop(columns='prospective').reset_index()))

# ------------------------------------------------- Figure 12: how large a head does accrual buy, and how long does each task take
FWIN, FHIN = FULL, 4.6
fig = plt.figure(figsize=(FWIN, FHIN))
axa = fig.add_axes([.95 / FWIN, .95 / FHIN, 2.60 / FWIN, 3.15 / FHIN])
axb = fig.add_axes([4.15 / FWIN, .95 / FHIN, 2.95 / FWIN, 3.15 / FHIN])
ax = axa; ygrid(ax)                                                               # (a) head size supported vs years of accrual
yrs = np.linspace(.5, 10, 40)
for v, c in zip(VOLS, (PAL['seq'][0], PAL['seq'][2], PAL['seq'][3])):
    ax.plot(yrs, [max_params(int(v * t), .72, .35) for t in yrs], color=c, lw=1.6, label=f'{v} a year, EHR timelines')
    ax.plot(yrs, [max_params(int(v * t * USABLE_FRACTION), .72, .35) for t in yrs], color=c, lw=1.3, ls='--')
ax.axhline(10, color=PAL['ink2'], lw=.7, ls=':'); ax.axvline(EHR_YEARS, color=PAL['ink2'], lw=.7, ls=':')
ax.set_xlabel('years of accrual'); ax.set_ylabel('largest head the cohort supports (parameters)'); ax.set_xlim(0, 10); ax.set_ylim(0, 60)
hs_ = [plt.Line2D([], [], color=c, lw=1.5) for c in (PAL['seq'][0], PAL['seq'][2], PAL['seq'][3])] + [plt.Line2D([], [], color=PAL['ink2'], lw=1.4), plt.Line2D([], [], color=PAL['ink2'], lw=1.2, ls='--')]
ax.legend(hs_, ['100 transplants a year', '200 a year', '300 a year', 'EHR timelines', 'banked specimens (70% usable)'], loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, columnspacing=1.0, handlelength=1.4, labelspacing=.3, handletextpad=.4)
ax.annotate('a', (0, 1), xycoords='axes fraction', xytext=(-.75 * 72, 32), textcoords='offset points', fontsize=9, fontweight='bold')
ROUTE_NAMES = ['100 a\nyear', '200 a\nyear', '300 a\nyear']
tbl = si_acc.sort_values('200/yr'); nr, nc = len(tbl), 4
def cls(v): return 0 if v <= 3 else 1 if v <= EHR_YEARS else 2
def txt(v): return fmt_y(v).replace(' y', ' y')
ax = axb
CELLC = [PAL['aqua'], PAL['seq'][0], PAL['orange']]; CELLA = [.85, .85, .55]
for i, (tid, row) in enumerate(tbl.iterrows()):
    ax.text(0.5, i, f'{int(row.need):,}', ha='center', va='center', fontsize=6.4, color=PAL['ink'])
    for j, vcol in enumerate(('100/yr', '200/yr', '300/yr')):
        v = row[vcol]; ax.add_patch(plt.Rectangle((j + 1.03, i - .46), .94, .92, color=CELLC[cls(v)], alpha=CELLA[cls(v)], lw=0))
        _t = ax.text(j + 1.5, i, txt(v), ha='center', va='center', fontsize=6.3, color=PAL['ink']); _t._ok_overlap = True
ax.set_xlim(0, nc); ax.set_ylim(nr - .5, -.5); ax.set_yticks(range(nr)); ax.set_yticklabels(tbl.index); ax.tick_params(axis='y', length=0)
ax.xaxis.tick_top(); ax.set_xticks(np.arange(nc) + .5); ax.set_xticklabels(['patients\nneeded'] + ROUTE_NAMES); ax.tick_params(axis='x', length=0)
for sp in ax.spines.values(): sp.set_visible(False)
ax.annotate('b', (0, 1), xycoords='axes fraction', xytext=(-.45 * 72, 32), textcoords='offset points', fontsize=9, fontweight='bold')
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c, alpha=a) for c, a in zip(CELLC, CELLA)], ['within 3 years', f'within {EHR_YEARS:g} years (the ceiling cohort at 200 a year)', 'longer'],
           ncol=1, loc='lower right', bbox_to_anchor=(.99, .0), columnspacing=1.2, handlelength=1.2, labelspacing=.4)
finish(fig, 12, 'one_institution_accrual',
       'What a single institution can reach, and how fast. (a) The largest head the cohort supports (Riley criteria, AUROC 0.72, prevalence 0.35) against years of accrual, at 100, 200 and 300 '
       'transplants a year, for EHR timelines (solid) and for prospectively banked specimens at 70% usable (dashed). A head of 10 parameters is reached within about 3 years of EHR history '
       'at 200 a year; specimens take longer. (b) Years of accrual each task needs for a head of 10 parameters and a +0.05 gain over its clinical baseline, by annual volume: green within '
       'three years, blue within the 7.5-year ceiling cohort, orange longer.')
''')

code(r'''
r3 = (si_acc['200/yr'] <= 3).sum(); r75 = (si_acc['200/yr'] <= EHR_YEARS).sum(); rl = si_acc.index[si_acc['200/yr'] > EHR_YEARS].tolist()
display(Markdown(f"""**Result.** Held to the *full* requirement — a head of 10 parameters **and** a +0.05 gain over the clinical baseline — **{r3} of the {len(si_acc)} numeric-target tasks** are reached within three years at 200 transplants a year and **{r75}** within the 7.5-year ceiling cohort (at 100 a year, {(si_acc['100/yr'] <= EHR_YEARS).sum()}; at 300 a year, {(si_acc['300/yr'] <= EHR_YEARS).sum()}). The tasks outside it are {', '.join(rl) if rl else 'none'}. This is the strict reading. Several of those tasks can still be *run* with a smaller head or a looser claim, and §9.8 applies that weaker rule (a head of at least 5 parameters and a detectable gain of at most 0.05) to say what the cohort actually supports. **The strategic ordering falls out of the arithmetic.** Start with the tasks that need only the EHR timeline (A, B3, B6, C3, D3, D4, E): the data exist and the only work is extraction. Start the prospective specimen collection on day one, because its clock is the longest (§9.9). Treat what does not fit as a reason to join a network later, not a reason to stop."""))
''')
