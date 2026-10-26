from part1 import CELLS, md, code

md(r"""
## 8.12 [SIM] Does the benchmark itself work? Its own operating characteristics

A benchmark makes claims about models, and it should be validated with the rigour it demands of them. Four questions a sceptical reviewer will ask — each answered below by simulation or by running the real harness code of §8.11 on cohorts where the truth is planted:

1. **Coverage.** Does the interval the gate relies on contain the truth 95 % of the time? (A percentile bootstrap over centres is the common choice. With 8–20 centres it is not safe.)
2. **False passes and power.** What is the probability that a model with *no* gain, or a trivial gain, passes the gate, and that a model with the designed-for gain does?
3. **Recovery.** If we plant an effect of known size, does the harness recover it, and does it report nothing when the planted effect is zero? Does a permuted outcome produce a null result?
4. **Shortcuts that look like signal.** Can a model score above chance by exploiting *whether* a test was ordered rather than what it showed?

**Setup for the gate simulation.** Each of *K* centres contributes a paired AUROC difference between candidate and baseline. That difference has three sources of variation: the true gain Δ, a **between-centre SD τ** of the gain (centres differ in case mix and practice; τ = 0.02 is our planning value and §9.6 varies it), and the **sampling noise of a paired AUROC difference at that centre's sample size** (computed from the closed form of §9.1 at baseline AUROC 0.70, score correlation 0.85, prevalence 0.35; centre sizes vary log-normally around a median of 300). We compare two gates. The **obvious** one: point estimate ≥ +0.03, percentile-bootstrap lower bound > 0, ≥ 70 % of centres improved. The **adopted** one (`spec/gate.json`): centre-level *t*-interval lower bound ≥ +0.01, ≥ 70 % of centres improved, at least 20 centres.
""")

code(r'''
# ----------------------------------------------------------------------------- [SIM] operating characteristics of the benchmark's own gate
def gate_oc(K, nc, d_true, tau=.02, reps=2000, B=300, seed=0, prev=.35, A1=.70, rho=.85, nc_cv=.5):
    """Simulate `reps` benchmark runs with K centres; return the interval coverage and the pass probability of both gates."""
    r = np.random.default_rng(seed); g = GATE['level2_gate']
    se_unit = se_delta(1, prev, A1, A1 + d_true, rho)                       # SE of a paired AUROC difference at n = 1 (scales as 1/sqrt(n))
    n_c = np.maximum((nc * np.exp(r.normal(0, nc_cv, (reps, K)) - nc_cv ** 2 / 2)).astype(int), 60)
    dh = d_true + r.normal(0, tau, (reps, K)) + r.standard_normal((reps, K)) * se_unit / np.sqrt(n_c)    # observed per-centre gains
    m = dh.mean(1); s = dh.std(1, ddof=1) / math.sqrt(K); tq = student_t.ppf(.975, K - 1)
    lo_t, hi_t = m - tq * s, m + tq * s; frac = (dh > 0).mean(1)
    lo_b = np.empty(reps); hi_b = np.empty(reps)
    for a in range(0, reps, 200):                                           # percentile bootstrap over centres, in chunks to bound memory
        sl = slice(a, min(a + 200, reps)); idx = r.integers(0, K, (sl.stop - sl.start, B, K))
        bm = np.take_along_axis(dh[sl][:, None, :], idx, axis=2).mean(2); lo_b[sl], hi_b[sl] = np.quantile(bm, [.025, .975], axis=1)
    return dict(K=K, nc=nc, tau=tau, d_true=d_true,
                cov_t=float(((lo_t <= d_true) & (d_true <= hi_t)).mean()), cov_boot=float(((lo_b <= d_true) & (d_true <= hi_b)).mean()),
                pass_obvious=float(((m >= .03) & (lo_b > 0) & (frac >= .70)).mean()),
                pass_adopted=float(((lo_t >= g['ci_lower_bound_min']) & (frac >= g['fraction_centres_improved_min'])).mean()))
OC_REPS = 2000 if not QUICK else 600
KS_OC, DS_OC = (8, 12, 20, 30, 50), (0., .01, .02, .03, .04, .05, .06)
oc = pd.DataFrame([gate_oc(K, 300, d, reps=OC_REPS, seed=11 * K + int(1000 * d)) for K in KS_OC for d in DS_OC])
cov = oc[oc.d_true == .03].set_index('K')[['cov_boot', 'cov_t']].round(3)
cov.columns = ['percentile bootstrap over centres', 'centre-level t-interval']
print('Coverage of the nominal 95 % interval for the true gain (Δ = 0.03, τ = 0.02, median 300 patients per centre):'); display(md_table(cov, index=True))
tab = lambda col: oc.pivot(index='K', columns='d_true', values=col).round(2).rename(columns=lambda d: f'Δ={d:g}')
print('Probability of passing the OBVIOUS gate (point estimate ≥ 0.03, bootstrap lower bound > 0, ≥ 70 % of centres improved):'); display(md_table(tab('pass_obvious'), index=True))
print('Probability of passing the ADOPTED gate (t-interval lower bound ≥ 0.01, ≥ 70 % of centres improved; evaluable only for K ≥ 20):'); display(md_table(tab('pass_adopted'), index=True))
# sensitivity of the adopted gate's power to how heterogeneous centres are and how large they are
sens_oc = pd.DataFrame([dict(tau=tau, **{f'median n per centre = {nc}': gate_oc(20, nc, .03, tau=tau, reps=OC_REPS // 2, seed=5)['pass_adopted'] for nc in (150, 300, 600)}) for tau in (.01, .02, .04)]).round(2)
print('Power of the adopted gate at K = 20 and true gain 0.03, against between-centre SD τ and centre size:'); display(md_table(sens_oc))
g20 = lambda d, col: float(oc[(oc.K == 20) & (oc.d_true == d)][col].iloc[0])
assert g20(.01, 'pass_adopted') <= .06 and g20(.03, 'pass_adopted') >= .78, 'the adopted gate does not meet its own design targets'
assert .35 <= g20(.03, 'pass_obvious') <= .65, 'the obvious gate was expected to pass a threshold-sized gain about half the time'
assert oc.cov_t.between(.92, .975).all(), 't-interval coverage is off nominal'
''')

code(r'''
cb = lambda K, col: float(oc[(oc.K == K) & (oc.d_true == .03)][col].iloc[0]); pw_ = lambda K, d: float(oc[(oc.K == K) & (oc.d_true == d)].pass_adopted.iloc[0])
display(Markdown(f"""**What the simulation says about the gate.**

* **A percentile bootstrap over centres under-covers**, badly when centres are few: its nominal 95 % interval contains the true gain {cb(8, 'cov_boot'):.0%} of the time at K = 8, {cb(20, 'cov_boot'):.0%} at K = 20 and still only {cb(50, 'cov_boot'):.0%} at K = 50. Its interval is too narrow, so it declares gains that are not there. The centre-level *t*-interval, which spends the degrees of freedom the data actually have, stays within a point or two of nominal at every K ({cb(8, 'cov_t'):.1%}–{cb(50, 'cov_t'):.1%}). The adopted gate uses it.
* **The obvious gate is a coin flip at its own threshold.** Requiring a point estimate of at least +0.03 means a model whose true gain is exactly +0.03 passes about half the time at any number of centres ({g20(.03, 'pass_obvious'):.0%} at K = 20) — the gate has ~50 % power by construction. The adopted gate asks for a *lower bound* ≥ +0.01 instead, so it is powered for a true gain of +0.03 ({pw_(20, .03):.0%} at K = 20) and nearly blind to a true gain of +0.01 or less (false-pass {pw_(20, .01):.0%}; {pw_(20, .0):.0%} at zero gain).
* **Twenty centres is where the adopted gate becomes usable**, and it is a derived number, not a custom: at τ = 0.02 and centres of ~300 patients power for a +0.03 gain is {pw_(12, .03):.0%} at K = 12, {pw_(20, .03):.0%} at K = 20 and {pw_(30, .03):.0%} at K = 30. If centres are more heterogeneous the same design needs more centres: at τ = 0.04 and K = 20 the power is only {sens_oc.set_index('tau').loc[.04, 'median n per centre = 300']:.0%}, which is why §9.6 reports the required K against τ.
* **A gain of +0.02 is not reliably detectable at 20 centres** ({pw_(20, .02):.0%}) and still only {pw_(50, .02):.0%} at 50. The benchmark is designed to establish gains of +0.03 or more, and says so."""))
''')

code(r'''
# ----------------------------------------------------------------------------- [SIM] recovery of planted effects and a permuted-outcome control, through the real harness code
def lco_ipcw(df, cols, held_, y_fn=y_b1):
    """Leave-centre-out IPCW td-AUROC at day 100 for one input set: exactly the procedure of the §8.11 harness."""
    X = prep(df, cols); y = y_fn(df); out = {}
    for c in held_:
        tr, te = (df.centre != c).values, (df.centre == c).values
        if y[tr].sum() < 10 or te.sum() < 40: continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000, class_weight='balanced')).fit(X[tr], y[tr])
        out[c] = ipcw_td_auroc(df.time.values[te], df.cause.values[te], m.decision_function(X[te]), 100)
    return pd.Series(out)

FULL_COLS = ARMS['full (clinical + day-7 + biomarkers)']
base_full = piv['full (clinical + day-7 + biomarkers)']
ladder = []
for am in (.50, .60, .70, .80, .90):
    rr = np.random.default_rng(int(am * 1000))
    coh2 = COH.assign(pc=rr.normal(0, 1, len(COH)) + math.sqrt(2) * norm.ppf(am) * COH['_latent_event'].values)    # planted marker with known univariate AUROC
    d = (lco_ipcw(coh2, FULL_COLS + ['pc'], held) - base_full).dropna().values
    m_, lo_, hi_ = centre_interval(d)
    ladder.append(dict(planted_marker_AUROC=am, recovered_delta=m_, lo=lo_, hi=hi_, centres_improved=f'{int((d > 0).sum())}/{len(d)}',
                       level2_gate=level2_verdict(lo_, float((d > 0).mean()), len(d))))
ladder = pd.DataFrame(ladder)
# negative control: outcomes permuted within centre (features and outcomes decoupled); every arm must fall to chance and no gain may appear
perm = COH.copy(); rp = np.random.default_rng(7)
for c in centres:
    idx = np.where(COH.centre.values == c)[0]; sh = rp.permutation(idx)
    perm.loc[idx, ['time', 'cause']] = COH.loc[sh, ['time', 'cause']].values
perm_clin, perm_full = lco_ipcw(perm, CLIN, held), lco_ipcw(perm, FULL_COLS, held)
dperm = (perm_full - perm_clin).dropna().values; mp_, lop_, hip_ = centre_interval(dperm)
negctl = dict(clinical_auroc=float(perm_clin.mean()), full_auroc=float(perm_full.mean()), delta=mp_, lo=lop_, hi=hip_,
              gate=level2_verdict(lop_, float((dperm > 0).mean()), len(dperm)))
print('PLANTED-EFFECT LADDER (a marker with known univariate AUROC added to the full model; recovered gain over the full model):')
display(md_table(ladder.round(3)))
print(f"NEGATIVE CONTROL (outcomes permuted within centre): clinical AUROC {negctl['clinical_auroc']:.3f}, full AUROC {negctl['full_auroc']:.3f}, "
      f"Δ {negctl['delta']:+.3f} [{negctl['lo']:+.3f}, {negctl['hi']:+.3f}], gate: {negctl['gate']}")
assert abs(negctl['clinical_auroc'] - .5) < .03 and abs(negctl['full_auroc'] - .5) < .03, 'a permuted outcome must give chance-level AUROC'
assert not negctl['gate'].startswith('PASS'), 'a permuted outcome must never pass the gate'
from scipy.stats import spearmanr
assert spearmanr(ladder.planted_marker_AUROC, ladder.recovered_delta)[0] >= .9 and ladder.recovered_delta.iloc[-1] > ladder.recovered_delta.iloc[0] + .05, 'recovered gain should rise with the planted effect'
assert ladder.lo.iloc[0] <= 0 <= ladder.hi.iloc[0], 'a planted null must be covered by its interval'
''')

code(r'''
# ----------------------------------------------------------------------------- [SIM] informative presence: can a model score by what was ordered rather than what it showed?
def presence_audit(df, held_, thresh=.02):
    """Presence-only comparator: indicator that the biomarker panel was ordered, no values. WARN if it beats chance by > thresh."""
    a = float(lco_ipcw(df, ['biomarker_measured'], held_).mean()); return a, ('WARN' if a - .5 > thresh else 'pass')
GAMMAS = (0., .5, 1., 1.5, 2.) if not QUICK else (0., 1., 2.)
prow = []
for gm in GAMMAS:
    cg = simulate_cohort(presence_gamma=gm); cg = cg[cg.time > LANDMARK].reset_index(drop=True)
    a_pres, flag = presence_audit(cg, held)
    prow.append(dict(ordering_bias_gamma=gm, presence_only_auroc=a_pres, audit=flag,
                     full_model=float(lco_ipcw(cg, FULL_COLS, held).mean()),
                     full_without_presence_feature=float(lco_ipcw(cg, [c for c in FULL_COLS if c != 'biomarker_measured'], held).mean())))
pres = pd.DataFrame(prow)
print('INFORMATIVE-PRESENCE TEST (leave-centre-out IPCW td-AUROC; γ = how strongly clinicians order the biomarker panel when they suspect GvHD):')
display(md_table(pres.round(3)))
assert abs(pres.presence_only_auroc.iloc[0] - .5) < .02 and pres.presence_only_auroc.iloc[-1] > pres.presence_only_auroc.iloc[0] + .02, 'presence-only AUROC should track the ordering bias'
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 9: the benchmark's own operating characteristics
FWIN, FHIN = FULL, 6.0
fig = plt.figure(figsize=(FWIN, FHIN))
def A(l, b, w, h): return fig.add_axes([l / FWIN, b / FHIN, w / FWIN, h / FHIN])
axa, axb = A(.95, 3.75, 2.50, 1.75), A(4.45, 3.75, 2.55, 1.75)
axc, axd = A(.95, .80, 2.50, 1.75), A(4.45, .80, 2.55, 1.75)
CB, CT, CO = PAL['seq'][2], PAL['orange'], PAL['seq'][3]
ax = axa; ygrid(ax)                                                      # (a) coverage of the interval
ax.plot(cov.index, cov['centre-level t-interval'], 'o-', color=CB, ms=3.4, lw=1.5, label='centre-level t-interval')
ax.plot(cov.index, cov['percentile bootstrap over centres'], 's-', color=CT, ms=3.2, lw=1.5, label='percentile bootstrap over centres')
ax.axhline(.95, color=PAL['ink2'], ls=':', lw=.8, label='nominal 0.95')
ax.set_xscale('log'); ax.set_xticks(list(cov.index)); ax.set_xticklabels(list(cov.index)); ax.minorticks_off(); ax.set_ylim(.83, .985)
ax.set_xlabel('held-out centres $K$'); ax.set_ylabel('coverage of the true gain'); panel(ax, 'a')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.30), ncol=3, columnspacing=1.0, handlelength=1.4, handletextpad=.4)
ax = axb; ygrid(ax)                                                      # (b) probability of passing the gate
pj = lambda K, col: [float(oc[(oc.K == K) & (oc.d_true == d)][col].iloc[0]) for d in DS_OC]
ax.plot(DS_OC, pj(20, 'pass_obvious'), 's-', color=CT, ms=3.2, lw=1.5, label='obvious gate, K = 20')
for K_, c_, ls_ in ((12, PAL['seq'][0], '--'), (20, CB, '-'), (30, CO, ':')):
    ax.plot(DS_OC, pj(K_, 'pass_adopted'), 'o', ls=ls_, color=c_, ms=3.2, lw=1.5, label=f'adopted gate, K = {K_}')
ax.axvline(.03, color=PAL['ink2'], lw=.6, ls=':'); ax.axhline(.8, color=PAL['bad'], lw=.7, ls=':'); ax.axhline(.05, color=PAL['bad'], lw=.7, ls=':')
ax.set_ylim(-.03, 1.05); ax.set_xlabel('true gain in AUROC'); ax.set_ylabel('probability of passing'); panel(ax, 'b')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.30), ncol=2, columnspacing=1.0, handlelength=1.4, handletextpad=.4, labelspacing=.3)
ax = axc; ygrid(ax)                                                      # (c) planted effects recovered through the real harness
lc = {'PASS': PAL['aqua'], 'fail': PAL['grey'], 'not evaluable': PAL['grey']}
for _, r_ in ladder.iterrows():
    col = lc['PASS'] if r_.level2_gate == 'PASS' else lc['fail']
    ax.plot([r_.planted_marker_AUROC] * 2, [r_.lo, r_.hi], color=col, lw=1.6, solid_capstyle='round'); ax.plot(r_.planted_marker_AUROC, r_.recovered_delta, 'o', color=col, ms=4.2, zorder=4)
ax.axhline(0, color=PAL['ink'], lw=.7); ax.axhline(GATE['level2_gate']['ci_lower_bound_min'], color=PAL['bad'], lw=.7, ls=':')
ax.set_xlabel('univariate AUROC of the planted marker'); ax.set_ylabel('recovered Δ AUROC\n(centre-level 95% t-interval)'); ax.set_xticks(ladder.planted_marker_AUROC); panel(ax, 'c')
ax.legend([plt.Line2D([], [], color=PAL['aqua'], marker='o', ms=4, lw=1.4), plt.Line2D([], [], color=PAL['grey'], marker='o', ms=4, lw=1.4), plt.Line2D([], [], color=PAL['bad'], ls=':', lw=.9)],
          ['gate passes', 'gate does not pass', 'gate margin +0.01'], loc='upper center', bbox_to_anchor=(.5, -.30), ncol=3, columnspacing=1.0, handlelength=1.4, handletextpad=.4)
ax = axd; ygrid(ax)                                                      # (d) informative presence
ax.plot(pres.ordering_bias_gamma, pres.full_model, 'o-', color=CO, ms=3.4, lw=1.5, label='full model')
ax.plot(pres.ordering_bias_gamma, pres.full_without_presence_feature, 'o-', color=PAL['seq'][0], ms=3.4, lw=1.5, label='full model, presence feature removed')
ax.plot(pres.ordering_bias_gamma, pres.presence_only_auroc, 's-', color=CT, ms=3.2, lw=1.5, label='presence-only')
ax.axhline(.5, color=PAL['ink2'], ls='--', lw=.7)
ax.set_xticks([0, .5, 1, 1.5, 2]); ax.set_xlabel('ordering bias $\\gamma$ (suspected cases get the panel more often)'); ax.set_ylabel('IPCW td-AUROC (leave-centre-out)'); panel(ax, 'd')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.30), ncol=2, columnspacing=1.0, handlelength=1.4, handletextpad=.4, labelspacing=.3)
finish(fig, 9, 'benchmark_self_validation',
       "[SIM] The benchmark tested like a model. (a) Coverage of the nominal 95% interval for the true gain: the centre-level t-interval holds its "
       "level at every number of centres; the percentile bootstrap over centres does not. (b) Probability of passing the gate against the true gain. "
       "The obvious gate (point estimate at least +0.03, bootstrap bound above 0) passes a threshold-sized gain about half the time; the adopted gate "
       "(t-interval lower bound at least +0.01) is powered for +0.03 at 20 centres and almost never passes +0.01 or less. Dotted lines mark 80% power, "
       "the 5% false-pass level and the designed-for gain. (c) Effects of known size planted into the simulated cohort and recovered through the "
       "real harness code; the planted null is covered by its interval and does not pass. (d) Informative presence: a model that sees only whether "
       "the biomarker panel was ordered scores above chance as soon as clinicians order it selectively, which is why presence-only is a mandatory baseline.")
''')

md(r"""
**What the self-validation changes in the specification.** (i) The Level-2 interval is the centre-level *t*-interval, never a percentile bootstrap (for fewer than ~40 centres). (ii) The Level-2 rule is a *lower bound* of +0.01, which makes the design powered for +0.03; the point estimate alone is not a gate. (iii) **Presence-only** and **values-masked** comparators are mandatory, and the audit warns when the presence-only model beats chance by more than 0.02. (iv) Task **G7** runs every submission through permuted outcomes and a negative-control exposure and reports the pipeline's false-pass rate. (v) Task **G6** anchors the counterfactual claims to a randomised result.

## 8.13 [SIM] Clinical translation: drift, recalibration, enrichment and alert burden

An AUROC does not tell a transplant physician what to do. The benchmark therefore reports four translations of the same score, each computed here from first principles.

* **Drift and recalibration (task G1).** Practice changes — letermovir, PtCy for matched donors, ruxolitinib — so a model's *calibration* decays long before its *discrimination* does. We simulate a cohort whose baseline aGvHD risk falls by 0.10 on the logit scale per year, train on 2014–2018, and score each later year with and without a simple recalibration. The benchmark reports observed/expected incidence (O/E) per year, not just AUROC.
* **Prognostic enrichment.** The commonest clinical use of a risk score is to enrol the high-risk fraction into a pre-emptive-therapy trial. A trial of a treatment that cuts risk by 30 % needs fewer patients if enrolment is restricted to the top-scoring fraction *q*, but more patients must be screened to find them. We compute the **total number of patients screened** as a function of *q* and of the model's AUROC.
* **Alert burden (task B3).** For a rolling 7-day onset alarm the quantity that decides adoption is not AUROC but **alerts per 100 patient-days at a given sensitivity**, because every alert costs someone's time.
* **Net benefit** (computed in §8.11) completes the set: it is the only one of the four that depends on how the clinician values a missed case against a false alarm.
""")

code(r'''
# ----------------------------------------------------------------------------- [SIM] temporal drift and recalibration (task G1)
DRIFT = -.10                                                       # logit change in baseline aGvHD risk per calendar year (planted)
CD_ALL = simulate_cohort(n=24000, seed=21, drift=DRIFT); CD = CD_ALL[CD_ALL.time > LANDMARK].reset_index(drop=True)
Xd = prep(CD, FULL_COLS); yd = y_b1(CD); tr_ = (CD.era <= 2018).values
mdl = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000)).fit(Xd[tr_], yd[tr_])
pd_ = mdl.predict_proba(Xd)[:, 1]; lg = lambda p: np.log(p / (1 - p))
drift_rows = []; prev_obs = None
for yr in range(2019, 2024):
    te = (CD.era == yr).values; tprev = (CD.era == yr - 1).values
    obs = aalen_johansen(CD.time.values[te], CD.cause.values[te], 100, 1)
    obs_prev = aalen_johansen(CD.time.values[tprev], CD.cause.values[tprev], 100, 1)
    exp_static = float(pd_[te].mean())
    # recalibration-in-the-large using ONLY the previous year's outcomes: shift the logit so mean predicted risk = last year's observed incidence
    shift = 0.0
    for _ in range(30):
        f = float((1 / (1 + np.exp(-(lg(pd_[tprev]) + shift)))).mean()); shift += (lg(np.clip(obs_prev, 1e-4, .99)) - lg(f))
    exp_recal = float((1 / (1 + np.exp(-(lg(pd_[te]) + shift)))).mean())
    drift_rows.append(dict(year=yr, n=int(te.sum()), observed=obs, expected_static=exp_static, OE_static=obs / exp_static,
                           expected_recalibrated=exp_recal, OE_recalibrated=obs / exp_recal,
                           ipcw_auroc=ipcw_td_auroc(CD.time.values[te], CD.cause.values[te], mdl.decision_function(Xd[te]), 100)))
drift_df = pd.DataFrame(drift_rows)
print(f'[SIM] drift of {DRIFT:+.2f} logit/year; model trained on 2014-2018 (n = {int(tr_.sum()):,}), scored year by year:'); display(md_table(drift_df.round(3)))
oe_slope = float(np.polyfit(drift_df.year, np.log(drift_df.OE_static), 1)[0])
print(f'log O/E of the static model changes by {oe_slope:+.3f} per year; AUROC range across years {drift_df.ipcw_auroc.min():.3f}-{drift_df.ipcw_auroc.max():.3f}')
assert drift_df.OE_static.iloc[-1] < drift_df.OE_static.iloc[0] - .1, 'the planted drift must show up as miscalibration of the static model'
assert drift_df.OE_recalibrated.sub(1).abs().mean() < drift_df.OE_static.sub(1).abs().mean(), 'recalibration must reduce the O/E error'
''')

code(r'''
# ----------------------------------------------------------------------------- prognostic enrichment and alert burden from the binormal model (closed form)
def _cut(q, A, prev):
    """Score threshold c such that a fraction q of patients scores above it, for a binormal score with AUROC A and prevalence prev."""
    D = math.sqrt(2) * norm.ppf(A); lo, hi = -8., D + 8
    for _ in range(80):
        mid = (lo + hi) / 2
        if prev * norm.sf(mid - D) + (1 - prev) * norm.sf(mid) > q: lo = mid
        else: hi = mid
    return (lo + hi) / 2, D
def trial_for_top_fraction(q, A, prev, rrr=.30, alpha=.05, power=.8):
    """Two-arm trial of a therapy with relative risk reduction rrr, enrolling the top fraction q of the risk ranking.
    Returns (patients enrolled, pool that must be ranked to find them, event rate among those enrolled)."""
    c, D = _cut(q, A, prev); p_sel = prev * norm.sf(c - D) / q
    pc_, pt_ = p_sel, p_sel * (1 - rrr)
    n_arm = (norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2 * (pc_ * (1 - pc_) + pt_ * (1 - pt_)) / (pc_ - pt_) ** 2
    return 2 * n_arm, 2 * n_arm / q, p_sel
def alerts_per_100_patient_days(sens, A, prev_window):
    """Alerts per 100 patient-days when the alarm is set to catch `sens` of upcoming onsets (binormal score, onset-within-7-days rate prev_window)."""
    D = math.sqrt(2) * norm.ppf(A); c = D + norm.isf(sens)                         # threshold at which P(score > c | event) = sens
    fpr = norm.sf(c); return 100 * (prev_window * sens + (1 - prev_window) * fpr)
QS = np.linspace(.05, 1.0, 40); PREV_ENR = .13                                  # grade III-IV aGvHD (task B2): where enrichment would actually be used
ENR = {A: [trial_for_top_fraction(q, A, PREV_ENR) for q in QS] for A in (.65, .72, .80)}
enr_rows = []
for A in (.65, .72, .80):
    n_un = trial_for_top_fraction(1.0, A, PREV_ENR)[0]
    for q in (.5, .3, .2):
        n_e, pool, ps = trial_for_top_fraction(q, A, PREV_ENR)
        enr_rows.append({'AUROC': A, 'fraction enrolled q': q, 'event rate among enrolled': f'{ps:.2f}', 'trial size, enriched': f'{n_e:,.0f}', 'trial size, unselected': f'{n_un:,.0f}',
                         'saving in trial size': f'{1 - n_e / n_un:.0%}', 'pool that must be ranked': f'{pool:,.0f}', 'pool / unselected trial': f'{pool / n_un:.1f}x'})
enr_tab = pd.DataFrame(enr_rows)
SS = np.linspace(.3, .95, 30)
ALERT = {A: [alerts_per_100_patient_days(s, A, .04) for s in SS] for A in (.70, .75, .80, .85)}
al_tab = pd.DataFrame([dict(AUROC=A, **{f'alerts / 100 patient-days at sens {s:.0%}': alerts_per_100_patient_days(s, A, .04) for s in (.5, .7)},
                            **{'PPV at sens 50%': (.04 * .5) / (.04 * .5 + .96 * norm.sf(math.sqrt(2) * norm.ppf(A) + norm.ppf(.5)))}) for A in (.70, .75, .80, .85)])
print(f'PROGNOSTIC ENRICHMENT for a therapy that cuts risk by 30 % (outcome prevalence {PREV_ENR}; α = 0.05, power 0.8):'); display(md_table(enr_tab))
print('ALERT BURDEN for a rolling 7-day onset alarm (event rate 0.04 per alarm-day):'); display(md_table(al_tab.round(2)))
# a random model (AUROC .5) must give no saving and an alert rate equal to the base rate
assert abs(trial_for_top_fraction(.5, .501, .35)[0] - trial_for_top_fraction(1.0, .501, .35)[0]) / trial_for_top_fraction(1.0, .501, .35)[0] < .02    # a random score cannot enrich
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 10: from AUROC to what a clinician would see
fig = plt.figure(figsize=(FULL, 3.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0], left=.075, right=.985, bottom=.255, top=.875, wspace=.40)
ax = fig.add_subplot(gs[0, 0]); ygrid(ax)                                       # (a) drift and recalibration
ax.axhline(1, color=PAL['ink2'], lw=.7, ls='--')
ax.plot(drift_df.year, drift_df.OE_static, 'o-', color=PAL['orange'], ms=3.4, lw=1.5, label='static model (trained 2014–2018)')
ax.plot(drift_df.year, drift_df.OE_recalibrated, 'o-', color=PAL['seq'][3], ms=3.4, lw=1.5, label='re-calibrated on the previous year')
ax.set_xticks(drift_df.year); ax.set_xticklabels([str(y)[2:] for y in drift_df.year]); ax.set_xlabel('calendar year (20xx)'); ax.set_ylabel('observed / expected incidence'); ax.set_ylim(.55, 1.25)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.26), ncol=1, handlelength=1.4, labelspacing=.3); panel(ax, 'a')
ax = fig.add_subplot(gs[0, 1]); ygrid(ax)                                       # (b) prognostic enrichment
for A, c in zip((.65, .72, .80), (PAL['seq'][0], PAL['seq'][2], PAL['seq'][3])):
    ax.plot(QS, [e[0] for e in ENR[A]], color=c, lw=1.5, label=f'AUROC {A:.2f}')
    ax.plot(QS, [e[1] for e in ENR[A]], color=c, lw=1.2, ls=':')
ax.set_yscale('log'); ax.set_yticks([300, 1000, 3000, 10000]); ax.set_yticklabels(['300', '1,000', '3,000', '10,000']); ax.minorticks_off()
ax.set_xlabel('fraction of the ranked pool enrolled, $q$'); ax.set_ylabel('patients (solid: enrolled; dotted: pool)'); ax.set_xlim(.03, 1.02)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.26), ncol=3, columnspacing=1.0, handlelength=1.4, handletextpad=.4); panel(ax, 'b')
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                       # (c) alert burden
for A, c in zip((.70, .75, .80, .85), (PAL['seq'][0], PAL['seq'][1], PAL['seq'][2], PAL['seq'][3])):
    ax.plot(SS, ALERT[A], color=c, lw=1.5, label=f'AUROC {A:.2f}')
ax.set_yscale('log'); ax.set_yticks([3, 5, 10, 20, 50]); ax.set_yticklabels(['3', '5', '10', '20', '50']); ax.minorticks_off()
ax.set_xlabel('sensitivity for onset within 7 days'); ax.set_ylabel('alerts per 100 patient-days')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.26), ncol=2, columnspacing=1.0, handlelength=1.4, handletextpad=.4, labelspacing=.3); panel(ax, 'c')
finish(fig, 10, 'clinical_translation',
       '[SIM / closed form] Three translations of the same score. (a) Calibration decays under practice drift while discrimination barely moves: '
       'observed/expected incidence of a static model against calendar year, and after re-calibration on the previous year. (b) Prognostic enrichment: '
       'patients enrolled (solid) and pool that must be ranked to find them (dotted) for a two-arm trial of a therapy that cuts risk by 30% in a '
       'population with a 13% event rate, against the fraction enrolled from the top of the risk ranking. Even AUROC 0.65 shrinks the trial substantially, at the price of a pool about twice as large; the better the score, the smaller the pool needed. '
       '(c) Alert burden of a rolling 7-day onset alarm at an onset rate of 0.04 per alarm-day, which decides whether anyone will use it.')
''')

code(r'''
gn = lambda A, q, col: enr_tab[(enr_tab.AUROC == A) & (enr_tab['fraction enrolled q'] == q)][col].iloc[0]; a80 = al_tab.set_index('AUROC').loc[.80]; a70 = al_tab.set_index('AUROC').loc[.70]
over_ = 1 / drift_df.OE_static.iloc[-1] - 1
display(Markdown(f"""**Result.** The three translations are more demanding than the AUROC suggests.

* **Drift** (panel a): with a planted decline in baseline risk the static model's discrimination barely moves (AUROC {drift_df.ipcw_auroc.min():.3f}–{drift_df.ipcw_auroc.max():.3f} across years) but its O/E falls from {drift_df.OE_static.iloc[0]:.2f} to **{drift_df.OE_static.iloc[-1]:.2f}** — by 2023 it **over-predicts risk by about {100 * over_:.0f} %** — while a one-line recalibration on the previous year keeps O/E within {100 * drift_df.OE_recalibrated.sub(1).abs().max():.0f} % of 1. Task G1 therefore reports O/E per year, and Level 3 requires a **recalibration plan**, not a one-off validation.
* **Enrichment** (panel b): at an event rate of 13 %, enrolling the top 30 % of the ranking shrinks a 30 %-risk-reduction trial by **{gn(.65, .3, 'saving in trial size')}** at AUROC 0.65, **{gn(.72, .3, 'saving in trial size')}** at 0.72 and **{gn(.80, .3, 'saving in trial size')}** at 0.80. Even a modest score therefore buys a much smaller trial; what a better score buys is a smaller *pool*: the ranking must cover {gn(.65, .3, 'pool / unselected trial')} the size of the unselected trial at AUROC 0.65, {gn(.72, .3, 'pool / unselected trial')} at 0.72 and {gn(.80, .3, 'pool / unselected trial')} at 0.80. Enrichment is attractive only where the pool is plentiful and the score is free to compute, which describes registry-scale pools and no single centre. This is the clinically meaningful reading of "is AUROC 0.72 good enough": it is enough to design a pre-emptive-therapy trial with roughly half the patients, provided a pool of about {gn(.72, .3, 'pool that must be ranked')} can be ranked.
* **Alerts** (panel c): a rolling alarm of AUROC 0.80 produces **{a80['alerts / 100 patient-days at sens 50%']:.0f} alerts per 100 patient-days** at 50 % sensitivity with a PPV of **{a80['PPV at sens 50%']:.0%}** — about {a80['alerts / 100 patient-days at sens 50%'] * .2:.1f} alerts a day on a 20-bed transplant unit; at AUROC 0.70 it is {a70['alerts / 100 patient-days at sens 50%']:.0f} per 100 patient-days at a PPV of {a70['PPV at sens 50%']:.0%}. B3 is therefore reported as alerts per 100 patient-days at fixed sensitivity, and its acceptance criterion is a **clinician-agreed alert budget**, not an AUROC."""))
''')

md(r"""
## 8.14 Rules the self-validation adds to the benchmark

| Rule | Why | Where it is enforced |
|---|---|---|
| Level-2 interval = centre-level *t* (K−1 d.f.), lower bound ≥ +0.01, ≥ 20 centres | percentile bootstrap under-covers; point-estimate gates are coin flips at threshold (§8.12) | `spec/gate.json` |
| Presence-only and values-masked comparators; audit warns if presence-only beats chance by > 0.02 | informative ordering looks like signal (§8.12) | harness component 5–6 |
| Task G7: permuted-outcome and negative-exposure controls on every submission | measures the pipeline's false-pass rate | harness component 7 |
| Task G6: reproduce a published randomised effect (BMT CTN 1703, GRFS HR 0.64 ✓) from registry data | the only external anchor for counterfactual claims | task card G6 |
| Task G1 reports O/E per calendar year and the plan for recalibration | calibration, not discrimination, decays (§8.13) | task card G1 |
| Utility is reported as screened-patient saving (enrichment) and alerts per 100 patient-days, alongside AUROC | what a clinician or trial sponsor actually acts on (§8.13) | tasks B1/B3/C3 |
| Subgroup rule: a subgroup is **assessed** only if it has the events §9.7 requires, otherwise reported as *not assessable* | prevents false reassurance on small groups | task G3 |
| Label reliability: ≥ 10 % of outcomes double-read, κ ≥ 0.8, else the attenuation of §9.7 is applied to every n | label noise caps discrimination and inflates n | §8.1 cohort definition |
""")
