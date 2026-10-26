from part1 import CELLS, md, code

# =====================================================================================================  9.4  TRANSCRIPTOME AT DIAGNOSIS
md(r'''
## 9.4 The transcriptome at diagnosis: how many labelled profiles?

§3.3 said what a transcriptomic profile can support. This section says **how many labelled profiles it takes**, and the answer turns on one quantity: the number of free parameters the model has to estimate from those labels.

**The model.** Take a linear head trained on *n* labelled transcriptomes pooled over *K* centres and scored at a *new* centre. Let the signal it is trying to learn have squared separation $I$ (so that a head that knew it exactly would reach AUROC $=\Phi(\sqrt{I/2})$), let $D_0^2$ be the separation already carried by a baseline score it is added to (zero if there is none), let $\rho$ be the fraction of the signal shared across centres, and let $p_{\rm eff}$ be the **effective number of parameters the head must estimate**. Then, for prevalence $\pi$,

$$\mathrm{AUROC}(n)\;=\;\Phi\!\left(\frac{D_0^2+\rho I}{\sqrt2\,\sqrt{D_0^2+\rho I+(1-\rho)I/K+p_{\rm eff}/\big(\pi(1-\pi)\,n\big)}}\right).$$

This is the learning curve of a plug-in linear classifier (Raudys–Duin type): the numerator is the signal the head captures, and the last term in the denominator is the noise from estimating $p_{\rm eff}$ weights from *n* patients. Three things follow at once. **(i)** The required *n* is proportional to $p_{\rm eff}$ — the same proportionality as Riley's *P* in §9.1. **(ii)** With $\rho<1$ there is a **plateau** below the within-centre ceiling that no amount of data removes. **(iii)** What a foundation model offers is a way to shrink $p_{\rm eff}$ — a 64-dimensional embedding instead of 15,000 genes — at the price of keeping only a fraction $\kappa$ of the signal.

We check the formula against simulation before using it.
''')

code(r'''
# ----------------------------------------------------------------------------- 9.4: learning curve of a linear head trained on n labelled transcriptomes
def sep2(A): return 2 * norm.ppf(A) ** 2                      # squared Mahalanobis separation of a binormal score with AUROC A
def auc_lc(n, D0sq, Isq, p_eff, prev=.30, rho=1.0, K=8):
    """AUROC at a NEW centre of [known baseline score] + [linear head learned from n labelled transcriptomes pooled over K centres]."""
    num = D0sq + rho * Isq
    den = math.sqrt(D0sq + rho * Isq + (1 - rho) * Isq / K + p_eff / (prev * (1 - prev) * n))
    return float(norm.cdf(num / (math.sqrt(2) * den)))
def n_for_target(target, A0, A1, p_eff, prev=.30, rho=1.0, K=8, kappa=1.0):
    """Smallest n with AUROC(n) >= target for a head whose signal is a fraction kappa of the ceiling gain A0 -> A1 (inf if the plateau is below target)."""
    D0, I = sep2(A0) if A0 > .5 else 0.0, kappa * (sep2(A1) - (sep2(A0) if A0 > .5 else 0.0))
    if auc_lc(1e15, D0, I, p_eff, prev, rho, K) < target - 1e-9: return float('inf')
    lo, hi = math.log(5), math.log(1e10)
    for _ in range(80):
        mid = (lo + hi) / 2
        if auc_lc(math.exp(mid), D0, I, p_eff, prev, rho, K) < target: lo = mid
        else: hi = mid
    return math.exp(hi)

# ---------- validation 1: the formula against simulated nearest-centroid heads (isotropic Gaussian data; the case where it is exact)
def sim_lc(n, A1, A0, p, rho, K, prev=.30, reps=6, seed=0):
    D0, I = (math.sqrt(sep2(A0)) if A0 > .5 else 0.0), sep2(A1) - (sep2(A0) if A0 > .5 else 0.0); D = math.sqrt(I); out = []
    for s in range(reps):
        r = np.random.default_rng(seed + 97 * s); unit = lambda: (lambda v: v * D / np.linalg.norm(v))(r.standard_normal(p))
        mu0 = unit(); mus = [math.sqrt(rho) * mu0 + math.sqrt(1 - rho) * unit() for _ in range(K)]; mus = [m * D / np.linalg.norm(m) for m in mus]
        per = n // K; X, Y, Z = [], [], []
        for m in mus:
            y = (r.random(per) < prev).astype(int); X.append(r.standard_normal((per, p)) + np.outer(y, m)); Y.append(y)
        X, Y = np.vstack(X), np.concatenate(Y)
        if min(np.bincount(Y, minlength=2)) < 2: continue
        w = X[Y == 1].mean(0) - X[Y == 0].mean(0)
        mt = math.sqrt(rho) * mu0 + math.sqrt(1 - rho) * unit(); mt *= D / np.linalg.norm(mt)
        yt = (r.random(4000) < prev).astype(int); xt = r.standard_normal((4000, p)) + np.outer(yt, mt); zt = D0 * (r.standard_normal(4000) + D0 * yt)
        out.append(roc_auc_score(yt, xt @ w + (D0 * (zt / max(D0, 1e-9)) if D0 > 0 else 0)))
    return float(np.mean(out))
VAL_LC = [(.80, .5, 400, 1.0), (.80, .5, 400, .6), (.85, .5, 1000, 1.0), (.85, .5, 1000, .3), (.75, .5, 200, 1.0), (.78, .72, 64, 1.0), (.78, .72, 64, .7)]
vrows = []
for A1_, A0_, p_, rho_ in VAL_LC:
    for n_ in (64, 256, 1024, 4096):
        f_ = auc_lc(n_, sep2(A0_) if A0_ > .5 else 0, sep2(A1_) - (sep2(A0_) if A0_ > .5 else 0), p_, .30, rho_, 8)
        s_ = sim_lc(n_, A1_, A0_, p_, rho_, 8, reps=6 if not QUICK else 3); vrows.append(dict(ceiling=A1_, baseline=A0_, p=p_, rho=rho_, n=n_, formula=f_, simulated=s_, diff=s_ - f_))
vlc = pd.DataFrame(vrows)
print(f'learning-curve formula vs simulation: {len(vlc)} comparisons, mean |diff| {vlc["diff"].abs().mean():.3f}, max {vlc["diff"].abs().max():.3f}')
assert vlc['diff'].abs().max() < (.025 if not QUICK else .04), 'learning-curve formula disagrees with its simulation'

# ---------- validation 2 (stress test): correlated nuisance structure (cell composition, batch) and a ridge head with its penalty chosen by closed-form leave-one-out
def ridge_loo(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-8; Z = (X - mu) / sd; t = 2. * y - 1; tc = t - t.mean()
    U, s, Vt = np.linalg.svd(Z, full_matrices=False); Uty = U.T @ tc; best = (1e18, None)
    for a in np.logspace(0, 5, 11):
        f = s ** 2 / (s ** 2 + a); loo = np.mean(((tc - U @ (f * Uty)) / (1 - (U ** 2) @ f)) ** 2)
        if loo < best[0]: best = (loo, a)
    w = Vt.T @ (s / (s ** 2 + best[1]) * Uty); return lambda Xn: ((Xn - mu) / sd) @ w
def stress_curve(A1, m_info, p=600, q=40, K=8, prev=.30, ns=(64, 128, 256, 512, 1024, 2048), reps=3, seed=0):
    D = math.sqrt(sep2(A1)); res = {n: [] for n in ns}
    for s in range(reps):
        r = np.random.default_rng(seed + 31 * s); Lam = r.standard_normal((p, q)) * 1.2 / math.sqrt(q); Sig = Lam @ Lam.T + np.eye(p); Sinv = np.linalg.inv(Sig); Lc = np.linalg.cholesky(Sig)
        supp = r.choice(p, m_info, replace=False); mu = np.zeros(p); mu[supp] = r.standard_normal(m_info); mu *= D / math.sqrt(mu @ Sinv @ mu)
        def draw(n, rr):
            y = (rr.random(n) < prev).astype(int); return rr.standard_normal((n, p)) @ Lc.T + np.outer(y, mu) + rr.normal(0, .3, p), y
        pool = [draw(max(ns) // K, np.random.default_rng(1000 * s + c)) for c in range(K)]; xt, yt = draw(1500, np.random.default_rng(10 ** 6 + s))
        for n in ns:
            X = np.vstack([a[0][:n // K] for a in pool]); Y = np.concatenate([a[1][:n // K] for a in pool])
            res[n].append(roc_auc_score(yt, ridge_loo(X, Y)(xt)))
    return {n: float(np.mean(v)) for n, v in res.items()}
from scipy.optimize import minimize_scalar
def fit_peff(curve, A1, prev=.30):
    ns_, vs_ = list(curve), list(curve.values()); f = lambda lp: sum((auc_lc(n, 0, sep2(A1), math.exp(lp), prev, 1.0, 8) - v) ** 2 for n, v in zip(ns_, vs_))
    r = minimize_scalar(f, bounds=(0, 14), method='bounded'); return math.exp(r.x), math.sqrt(r.fun / len(ns_))
srows = []
for m_ in (15, 60, 250):
    c_ = stress_curve(.80, m_, reps=3 if not QUICK else 2); pe_, rm_ = fit_peff(c_, .80)
    srows.append(dict(informative_genes=m_, genes=600, fitted_p_eff=pe_, p_eff_over_p=pe_ / 600, rmse_of_fit=rm_))
stress = pd.DataFrame(srows); MULT = float(stress.p_eff_over_p.mean())
print('stress test (correlated nuisance structure; ridge head with LOO-chosen penalty): the learning curve is still described by the same formula,\n'
      'with an effective dimension of about p x', f'{MULT:.1f}', '- regardless of how sparse the signal is, because a ridge penalty does not exploit sparsity:')
display(md_table(stress.round(3)))
''')

code(r'''
# ----------------------------------------------------------------------------- 9.4: what a foundation-model adapter buys, in labelled profiles
GENES = 15_000
HEADS = [  # name, p_eff, kappa (fraction of the signal retained), note
 ('pre-specified pathway scores (≈10)',            10,              .70, 'head sees a handful of fixed scores; keeps part of the signal'),
 ('frozen pretrained adapter, 10-parameter head',  10,              .90, 'a released encoder, frozen; only a 10-parameter head is fitted; kappa is an ASSUMPTION that task F2 / G4 measure'),
 ('foundation-model adapter embedding (64-d)',     64,              .90, 'head sees a learned 64-d embedding; kappa is an ASSUMPTION that task G4 / F2 measure'),
 ('selected gene panel (500 genes)',               500 * MULT,      .95, 'a ridge/logistic head on 500 genes; effective dimension inflated by nuisance structure'),
 ('whole transcriptome (15,000 genes)',            GENES * MULT,    1.0, 'a ridge/logistic head on every gene'),
]
def plan_row(name, p_eff, kappa, A1, prev=.30, eps=.02, rho=1.0, K=8):
    I = kappa * sep2(A1); plateau = auc_lc(1e15, 0, I, p_eff, prev, rho, K)
    return plateau, n_for_target(plateau - eps, .5, A1, p_eff, prev, rho, K, kappa)
plan = []
for name, pe, kap, note in HEADS:
    row = dict(head=name, p_eff=f'{pe:,.0f}', kappa=kap)
    for A1 in (.75, .88):
        pl, n_ = plan_row(name, pe, kap, A1); row[f'ceiling {A1:.2f}: plateau'] = f'{pl:.3f}'; row[f'ceiling {A1:.2f}: n within 0.02 of it'] = capn(int(round(n_)))
    plan.append(row)
plan = pd.DataFrame(plan)
print(f'LABELLED TRANSCRIPTOMES NEEDED to come within 0.02 AUROC of what each head can reach (prevalence 0.30, single centre; "> {N_CAP:,}" = more than the ceiling cohort):'); display(md_table(plan))
HD = {h[0]: h[:3] for h in HEADS}
n_whole = plan_row(*HD['whole transcriptome (15,000 genes)'], .88)[1]; n_whole75 = plan_row(*HD['whole transcriptome (15,000 genes)'], .75)[1]
n_adapt = plan_row(*HD['foundation-model adapter embedding (64-d)'], .88)[1]; n_adapt75 = plan_row(*HD['foundation-model adapter embedding (64-d)'], .75)[1]
n_frozen = plan_row(*HD['frozen pretrained adapter, 10-parameter head'], .88)[1]; n_frozen75 = plan_row(*HD['frozen pretrained adapter, 10-parameter head'], .75)[1]
print(f'at ceiling 0.88 a 10-parameter head on a frozen adapter needs {n_frozen:,.0f} labelled profiles and a 64-d adapter head {n_adapt:,.0f}; a whole-transcriptome head needs more than {N_CAP:,} by orders of magnitude')

# the incremental question: labelled profiles for a transcriptome adapter to ADD to a strong clinical baseline (ρ = 0.8 across K = 8 centres)
TXT = [  # task, what is added, baseline AUROC, ceiling AUROC with the transcriptome, prevalence
 ('C1/C2  GvHD vs infection / drug (tissue + clinical)', .75, .86, .45), ('C4  steroid refractoriness (onset sample)', .66, .74, .30),
 ('C5  day-28 response (onset sample)', .64, .72, .35), ('B1  + day-7 blood transcriptome on MAP + clinical', .72, .76, .35),
 ('E1  + day-100 blood transcriptome (cGvHD)', .65, .69, .30)]
trows = []
for name, A0, A1, pv in TXT:
    r_ = dict(task=name, baseline=A0, ceiling=A1, prevalence=pv)
    for lab, rho_, K_ in (('ρ = 1', 1.0, 8), ('ρ = 0.8, K = 8', .8, 8)):
        plat = auc_lc(1e15, sep2(A0), .9 * (sep2(A1) - sep2(A0)), 64, pv, rho_, K_)
        r_[f'n for +0.02 ({lab})'] = n_for_target(A0 + .02, A0, A1, 64, pv, rho_, K_, .9)
        r_[f'realisable gain ({lab})'] = plat - A0
        r_[f'n for 80% of it ({lab})'] = n_for_target(A0 + .80 * (plat - A0), A0, A1, 64, pv, rho_, K_, .9)
        r_[f'n for 95% of it ({lab})'] = n_for_target(A0 + .95 * (plat - A0), A0, A1, 64, pv, rho_, K_, .9)
    r_['10-parameter head, n for +0.02 (ρ = 1)'] = n_for_target(A0 + .02, A0, A1, 10, pv, 1.0, 8, .9)
    trows.append(r_)
txt_tab = pd.DataFrame(trows)
txt_disp = txt_tab.copy()
for c in txt_disp.columns[4:]: txt_disp[c] = txt_disp[c].map(lambda v: ('+' + f'{v:.3f}') if 'gain' in c else ('not reachable' if not np.isfinite(v) else capn(int(round(v)))))
print(f'LABELLED PROFILES FOR AN ADAPTER (64-d, κ = 0.9) TO ADD TO A BASELINE, and for a 10-parameter head ("realisable gain" = the plateau of the added AUROC at that ρ; "> {N_CAP:,}" = more than the ceiling cohort):'); display(md_table(txt_disp))
def min_ceiling_gain(A0, want=.03, rho=.8, K=8, kappa=.9, pv=.30):
    for g in np.arange(.02, .30, .0025):
        if np.isfinite(n_for_target(A0 + want, A0, A0 + g, 64, pv, rho, K, kappa)): return float(g)
G_STAR = min_ceiling_gain(.72)
''')

code(r'''
# ----------------------------------------------------------------------------- 9.4: the assay specification (depth and detection)
# depth and detection: lognormal gene abundance, negative-binomial counts; reliability of a 50-gene score between technical replicates
rg = np.random.default_rng(3); frac_g = np.exp(rg.normal(math.log(1e-5), 1.8, 20000)); frac_g /= frac_g.sum() / 1.0
sig = rg.choice(np.where(5e6 * frac_g >= 1)[0], 50, replace=False)             # one fixed 50-gene signature, scored at every depth
drows = []
for R in (5e6, 10e6, 20e6, 40e6, 80e6):
    mu_g = R * frac_g; det = float((mu_g >= 10).mean())
    tech = float(np.mean(1 / mu_g[sig] + .02) / 50)                                                                    # technical variance of the mean log-score
    bio = .16 * (.3 + .7 / 50)                                                                                          # between-patient variance of a pathway score (gene SD .4, mean inter-gene correlation .3)
    drows.append({'reads per sample': f'{R / 1e6:.0f} M', 'genes with >= 10 counts': f'{det:.0%}', 'reliability of a 50-gene pathway score': f'{bio / (bio + tech):.3f}'})
print('DEPTH (assumed gene-abundance model †):'); display(md_table(pd.DataFrame(drows)))
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 13: how many labelled transcriptomes at one institution?
fig = plt.figure(figsize=(FULL, 3.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0], left=.075, right=.985, bottom=.265, top=.875, wspace=.40)
ax = fig.add_subplot(gs[0, 0]); ygrid(ax)                                       # (a) learning curves inside the ceiling cohort
nn = np.geomspace(20, N_CAP, 80); A1_ = .80; I_ = sep2(A1_)
ax.plot(nn, [auc_lc(n, 0, I_, GENES * MULT) for n in nn], color=PAL['grey'], lw=1.6, label='whole transcriptome')
ax.plot(nn, [auc_lc(n, 0, .9 * I_, 64) for n in nn], color=PAL['seq'][1], lw=1.5, ls='--', label='64-d adapter head ($\\kappa$ = 0.9)')
ax.plot(nn, [auc_lc(n, 0, .9 * I_, 10) for n in nn], color=PAL['seq'][3], lw=1.6, label='10-parameter head, frozen adapter ($\\kappa$ = 0.9)')
ax.plot(nn, [auc_lc(n, 0, .7 * I_, 10) for n in nn], color=PAL['seq'][2], lw=1.5, ls=':', label='10-parameter head ($\\kappa$ = 0.7)')
ax.axhline(A1_, color=PAL['ink2'], lw=.6, ls='--'); ax.axvline(N_SPEC, color=PAL['ink2'], lw=.6, ls=':')
ax.set_xscale('log'); ax.set_xlim(20, N_CAP); ax.set_ylim(.5, .83); ax.set_xticks([20, 50, 100, 300, 1000]); ax.set_xticklabels(['20', '50', '100', '300', '1,000']); ax.minorticks_off()
ax.set_xlabel('labelled transcriptomes $n$'); ax.set_ylabel('AUROC at a new patient set'); panel(ax, 'a')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=1, handlelength=1.4, handletextpad=.4, labelspacing=.3)
ax = fig.add_subplot(gs[0, 1]); ygrid(ax)                                       # (b) labelled profiles needed to add a gain to a baseline
gains = np.linspace(.035, .14, 40)
for A0, c in ((.65, PAL['seq'][0]), (.72, PAL['seq'][2]), (.80, PAL['seq'][3])):
    ys = [n_for_target(A0 + .03, A0, A0 + g, 10, .30, 1.0, 8, .9) for g in gains]; ys = [y if np.isfinite(y) and y <= 4000 else np.nan for y in ys]
    ax.plot(gains, ys, color=c, lw=1.5, label=f'baseline AUROC {A0:.2f}')
ax.axhline(N_CAP, color=PAL['bad'], lw=.8, ls=':', label=f'ceiling cohort ({N_CAP:,})'); ax.axhline(N_SPEC, color=PAL['ink2'], lw=.8, ls='--', label=f'with specimens ({N_SPEC})')
ax.set_yscale('log'); ax.set_ylim(40, 4000); ax.set_yticks([50, 100, 420, 1500, 4000]); ax.set_yticklabels(['50', '100', '420', '1,500', '4,000']); ax.minorticks_off()
ax.set_xlabel('gain the transcriptome could add at its ceiling'); ax.set_ylabel('labelled profiles to realise +0.03\n(10-parameter head)'); panel(ax, 'b')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2, columnspacing=1.0, handlelength=1.4, handletextpad=.4, labelspacing=.3)
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                       # (c) the plateau set by cross-centre sharing (what a second centre would see)
rr_ = np.linspace(0, 1, 60); D_ = math.sqrt(sep2(.85))
for K_, c in ((4, PAL['seq'][0]), (8, PAL['seq'][2]), (20, PAL['seq'][3])):
    ax.plot(rr_, [norm.cdf((D_ / math.sqrt(2)) * r / math.sqrt(r + (1 - r) / K_)) if r > 0 else .5 for r in rr_], color=c, lw=1.5)
ax.axhline(.85, color=PAL['ink2'], lw=.6, ls='--'); ax.set_xlabel('shared fraction of the signal across centres $\\rho$'); ax.set_ylabel('AUROC plateau at a new centre'); ax.set_ylim(.5, .88); ax.set_xlim(0, 1)
ax.legend([plt.Line2D([], [], color=c, lw=1.5) for c in (PAL['seq'][0], PAL['seq'][2], PAL['seq'][3])], ['4 training centres', '8 training centres', '20 training centres'],
          loc='upper center', bbox_to_anchor=(.5, -.25), ncol=1, handlelength=1.4, labelspacing=.3); panel(ax, 'c')
finish(fig, 13, 'transcriptome_requirements',
       "Labelled transcriptomes at diagnosis, inside the ceiling cohort (closed form, validated against simulation in §9.4). (a) Learning curves at a ceiling AUROC of 0.80: a head on the whole "
       "transcriptome is still at chance over the whole range a centre can reach, a 64-dimensional adapter head is slow, and a head of 10 parameters on a frozen adapter is close to its "
       "plateau within a few hundred profiles; the dotted line marks the 420 patients with banked specimens. (b) Labelled profiles a 10-parameter head needs to add +0.03 AUROC to a baseline, against "
       "the gain the transcriptome could add at its ceiling, with the 1,500-patient ceiling and the 420 specimen patients marked: small ceiling gains cannot be realised at one centre. "
       "(c) What a second centre would see: the plateau falls as the shared fraction rho falls, and more training centres push it toward its limit; this is the reason a single-centre result is "
       "a claim about that centre.")
''')

code(r'''
b1 = txt_tab[txt_tab.task.str.startswith('B1')].iloc[0]; c4_ = txt_tab[txt_tab.task.str.startswith('C4')].iloc[0]; c12 = txt_tab.iloc[0]
min_n_b1 = int(round(b1['10-parameter head, n for +0.02 (ρ = 1)'])) if np.isfinite(b1['10-parameter head, n for +0.02 (ρ = 1)']) else None
display(Markdown(f"""**Reading §9.4.** Three results, each computed above, and each restated for what one centre can reach (≤ {N_CAP:,} patients, ≤ {N_SPEC} with specimens).

1. **A head on the whole transcriptome is out of reach at any single-centre scale** — the planning table shows "> {N_CAP:,}" for it at both ceilings, because cell composition and batch inflate its effective dimension by a factor of {stress.p_eff_over_p.min():.1f}–{stress.p_eff_over_p.max():.1f}. This is the quantitative reason for the pilot's negative result on blood (§10), and it is independent of the architecture.
2. **What a centre can do is shrink the head.** A **10-parameter head on a frozen pretrained adapter that keeps 90 % of the signal needs {n_frozen:,.0f}–{n_frozen75:,.0f} labelled profiles** (ceiling 0.88 and 0.75): inside the {N_SPEC} patients with specimens, and inside the biopsies a centre collects in three to four years. A pre-specified pathway score (no fitted parameters beyond a threshold) needs fewer still. A 64-dimensional adapter head needs {n_adapt:,.0f}–{n_adapt75:,.0f}, which is the edge of what one centre holds, and is the point at which pooling with other centres begins to matter.
3. **A transcriptome adds a *small* increment to a *strong* baseline, and small increments are expensive twice over**: the head has to *learn* the increment, and the cohort has to *show* it against the baseline with a paired test, and the claim needs the larger of the two. §9.8 computes both for each claim. The most a day-7 blood transcriptome could ever add to MAP plus clinical covariates here is +{b1['realisable gain (ρ = 1)']:.3f}. In tissue the picture is different (a high ceiling, and the tissue is in hand), which is why the paper claims the transcriptome adds value at diagnosis and does not claim it for pre-onset blood prognosis."""))
''')

# =====================================================================================================  9.5  LONGITUDINAL DESIGN
md(r'''
## 9.5 Longitudinal design: how many draws, how often, how many specimens?

The sampling schedule of §4.4 is a minimum. This section asks what is needed for the *trajectory*, which is what a foundation model on a timeline can exploit and a single draw cannot. Three questions, three calculations.

1. **How many draws in a window?** If one draw measures a patient's state with single-draw reliability λ (the fraction of between-patient variance that is signal), the mean of *k* draws has reliability $\lambda_k=k\lambda/(1+(k-1)\lambda)$ (Spearman–Brown), and a predictor whose oracle AUROC is $A^*$ reaches $\Phi\!\big(\Phi^{-1}(A^*)\sqrt{\lambda_k}\big)$. We validate that against simulation.
2. **How often?** Many biomarkers rise only in the days before clinical onset (a *prodrome* of length τ). If draws are Δ days apart, the number that land in the prodrome is $\lfloor\tau/\Delta\rfloor$ or one more, so the expected AUROC of a pre-onset alarm depends on Δ through the draws it gets.
3. **How many specimens?** The protocol of §4.4 is evaluated against a realistic distribution of onset days: of the cases, what fraction have at least one, and at least two, specimens in the 14 days before onset? And how many patients must be enrolled to have **100** such cases (enough for an AUROC with a half-width of about ±0.09)?
''')

code(r'''
# ----------------------------------------------------------------------------- 9.5: draws per window (Spearman-Brown), validated by simulation
def auc_k(A_star, lam1, k):
    lamk = k * lam1 / (1 + (k - 1) * lam1); return float(norm.cdf(norm.ppf(A_star) * math.sqrt(lamk)))
def sim_auc_k(A_star, lam1, k, n=40000, prev=.35, seed=0):
    r = np.random.default_rng(seed); D = math.sqrt(2) * norm.ppf(A_star); y = (r.random(n) < prev).astype(int)
    b = r.standard_normal(n) + D * y; se = math.sqrt((1 - lam1) / lam1)
    return float(roc_auc_score(y, (b[:, None] + se * r.standard_normal((n, k))).mean(1)))
LAMS, KDR = (.3, .5, .7, .9), (1, 2, 3, 5, 8)
dchk = max(abs(auc_k(.80, l, k) - sim_auc_k(.80, l, k, n=40000 if not QUICK else 15000)) for l in LAMS for k in KDR)
print(f'Spearman-Brown AUROC vs simulation over {len(LAMS) * len(KDR)} (reliability, draws) pairs: max |diff| = {dchk:.4f}')
assert dchk < (.012 if not QUICK else .02)
def k_needed(A_star, lam1, tol):
    for k in range(1, 400):
        if auc_k(A_star, lam1, k) >= A_star - tol: return k
    return None
kn = pd.DataFrame([dict(single_draw_reliability=l, **{'AUROC with 1 draw': round(auc_k(.80, l, 1), 3), 'draws for within 0.02 of the oracle': k_needed(.80, l, .02),
                                                      'draws for within 0.01': k_needed(.80, l, .01)}) for l in (.2, .3, .5, .7, .9)])
display(md_table(kn))

# ----------------------------------------------------------------------------- expected AUROC of a pre-onset alarm against draw interval and prodrome length
def expected_prodrome_auc(tau, dt, A_star=.80, lam1=.5, A_none=.60):
    f = tau / dt; fl = math.floor(f); frac = f - fl
    one = lambda n: A_none if n == 0 else auc_k(A_star, lam1, n)
    return (1 - frac) * one(fl) + frac * one(fl + 1)
def p_draws_ge(tau, dt, m):
    f = tau / dt; fl = math.floor(f); frac = f - fl
    return 1.0 if fl >= m else (frac if fl == m - 1 else 0.0)
DTS, TAUS = (1, 2, 3.5, 7, 14), (5, 10, 21)
prod = pd.DataFrame([dict(prodrome_days=tau, **{f'every {dt:g} d': round(expected_prodrome_auc(tau, dt), 3) for dt in DTS}) for tau in TAUS])
prod2 = pd.DataFrame([dict(prodrome_days=tau, **{f'every {dt:g} d': round(p_draws_ge(tau, dt, 2), 2) for dt in DTS}) for tau in TAUS])
print('EXPECTED AUROC of a pre-onset alarm (oracle 0.80, single-draw reliability 0.5, 0.60 when no draw lands in the prodrome):'); display(md_table(prod))
print('FRACTION OF CASES WITH >= 2 DRAWS INSIDE THE PRODROME:'); display(md_table(prod2))
''')

code(r'''
# ----------------------------------------------------------------------------- 9.5: the §4.4 schedule evaluated against onset times
VISIT_DAY = {'d-30/pre': -30, 'd0 graft': 0, 'd+7': 7, 'd+14': 14, 'd+21': 21, 'd+28': 28, 'd+60': 60, 'd+100': 100, 'd+180': 180, 'd+365': 365}
rng_on = np.random.default_rng(5); ONSET = np.clip(np.exp(rng_on.normal(math.log(28), .45, 200_000)), 8, 100)   # grade II-IV onset day: median 28, IQR ~ 21-38 †
WIN = 14
def pre_counts(days):
    d = np.asarray(days, float); return ((d[None, :] >= ONSET[:, None] - WIN) & (d[None, :] < ONSET[:, None])).sum(1)
DENSE = [-30, 0, 7, 14, 21, 28, 35, 42, 49, 56, 100, 180, 365]                        # weekly to day +56 for the nested trajectory cohort
SPEC_TYPES = [('Whole blood PAXgene RNA', 'Whole blood PAXgene RNA (2.5 mL)'), ('Plasma / serum', 'Plasma / serum (EDTA + serum, 2×1 mL aliquots)'),
              ('Stool', 'Stool (−80 °C, no preservative or OMNIgene)'), ('Flow phenotyping', 'Flow immune phenotyping (fresh, harmonised panel)'),
              ('Sorted T-cell RNA', 'Sorted T-cell subsets → RNA (CD4 Tconv/CD8/Treg)')]
srow = []
for short, full in SPEC_TYPES:
    row_ = sched[sched.specimen == full].iloc[0]; prot = [VISIT_DAY[c] for c in SCHED_DAYS if c in VISIT_DAY and row_[c] == 'X']
    for sch, days in (('protocol §4.4', prot), ('weekly to day +56', DENSE if short in ('Whole blood PAXgene RNA', 'Plasma / serum', 'Stool', 'Flow phenotyping') else prot)):
        c_ = pre_counts(days); srow.append(dict(specimen=short, schedule=sch, draws_per_patient_to_d100=int(sum(d_ <= 100 for d_ in days)),
                                                 P_at_least_1=float((c_ >= 1).mean()), P_at_least_2=float((c_ >= 2).mean())))
sdf = pd.DataFrame(srow)
USABLE, PREV_AG, N_CASES_TARGET = USABLE_FRACTION, .35, 100
sdf['patients to enrol for 100 cases with >= 2 pre-onset draws'] = [int(np.ceil(N_CASES_TARGET / (PREV_AG * USABLE * p))) if p > 0 else np.nan for p in sdf.P_at_least_2]
disp_sched = sdf.copy()
for c in ('P_at_least_1', 'P_at_least_2'): disp_sched[c] = disp_sched[c].map(lambda v: f'{v:.0%}')
disp_sched['patients to enrol for 100 cases with >= 2 pre-onset draws'] = disp_sched['patients to enrol for 100 cases with >= 2 pre-onset draws'].map(lambda v: 'not reachable' if pd.isna(v) else f'{v:,.0f}')
print(f'PRE-ONSET SPECIMENS (onset = day {np.median(ONSET):.0f} median, window = the {WIN} days before onset, 70 % usable, grade II-IV prevalence {PREV_AG}):'); display(md_table(disp_sched))
# B3 evaluation precision
b3_events = int(np.ceil((1.96 / .07) ** 2 * .25)); b3_alerts = int(np.ceil((1.96 / .2) ** 2))
print(f'B3 precision: sensitivity at a fixed alert budget to within ±0.07 needs {b3_events} onset events; an alert rate to within ±20 % needs {b3_alerts} alerts '
      f'= {100 * b3_alerts / alerts_per_100_patient_days(.5, .80, .04):,.0f} patient-days at an AUROC-0.80 alarm.')
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 14: longitudinal design
fig = plt.figure(figsize=(FULL, 3.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, .95], left=.075, right=.985, bottom=.265, top=.875, wspace=.55)
ax = fig.add_subplot(gs[0, 0]); ygrid(ax)                                       # (a) draws per window
kk = np.arange(1, 13)
for lam, c in zip((.3, .5, .7, .9), (PAL['seq'][0], PAL['seq'][1], PAL['seq'][2], PAL['seq'][3])):
    ax.plot(kk, [auc_k(.80, lam, k) for k in kk], color=c, lw=1.5, label=f'single-draw reliability {lam:g}')
    ax.plot(KDR, [sim_auc_k(.80, lam, k, n=60000) for k in KDR], 'o', ms=2.6, mfc='white', mec=c, mew=.8, ls='None')
ax.axhline(.80, color=PAL['ink2'], ls='--', lw=.6); ax.set_xlabel('draws averaged in the window $k$'); ax.set_ylabel('AUROC (oracle 0.80)'); ax.set_ylim(.62, .82); ax.set_xlim(.7, 12.3); ax.set_xticks([1, 2, 3, 5, 8, 12])
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=1, handlelength=1.4, labelspacing=.3); panel(ax, 'a')
ax = fig.add_subplot(gs[0, 1]); ygrid(ax)                                       # (b) how often
dts = np.geomspace(1, 14, 60)
for tau, c in zip(TAUS, (PAL['seq'][0], PAL['seq'][2], PAL['seq'][3])):
    ax.plot(dts, [expected_prodrome_auc(tau, d) for d in dts], color=c, lw=1.5, label=f'prodrome {tau} days')
ax.set_xscale('log'); ax.set_xticks([1, 2, 3.5, 7, 14]); ax.set_xticklabels(['1', '2', '3.5', '7', '14']); ax.minorticks_off()
ax.set_xlabel('interval between draws (days)'); ax.set_ylabel('expected AUROC of a pre-onset alarm'); ax.set_ylim(.58, .80)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=1, handlelength=1.4, labelspacing=.3); panel(ax, 'b')
ax = fig.add_subplot(gs[0, 2]); ygrid(ax, 'x')                                  # (c) specimens: cases with >= 2 pre-onset draws under two schedules
types = ['Whole blood PAXgene RNA', 'Plasma / serum', 'Stool', 'Flow phenotyping']; short_lab = ['blood RNA', 'plasma', 'stool', 'flow']; yy = np.arange(len(types)); hh = .36
for j, (sch, c) in enumerate((('protocol §4.4', PAL['seq'][1]), ('weekly to day +56', PAL['seq'][3]))):
    v = [float(sdf[(sdf.specimen == t) & (sdf.schedule == sch)].P_at_least_2.iloc[0]) for t in types]
    ax.barh(yy + (j - .5) * hh, v, height=hh, color=c, zorder=3, label=sch)
ax.set_yticks(yy); ax.set_yticklabels(short_lab); ax.invert_yaxis(); ax.tick_params(axis='y', length=0); ax.set_xlim(0, 1); ax.set_xlabel('cases with ≥ 2 specimens\nin the 14 days before onset')
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.30), ncol=1, handlelength=1.4, handletextpad=.4, labelspacing=.3); panel(ax, 'c')
finish(fig, 14, 'longitudinal_design',
       "Longitudinal sampling design. (a) AUROC against the number of draws averaged in a window, for four single-draw reliabilities (oracle AUROC 0.80); lines are "
       "the Spearman-Brown closed form, circles are simulation. (b) Expected AUROC of a pre-onset alarm against the interval between draws, for prodromes of 5, 10 "
       "and 21 days: sampling less often than the prodrome is long loses the trajectory. (c) The fraction of cases that have at least two specimens in the 14 days "
       "before onset under the minimum protocol of section 4.4 and under a weekly schedule to day +56, for each specimen type, against a realistic onset distribution.")
''')

code(r'''
WEEKLY_ENROL = int(sdf[(sdf.specimen == 'Whole blood PAXgene RNA') & (sdf.schedule == 'weekly to day +56')]['patients to enrol for 100 cases with >= 2 pre-onset draws'].iloc[0]); WEEKLY_N = int(round(WEEKLY_ENROL * USABLE_FRACTION))
display(Markdown(f"""**Reading §9.5.** (1) **Draws:** when a single draw is a noisy measurement (λ ≈ 0.3–0.5, typical of a dynamic blood signature), three to five draws in the window recover most of the oracle AUROC, and eight recover essentially all of it; when a draw is already reliable (λ ≥ 0.9), one suffices. Replicate draws are therefore worth buying exactly where the single-draw reliability is low — which is a *measurable* property of each assay (bridging and replicate samples, §4.4), not a guess. (2) **Interval:** the alarm degrades as the draw interval approaches the prodrome length; for a 10-day prodrome, draws every 7 or 14 days miss it in a large share of cases, while every 2–3.5 days keeps at least two draws inside it. (3) **Specimens:** the minimum protocol of §4.4 is designed for diagnosis and prognosis at fixed landmarks; it yields **few cases with two or more pre-onset specimens**, so it cannot by itself support trajectory tasks. A **nested trajectory cohort** (weekly to day +56 for blood RNA and plasma) is what makes B3 and the longitudinal objectives testable, and the table above gives the enrolment it needs for 100 evaluable cases: about **{WEEKLY_ENROL:,} patients enrolled** ({WEEKLY_N} usable) — inside the cohort of §9.9."""))
''')

# =====================================================================================================  9.6  HOW MANY CENTRES
md(r'''
## 9.6 How many centres, *if a network forms*? (context — not the ask)

**This subsection is about the benchmark's Level 2, which any group may attempt with a network of centres; the plan of this paper is a single institution (§9.8–9.9) and makes no out-of-centre claim.** Patients are not the only unit. A benchmark that claims *out-of-centre* generalisation has an effective sample size that is partly a **number of centres**, because centres differ in case mix, prophylaxis practice, grading habits and assay handling in ways a model cannot see. The paired gain of a candidate over a baseline therefore varies from centre to centre with some **between-centre standard deviation τ**, on top of the sampling noise within each centre. The gate of §8.12 averages over *K* held-out centres with a *t*-interval, so its power is

$$\Pr\big[\text{pass}\big]=\Pr\!\Big[t_{K-1}\!\big(\text{ncp}=\tfrac{\Delta-m}{\sqrt{(\tau^2+\mathrm{se}_c^2)/K}}\big)\ge t_{0.975,K-1}\Big],$$

with $m=+0.01$ the margin and $\mathrm{se}_c$ the standard error of a paired AUROC difference at the centre's sample size. This is closed-form; we check it against the §8.12 simulation, then solve for the number of centres.
''')

code(r'''
# ----------------------------------------------------------------------------- 9.6: centres needed for the gate (closed form, validated against the 8.12 simulation)
from scipy.stats import nct
def gate_power(K, n_c, d_true, tau, prev=.35, A1=.70, rho=.85):
    """Power of the adopted gate (t-interval lower bound >= margin) with K centres of n_c patients each."""
    m = GATE['level2_gate']['ci_lower_bound_min']; se_c = se_delta(n_c, prev, A1, A1 + d_true, rho); se = math.sqrt((tau ** 2 + se_c ** 2) / K)
    return float(nct.sf(student_t.ppf(.975, K - 1), K - 1, (d_true - m) / se))
def K_required(d_true, tau, n_c, power=.8, prev=.35, A1=.70, rho=.85, Kmax=400):
    for K in range(3, Kmax):
        if gate_power(K, n_c, d_true, tau, prev, A1, rho) >= power: return K
    return None
chk = pd.DataFrame([dict(K=K, closed_form=gate_power(K, 300, .03, .02), simulated=float(oc[(oc.K == K) & (oc.d_true == .03)].pass_adopted.iloc[0])) for K in (12, 20, 30, 50)])
chk['diff'] = chk.simulated - chk.closed_form
print('closed-form power of the adopted gate vs the §8.12 simulation (Δ = 0.03, τ = 0.02, 300 patients per centre; the simulation also requires 70 % of centres improved):'); display(md_table(chk.round(3)))
assert chk['diff'].abs().max() < .09, 'closed-form gate power disagrees with the simulation'
krow = []
for d in (.03, .04, .05):
    for nc in (150, 300, 600):
        krow.append({'true gain': d, 'patients per centre': nc, **{f'τ = {tau}': K_required(d, tau, nc) for tau in (.01, .02, .03, .04)}})
ktab = pd.DataFrame(krow)
print('CENTRES REQUIRED for 80 % power of the adopted gate:'); display(md_table(ktab))
# what the simulated harness cohort of 8.11 implies for tau (it is a planted property of the simulation, not a measurement of the real world)
dvec = (piv['full (clinical + day-7 + biomarkers)'] - piv['clinical baseline (registry covariates)']).dropna().values
se_c_sim = float(np.mean([se_delta(int(n_), .35, .70, .75, .85) for n_ in COH.groupby('centre').size().values]))
tau_sim = math.sqrt(max(dvec.var(ddof=1) - se_c_sim ** 2, 0.0))
print(f'[SIM] the 8.11 cohort: SD of the per-centre gain {dvec.std(ddof=1):.3f}, expected sampling SD {se_c_sim:.3f} -> implied between-centre tau ≈ {tau_sim:.3f}')
# design effect for centre-level exposures (practice, policy): patients are clustered in centres
dex = pd.DataFrame([{'intra-centre correlation': icc, **{f'{m} patients per centre': f'{1 + (m - 1) * icc:.1f}' for m in (100, 300, 1000)}} for icc in (.01, .03, .05)])
print('DESIGN EFFECT 1 + (m - 1) × ICC for any effect that varies BETWEEN centres (prophylaxis policy, biopsy habit); effective n = n / DE:'); display(md_table(dex))
print(f'with 20 centres of 300 patients each (no centre holds more than the one-institution plan) and an ICC of 0.03, a centre-level effect has an effective sample of {20 * 300 / (1 + 299 * .03):.0f} patients; the pooled count is never needed in one place, because each centre can run the harness locally and share only summary statistics (§11).')
''')

code(r'''
k_tab = lambda d, nc, tau: int(ktab[(ktab['true gain'] == d) & (ktab['patients per centre'] == nc)][f'τ = {tau}'].iloc[0])
display(Markdown(f"""**Reading §9.6.** With the planning value τ = 0.02 and centres of ~300 patients, the closed form says **{k_tab(.03, 300, .02)} centres** give the adopted gate 80 % power for a +0.03 gain, and the full simulation of §8.12 — which also requires 70 % of centres to improve and lets centre sizes vary — says **20** (power {g20(.03, 'pass_adopted'):.2f}); the gate's minimum is set to the more conservative 20. The table shows how fast the requirement moves: at τ = 0.04 the same gain needs **{k_tab(.03, 300, .04)}** centres, and **adding patients within existing centres barely helps** once the between-centre term dominates (at τ = 0.04, going from 150 to 600 patients per centre lowers the requirement only from {k_tab(.03, 150, .04)} to {k_tab(.03, 600, .04)} centres). τ is a property of the real world that no simulation can supply (the simulated cohort of §8.11 happens to imply about {tau_sim:.2f}); **stage 1 of the roadmap measures it**, and the required number of centres is then read from this table. The design effect makes the same point for any *centre-level* explanatory variable — prophylaxis platform, biopsy policy, grading habit: their effective sample size is closer to the number of centres than to the number of patients."""))
''')

# =====================================================================================================  9.7  VALIDATION, SUBGROUPS, LABEL NOISE
md(r'''
## 9.7 How many patients to validate a claim, to audit a subgroup, and when the labels are imperfect

Three further requirements that decide whether a "Level 3" claim is believable.

**Validation sample size (Level 3).** Discrimination is the easy part. For a calibrated model the *calibration slope* is the quantity that needs the most patients, because its information comes from the spread of predicted risks. For a logistic recalibration $y\sim a+b\,\mathrm{lp}$ at $(a,b)=(0,1)$, the large-sample variance of $\hat b$ is the corresponding element of the inverse Fisher information, which we integrate over the distribution of predicted risks implied by a binormal score. We ask for SE(slope) ≤ 0.10 (so the 95 % interval is ±0.2), SE(ln O/E) ≤ 0.10, and an AUROC half-width ≤ 0.05.

**Subgroups (task G3).** A subgroup of share *s* of the cohort has *s·N* patients. We ask how large *N* must be for the subgroup to support (i) an AUROC half-width ≤ 0.07, (ii) SE(calibration slope) ≤ 0.15, and (iii) detection of an equal-opportunity gap of 0.10 (or 0.05) in sensitivity at fixed specificity (80 % power, sensitivities around 0.7).

**Label noise.** If the recorded label has sensitivity *se* and specificity *sp* against the truth (inter-centre grading differences, missed biopsies), a score with true AUROC *A* has observed AUROC
$$A_{\rm obs}=\frac{w_{p1}w_{n0}\,A+w_{p0}w_{n1}\,(1-A)+\tfrac12\,(w_{p1}w_{n1}+w_{p0}w_{n0})}{W_pW_n},$$
with $w_{p1}=\pi\,se,\;w_{p0}=(1-\pi)(1-sp),\;w_{n1}=\pi(1-se),\;w_{n0}=(1-\pi)\,sp$. Both models in a comparison are attenuated, so the observable gain shrinks and the required *n* grows.
''')

code(r'''
# ----------------------------------------------------------------------------- 9.7: validation n, subgroup precision, label-noise attenuation
@functools.lru_cache(maxsize=None)
def _val_parts(A, prev, N=800_000, seed=1):
    r = np.random.default_rng(seed); D = math.sqrt(2) * norm.ppf(A); y = r.random(N) < prev
    x = np.where(y, r.normal(D, 1, N), r.normal(0, 1, N)); lp = math.log(prev / (1 - prev)) + D * x - D * D / 2
    p = 1 / (1 + np.exp(-lp)); w = p * (1 - p)
    I = np.array([[w.mean(), (w * lp).mean()], [(w * lp).mean(), (w * lp * lp).mean()]]); return float(np.linalg.inv(I)[1, 1]), float(w.mean())
def validation_n(A, prev, se_slope=.10, se_oe=.10, hw_auc=.05):
    """Patients so that SE(calibration slope) <= se_slope, SE(ln O/E) <= se_oe and the AUROC half-width <= hw_auc (large-sample)."""
    ib, ew = _val_parts(round(A, 3), round(prev, 3))
    n_slope, n_citl, n_oe, n_auc = ib / se_slope ** 2, 1 / (ew * se_oe ** 2), (1 - prev) / (prev * se_oe ** 2), n_for_ci(A, prev, hw_auc)
    comp = dict(slope=n_slope, intercept=n_citl, oe=n_oe, auroc=n_auc); k = max(comp, key=comp.get)
    return dict(n=int(math.ceil(comp[k])), events=int(math.ceil(comp[k] * prev)), binding=k)
def sim_slope_se(A, prev, n, reps=300, seed=2):
    r = np.random.default_rng(seed); D = math.sqrt(2) * norm.ppf(A); out = []
    for _ in range(reps):
        y = (r.random(n) < prev).astype(int); x = np.where(y == 1, r.normal(D, 1, n), r.normal(0, 1, n)); lp = math.log(prev / (1 - prev)) + D * x - D * D / 2
        m = LogisticRegression(C=1e6, max_iter=500).fit(lp[:, None], y); out.append(m.coef_[0, 0])
    return float(np.std(out))
fi_se = math.sqrt(_val_parts(.75, .15)[0] / 1500); sm_se = sim_slope_se(.75, .15, 1500, reps=150 if QUICK else 300)
print(f'calibration-slope SE at n = 1500 (AUROC 0.75, prevalence 0.15): Fisher {fi_se:.3f} vs simulated {sm_se:.3f}')
assert abs(fi_se - sm_se) / fi_se < (.25 if QUICK else .15)
vtab = pd.DataFrame([dict(AUROC=A, prevalence=pv, **validation_n(A, pv)) for A in (.70, .75, .80) for pv in (.10, .20, .35, .50)])
vtab = vtab.rename(columns={'n': 'patients', 'binding': 'binding criterion'})
vtab['years of silent-mode at one centre'] = (vtab.patients / TRANSPLANTS_PER_YEAR).round(1)
v_show = vtab.copy(); v_show['patients'] = v_show.patients.map(capn)
print(f'PATIENTS (AND EVENTS) FOR A LEVEL-3 SILENT-MODE VALIDATION — all transplants of the centre count, so {TRANSPLANTS_PER_YEAR} a year ("> {N_CAP:,}" = more than the ceiling cohort):'); display(md_table(v_show))

# ---------- subgroups at one institution (shares are planning assumptions †)
SUBGROUPS = [('cord-blood graft', .05), ('Black / African ancestry (US centre)', .06), ('Asian / Pacific Islander (US centre)', .07),
             ('Hispanic / Latino (US centre)', .10), ('paediatric (< 18 y)', .15), ('haploidentical donor', .20)]
A_sg, pv_sg = .72, .35
n_gap_events = {g: int(np.ceil((norm.ppf(.975) + norm.ppf(.8)) ** 2 * (.70 * .30 + (.70 - g) * (.30 + g)) / g ** 2)) for g in (.10, .05)}
n_gap_pat = {g: int(np.ceil(e / pv_sg)) for g, e in n_gap_events.items()}
sg_base = max(n_for_ci(A_sg, pv_sg, .07), int(np.ceil(_val_parts(A_sg, pv_sg)[0] / .15 ** 2)))
sg = pd.DataFrame([{'subgroup': name, 'share †': f'{s:.0%}', f'patients at {N_CAP:,}': int(s * N_CAP),
                    'AUROC to ±0.07 and slope SE 0.15 need': f'{sg_base:,}', 'a 0.10 sensitivity gap needs': f'{max(sg_base, n_gap_pat[.10]):,}',
                    'assessable at one centre?': 'no — descriptive only' if s * N_CAP < max(sg_base, n_gap_pat[.10]) else 'yes'} for name, s in SUBGROUPS])
print(f'SUBGROUPS AT ONE INSTITUTION (AUROC {A_sg}, prevalence {pv_sg}; an equal-opportunity gap of 0.10 in sensitivity needs {n_gap_events[.10]} events per group = {n_gap_pat[.10]:,} patients in the subgroup, a gap of 0.05 needs {n_gap_pat[.05]:,}):'); display(md_table(sg))

# ---------- label noise
def auc_obs(A, prev, se, sp):
    wp1, wp0, wn1, wn0 = prev * se, (1 - prev) * (1 - sp), prev * (1 - se), (1 - prev) * sp
    return (wp1 * wn0 * A + wp0 * wn1 * (1 - A) + .5 * (wp1 * wn1 + wp0 * wn0)) / ((wp1 + wp0) * (wn1 + wn0))
def sim_auc_obs(A, prev, se, sp, n=400000, seed=4):
    r = np.random.default_rng(seed); D = math.sqrt(2) * norm.ppf(A); y = (r.random(n) < prev).astype(int); s_ = np.where(y == 1, r.normal(D, 1, n), r.normal(0, 1, n))
    yo = np.where(y == 1, r.random(n) < se, r.random(n) < 1 - sp).astype(int); return float(roc_auc_score(yo, s_))
assert abs(auc_obs(.75, .35, .9, .9) - sim_auc_obs(.75, .35, .9, .9)) < .004, 'label-noise attenuation formula disagrees with simulation'
lrow = []
for se_, sp_ in ((1.0, 1.0), (.95, .95), (.90, .90), (.85, .85), (.80, .95)):
    po = .35 * se_ + .65 * sp_; pexp = (.35 * se_ + .65 * (1 - sp_)) * .35 + (.35 * (1 - se_) + .65 * sp_) * .65; kappa = (po - pexp) / (1 - pexp)
    a0, a1 = auc_obs(.72, .35, se_, sp_), auc_obs(.75, .35, se_, sp_)
    lrow.append({'label sensitivity': se_, 'label specificity': sp_, 'kappa vs truth': round(kappa, 2), 'observed AUROC of a true 0.72': round(a0, 3), 'observed AUROC of a true 0.75': round(a1, 3),
                 'observed gain of a true +0.03': round(a1 - a0, 3), 'n to show it vs noise-free n': round(n_for_delta(a0, a1 - a0, .35) / n_for_delta(.72, .03, .35), 2)})
ltab = pd.DataFrame(lrow); print('LABEL NOISE (prevalence 0.35):'); display(md_table(ltab))
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 15: centres (if a network forms), validation size and subgroup size
fig = plt.figure(figsize=(FULL, 3.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0], left=.075, right=.985, bottom=.265, top=.875, wspace=.42)
ax = fig.add_subplot(gs[0, 0]); ygrid(ax)                                       # (a) centres needed for the gate
taus = np.linspace(.005, .05, 30)
for nc, c in zip((150, 300, 600), (PAL['seq'][0], PAL['seq'][2], PAL['seq'][3])):
    ax.plot(taus, [K_required(.03, t, nc) for t in taus], color=c, lw=1.5, label=f'{nc} patients per centre')
ax.axhline(GATE['level2_gate']['min_centres'], color=PAL['bad'], lw=.8, ls=':', label='gate minimum (20)')
ax.set_xlabel('between-centre SD of the gain, $\\tau$'); ax.set_ylabel('centres for 80% power (gain +0.03)'); ax.set_xlim(.005, .05)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2, columnspacing=1.0, handlelength=1.4, handletextpad=.4, labelspacing=.3); panel(ax, 'a')
ax = fig.add_subplot(gs[0, 1]); ygrid(ax)                                       # (b) Level-3 validation size
prevs = np.geomspace(.10, .5, 24)
for A, c in zip((.70, .75, .80), (PAL['seq'][0], PAL['seq'][2], PAL['seq'][3])):
    ax.plot(prevs, [validation_n(A, pv)['n'] for pv in prevs], color=c, lw=1.5, label=f'AUROC {A:.2f}')
ax.axhline(N_CAP, color=PAL['bad'], lw=.8, ls=':', label=f'ceiling cohort ({N_CAP:,})'); ax.axhline(TRANSPLANTS_PER_YEAR * 3, color=PAL['ink2'], lw=.8, ls='--', label='3 years of silent mode (600)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([.1, .2, .35, .5]); ax.set_xticklabels(['0.1', '0.2', '0.35', '0.5']); ax.minorticks_off()
ax.set_yticks([300, 600, 1000, 1500, 2500]); ax.set_yticklabels(['300', '600', '1,000', '1,500', '2,500']); ax.set_ylim(300, 2800)
ax.set_xlabel('outcome prevalence'); ax.set_ylabel('patients for a Level-3 validation')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2, columnspacing=1.0, handlelength=1.4, handletextpad=.4, labelspacing=.3); panel(ax, 'b')
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                       # (c) subgroup size at one centre against what an equity audit needs
Ns = np.linspace(150, N_CAP, 30)
for s, c in ((.05, PAL['seq'][0]), (.10, PAL['seq'][1]), (.20, PAL['seq'][3])):
    ax.plot(Ns, s * Ns, color=c, lw=1.5, label=f'subgroup of {s:.0%}')
ax.axhline(max(sg_base, n_gap_pat[.10]), color=PAL['bad'], lw=.9, ls=':', label=f'needed for a 0.10 gap ({max(sg_base, n_gap_pat[.10]):,})')
ax.set_xlabel('patients at the centre'); ax.set_ylabel('patients in the subgroup'); ax.set_xlim(150, N_CAP); ax.set_ylim(0, 1150)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2, columnspacing=1.0, handlelength=1.4, handletextpad=.4, labelspacing=.3); panel(ax, 'c')
finish(fig, 15, 'centres_validation_subgroups',
       "Centres, validation, and subgroups. (a) Context for the benchmark's Level 2, not the plan of this paper: centres needed for the gate to have 80% power for a true gain of +0.03, against the "
       "between-centre SD of the gain, for three centre sizes; once heterogeneity dominates, more centres help and more patients per centre do not. (b) Patients for a Level-3 silent-mode validation "
       "(calibration slope SE 0.10, O/E SE 0.10, AUROC half-width 0.05) against prevalence, with the 1,500-patient ceiling and three years of one centre's transplants marked. (c) The size of a "
       "subgroup at one centre against the roughly 1,000 patients that detecting a 0.10 sensitivity gap needs: no subgroup of a single centre reaches it, so a single-centre equity audit is descriptive.")
''')

# =====================================================================================================  9.8  WHAT ONE INSTITUTION CAN RUN
md(r'''
## 9.8 What can one institution run?

This is the subsection that matters for the plan. It takes the calculators above and asks them, for the cohort sizes of one centre, what can be **learned** (the largest head), **shown** (the smallest detectable gain) and **estimated** (the width of an interval), and then goes through the benchmark task by task. The sizes are those a centre actually has: the biopsies it collects in three years, the patients with banked specimens, the temporal test set and development set of a 60/40 split of the ceiling cohort, and the ceiling cohort itself.
''')

code(r'''
# ----------------------------------------------------------------------------- 9.8 (A) capacity of the cohort sizes one centre has  (Table 9.1)
N_BIOPSY = int(SPECIMEN_YEARS * TRANSPLANTS_PER_YEAR * .20)                 # patients biopsied for suspected GvHD over three years (about a fifth of transplants) †
N_DEV, N_TEST = int(.6 * N_CAP), int(.4 * N_CAP)
SIZES = [(f'biopsies in {SPECIMEN_YEARS} years', N_BIOPSY), ('patients with banked specimens', N_SPEC), ('temporal test set (latest 40 %)', N_TEST), ('temporal development set (earliest 60 %)', N_DEV), ('ceiling cohort', N_CAP)]
cap_rows = []
for lab, n in SIZES:
    n1 = int(n * .35)
    cap_rows.append({'cohort': lab, 'patients': n, '95% CI half-width of one AUROC (0.72, prev 0.35)': round(1.96 * hm_se(.72, n1, n - n1), 3),
                     'smallest paired gain shown (80% power, r = 0.85, baseline 0.70)': round(min_detectable_gain(n, .70, .35), 3),
                     'largest head the cohort supports (AUROC 0.72)': max_params(n, .72, .35)})
cap_si = pd.DataFrame(cap_rows)
print('TABLE 9.1 — WHAT THE COHORT SIZES OF ONE CENTRE CAN LEARN, SHOW AND ESTIMATE:'); display(md_table(cap_si))
mg = lambda n: min_detectable_gain(n, .70, .35)
''')

code(r'''
# ----------------------------------------------------------------------------- 9.8 (B) the benchmark, task by task, at one centre
feas = cards_df[cards_df.one_inst_verdict != '-'][['id', 'readiness', 'tier needed', 'one_inst_n', 'max_head_params', 'min_detectable_gain', 'ci_halfwidth', 'one_inst_verdict']].copy()
feas = feas.rename(columns={'one_inst_n': 'patients at one centre', 'max_head_params': 'largest head', 'min_detectable_gain': 'smallest gain shown', 'ci_halfwidth': 'AUROC ± (95%)', 'one_inst_verdict': 'verdict at one centre'})
feas['patients at one centre'] = feas['patients at one centre'].astype(int)
print(f"TASK FEASIBILITY AT ONE INSTITUTION ({N_CAP:,} EHR timelines, {N_SPEC} with specimens; RUNNABLE = a head of at least 5 parameters and a smallest detectable gain of at most 0.05; ESTIMABLE = AUROC to ±0.10 but no comparison; otherwise OUT OF REACH):")
display(md_table(feas))
vs_ids = {v: cards_df[cards_df.one_inst_verdict == v].id.tolist() for v in ('RUNNABLE', 'ESTIMABLE', 'OUT OF REACH')}
display(Markdown(f"""**Result.** Of the {int((cards_df.one_inst_verdict != '-').sum())} tasks with a numeric target, **{len(vs_ids['RUNNABLE'])} are runnable at one institution** ({', '.join(vs_ids['RUNNABLE'])}), **{len(vs_ids['ESTIMABLE'])} are estimable but cannot support a comparison** ({', '.join(vs_ids['ESTIMABLE'])}), and **{len(vs_ids['OUT OF REACH'])} are out of reach** ({', '.join(vs_ids['OUT OF REACH'])}). The runnable tasks are the EHR-timeline tasks of families A and E and of B6, C3 and D4, plus B1 on the {N_SPEC} patients with specimens; the estimable ones are the specimen-dependent response and prognosis tasks, which a single centre can *describe* with an honest interval and which pooling with other centres would turn into comparisons."""))
''')

code(r'''
# ----------------------------------------------------------------------------- 9.8 (C) transcriptome claims at one centre
ONE = lambda tid: int(cards_df[cards_df.id == tid].one_inst_n.iloc[0])
TC_NUM = []                                                                 # (learn, show, need) of each fitted-head claim, in order
def claim_row(claim, head, p_eff, A0, A1, prev, gain, avail, kappa=.9):
    """A claim 'a head of p_eff parameters adds `gain` AUROC over baseline A0' needs the head to LEARN that gain (learning curve) and the cohort to SHOW it (paired DeLong)."""
    n_learn = n_for_target(A0 + gain, A0, A1, p_eff, prev, 1.0, 8, kappa); n_show = n_for_delta(A0, gain, prev)
    need = max(n_learn, n_show) if np.isfinite(n_learn) else float('inf'); TC_NUM.append((n_learn, n_show, need))
    return {'claim': claim, 'head (effective parameters)': head, 'to learn it': capn(int(math.ceil(n_learn))) if np.isfinite(n_learn) else 'never', 'to show it': capn(n_show),
            'patients needed': capn(int(math.ceil(need))) if np.isfinite(need) else 'never', 'patients available': avail,
            'feasible at one centre?': 'yes' if avail >= need else ('marginal' if avail >= .7 * need else 'no')}
TC_ROWS = [
 {'claim': 'validate a pre-specified IFN-γ / allograft-rejection tissue score to ±0.10 (AUROC 0.85, prevalence 0.5)', 'head (effective parameters)': '0 (a fixed score)', 'to learn it': '—', 'to show it': capn(n_for_ci(.85, .5, .10)),
  'patients needed': capn(n_for_ci(.85, .5, .10)), 'patients available': N_BIOPSY, 'feasible at one centre?': 'yes' if N_BIOPSY >= n_for_ci(.85, .5, .10) else 'no'},
 {'claim': 'same, to ±0.07', 'head (effective parameters)': '0 (a fixed score)', 'to learn it': '—', 'to show it': capn(n_for_ci(.85, .5, .07)),
  'patients needed': capn(n_for_ci(.85, .5, .07)), 'patients available': N_BIOPSY, 'feasible at one centre?': 'yes' if N_BIOPSY >= n_for_ci(.85, .5, .07) else ('marginal' if N_BIOPSY >= .7 * n_for_ci(.85, .5, .07) else 'no')},
 claim_row('tissue differential diagnosis (C1/C2): +0.05 over a 0.75 clinical baseline', '10, frozen adapter', 10, .75, .86, .45, .05, ONE('C1')),
 claim_row('tissue differential diagnosis (C1/C2): +0.05', '64, adapter embedding', 64, .75, .86, .45, .05, ONE('C1')),
 claim_row('steroid refractoriness (C4): +0.05 over a 0.66 baseline', '10, frozen adapter', 10, .66, .74, .30, .05, ONE('C4')),
 claim_row('steroid refractoriness (C4): +0.03', '10, frozen adapter', 10, .66, .74, .30, .03, ONE('C4')),
 claim_row('day-7 blood transcriptome: +0.03 over MAP + clinical (0.72)', '10, frozen adapter', 10, .72, .76, .35, .03, ONE('B1')),
 {'claim': 'any claim from a head on the whole transcriptome', 'head (effective parameters)': '≈ 40,000 effective', 'to learn it': f'> {N_CAP:,}', 'to show it': '—', 'patients needed': f'> {N_CAP:,}', 'patients available': N_CAP, 'feasible at one centre?': 'no'}]
tc = pd.DataFrame(TC_ROWS)
print('TRANSCRIPTOME CLAIMS AT ONE INSTITUTION:'); display(md_table(tc))
_fmt = lambda x: capn(int(math.ceil(x))) if np.isfinite(x) else 'never'
display(Markdown(f"""**Reading.** (1) A **pre-specified** tissue score has no fitted parameters, so it needs only enough biopsies to *estimate* its AUROC: {n_for_ci(.85, .5, .10)} for ±0.10 and {n_for_ci(.85, .5, .07)} for ±0.07, against {N_BIOPSY} biopsies in three years — **feasible, and the cleanest transcriptome claim one centre can make.** (2) **Tissue differential diagnosis (C1/C2):** a 10-parameter head *learns* +0.05 over a 0.75 clinical baseline from {_fmt(TC_NUM[0][0])} profiles, but *showing* the gain takes {_fmt(TC_NUM[0][1])}, against {ONE('C1')} eligible patients; a 64-dimensional head needs {_fmt(TC_NUM[1][2])}. (3) **Steroid refractoriness (C4)** needs {_fmt(TC_NUM[2][2])} profiles for +0.05 and {_fmt(TC_NUM[3][2])} for +0.03, against {ONE('C4')}; the **day-7 blood transcriptome** needs {_fmt(TC_NUM[4][2])} for +0.03 over MAP plus clinical covariates, against {ONE('B1')}. Neither is feasible at one centre. (4) **A head on the whole transcriptome** is out of reach. The honest single-centre transcriptome result is therefore: validate a pre-specified tissue score, report the differential-diagnosis gain with its interval, and describe the rest."""))

# ----------------------------------------------------------------------------- 9.8 (D) longitudinal claims at one centre
P2 = float(sdf[(sdf.specimen == 'Whole blood PAXgene RNA') & (sdf.schedule == 'weekly to day +56')].P_at_least_2.iloc[0])
lg_rows = []
for n in (150, 300, N_SPEC):
    cases = int(n * .35); ev = int(cases * P2)
    lg_rows.append({'patients with weekly draws to day +56': n, 'expected cases': cases, 'cases with ≥ 2 pre-onset draws': ev, 'AUROC ± (95%) with these cases': round(1.96 * hm_se(.75, ev, n - cases), 3)})
lg = pd.DataFrame(lg_rows)
def icc_n(icc, k=3, se=.05):
    """Patients with k draws in a clinically stable window so that the intraclass correlation (single-draw reliability) has standard error se (Fisher approximation)."""
    return int(math.ceil(1 + 2 * (1 - icc) ** 2 * (1 + (k - 1) * icc) ** 2 / (k * (k - 1) * se ** 2)))
icc_tab = pd.DataFrame([{'single-draw reliability': r, 'patients with 3 draws for SE 0.05': icc_n(r), 'with 5 draws': icc_n(r, 5)} for r in (.3, .5, .7)])
print('LONGITUDINAL (weekly draws to day +56) AT ONE INSTITUTION:'); display(md_table(lg))
print('MEASURING THE ASSAY RELIABILITY THAT §9.5 NEEDS (replicate draws in a stable window):'); display(md_table(icc_tab))
''')

code(r'''
# ----------------------------------------------------------------------------- 9.8 (E) the single-centre gate (Level 1C): operating characteristics
def level1c_power(n_test, delta, m=None, prev=.35, A1=.70, rho=.85):
    """P(lower 95 % bound of the paired AUROC difference >= m) on n_test temporal-test patients (closed form)."""
    m = GATE['level1c_single_centre']['ci_lower_bound_min'] if m is None else m
    return float(norm.cdf((delta - m) / se_delta(n_test, prev, A1, A1 + delta, rho) - 1.96))
def delong_est(y, s1, s2):
    """(difference of AUROCs, its DeLong standard error) for two scores on the same patients."""
    m, n = int((y == 1).sum()), int((y == 0).sum()); aucs, v01, v10 = [], [], []
    for s in (s1, s2):
        X, Y = s[y == 1], s[y == 0]; tx, ty, tz = rankdata(X), rankdata(Y), rankdata(np.r_[X, Y])
        aucs.append(tz[:m].sum() / (m * n) - (m + 1) / (2 * n)); v01.append((tz[:m] - tx) / n); v10.append(1 - (tz[m:] - ty) / m)
    S = np.cov(np.vstack(v01)) / m + np.cov(np.vstack(v10)) / n
    return aucs[1] - aucs[0], math.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-12))
def sim_level1c(n_test, delta, reps, prev=.35, A1=.70, rho=.85, seed=0):
    rng = np.random.default_rng(seed); n1 = int(round(n_test * prev)); n0 = n_test - n1; d1, d2 = math.sqrt(2) * norm.ppf(A1), math.sqrt(2) * norm.ppf(A1 + delta)
    cov = np.array([[1, rho], [rho, 1]]); hit = 0; m = GATE['level1c_single_centre']['ci_lower_bound_min']
    for _ in range(reps):
        sp = rng.multivariate_normal([d1, d2], cov, n1); sn = rng.multivariate_normal([0, 0], cov, n0); y = np.r_[np.ones(n1), np.zeros(n0)]
        dd, se = delong_est(y, np.r_[sp[:, 0], sn[:, 0]], np.r_[sp[:, 1], sn[:, 1]]); hit += (dd - 1.96 * se) >= m
    return hit / reps
reps_ = 400 if not QUICK else 60
chk1c = pd.DataFrame([dict(n_test=N_TEST, true_gain=d, closed_form=level1c_power(N_TEST, d), simulated=sim_level1c(N_TEST, d, reps_)) for d in (.0, .03, .05)])
chk1c['diff'] = chk1c.simulated - chk1c.closed_form
print('Level-1C gate: closed-form pass probability vs simulation:'); display(md_table(chk1c.round(3)))
assert chk1c["diff"].abs().max() < (.08 if not QUICK else .16), 'closed-form Level-1C power disagrees with simulation'
DS1C = (0., .01, .02, .03, .04, .05, .06)
oc1c = pd.DataFrame([{'true gain': d, **{f'n_test = {n}': round(level1c_power(n, d), 2) for n in (400, N_TEST, 800)}} for d in DS1C])
print(f'LEVEL-1C GATE (paired lower bound ≥ +0.01 on a temporal test set) — probability of passing:'); display(md_table(oc1c))
def n_test_for(delta, power=.8):
    for n in range(100, 5000, 10):
        if level1c_power(n, delta) >= power: return n
n1c = {d: n_test_for(d) for d in (.04, .05, .06)}
print('test patients for 80 % pass probability:', {d: n1c[d] for d in n1c})
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 16: what one institution can resolve
SZ = [N_BIOPSY, N_SPEC, N_TEST, N_DEV, N_CAP]; SZL = [f'{N_BIOPSY}', f'{N_SPEC}', f'{N_TEST}', f'{N_DEV}', f'{N_CAP:,}']
fig = plt.figure(figsize=(FULL, 3.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], left=.075, right=.985, bottom=.285, top=.875, wspace=.42)
nn = np.geomspace(100, N_CAP, 40)
ax = fig.add_subplot(gs[0, 0]); ygrid(ax)                                       # (a) what the sizes resolve
ax.plot(nn, [min_detectable_gain(int(n), .70, .35) for n in nn], color=PAL['seq'][3], lw=1.6, label='smallest paired gain shown (80% power)')
ax.plot(nn, [1.96 * hm_se(.72, int(n * .35), int(n) - int(n * .35)) for n in nn], color=PAL['orange'], lw=1.5, ls='--', label='95% CI half-width of one AUROC')
ax.set_xscale('log'); ax.set_xticks(SZ); ax.set_xticklabels(SZL, fontsize=6.0); ax.minorticks_off(); ax.set_xlim(100, N_CAP * 1.05); ax.set_ylim(0, .12)
ax.set_xlabel('patients in the cohort'); ax.set_ylabel('AUROC units'); panel(ax, 'a'); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.34), ncol=1, handlelength=1.4, labelspacing=.3)
ax = fig.add_subplot(gs[0, 1]); ygrid(ax)                                       # (b) largest head
for A, c in zip((.68, .72, .75), (PAL['seq'][0], PAL['seq'][2], PAL['seq'][3])):
    ax.plot(nn, [max_params(int(n), A, .35) for n in nn], color=c, lw=1.5, label=f'target AUROC {A:.2f}')
ax.set_xscale('log'); ax.set_xticks(SZ); ax.set_xticklabels(SZL, fontsize=6.0); ax.minorticks_off(); ax.set_xlim(100, N_CAP * 1.05); ax.set_ylim(0, 32)
ax.set_xlabel('patients in the cohort'); ax.set_ylabel('largest head the cohort supports\n(parameters)'); panel(ax, 'b'); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.34), ncol=1, handlelength=1.4, labelspacing=.3)
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                       # (c) the single-centre gate
dg = np.linspace(0, .07, 50)
for n, c in ((400, PAL['seq'][0]), (N_TEST, PAL['seq'][2]), (800, PAL['seq'][3])):
    ax.plot(dg, [level1c_power(n, d) for d in dg], color=c, lw=1.5, label=f'{n} test patients')
ax.axhline(.8, color=PAL['bad'], ls=':', lw=.8); ax.axhline(.05, color=PAL['bad'], ls=':', lw=.8); ax.axvline(.05, color=PAL['ink2'], ls=':', lw=.7)
ax.set_xlabel('true gain in AUROC'); ax.set_ylabel('probability of passing Level 1C'); ax.set_xlim(0, .07); ax.set_ylim(-.02, 1.03)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=1, handlelength=1.4, labelspacing=.3); panel(ax, 'c')
finish(fig, 16, 'one_institution_capacity',
       "What one institution can resolve. (a) The smallest paired AUROC gain over a correlated baseline shown with 80% power (solid) and the half-width of one AUROC's 95% interval (dashed), at "
       "the cohort sizes a centre actually has: 120 biopsies, 420 patients with banked specimens, a 600-patient temporal test set, a 900-patient development set, and the 1,500-patient ceiling. (b) The largest head, in candidate parameters, that those cohorts support: "
       f"about {max_params(N_SPEC, .72, .35)} on the {N_SPEC} patients with specimens and about {max_params(N_DEV, .72, .35)} on the {N_DEV}-patient development set (AUROC 0.72, prevalence 0.35). (c) Operating characteristics of the single-centre gate (lower bound of the paired interval at least +0.01 on a temporal test set): at "
       f"{N_TEST} test patients it passes a true gain of +0.05 {level1c_power(N_TEST, .05):.0%} of the time, +0.03 {level1c_power(N_TEST, .03):.0%} and +0.01 or less about {level1c_power(N_TEST, .01):.0%}. Dotted lines mark 80% and 5%.")
''')

# =====================================================================================================  9.9  THE ONE-INSTITUTION DATA PLAN
md(r'''
## 9.9 The one-institution data plan

Three nested packages, each a subset of the next, none above the 1,500-patient ceiling. **S0** needs no new specimens and can start now. **S1** adds specimens banked on the routine blood draws of the next three years. **S2** adds a weekly trajectory cohort inside S1. Specimen counts are computed from the schedule of §4.4, the onset distribution of §9.5 and a case-cohort design for sequencing (every case and a random subcohort of controls), which is the standard way to keep an expensive assay proportional to the number of events rather than the number of patients. No costs are estimated here: the paper's argument does not depend on them, and an institution can price its own assays.
''')

code(r'''
# ----------------------------------------------------------------------------- 9.9: the three nested packages
PREV_AG_ = .35; CASES = int(N_SPEC * PREV_AG_); CTRL_SUB = 100; W_CASES = int(WEEKLY_N * PREV_AG_); W_CTRL = 60
GB_RNA, GB_TISSUE = 6, 8                                                          # FASTQ per bulk RNA-seq library; RNA-seq + whole-slide image per biopsy (GB) †
s1_plasma = N_SPEC * 3 + CASES                                                    # d+7, d+14, d+28 on everyone, plus the onset sample of every case
s1_rna_drawn = N_SPEC * 3 + CASES
s1_rna_seq = (CASES + CTRL_SUB) * 3 + CASES                                      # case-cohort: all cases + a random subcohort of controls
s2_plasma = s1_plasma + WEEKLY_N * (11 - 3)                                       # weekly to day +56 on the first WEEKLY_N patients
s2_rna_seq = s1_rna_seq + (W_CASES + W_CTRL) * (11 - 3)
plan_si = pd.DataFrame([
 dict(package='S0  EHR timeline', patients=N_CAP, plasma=0, rna=0, biopsies=0, storage_TB=N_CAP * 10e6 / 1e12, time='12–18 months'),
 dict(package='S1  + banked specimens', patients=N_CAP, plasma=s1_plasma, rna=s1_rna_seq, biopsies=N_BIOPSY, storage_TB=(s1_rna_seq * GB_RNA + N_BIOPSY * GB_TISSUE) / 1000, time=f'{SPECIMEN_YEARS} years of accrual + 6 months'),
 dict(package='S2  + weekly trajectory cohort', patients=N_CAP, plasma=s2_plasma, rna=s2_rna_seq, biopsies=N_BIOPSY, storage_TB=(s2_rna_seq * GB_RNA + N_BIOPSY * GB_TISSUE) / 1000, time=f'{SPECIMEN_YEARS} years of accrual + 6 months (weekly draws in the first {WEEKLY_N / (TRANSPLANTS_PER_YEAR * USABLE_FRACTION):.1f} years)')])
ehr_run = cards_df[(cards_df.readiness == 'R1') & (cards_df.one_inst_verdict == 'RUNNABLE')].id.tolist()
spec_run = cards_df[(cards_df.readiness != 'R1') & (cards_df.one_inst_verdict.isin(['RUNNABLE']))].id.tolist(); spec_est = cards_df[(cards_df.readiness != 'R1') & (cards_df.one_inst_verdict == 'ESTIMABLE')].id.tolist()
plan_si['tasks it makes runnable'] = [', '.join(ehr_run), ', '.join(spec_run) or '—', 'B3 trajectory claims (≈ 100 evaluable cases); assay reliability (§9.8)']
plan_si['plus, estimable only'] = ['—', ', '.join(spec_est), '—']
disp_si = plan_si.copy()
for c in ('plasma', 'rna'): disp_si[c] = disp_si[c].map(lambda v: f'{v:,}' if v else '—')
disp_si['biopsies'] = disp_si.biopsies.map(lambda v: f'{v:,}' if v else '—'); disp_si['storage_TB'] = disp_si.storage_TB.map(lambda v: f'{v:.2f}' if v < 1 else f'{v:.1f}')
disp_si = disp_si.rename(columns={'plasma': 'plasma aliquots assayed', 'rna': 'RNA-seq libraries', 'biopsies': 'biopsies (RNA + slide)', 'storage_TB': 'storage (TB)'})
print(f'THE ONE-INSTITUTION DATA PLAN (≤ {N_CAP:,} patients; {N_SPEC} with specimens; {CASES} expected cases; weekly cohort {WEEKLY_N}):'); display(md_table(disp_si))
assert max(plan_si.patients) <= N_CAP

# ----------------------------------------------------------------------------- Figure 17: the data plan
FW_, FH_ = FULL, 3.6
fig = plt.figure(figsize=(FW_, FH_))
axa = fig.add_axes([1.75 / FW_, .55 / FH_, 3.55 / FW_, 2.50 / FH_]); axb = fig.add_axes([5.85 / FW_, .55 / FH_, 1.25 / FW_, 2.50 / FH_])
ax = axa                                                                        # (a) the plan as a table
ROWS = [('EHR timeline', [f'{N_CAP:,}'] * 3), ('plasma aliquots assayed', ['—', f'{s1_plasma:,}', f'{s2_plasma:,}']), ('RNA-seq libraries', ['—', f'{s1_rna_seq:,}', f'{s2_rna_seq:,}']),
        ('biopsies (RNA + slide)', ['—', f'{N_BIOPSY}', f'{N_BIOPSY}']), ('patients with weekly draws', ['—', '—', f'{WEEKLY_N}']), ('storage (TB)', [f'{plan_si.storage_TB.iloc[i]:.2f}' if plan_si.storage_TB.iloc[i] < 1 else f'{plan_si.storage_TB.iloc[i]:.1f}' for i in range(3)]),
        ('tasks runnable', [str(len(ehr_run)), str(len(ehr_run) + len(spec_run)), str(len(ehr_run) + len(spec_run))]), ('calendar time', ['1–1.5 y', '3.5 y', '3.5 y'])]
CELLC = [PAL['seq'][0], PAL['seq'][1], PAL['seq'][2]]
for i, (lab, vals_) in enumerate(ROWS):
    for j, v_ in enumerate(vals_):
        ax.add_patch(plt.Rectangle((j + .03, i - .46), .94, .92, color=CELLC[j], alpha=.5, lw=0))
        t_ = ax.text(j + .5, i, v_, ha='center', va='center', fontsize=6.8, fontweight='bold', color=PAL['ink']); t_._ok_overlap = True
ax.set_xlim(0, 3); ax.set_ylim(len(ROWS) - .5, -.5); ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS]); ax.tick_params(axis='y', length=0)
ax.xaxis.tick_top(); ax.set_xticks([.5, 1.5, 2.5]); ax.set_xticklabels(['S0\nEHR timeline', 'S1\n+ banked specimens', 'S2\n+ weekly cohort']); ax.tick_params(axis='x', length=0)
for sp in ax.spines.values(): sp.set_visible(False)
ax.annotate('a', (0, 1), xycoords='axes fraction', xytext=(-1.7 * 72, 30), textcoords='offset points', fontsize=9, fontweight='bold')
ax = axb; ygrid(ax, 'x')                                                         # (b) where the patients and specimens come from, over time
yrs_ = np.arange(0, 8.01, .5); ax.plot(yrs_, [min(N_CAP, TRANSPLANTS_PER_YEAR * t) for t in yrs_], color=PAL['seq'][3], lw=1.6, label='EHR timelines')
ax.plot(yrs_, [min(N_SPEC, TRANSPLANTS_PER_YEAR * USABLE_FRACTION * t) for t in yrs_], color=PAL['orange'], lw=1.5, ls='--', label='with specimens')
ax.set_xlabel('years from start'); ax.set_ylabel('patients accrued'); ax.set_xlim(0, 8); ax.set_ylim(0, 1650)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.2), ncol=1, handlelength=1.4, labelspacing=.3)
ax.annotate('b', (0, 1), xycoords='axes fraction', xytext=(-.5 * 72, 30), textcoords='offset points', fontsize=9, fontweight='bold')
finish(fig, 17, 'one_institution_data_plan',
       "The one-institution data plan. (a) Three nested packages, none above the 1,500-patient ceiling: S0 uses the EHR timeline only; S1 adds specimens banked on routine draws for three years, "
       "sequenced as a case-cohort; S2 adds a weekly trajectory cohort inside S1. Counts are computed from the sampling schedule of section 4.4 and the onset distribution of section 9.5. "
       "(b) Accrual at 200 transplants a year: EHR timelines reach the ceiling in 7.5 years (the first 4.5 years are retrospective), and patients with specimens (70% usable) reach 420 in three.")
''')

code(r'''
b1r = cards_df[cards_df.id == 'B1'].iloc[0]
display(Markdown(f"""**What the plan buys, and what it does not.** S0 alone — the EHR timelines of {N_CAP:,} transplants, no new specimens — runs the whole EHR-based set of tasks at one centre ({', '.join(ehr_run)}) with a head of 10–25 parameters. On the {N_TEST}-patient temporal test set a paired comparison detects a gain of about **+{min_detectable_gain(N_TEST, .70, .35):.3f}** (80 % power), and the stricter Level-1C gate is passed with probability **{level1c_power(N_TEST, .05):.0%} for a true gain of +0.05** and {level1c_power(N_TEST, .03):.0%} for +0.03. S1 adds the specimen-dependent tasks, notably **B1 against the MAGIC algorithm on {int(b1r.one_inst_n)} patients** (a gain of +{b1r.min_detectable_gain:.3f} or more, with a head of about {int(b1r.max_head_params)} parameters) and the tissue-transcriptome claims of §9.4 on about {N_BIOPSY} biopsies. S2 adds the trajectory claims on about 100 evaluable cases and the assay reliability that §9.5 needs. **It does not buy** any claim about another centre, a subgroup audit, a gain below about +0.03, or a Level-3 claim: the single-centre ladder stops at Level 1C (§8.10), and that is a complete, publishable result.

**What to do about the claims it cannot make.** Publish the data card and the harness, keep the event schema identical to the data dictionary of §4, and let other centres run the same code locally and share only summary statistics (§11.0, federated mode). The benchmark's Level 2 and Level 3 then become something a network can attempt without anyone moving patient-level data — and the single-centre result is the first point on that curve."""))
''')
