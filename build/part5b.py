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
        pl, n_ = plan_row(name, pe, kap, A1); row[f'ceiling {A1:.2f}: plateau'] = f'{pl:.3f}'; row[f'ceiling {A1:.2f}: n within 0.02 of it'] = f'{n_:,.0f}'
    plan.append(row)
plan = pd.DataFrame(plan)
print('LABELLED TRANSCRIPTOMES NEEDED to come within 0.02 AUROC of what each head can reach (prevalence 0.30, single-centre ρ = 1, K = 8):'); display(md_table(plan))
n_whole = plan_row(*HEADS[3][:3], .88)[1]; n_adapt = plan_row(*HEADS[1][:3], .88)[1]; n_whole75 = plan_row(*HEADS[3][:3], .75)[1]; n_adapt75 = plan_row(*HEADS[1][:3], .75)[1]
print(f'at ceiling 0.88 the adapter head needs {n_adapt:,.0f} labelled profiles versus {n_whole:,.0f} for the whole-transcriptome head ({n_whole / n_adapt:,.0f}x fewer)')

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
    trows.append(r_)
txt_tab = pd.DataFrame(trows)
txt_disp = txt_tab.copy()
for c in txt_disp.columns[4:]: txt_disp[c] = txt_disp[c].map(lambda v: ('+' + f'{v:.3f}') if 'gain' in c else ('not reachable' if not np.isfinite(v) else f'{v:,.0f}'))
print('LABELLED PROFILES FOR AN ADAPTER (64-d, κ = 0.9) TO ADD TO A BASELINE ("realisable gain" = the plateau of the added AUROC at that ρ):'); display(md_table(txt_disp))
def min_ceiling_gain(A0, want=.03, rho=.8, K=8, kappa=.9, pv=.30):
    for g in np.arange(.02, .30, .0025):
        if np.isfinite(n_for_target(A0 + want, A0, A0 + g, 64, pv, rho, K, kappa)): return float(g)
G_STAR = min_ceiling_gain(.72)
''')

code(r'''
# ----------------------------------------------------------------------------- 9.4: unlabelled pretraining corpora and the assay specification
CORP = [('public bulk RNA-seq, all tissues (ARCHS4 / recount3 scale) †', 1_000_000), ('public blood and immune-cell bulk RNA-seq †', 200_000), ('GvHD-specific public series (the pilot of §10)', 838)]
TOK_PER_SAMPLE = 2000                                  # one token per expressed gene, rank-encoded, top 2,000 per sample
crow = []
for name, ns_ in CORP:
    Dc = ns_ * TOK_PER_SAMPLE; Nc = Dc / 20; hrs = a100_hours(train_flops(Nc, Dc))
    crow.append({'unlabelled corpus': name, 'samples': f'{ns_:,}', 'tokens': fmt_n(Dc), 'compute-optimal adapter size': fmt_n(Nc), 'A100-hours': fmt_h(hrs)})
print('UNLABELLED PRETRAINING CORPUS FOR A TRANSCRIPTOME ADAPTER:'); display(md_table(pd.DataFrame(crow)))
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
# ----------------------------------------------------------------------------- Figure 13: how many labelled transcriptomes?
fig = plt.figure(figsize=(FULL, 3.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0], left=.075, right=.985, bottom=.265, top=.875, wspace=.40)
ax = fig.add_subplot(gs[0, 0]); ygrid(ax)                                       # (a) learning curves: raw genes vs adapter embeddings
nn = np.geomspace(60, 3e5, 80); A1_ = .80
raw_p = GENES * MULT; I_ = sep2(A1_)
ax.plot(nn, [auc_lc(n, 0, I_, raw_p) for n in nn], color=PAL['grey'], lw=1.6, label='whole transcriptome')
for kap, c, ls_ in ((1.0, PAL['seq'][3], '-'), (.85, PAL['seq'][2], '--'), (.7, PAL['seq'][1], ':')):
    ax.plot(nn, [auc_lc(n, 0, kap * I_, 64) for n in nn], color=c, lw=1.5, ls=ls_, label=f'64-d adapter, $\\kappa$ = {kap:g}')
ax.axhline(A1_, color=PAL['ink2'], lw=.6, ls='--'); ax.set_xscale('log'); ax.set_xlim(60, 3e5); ax.set_ylim(.5, .84)
ax.set_xlabel('labelled transcriptomes $n$'); ax.set_ylabel('AUROC at a new centre'); panel(ax, 'a')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2, columnspacing=1.0, handlelength=1.4, handletextpad=.4, labelspacing=.3)
ax = fig.add_subplot(gs[0, 1]); ygrid(ax)                                       # (b) labelled profiles needed to add a gain to a baseline
gains = np.linspace(.035, .14, 40)
for A0, c in ((.65, PAL['seq'][0]), (.72, PAL['seq'][2]), (.80, PAL['seq'][3])):
    ys = [n_for_target(A0 + .03, A0, A0 + g, 64, .30, .8, 8, .9) for g in gains]; ys = [y if np.isfinite(y) else np.nan for y in ys]
    ax.plot(gains, ys, color=c, lw=1.5, label=f'baseline AUROC {A0:.2f}')
ax.axhline(1287, color=PAL['ink2'], lw=.6, ls=':'); ax.set_yscale('log'); ax.set_ylim(80, 3e5)
ax.set_xlabel('gain the transcriptome could add at its ceiling'); ax.set_ylabel('labelled profiles to realise +0.03'); panel(ax, 'b')
ax.annotate('MAGIC-scale cohort', (.14, 1287), xytext=(0, 3), textcoords='offset points', fontsize=6.3, color=PAL['ink2'], ha='right', va='bottom')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=1, handlelength=1.4, labelspacing=.3)
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                       # (c) the plateau set by cross-centre sharing
rr_ = np.linspace(0, 1, 60); D_ = math.sqrt(sep2(.85))
for K_, c in ((4, PAL['seq'][0]), (8, PAL['seq'][2]), (20, PAL['seq'][3])):
    ax.plot(rr_, [norm.cdf((D_ / math.sqrt(2)) * r / math.sqrt(r + (1 - r) / K_)) if r > 0 else .5 for r in rr_], color=c, lw=1.5)
    ax.annotate(f'K = {K_}', (1.0, norm.cdf((D_ / math.sqrt(2)) * (.30) / math.sqrt(.30 + .70 / K_))), xytext=(0, 0), textcoords='offset points', fontsize=6.3, color=c, ha='left', va='center', annotation_clip=False) if False else None
ax.axhline(.85, color=PAL['ink2'], lw=.6, ls='--'); ax.set_xlabel('shared fraction of the signal across centres $\\rho$'); ax.set_ylabel('AUROC plateau at a new centre'); ax.set_ylim(.5, .88); ax.set_xlim(0, 1)
ax.legend([plt.Line2D([], [], color=c, lw=1.5) for c in (PAL['seq'][0], PAL['seq'][2], PAL['seq'][3])], ['4 training centres', '8 training centres', '20 training centres'],
          loc='upper center', bbox_to_anchor=(.5, -.25), ncol=1, handlelength=1.4, labelspacing=.3); panel(ax, 'c')
finish(fig, 13, 'transcriptome_requirements',
       "Labelled transcriptomes needed at diagnosis (closed form, validated against simulation in §9.4). (a) Learning curve of a head on the whole transcriptome "
       "against heads on a 64-dimensional adapter embedding that retains a fraction kappa of the signal: the embedding wins at the label counts a consortium can "
       "reach, and the whole-transcriptome head approaches its ceiling only at several hundred thousand profiles. Dashed: the within-centre ceiling. (b) Labelled "
       "profiles an adapter needs to add +0.03 AUROC to a baseline, against the gain the transcriptome could add at its ceiling; a weak ceiling gain cannot be "
       "realised at any n once centres only partly share the signal. (c) The plateau at a new centre falls as the shared fraction rho falls, and more training "
       "centres push it toward its limit.")
''')

code(r'''
b1 = txt_tab[txt_tab.task.str.startswith('B1')].iloc[0]; c4_ = txt_tab[txt_tab.task.str.startswith('C4')].iloc[0]
display(Markdown(f"""**Reading §9.4.** Three results, each computed above.

1. **The transcriptome is label-hungry unless something shrinks the dimension.** A head on the whole transcriptome needs **{min(n_whole, n_whole75):,.0f}–{max(n_whole, n_whole75):,.0f}** labelled profiles (ceiling 0.88 and 0.75) to come within 0.02 AUROC of its ceiling, and the stress test shows the real requirement exceeds the isotropic formula because cell composition and batch inflate the effective dimension by a factor of **{stress.p_eff_over_p.min():.1f}–{stress.p_eff_over_p.max():.1f}**. No public GvHD resource and no single consortium can supply that. This is the quantitative reason for the pilot's negative result on blood (§10), and it is independent of the architecture.
2. **A good adapter changes the arithmetic.** If a pretrained 64-dimensional embedding retains 90 % of the signal, the same target needs **{min(n_adapt, n_adapt75):,.0f}–{max(n_adapt, n_adapt75):,.0f}** labelled profiles, **{n_whole / n_adapt:,.0f}×** fewer at ceiling 0.88. That *is* the label-efficiency claim of a foundation model, now with a number on it. Its price is the retained fraction κ: panel (a) shows an embedding that keeps only 70 % of the signal still wins at every label count a consortium can reach but plateaus lower, and tasks F2 and G4 measure κ directly.
3. **A transcriptome adds a *small* increment to a *strong* baseline, and small increments are expensive.** For a day-7 blood transcriptome to add +0.02 to MAP plus clinical covariates (baseline 0.72, ceiling 0.76) an adapter needs **{b1['n for +0.02 (ρ = 1)']:,.0f}** labelled profiles from one centre and **{b1['n for +0.02 (ρ = 0.8, K = 8)']:,.0f}** across eight centres sharing 80 % of the signal, and the gain it can ever realise there is only **+{b1['realisable gain (ρ = 0.8, K = 8)']:.3f}**. To add +0.03 to a 0.72 baseline at all, the transcriptome must carry a ceiling gain of at least **+{G_STAR:.3f}**. By contrast, in tissue the same machinery needs only **{txt_tab.iloc[0]['n for +0.02 (ρ = 0.8, K = 8)']:,.0f}** profiles for +0.02 (C1/C2) and **{c4_['n for +0.02 (ρ = 0.8, K = 8)']:,.0f}** for steroid refractoriness (C4). This is why the paper claims the transcriptome adds value at diagnosis (tissue in hand, high ceiling) and does not claim it for pre-onset blood prognosis."""))
''')

# =====================================================================================================  9.5  LONGITUDINAL DESIGN
md(r'''
## 9.5 Longitudinal design: how many draws, how often, how many specimens?

The sampling schedule of §4.4 is a minimum. This section asks what is needed for the *trajectory*, which is what a foundation model on a timeline can exploit and a single draw cannot. Three questions, three calculations.

1. **How many draws in a window?** If one draw measures a patient's state with single-draw reliability λ (the fraction of between-patient variance that is signal), the mean of *k* draws has reliability $\lambda_k=k\lambda/(1+(k-1)\lambda)$ (Spearman–Brown), and a predictor whose oracle AUROC is $A^*$ reaches $\Phi\!\big(\Phi^{-1}(A^*)\sqrt{\lambda_k}\big)$. We validate that against simulation.
2. **How often?** Many biomarkers rise only in the days before clinical onset (a *prodrome* of length τ). If draws are Δ days apart, the number that land in the prodrome is $\lfloor\tau/\Delta\rfloor$ or one more, so the expected AUROC of a pre-onset alarm depends on Δ through the draws it gets.
3. **How many specimens?** The protocol of §4.4 is evaluated against a realistic distribution of onset days: of the cases, what fraction have at least one, and at least two, specimens in the 14 days before onset? And how many patients must be enrolled to have 300 such cases?
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
USABLE, PREV_AG, N_CASES_TARGET = .70, .35, 300
sdf['patients to enrol for 300 cases with >= 2 pre-onset draws'] = [int(np.ceil(N_CASES_TARGET / (PREV_AG * USABLE * p))) if p > 0 else np.nan for p in sdf.P_at_least_2]
disp_sched = sdf.copy()
for c in ('P_at_least_1', 'P_at_least_2'): disp_sched[c] = disp_sched[c].map(lambda v: f'{v:.0%}')
disp_sched['patients to enrol for 300 cases with >= 2 pre-onset draws'] = disp_sched['patients to enrol for 300 cases with >= 2 pre-onset draws'].map(lambda v: 'not reachable' if pd.isna(v) else f'{v:,.0f}')
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

md(r'''
**Reading §9.5.** (1) **Draws:** when a single draw is a noisy measurement (λ ≈ 0.3–0.5, typical of a dynamic blood signature), three to five draws in the window recover most of the oracle AUROC, and eight recover essentially all of it; when a draw is already reliable (λ ≥ 0.9), one suffices. Replicate draws are therefore worth buying exactly where the single-draw reliability is low — which is a *measurable* property of each assay (bridging and replicate samples, §4.4), not a guess. (2) **Interval:** the alarm degrades as the draw interval approaches the prodrome length; for a 10-day prodrome, draws every 7 or 14 days miss it in a large share of cases, while every 2–3.5 days keeps at least two draws inside it. (3) **Specimens:** the minimum protocol of §4.4 is designed for diagnosis and prognosis at fixed landmarks; it yields **few cases with two or more pre-onset specimens**, so it cannot by itself support trajectory tasks. A **nested trajectory cohort** (weekly to day +56 for blood RNA, plasma, stool and flow) is what makes B3 and the longitudinal objectives testable, and the table above gives the enrolment it needs for 300 evaluable cases. Its cost is in §9.8.
''')

# =====================================================================================================  9.6  HOW MANY CENTRES
md(r'''
## 9.6 How many centres?

Patients are not the only unit. A benchmark that claims *out-of-centre* generalisation has an effective sample size that is partly a **number of centres**, because centres differ in case mix, prophylaxis practice, grading habits and assay handling in ways a model cannot see. The paired gain of a candidate over a baseline therefore varies from centre to centre with some **between-centre standard deviation τ**, on top of the sampling noise within each centre. The gate of §8.12 averages over *K* held-out centres with a *t*-interval, so its power is

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
print(f'with 6,000 patients in 20 centres of 300 and an ICC of 0.03 the effective sample for a centre-level effect is {6000 / (1 + 299 * .03):.0f} patients.')
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
# check the Fisher computation against a direct simulation of the calibration-slope SE
def sim_slope_se(A, prev, n, reps=300, seed=2):
    r = np.random.default_rng(seed); D = math.sqrt(2) * norm.ppf(A); out = []
    for _ in range(reps):
        y = (r.random(n) < prev).astype(int); x = np.where(y == 1, r.normal(D, 1, n), r.normal(0, 1, n)); lp = math.log(prev / (1 - prev)) + D * x - D * D / 2
        m = LogisticRegression(C=1e6, max_iter=500).fit(lp[:, None], y); out.append(m.coef_[0, 0])
    return float(np.std(out))
fi_se = math.sqrt(_val_parts(.75, .15)[0] / 1500); sm_se = sim_slope_se(.75, .15, 1500, reps=150 if QUICK else 300)
print(f'calibration-slope SE at n = 1500 (AUROC 0.75, prevalence 0.15): Fisher {fi_se:.3f} vs simulated {sm_se:.3f}')
assert abs(fi_se - sm_se) / fi_se < (.25 if QUICK else .15)
vtab = pd.DataFrame([dict(AUROC=A, prevalence=pv, **validation_n(A, pv)) for A in (.70, .75, .80) for pv in (.05, .15, .35)])
vtab = vtab.rename(columns={'n': 'patients', 'binding': 'binding criterion'})
print('PATIENTS (AND EVENTS) FOR A LEVEL-3 VALIDATION:'); display(md_table(vtab))
rate1, rateC = 200 * USABLE, 3000 * USABLE
vtab['months at one centre'] = (vtab.patients / rate1 * 12).round(0).astype(int); vtab['months, 20-centre consortium'] = (vtab.patients / rateC * 12).round(0).astype(int)
print('TIME TO COLLECT A PROSPECTIVE VALIDATION COHORT (70 % usable; one centre 200 transplants a year, a 20-centre consortium 3,000):')
display(md_table(vtab[['AUROC', 'prevalence', 'patients', 'months at one centre', 'months, 20-centre consortium']]))

# ---------- subgroups (shares are planning assumptions †)
SUBGROUPS = [('cord-blood graft', .05), ('Black / African ancestry (US registry)', .06), ('Asian / Pacific Islander (US registry)', .07), ('lowest-volume centre tertile', .08),
             ('Hispanic / Latino (US registry)', .10), ('paediatric (< 18 y)', .15), ('haploidentical donor', .20)]
A_sg, pv_sg = .72, .35
n_gap_events = {g: int(np.ceil((norm.ppf(.975) + norm.ppf(.8)) ** 2 * (.70 * .30 + (.70 - g) * (.30 + g)) / g ** 2)) for g in (.10, .05)}
n_gap_pat = {g: int(np.ceil(e / pv_sg)) for g, e in n_gap_events.items()}
sgrow = []
for name, s in SUBGROUPS:
    n_auc_ = n_for_ci(A_sg, pv_sg, .07); n_slope_ = int(np.ceil(_val_parts(A_sg, pv_sg)[0] / .15 ** 2)); base_need = max(n_auc_, n_slope_)
    row_ = {'subgroup': name, 'share †': f'{s:.0%}'}
    for g in (.10, .05):
        need_sub = max(base_need, n_gap_pat[g]); row_[f'patients in subgroup for a {g:.2f} gap'] = f'{need_sub:,}'; row_[f'total cohort for a {g:.2f} gap'] = int(np.ceil(need_sub / s))
    row_['assessable at 25,000 (gap 0.10 / 0.05)'] = ' / '.join('yes' if 25000 * s >= max(base_need, n_gap_pat[g]) else 'NO' for g in (.10, .05))
    row_['assessable at 100,000 (gap 0.10 / 0.05)'] = ' / '.join('yes' if 100000 * s >= max(base_need, n_gap_pat[g]) else 'NO' for g in (.10, .05))
    sgrow.append(row_)
sg = pd.DataFrame(sgrow); sg_disp = sg.copy()
for c in ('total cohort for a 0.10 gap', 'total cohort for a 0.05 gap'): sg_disp[c] = sg_disp[c].map(lambda v: f'{v:,}')
print(f'SUBGROUP AUDIT (AUROC {A_sg}, prevalence {pv_sg}; an equal-opportunity gap of 0.10 in sensitivity needs {n_gap_events[.10]} events per group = {n_gap_pat[.10]:,} patients, a gap of 0.05 needs {n_gap_events[.05]:,} = {n_gap_pat[.05]:,}):'); display(md_table(sg_disp))

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
# ----------------------------------------------------------------------------- Figure 15: centres, validation size and subgroup precision
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
prevs = np.geomspace(.05, .5, 24)
for A, c in zip((.70, .75, .80), (PAL['seq'][0], PAL['seq'][2], PAL['seq'][3])):
    ax.plot(prevs, [validation_n(A, pv)['n'] for pv in prevs], color=c, lw=1.5, label=f'AUROC {A:.2f}')
ax.axhline(200 * USABLE * 3, color=PAL['ink2'], lw=.8, ls=':', label='one centre, 3 years'); ax.axhline(3000 * USABLE, color=PAL['ink2'], lw=.8, ls='--', label='20 centres, 1 year')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([.05, .1, .2, .5]); ax.set_xticklabels(['0.05', '0.1', '0.2', '0.5']); ax.minorticks_off()
ax.set_yticks([300, 1000, 3000]); ax.set_yticklabels(['300', '1,000', '3,000'])
ax.set_xlabel('outcome prevalence'); ax.set_ylabel('patients for a Level-3 validation')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2, columnspacing=1.0, handlelength=1.4, handletextpad=.4, labelspacing=.3); panel(ax, 'b')
ax = fig.add_subplot(gs[0, 2]); ygrid(ax)                                       # (c) subgroup assessability
shares = np.geomspace(.02, .30, 30)
for g, c in ((.10, PAL['seq'][2]), (.05, PAL['seq'][3])):
    need_g = max(n_for_ci(A_sg, pv_sg, .07), int(np.ceil(_val_parts(A_sg, pv_sg)[0] / .15 ** 2)), n_gap_pat[g])
    ax.plot(shares, need_g / shares, color=c, lw=1.5, label=f'sensitivity gap {g:.2f}')
ax.axhline(25000, color=PAL['ink2'], lw=.8, ls=':', label='25,000-patient extract'); ax.axhline(100000, color=PAL['ink2'], lw=.8, ls='--', label='100,000-patient extract')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([.02, .05, .1, .2]); ax.set_xticklabels(['2%', '5%', '10%', '20%']); ax.minorticks_off()
ax.set_yticks([5000, 10000, 25000, 50000, 100000]); ax.set_yticklabels(['5k', '10k', '25k', '50k', '100k']); ax.set_xlabel('share of the cohort in the subgroup'); ax.set_ylabel('total cohort for the subgroup\nto be assessable')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2, columnspacing=1.0, handlelength=1.4, handletextpad=.4, labelspacing=.3); panel(ax, 'c')
finish(fig, 15, 'centres_validation_subgroups',
       "How many centres, how many validation patients, how large a subgroup. (a) Centres needed for the adopted gate to have 80% power for a true gain of +0.03, against the "
       "between-centre SD of the gain, for three centre sizes: once heterogeneity dominates, more centres help and more patients per centre do not. The dotted line is the gate's minimum. "
       "(b) Patients for a Level-3 validation (calibration slope SE 0.10, O/E SE 0.10, AUROC half-width 0.05) against prevalence, set against what one centre collects in three years "
       "and what a 20-centre consortium collects in one. (c) Total cohort at which a subgroup of a given share has enough patients to detect an equal-opportunity gap in sensitivity of "
       "0.10 or 0.05, against two registry-extract sizes.")
''')

# =====================================================================================================  9.8  DATA BILL OF MATERIALS
md(r'''
## 9.8 The data bill of materials: minimum viable, credible, definitive

Everything above is a requirement on one quantity. This subsection puts them together into **what to collect**, in three packages, with the constraint that binds each number stated. The packages are cumulative in what they must demonstrate:

* **Minimum viable (MVD)** — enough to *test the primary claim* once, honestly: for each data layer, the largest of the registry's per-task requirement (Riley development size for a 50-parameter head, and the paired-power size to show +0.03; §8.4) and the transcriptome learning-curve size of §9.4, spread over the number of centres at which the gate of §8.12 has 80 % power for a +0.03 gain with between-centre SD **τ = 0.02**.
* **Credible** — enough to support *several claims and an equity audit*: MVD plus the total cohort at which the smallest subgroup with a 10 % share becomes assessable (§9.7), plus, for the registry layers, the Riley size for a 768-parameter embedding head; centres planned at **τ = 0.03**.
* **Definitive** — enough for a *Level-3* claim: credible plus a prospective validation cohort sized by §9.7, planned at **τ = 0.04**.

For each layer the table reports patients, centres, specimens and what it costs. **Unit costs are planning figures †, not quotes**; they are in one dictionary in the cell below so a reader can change them, and the totals are shown with a ×0.5 / ×2 range.
''')

code(r'''
# ----------------------------------------------------------------------------- 9.8: the data bill of materials
UNIT = dict(collect_per_specimen=60, plasma_panel=100, bulk_rnaseq=250, tissue_rna_wsi=450, flow_panel=200, stool_16s=120, scrna_vdj=3000,
            coordination_per_patient=1500, centre_setup=50_000, registry_access=150_000, ehr_etl_per_centre=25_000, harmonisation_central=200_000)   # USD †
nreq = need_tbl.set_index('id')['n required']                                  # per-task requirement of §8.4 / §9.1 (max of Riley-50 and paired power for +0.03)
MIN_NC, K_MIN = 60, GATE['level2_gate']['min_centres']
def N_for_gate(N_task, tau, d=.03):
    """Smallest total N >= N_task, and the fewest centres K (>= the gate's minimum), at which the adopted gate has 80 % power with n_c = N/K per centre."""
    N = max(N_task, K_MIN * MIN_NC)
    for _ in range(200):
        for K in range(K_MIN, 250):
            if N / K < MIN_NC: break
            if gate_power(K, N / K, d, tau) >= .8: return int(math.ceil(N)), K
        N *= 1.08
    return None, None
sub_need = lambda gap=.10: max(n_for_ci(A_sg, pv_sg, .07), int(np.ceil(_val_parts(A_sg, pv_sg)[0] / .15 ** 2)), n_gap_pat[gap])
DENSE_N = int(sdf[(sdf.specimen == 'Whole blood PAXgene RNA') & (sdf.schedule == 'weekly to day +56')]['patients to enrol for 300 cases with >= 2 pre-onset draws'].iloc[0])
c4 = txt_tab.set_index('task')
def _tx(prefix, col):
    return [v for k, v in c4[col].items() if k.startswith(prefix)][0]
LAYERS = [  # key, name, n_task (MVD), primary task for FM-head and validation, smallest subgroup that must be assessable (share), draws per patient (landmark-only for the MVD, the §4.4 protocol, the weekly trajectory schedule), nested trajectory sub-cohort?, assay USD, GB per specimen
 dict(key='A',  name='registry core + outcomes (Tier A)',  n_task=max(nreq[t] for t in ('A1', 'A2', 'A3', 'A4', 'A5')), prim=(.68, .38), fm_head=True, share=.05, draws=(0, 0, 0), dense=False, assay=0, gb=0),
 dict(key='B',  name='EHR timeline (Tier B)',               n_task=max(nreq[t] for t in ('B3', 'B6', 'C3', 'D3', 'D4', 'E1', 'E2', 'E4')), prim=(.68, .30), fm_head=True, share=.05, draws=(0, 0, 0), dense=False, assay=0, gb=0),
 dict(key='C1', name='plasma biomarkers, scheduled draws (Tier C)', n_task=max(nreq[t] for t in ('B1', 'B2', 'B4', 'B5', 'C4', 'C5', 'C6', 'E3')), prim=(.74, .35), fm_head=False, share=.10, draws=(3, 8, 11), dense=True, assay=UNIT['plasma_panel'], gb=.001),
 dict(key='C2', name='blood transcriptome at diagnosis and prodrome (Tier C)', n_task=None, tx=('C4', .72, .30), prim=(.72, .30), fm_head=False, share=.20, draws=(3, 6, 11), dense=True, assay=UNIT['bulk_rnaseq'], gb=6),
 dict(key='C3', name='tissue transcriptome + whole-slide image at diagnosis (Tier C/D)', n_task=None, tx=('C1/C2', .86, .45), prim=(.86, .45), fm_head=False, share=.20, draws=(1.3, 1.3, 1.3), dense=False, assay=UNIT['tissue_rna_wsi'], gb=8),
 dict(key='D',  name='TCR repertoire / single-cell / spatial (Tier D)', n_task=None, prim=None, fm_head=False, share=None, draws=(6, 6, 6), dense=False, assay=UNIT['scrna_vdj'], gb=120),
]
TIER_D = {'minimum viable': 80, 'credible': 300, 'definitive': 1000}               # F1 / F2 accessible-n scale of §8.4: nested research cohort
PKG = (('minimum viable', .02), ('credible', .03), ('definitive', .04))
TXCOL = {'minimum viable': 'n for +0.02', 'credible': 'n for 80% of it', 'definitive': 'n for 95% of it'}
bom = []
for L in LAYERS:
    for pkg, tau in PKG:
        if L['key'] == 'D':
            N, K = TIER_D[pkg], max(1, int(round(TIER_D[pkg] / 100))); binding = 'research cohort size (F1/F2)'; spec_n = N * L['draws'][1]
        else:
            if L['n_task']: N_task, binding = L['n_task'], 'per-task n (Riley-50 / paired power)'
            else:
                N_task = _tx(L['tx'][0], f"{TXCOL[pkg]} (ρ = 0.8, K = 8)"); binding = {'minimum viable': 'transcriptome learning curve: +0.02 (§9.4)', 'credible': 'transcriptome learning curve: 80% of realisable gain (§9.4)',
                                                                                        'definitive': 'transcriptome learning curve: 95% of realisable gain (§9.4)'}[pkg]
            if not np.isfinite(N_task): N_task = _tx(L['tx'][0], 'n for +0.02 (ρ = 0.8, K = 8)')
            if pkg in ('credible', 'definitive'):
                cands = {binding: N_task, f'subgroup audit ({L["share"]:.0%} subgroup, §9.7)': int(np.ceil(sub_need(.10) / L['share']))}
                if L['fm_head']: cands['768-parameter embedding head (Riley)'] = riley_n(768, *L['prim'])['n']
                binding = max(cands, key=cands.get); N_task = cands[binding]
            if pkg == 'definitive' and L['prim'] is not None:
                N_task += validation_n(*L['prim'])['n']; binding += ' + prospective validation (§9.7)'
            N, K = N_for_gate(N_task, tau)
            if N != N_task: binding += ' + centres for the gate'
            draws = L['draws'][0] if pkg == 'minimum viable' else L['draws'][1]
            spec_n = N * draws + (DENSE_N * (L['draws'][2] - L['draws'][1]) if (L['dense'] and pkg != 'minimum viable') else 0)      # weekly trajectory draws on a nested sub-cohort
        bom.append(dict(layer=L['name'], key=L['key'], package=pkg, patients=N, centres=K, binding=binding, specimens=spec_n,
                        specimens_per_patient=spec_n / N, assay_usd=spec_n * L['assay'], collect_usd=spec_n * UNIT['collect_per_specimen'] if L['key'] not in ('A', 'B') else 0.0,
                        storage_TB=spec_n * L['gb'] / 1000))
bom = pd.DataFrame(bom)
# shared prospective infrastructure and retrospective data work, once per package
shared = []
for pkg, tau in PKG:
    q = bom[bom.package == pkg]; pro = q[q.key.isin(['C1', 'C2', 'C3'])]
    n_pro = int(pro.patients.max()); k_pro = int(pro.centres.max())
    shared.append(dict(package=pkg, prospective_patients=n_pro, centres=k_pro,
        coordination=n_pro * UNIT['coordination_per_patient'], centre_setup=k_pro * UNIT['centre_setup'],
        registry_and_etl=UNIT['registry_access'] + UNIT['harmonisation_central'] + int(q[q.key == 'B'].centres.iloc[0]) * UNIT['ehr_etl_per_centre'],
        years_consortium=1 + n_pro / (3000 * USABLE)))
shared = pd.DataFrame(shared).set_index('package')
cost = pd.DataFrame({pkg: dict(collection=bom[(bom.package == pkg)].collect_usd.sum() + shared.loc[pkg, 'coordination'],
                               assays=bom[(bom.package == pkg)].assay_usd.sum(), centres_and_data=shared.loc[pkg, 'centre_setup'] + shared.loc[pkg, 'registry_and_etl'],
                               compute=PROG_H * USD_PER_H) for pkg, _ in PKG}).T
cost['total'] = cost.sum(axis=1)
show = bom.copy()
show['patients'] = show.patients.map(lambda v: f'{v:,}'); show['specimens'] = show.specimens.map(lambda v: f'{v:,.0f}'); show['storage_TB'] = show.storage_TB.map(lambda v: f'{v:,.1f}')
show['specimens per patient'] = show.specimens_per_patient.map(lambda v: f'{v:.1f}')
show['assays + collection (USD)'] = [fmt_usd(a + c) for a, c in zip(bom.assay_usd, bom.collect_usd)]
display(md_table(show[['layer', 'package', 'patients', 'centres', 'specimens per patient', 'specimens', 'storage_TB', 'assays + collection (USD)', 'binding']], max_w=90))
cd = cost.copy()
for c in cd.columns: cd[c] = cd[c].map(fmt_usd)
cd['years to accrue (20-centre consortium, prospective layers)'] = shared.years_consortium.map(lambda v: f'{v:.1f}')
cd['range (x0.5 / x2)'] = [f'{fmt_usd(.5 * t)} – {fmt_usd(2 * t)}' for t in cost.total]
print('PROGRAMME COST BY PACKAGE (all unit costs are planning figures †):'); display(md_table(cd, index=True))
''')

code(r'''
BANKED = 1287 + 431                                                         # MAGIC day-7 cohort + BMT CTN 1703 (§9.2)
save_banked = BANKED * (3 * UNIT['collect_per_specimen'] + UNIT['coordination_per_patient']) + BANKED * UNIT['plasma_panel']
bg = lambda k, pkg, col: bom[(bom.key == k) & (bom.package == pkg)][col].iloc[0]
c_ = lambda pkg: cost.loc[pkg, 'total']
display(Markdown(f"""**Reading §9.8 — the answer to "how much, of what, how often".**

* **Minimum viable ({fmt_usd(c_('minimum viable'))}, ≈ {shared.loc['minimum viable', 'years_consortium']:.1f} years of consortium accrual).** Registry core on **{bg('A', 'minimum viable', 'patients'):,}** patients; an EHR timeline on **{bg('B', 'minimum viable', 'patients'):,}**; day-7/day-14/onset plasma biomarkers on **{bg('C1', 'minimum viable', 'patients'):,}** patients ({bg('C1', 'minimum viable', 'specimens'):,.0f} specimens); a blood transcriptome (landmark draws only) on **{bg('C2', 'minimum viable', 'patients'):,}** patients ({bg('C2', 'minimum viable', 'specimens'):,.0f} libraries, {bg('C2', 'minimum viable', 'storage_TB'):,.0f} TB); tissue transcriptome plus whole-slide image on **{bg('C3', 'minimum viable', 'patients'):,}** biopsy-adjudicated patients; and **{bg('D', 'minimum viable', 'patients'):,}** deeply phenotyped patients. It tests the primary claim once and has **no trajectory cohort**, so task B3 and the longitudinal objectives are not testable at this size.
* **Credible ({fmt_usd(c_('credible'))}).** Adds the **nested weekly trajectory cohort** of **{DENSE_N:,}** patients (§9.5), an equity audit that makes a {sub_need(.10):,}-patient subgroup assessable, and the registry/EHR scale ({bg('A', 'credible', 'patients'):,} and {bg('B', 'credible', 'patients'):,} patients) at which a 768-parameter embedding head can be developed. Blood transcriptome: **{bg('C2', 'credible', 'patients'):,}** patients, {bg('C2', 'credible', 'specimens'):,.0f} libraries.
* **Definitive ({fmt_usd(c_('definitive'))}).** Adds a prospective Level-3 validation cohort and the heterogeneity (τ = 0.04) that needs **{bg('C1', 'definitive', 'centres')}** centres.
* **Where the money goes.** Collection and assays are ≈ {100 * (cost.loc['credible', 'collection'] + cost.loc['credible', 'assays']) / c_('credible'):.0f} % of the credible package; compute is **{fmt_usd(PROG_H * USD_PER_H)}**, about {math.log10(c_('credible') / (PROG_H * USD_PER_H)):.0f} orders of magnitude less. Reusing the {BANKED:,} banked MAGIC and BMT CTN 1703 specimens would save about **{fmt_usd(save_banked)}** of collection and day-7 assay cost.
* **Ranges, not budgets.** Every figure moves with the unit costs (†); the ×0.5/×2 range is in the table, and the *patient* numbers do not depend on the costs at all."""))
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 16: the data bill of materials
fig = plt.figure(figsize=(FULL, 3.9))
FW_, FH_ = FULL, 3.9
axa = fig.add_axes([1.95 / FW_, .50 / FH_, 3.20 / FW_, 2.85 / FH_]); axb = fig.add_axes([5.80 / FW_, .50 / FH_, 1.30 / FW_, 2.85 / FH_])
ax = axa                                                                        # (a) patients by layer and package
pk = [p for p, _ in PKG]; keys = [L['key'] for L in LAYERS]
CELLC = [PAL['seq'][0], PAL['seq'][1], PAL['seq'][2]]
for i, L in enumerate(LAYERS):
    for j, p_ in enumerate(pk):
        r_ = bom[(bom.key == L['key']) & (bom.package == p_)].iloc[0]
        ax.add_patch(plt.Rectangle((j + .03, i - .46), .94, .92, color=CELLC[j], alpha=.55, lw=0))
        t_ = ax.text(j + .5, i - .08, f'{r_.patients:,}', ha='center', va='center', fontsize=6.6, fontweight='bold', color=PAL['ink']); t_._ok_overlap = True
        t2 = ax.text(j + .5, i + .22, f'{int(r_.centres)} centres' if L['key'] != 'D' else f'{int(r_.centres)} site(s)', ha='center', va='center', fontsize=6.0, color=PAL['ink2']); t2._ok_overlap = True
ax.set_xlim(0, 3); ax.set_ylim(len(LAYERS) - .5, -.5); ax.set_yticks(range(len(LAYERS)))
ax.set_yticklabels([L['name'].split(' (')[0].replace(' at diagnosis and prodrome', '').replace(' at diagnosis', '').replace('registry core + outcomes', 'registry core').replace('plasma biomarkers, scheduled draws', 'plasma biomarkers').replace('TCR repertoire / single-cell / spatial', 'Tier D research') for L in LAYERS])
ax.tick_params(axis='y', length=0); ax.xaxis.tick_top(); ax.set_xticks([.5, 1.5, 2.5]); ax.set_xticklabels(['minimum\nviable', 'credible', 'definitive']); ax.tick_params(axis='x', length=0)
for sp in ax.spines.values(): sp.set_visible(False)
ax.annotate('a', (0, 1), xycoords='axes fraction', xytext=(-1.85 * 72, 24), textcoords='offset points', fontsize=9, fontweight='bold')
ax = axb; ygrid(ax, 'x')                                                         # (b) cost
parts = [('collection', PAL['seq'][1]), ('assays', PAL['seq'][3]), ('centres_and_data', PAL['aqua']), ('compute', PAL['orange'])]
left = np.zeros(3)
for nm, c in parts:
    v = np.array([cost.loc[p, nm] / 1e6 for p in pk]); ax.barh(range(3), v, left=left, color=c, height=.55, zorder=3, label=nm.replace('_', ' ')); left += v
for i, p_ in enumerate(pk): ax.text(left[i] + .02 * left.max(), i, f'${cost.loc[p_, "total"] / 1e6:.1f}M', va='center', fontsize=6.4, fontweight='bold')
ax.set_yticks(range(3)); ax.set_yticklabels(['MVD', 'credible', 'definitive']); ax.invert_yaxis(); ax.tick_params(axis='y', length=0); ax.set_xlim(0, left.max() * 1.45)
ax.set_xlabel('USD millions †'); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=2, columnspacing=.9, handlelength=1.2, handletextpad=.4, labelspacing=.3)
ax.annotate('b', (0, 1), xycoords='axes fraction', xytext=(0, 24), textcoords='offset points', fontsize=9, fontweight='bold')
finish(fig, 16, 'data_bill_of_materials',
       "The data bill of materials. (a) Patients required in each data layer for three packages, with the number of centres at which the gate of section 8.12 has "
       "80% power; the packages differ in what they must demonstrate (a single honest test of the primary claim; several claims with a subgroup audit; a Level-3 claim "
       "with prospective validation) and in the between-centre heterogeneity planned for (tau = 0.02, 0.03, 0.04). (b) Programme cost by package, split into specimen "
       "collection and coordination, assays, centre set-up and data engineering, and compute; unit costs are planning figures. Compute is the smallest bar by orders of magnitude.")
''')
