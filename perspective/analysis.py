"""Real-data check for the perspective: the one public longitudinal allo-HCT cohort with daily timelines and GvHD labels.

Data   aGOAT cohort, OMIX001095 (Liu et al., Nat Comput Sci 2022; https://ngdc.cncb.ac.cn/omix/release/OMIX001095/):
       one centre (Tianjin), 584 adults, 194 dynamic variables per patient on days -14..100, severe (grade III-IV) aGvHD with onset day.
       Open access, but sharing is governed by PRC human-genetic-resources rules: this script downloads it from the source and never redistributes it.
Design every evaluation is temporal: the cohort's `fold` column orders patients by transplant date, and a block is always predicted by models fitted
       on EARLIER blocks only (forward chaining, blocks 6-9; training sizes 142 / 224 / 351 / 519 patients).  Every arm shares one regularisation grid.
       Nothing here is tuned on the evaluation blocks.  Label-free summaries (PCA, a small forecaster) are fitted on earlier patients' timelines only.
Tasks  EARLY    at landmark day L = 7 or 14: severe aGvHD by day 100 from days 1..L            (prognosis)
       DYNAMIC  at landmark day L = 14, 21, ..., 63: severe aGvHD onset in (L, L+14]              (the task the cohort's own authors forecast)
Usage  python3 analysis.py          (about 6 min on 4 CPU cores; PERSPECTIVE_QUICK=1 cuts the draws for a smoke test)
Output results.json (every number quoted in perspective_draft.md and drawn in make_figure.py)
"""
import os, io, json, math, re, time, glob, zipfile, urllib.request, warnings
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.simplefilter('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); QUICK = os.environ.get('PERSPECTIVE_QUICK', '0') == '1'
URL = 'https://download.cncb.ac.cn/OMIX/OMIX001095/OMIX001095-01-01.zip'
CG = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)                  # one regularisation grid for every arm
BLOCKS = (6, 7, 8, 9); H = 14; LANDMARKS = list(range(14, 64, 7)); BOOT = 150 if QUICK else 400
GROUPS = ['CBC', 'biochemistry', 'electrolytes', 'vitalsigns', 'fluidloss', 'flowcytometry', 'cytokines', 'antibodies']; DENSE = GROUPS[:5]
RES = {}; t0 = time.time()
log = lambda *a: print(f'[{time.time() - t0:5.0f}s]', *a, flush=True)

# ------------------------------------------------------------------------------------------------ data
def download():
    os.makedirs(DATA, exist_ok=True)
    if glob.glob(os.path.join(DATA, '**', 'adult', 'encoded.csv'), recursive=True): return
    z = os.path.join(DATA, 'agoat.zip')
    for attempt in range(5):                                             # the host resets connections now and then
        try:
            with urllib.request.urlopen(URL, timeout=180) as r, open(z, 'wb') as f: f.write(r.read())
            zipfile.ZipFile(z).extractall(DATA); return
        except Exception as e: log('download attempt', attempt + 1, 'failed:', repr(e)[:80]); time.sleep(4)
    raise RuntimeError('could not download OMIX001095; fetch it manually from https://ngdc.cncb.ac.cn/omix/release/OMIX001095/ and unzip into perspective/data/')
download(); ROOT = os.path.dirname(os.path.dirname(glob.glob(os.path.join(DATA, '**', 'adult', 'encoded.csv'), recursive=True)[0]))
enc = pd.read_csv(f'{ROOT}/adult/encoded.csv'); keys = list(pd.read_csv(f'{ROOT}/adult/keys.csv')['keys'])
X = np.full((len(enc), 115, len(keys)), np.nan, dtype=np.float32)       # patients x days(-14..100) x variables
for i, pid in enumerate(enc.ID):
    X[i] = pd.read_csv(f'{ROOT}/adult/processed/pt_{pid}.csv').set_index('day').reindex(range(-14, 101))[keys].to_numpy(dtype=np.float32)
grp = np.array([re.sub(r'_\d+$', '', k) for k in keys]); W = len(keys); dv = np.where(np.isin(grp, DENSE))[0]
CAT = ['Sex', 'Primary.disease', 'Conditioning.regimen', 'Stem.cell.source', 'Sex.match', 'ABO.match', 'GVHD.prophylaxis', 'Antithymocyte.globulin.in.conditioning', 'Donor.type']
NUM = ['Age', 'BMI', 'TNC.count', 'CD34.count', 'HLA.mismatch']          # peri-transplant covariates only: no year, no engraftment day (known only later), no outcome
S = pd.concat([pd.get_dummies(enc[CAT].astype(str).apply(lambda s: s.str.strip()), drop_first=True).astype(float), enc[NUM].astype(float)], axis=1)
S = S.loc[:, S.nunique() > 1].to_numpy(np.float32)
sev = enc['Severe.aGVHD.within.100.days'].to_numpy(); onset = enc['Severe.aGVHD.onset.day'].to_numpy(); fold = enc.fold.to_numpy(); year = enc.Year.to_numpy().astype(np.float32)
anyg = enc['aGVHD.within.100.days'].to_numpy(); any_onset = enc['aGVHD.onset.day'].to_numpy()
death_t = np.where(enc.Death == 1, enc['Last.followup.time.for.death'], np.inf)
RES['cohort'] = dict(n=int(len(enc)), severe=int(sev.sum()), severe_onset_quartiles=[float(q) for q in np.nanpercentile(onset[sev == 1], [25, 50, 75])], severe_onset_min=float(np.nanmin(onset)),
                     any_agvhd=int((anyg >= 1).sum()), years=[int(enc.Year.min()), int(enc.Year.max())], deaths_by_day100=int(((enc.Death == 1) & (enc['Last.followup.time.for.death'] <= 100)).sum()),
                     donor_haplo=int((enc['Donor.type'].str.strip() == 'Haploidentical').sum()), temporal_blocks={int(k): int(v) for k, v in enc.fold.value_counts().sort_index().items()})
# panel A inputs: when does severe aGvHD start, and which data exist on which day
RES['onset_hist'] = np.histogram(onset[sev == 1], bins=np.arange(0, 101, 5))[0].tolist()
RES['coverage'] = {g: [float(np.isfinite(X[:, 15 + d - 1, grp == g]).any(1).mean()) for d in range(1, 61)] for g in GROUPS}
RES['n_vars'] = {g: int((grp == g).sum()) for g in GROUPS}

# ------------------------------------------------------------------------------------------------ models
def lr(C): return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=C, max_iter=3000))
def fit_predict(Xtr, ytr, Xte, groups=None, Cs=CG):
    """L2-logistic head; C chosen by patient-grouped inner CV on the TRAINING rows only, same grid for every arm."""
    keep = np.array([len(np.unique(Xtr[:, j][np.isfinite(Xtr[:, j])])) >= 2 for j in range(Xtr.shape[1])]); Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    cv = GroupKFold(5).split(Xtr, ytr, groups) if groups is not None else StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, ytr)
    cv = list(cv); best, bc = -1, Cs[0]
    for C in Cs:
        p = cross_val_predict(lr(C), Xtr, ytr, cv=cv, method='predict_proba')[:, 1]; a = roc_auc_score(ytr, p)
        if a > best: best, bc = a, C
    return lr(bc).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
def pca_embed(R, tr, k):
    imp = SimpleImputer(strategy='median').fit(R[tr]); sc = StandardScaler().fit(imp.transform(R[tr])); pc = PCA(k, random_state=0).fit(sc.transform(imp.transform(R[tr])))
    return pc.transform(sc.transform(imp.transform(R)))
def strat_auc(y, p, strata):
    """AUROC over pairs that share a landmark day only (weights n1 x n0 per landmark): who, among patients at risk on the SAME day, is next?
    The pooled AUROC also rewards anything that merely tracks days-since-transplant, because the hazard peaks around day 28."""
    num = den = 0.
    for s_ in np.unique(strata):
        m = strata == s_; n1 = int(y[m].sum()); n0 = int(m.sum()) - n1
        if n1 and n0: num += roc_auc_score(y[m], p[m]) * n1 * n0; den += n1 * n0
    return num / den
def cluster_boot(ids, y, p, p2=None, B=BOOT, seed=0, strata=None):
    """95 % interval of the metric (p2 is None) or of metric(p2) - metric(p), resampling PATIENTS (all of a patient's landmarks together)."""
    f = (lambda yy, pp, ss: strat_auc(yy, pp, ss)) if strata is not None else (lambda yy, pp, ss: roc_auc_score(yy, pp))
    r = np.random.default_rng(seed); u = np.unique(ids); by = {i: np.where(ids == i)[0] for i in u}; out = []
    for _ in range(B):
        s = np.concatenate([by[i] for i in r.choice(u, len(u))])
        if y[s].min() == y[s].max(): continue
        ss = strata[s] if strata is not None else None
        out.append(f(y[s], p[s], ss) if p2 is None else f(y[s], p2[s], ss) - f(y[s], p[s], ss))
    return [float(v) for v in np.percentile(out, [2.5, 97.5])]
def summarise(ids, y, preds, ref, strata=None):
    f = (lambda yy, pp, ss: strat_auc(yy, pp, ss)) if strata is not None else (lambda yy, pp, ss: roc_auc_score(yy, pp)); out = {}
    for nm, p in preds.items():
        m = np.isfinite(p); st = strata[m] if strata is not None else None; a = f(y[m], p[m], st); lo, hi = cluster_boot(ids[m], y[m], p[m], strata=st)
        d = dict(auroc=float(a), lo=lo, hi=hi, n=int(m.sum()), events=int(y[m].sum()))
        if strata is not None: d['auroc_pooled'] = float(roc_auc_score(y[m], p[m]))
        if nm != ref:
            dd = a - f(y[m], preds[ref][m], st); dlo, dhi = cluster_boot(ids[m], y[m], preds[ref][m], p2=p[m], strata=st); d.update(delta=float(dd), dlo=dlo, dhi=dhi)
        out[nm] = d
    return out

# ------------------------------------------------------------------------------------------------ 1. EARLY prognosis: severe aGvHD by day 100 from days 1..L
def summaries(L):
    Wd = X[:, 15:15 + L, :]
    with np.errstate(all='ignore'):
        last = np.full((X.shape[0], W), np.nan, np.float32)
        for t in range(L): v = Wd[:, t, :]; last = np.where(np.isfinite(v), v, last)
        return np.hstack([last, np.nanmean(Wd, 1), np.nanmin(Wd, 1), np.nanmax(Wd, 1), np.isfinite(Wd).sum(1).astype(np.float32)])
RES['early'] = {}
for L in (7, 14):
    R = summaries(L)[:, np.tile(np.isin(grp, DENSE), 5)]; el = (death_t > L) & ~((sev == 1) & (onset <= L)); P = {}
    for a in ('year only', 'static covariates', 'timeline, all dense summaries', 'static + all dense summaries', 'static + 5-d label-free summary'): P[a] = np.full(len(sev), np.nan)
    for tb in BLOCKS:
        tr = el & (fold < tb); te = el & (fold == tb); Z = pca_embed(R, tr, 5)
        F = {'year only': year[:, None], 'static covariates': S, 'timeline, all dense summaries': R, 'static + all dense summaries': np.hstack([S, R]), 'static + 5-d label-free summary': np.hstack([S, Z])}
        for a in P: P[a][te] = fit_predict(F[a][tr], sev[tr], F[a][te])
    m = np.isfinite(P['static covariates']); RES['early'][f'L{L}'] = summarise(np.arange(len(sev))[m], sev[m], {k: v[m] for k, v in P.items()}, 'static covariates')
    log(f'early L={L}', {k: round(v['auroc'], 3) for k, v in RES['early'][f'L{L}'].items()})

# ------------------------------------------------------------------------------------------------ 2. DYNAMIC: severe aGvHD onset in the next 14 days
def recent_window(L, w=3):
    """last / mean / max / min over days L-w+1..L and the trend against the preceding week; dense modalities only"""
    Wd = X[:, 14 + L - w + 1:15 + L, :]
    with np.errstate(all='ignore'):
        last = np.full((X.shape[0], W), np.nan, np.float32)
        for t in range(Wd.shape[1]): v = Wd[:, t, :]; last = np.where(np.isfinite(v), v, last)
        mean = np.nanmean(Wd, 1); prev = np.nanmean(X[:, max(15, 14 + L - w - 6):14 + L - w + 1, :], 1)
        return np.hstack([last, mean, np.nanmax(Wd, 1), np.nanmin(Wd, 1), mean - prev])
rows = dict(i=[], L=[], y=[], f=[], R=[], known=[])
for L in LANDMARKS:
    el = (death_t > L) & ~((sev == 1) & (onset <= L)); yL = ((sev == 1) & (onset > L) & (onset <= L + H)).astype(int)   # a death inside the window stays in as a non-event
    R = recent_window(L)[:, np.tile(np.isin(grp, DENSE), 5)]; w = np.where(el)[0]
    rows['i'] += list(w); rows['L'] += [L] * len(w); rows['y'] += list(yL[w]); rows['f'] += list(fold[w]); rows['R'].append(R[w])
    rows['known'].append(((any_onset[w] <= L) & (anyg[w] >= 1)).astype(float)[:, None])    # aGvHD of any grade already treated by L is known at L; the FINAL grade is never used
idx, Ls, ys, fl = (np.array(rows[k]) for k in ('i', 'L', 'y', 'f')); R = np.vstack(rows['R']); KN = np.vstack(rows['known']); Ss = S[idx]; Lf = Ls[:, None].astype(np.float32)
gw = grp[np.isin(grp, DENSE)]; gcol = lambda g: np.where(np.tile(gw, 5) == g)[0]          # columns of R belonging to a modality group
mg = lambda g: np.r_[gcol(g)[len(gcol(g)) // 5 * 1:len(gcol(g)) // 5 * 3]]                  # mean + max over the last 3 days (blocks 2 and 3 of 5)
BASE = 'static + time + known state'
ARMS = {'time since HCT only': lambda tr: Lf, 'static + time': lambda tr: np.hstack([Ss, Lf]), BASE: lambda tr: np.hstack([Ss, Lf, KN]),
        '+ fluid loss': lambda tr: np.hstack([Ss, Lf, KN, R[:, mg('fluidloss')]]), '+ vital signs': lambda tr: np.hstack([Ss, Lf, KN, R[:, mg('vitalsigns')]]),
        '+ electrolytes': lambda tr: np.hstack([Ss, Lf, KN, R[:, mg('electrolytes')]]), '+ blood count': lambda tr: np.hstack([Ss, Lf, KN, R[:, mg('CBC')]]),
        '+ chemistry': lambda tr: np.hstack([Ss, Lf, KN, R[:, mg('biochemistry')]]),
        '+ 3-d label-free summary': lambda tr: np.hstack([Ss, Lf, KN, pca_embed(R, tr, 3)]), '+ 5-d label-free summary': lambda tr: np.hstack([Ss, Lf, KN, pca_embed(R, tr, 5)]),
        '+ 10-d label-free summary': lambda tr: np.hstack([Ss, Lf, KN, pca_embed(R, tr, 10)]), '+ raw recent window': lambda tr: np.hstack([Ss, Lf, KN, R])}
NPAR = {'+ fluid loss': len(mg('fluidloss')), '+ vital signs': len(mg('vitalsigns')), '+ electrolytes': len(mg('electrolytes')), '+ blood count': len(mg('CBC')), '+ chemistry': len(mg('biochemistry')),
        '+ 3-d label-free summary': 3, '+ 5-d label-free summary': 5, '+ 10-d label-free summary': 10, '+ raw recent window': R.shape[1]}
def run_dynamic(y, only=None):
    P = {}
    for nm, f in ARMS.items():
        if only is not None and nm not in only: continue
        p = np.full(len(y), np.nan)
        for tb in BLOCKS:
            tr = fl < tb; te = fl == tb; F = f(tr); p[te] = fit_predict(F[tr], y[tr], F[te], groups=idx[tr])
        P[nm] = p
    return P
P = run_dynamic(ys); m = np.isfinite(P[BASE]); STRATA = Ls * 10 + fl        # a stratum = one landmark day predicted by ONE model (one temporal block): only like-for-like pairs are compared
RES['dynamic'] = dict(rows=int(m.sum()), patients=int(len(np.unique(idx[m]))), events=int(ys[m].sum()), prevalence=float(ys[m].mean()), base=BASE, n_params=NPAR, arms=summarise(idx[m], ys[m], {k: v[m] for k, v in P.items()}, BASE, strata=STRATA[m]))
RES['dynamic']['by_block'] = {int(tb): dict(n=int((fl == tb).sum()), events=int(ys[fl == tb].sum()), base=float(strat_auc(ys[fl == tb], P[BASE][fl == tb], Ls[fl == tb])), pca5=float(strat_auc(ys[fl == tb], P['+ 5-d label-free summary'][fl == tb], Ls[fl == tb]))) for tb in BLOCKS}
log('dynamic', {k: (round(v['auroc'], 3), round(v.get('delta', 0), 3)) for k, v in RES['dynamic']['arms'].items()})
# negative controls on the same pipeline.  (a) permute the 5-d summary across rows; (b) permute the OUTCOME across patients within each landmark, many times, to get the null distribution of
# the very quantity we report (the gain of an arm over the baseline).  An observed gain is only a finding if it sits outside that null.
Z5 = {tb: pca_embed(R, fl < tb, 5) for tb in BLOCKS}; dl = []; ST = lambda y_, p_: strat_auc(y_[m], p_[m], STRATA[m])
for sd in range(3 if QUICK else 8):
    pp = np.full(len(ys), np.nan)
    for tb in BLOCKS:
        tr = fl < tb; te = fl == tb; F = np.hstack([Ss, Lf, KN, Z5[tb][np.random.default_rng(sd).permutation(len(ys))]]); pp[te] = fit_predict(F[tr], ys[tr], F[te], groups=idx[tr])
    dl.append(float(ST(ys, pp) - ST(ys, P[BASE])))
RES['control_permuted_summary'] = dict(deltas=dl, mean=float(np.mean(dl)))
NPERM = 4 if QUICK else 20; null = {'+ 5-d label-free summary': []}; null_base = []
for sd in range(NPERM):
    rp = np.random.default_rng(100 + sd); yp = ys.copy()
    for L in LANDMARKS: w = np.where(Ls == L)[0]; yp[w] = rp.permutation(ys[w])
    Pp = run_dynamic(yp, only=(BASE, '+ 5-d label-free summary')); null_base.append(float(strat_auc(yp[m], Pp[BASE][m], STRATA[m])))
    for k in null: null[k].append(float(strat_auc(yp[m], Pp[k][m], STRATA[m]) - strat_auc(yp[m], Pp[BASE][m], STRATA[m])))
RES['control_permuted_outcome'] = dict(n_permutations=NPERM, base_auroc_mean=float(np.mean(null_base)), null_delta=null,
    null_delta_mean={k: float(np.mean(v)) for k, v in null.items()}, null_delta_sd={k: float(np.std(v)) for k, v in null.items()},
    p_value_ge_observed={k: float((1 + sum(x >= RES['dynamic']['arms'][k]['delta'] for x in v)) / (1 + len(v))) for k, v in null.items()})
log('controls', RES['control_permuted_summary']['mean'], RES['control_permuted_outcome']['null_delta_mean'], RES['control_permuted_outcome']['p_value_ge_observed'])
# the same arms inside single landmarks (removes the between-landmark time effect that dominates the pooled AUROC)
RES['dynamic']['by_landmark'] = {}
for L in (14, 21, 28):
    mm = (Ls == L) & m
    if mm.sum() and ys[mm].min() != ys[mm].max(): RES['dynamic']['by_landmark'][L] = dict(n=int(mm.sum()), events=int(ys[mm].sum()), **{k: float(roc_auc_score(ys[mm], P[k][mm])) for k in (BASE, '+ 5-d label-free summary', '+ raw recent window')})

# ------------------------------------------------------------------------------------------------ 3. FEW-SHOT (EHRSHOT protocol): k events + k non-events, label-free summaries fitted on earlier patients' unlabeled timelines
try:
    import torch; torch.set_num_threads(4); HAVE_TORCH = True
except Exception: HAVE_TORCH = False
daily_F, daily_T, daily_P = [], [], []
for L in range(14, 71):
    daily_F.append(recent_window(L)[:, np.tile(np.isin(grp, DENSE), 5)])
    with np.errstate(all='ignore'): daily_T.append(np.nanmean(X[:, 15 + L:15 + L + 7, :][:, :, dv], axis=1))
    daily_P.append(np.arange(X.shape[0]))
daily_F, daily_T, daily_P = np.vstack(daily_F), np.vstack(daily_T), np.concatenate(daily_P); daily_fold = fold[daily_P]
def pretrain_encoder(tb, d_lat=5, epochs=25, seed=0):
    """denoising autoencoder + next-7-day forecaster on EARLIER patients' daily windows (labels never touched); returns an embedding function"""
    torch.manual_seed(seed); mk = daily_fold < tb; Fm, Tm = daily_F[mk], daily_T[mk]
    mu, sd = np.nanmean(Fm, 0), np.nanstd(Fm, 0) + 1e-6; tm, ts = np.nanmean(Tm, 0), np.nanstd(Tm, 0) + 1e-6
    Zt = torch.tensor(np.nan_to_num((Fm - mu) / sd), dtype=torch.float32); Mk = torch.tensor(np.isfinite(Fm), dtype=torch.float32)
    Tz = (Tm - tm) / ts; Tmk = torch.tensor(np.isfinite(Tz), dtype=torch.float32); Tt = torch.tensor(np.nan_to_num(Tz), dtype=torch.float32)
    e = torch.nn.Sequential(torch.nn.Linear(Zt.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, d_lat)); dr = torch.nn.Sequential(torch.nn.Linear(d_lat, 64), torch.nn.ReLU(), torch.nn.Linear(64, Zt.shape[1]))
    df = torch.nn.Sequential(torch.nn.Linear(d_lat, 64), torch.nn.ReLU(), torch.nn.Linear(64, Tt.shape[1])); opt = torch.optim.Adam([*e.parameters(), *dr.parameters(), *df.parameters()], 2e-3, weight_decay=1e-4)
    for ep in range(epochs):
        perm = torch.randperm(len(Zt))
        for b in range(0, len(Zt), 256):
            i = perm[b:b + 256]; z = e(Zt[i] * (torch.rand(len(i), Zt.shape[1]) > .2)); loss = (((dr(z) - Zt[i]) ** 2) * Mk[i]).mean() + (((df(z) - Tt[i]) ** 2) * Tmk[i]).mean(); opt.zero_grad(); loss.backward(); opt.step()
    def emb(M):
        with torch.no_grad(): return e(torch.tensor(np.nan_to_num((M - mu) / sd), dtype=torch.float32)).numpy()
    return emb
def head_auc(Ftr, ytr, Fte, yte, seed, ste):
    best, bc = -1, .1
    for C in (.01, .1, 1.):
        try: a = roc_auc_score(ytr, cross_val_predict(lr(C), Ftr, ytr, cv=StratifiedKFold(4, shuffle=True, random_state=seed), method='predict_proba')[:, 1])
        except Exception: a = -1
        if a > best: best, bc = a, C
    return strat_auc(yte, lr(bc).fit(Ftr, ytr).predict_proba(Fte)[:, 1], ste)
KS = (8, 16, 32, 64); DRAWS = 6 if QUICK else 25; fs = {}
for tb in (7, 8, 9):
    tr = np.where(fl < tb)[0]; te = np.where(fl == tb)[0]; base_F = np.hstack([Ss, Lf, KN]); Zp = pca_embed(R, fl < tb, 5); Zr = None
    arms = {'static + time + known state': base_F, '+ 5-d label-free summary (PCA)': np.hstack([base_F, Zp]), '+ raw recent window': np.hstack([base_F, R])}
    if HAVE_TORCH: arms['+ 5-d label-free summary (pretrained encoder)'] = np.hstack([base_F, pretrain_encoder(tb)(R)])
    pos, neg = tr[ys[tr] == 1], tr[ys[tr] == 0]; rr = np.random.default_rng(tb)
    for k in KS:
        if k > len(pos): continue
        for d in range(DRAWS):
            sel = np.r_[rr.choice(pos, k, replace=False), rr.choice(neg, k, replace=False)]
            for nm, F in arms.items(): fs.setdefault((nm, k), []).append(head_auc(F[sel], ys[sel], F[te], ys[te], d, Ls[te]))
    log(f'few-shot block {tb} done')
RES['fewshot'] = {nm: {str(k): dict(mean=float(np.mean(v)), sd=float(np.std(v)), draws=len(v)) for (n2, k), v in fs.items() if n2 == nm} for nm in sorted({a for a, _ in fs})}

# ------------------------------------------------------------------------------------------------ 4. what a single centre can resolve (inputs to panel D) and the head-size limit
def r2cs(A, prev, n=400_000):
    rg = np.random.default_rng(0); d = math.sqrt(2) * norm.ppf(A); y = rg.random(n) < prev; x = np.where(y, rg.normal(d, 1, n), rg.normal(0, 1, n))
    lp = math.log(prev / (1 - prev)) + d * x - d * d / 2; p = 1 / (1 + np.exp(-lp)); ll1 = np.mean(np.where(y, np.log(p), np.log(1 - p))); ll0 = prev * math.log(prev) + (1 - prev) * math.log(1 - prev)
    return 1 - math.exp(-2 * (ll1 - ll0))
def max_params(n, A, prev, S_=.9, delta=.05):
    """largest head (candidate parameters) that n patients support under Riley et al.'s shrinkage and optimism criteria"""
    r2 = r2cs(A, prev); mx = 1 - math.exp(2 * (prev * math.log(prev) + (1 - prev) * math.log(1 - prev))); S2 = r2 / (r2 + delta * mx)
    return int(math.floor(min(n * (S_ - 1) * math.log(1 - r2 / S_), n * (S2 - 1) * math.log(1 - r2 / S2))))
a5 = RES['dynamic']['arms']['+ 5-d label-free summary']; se = (a5['dhi'] - a5['dlo']) / (2 * 1.96); n_pat = RES['dynamic']['patients']; margin = .01
need = lambda gain, power=.8: int(math.ceil(n_pat * (se / ((gain - margin) / (1.96 + norm.ppf(power)))) ** 2)) if gain > margin else None
RES['resolve'] = dict(se_observed=float(se), patients=n_pat, gain=a5['delta'], margin=margin, patients_for_80pct_pass=need(a5['delta']),
                      pass_probability_observed_gain={str(n): float(norm.cdf((a5['delta'] - margin) / (se * math.sqrt(n_pat / n)) - 1.96)) for n in (n_pat, 1500, 6000)},
                      head_limit_ever_severe=max_params(n_pat, .70, RES['cohort']['severe'] / RES['cohort']['n']), head_limit_events=RES['dynamic']['events'])
log('resolve', RES['resolve'])
json.dump(RES, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
log('wrote results.json')
