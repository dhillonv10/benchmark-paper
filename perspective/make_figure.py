"""The perspective's single figure (4 panels), drawn from results.json and gated for text collisions by the notebook's layout gate.
Usage  python3 analysis.py && python3 make_figure.py          ->  figures/Fig01_perspective.pdf / .png
"""
import json, os, sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figsys
from figsys import PAL, FULL, finish, panel, ygrid, plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(os.path.join(HERE, 'results.json')))
D = R['dynamic']; A = D['arms']; BASE = D['base']

fig = plt.figure(figsize=(FULL, 6.3))
gs = fig.add_gridspec(2, 2, left=.085, right=.985, bottom=.115, top=.945, wspace=.60, hspace=.46)

# ------------------------------------------------------------------ A  where the signal window is, and what data exist in it
gA = gs[0, 0].subgridspec(2, 1, height_ratios=[1, 1.35], hspace=.62)
ax = fig.add_subplot(gA[0]); ygrid(ax)
h = np.array(R['onset_hist']); ax.bar(np.arange(len(h)) * 5 + 2.5, h, width=4.2, color=PAL['seq'][2], zorder=3)
ax.axvspan(R['cohort']['severe_onset_quartiles'][0], R['cohort']['severe_onset_quartiles'][2], color=PAL['orange'], alpha=.16, lw=0, zorder=1)
ax.set_xlim(0, 100); ax.set_xticks([0, 14, 28, 50, 75, 100]); ax.set_ylabel('patients'); ax.set_xlabel('day of onset of severe (grade III–IV) aGvHD')
panel(ax, 'A'); ax.text(99, ax.get_ylim()[1] * .97, f"earliest day {R['cohort']['severe_onset_min']:.0f};\nmiddle half day {R['cohort']['severe_onset_quartiles'][0]:.0f}–{R['cohort']['severe_onset_quartiles'][2]:.0f}", ha='right', va='top', fontsize=6.6, color=PAL['ink2'])
ax2 = fig.add_subplot(gA[1]); labs = [('CBC', 'blood count'), ('biochemistry', 'chemistry'), ('electrolytes', 'electrolytes'), ('vitalsigns', 'vital signs'), ('fluidloss', 'fluid loss'), ('flowcytometry', 'flow cytometry'), ('cytokines', 'cytokines'), ('antibodies', 'antibodies')]
M = np.array([R['coverage'][g] for g, _ in labs]); im = ax2.imshow(M, aspect='auto', cmap='Blues', vmin=0, vmax=1, extent=(.5, 60.5, len(labs) - .5, -.5), interpolation='nearest')
ax2.set_yticks(range(len(labs))); ax2.set_yticklabels([f"{n} ({R['n_vars'][g]})" for g, n in labs]); ax2.set_xticks([1, 7, 14, 21, 28, 42, 60]); ax2.set_xlabel('day after transplant'); ax2.tick_params(axis='y', length=0)
for sp in ax2.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=ax2, fraction=.045, pad=.02, aspect=14); cb.set_label('patients with data', fontsize=6.4); cb.set_ticks([0, .5, 1]); cb.ax.tick_params(labelsize=6.2, length=2); cb.outline.set_visible(False)

# ------------------------------------------------------------------ B  head size: what each added piece of the timeline does to the next-14-day forecast
ax = fig.add_subplot(gs[0, 1]); ygrid(ax, 'x')
rows = [('+ fluid loss', 'fluid loss', 'raw'), ('+ vital signs', 'vital signs', 'raw'), ('+ electrolytes', 'electrolytes', 'raw'), ('+ blood count', 'blood count', 'raw'), ('+ chemistry', 'chemistry', 'raw'),
        ('+ 3-d label-free summary', '3-d summary', 'lf'), ('+ 5-d label-free summary', '5-d summary', 'lf'), ('+ 10-d label-free summary', '10-d summary', 'lf'), ('+ raw recent window', 'whole window', 'bad')]
col = dict(raw=PAL['grey'], lf=PAL['aqua'], bad=PAL['bad']); nrow = len(rows) + 1
for i, (k, nm, c) in enumerate(rows):
    a = A[k]; ax.plot([a['dlo'], a['dhi']], [i, i], color=col[c], lw=1.6, solid_capstyle='butt', zorder=3); ax.plot(a['delta'], i, 'o', color=col[c], ms=4.2, zorder=4)
nl = R['control_permuted_outcome']; mu = nl['null_delta_mean']['+ 5-d label-free summary']; sd = nl['null_delta_sd']['+ 5-d label-free summary']; nulls = nl['null_delta']['+ 5-d label-free summary']
ax.plot([np.percentile(nulls, 2.5), np.percentile(nulls, 97.5)], [len(rows), len(rows)], color=PAL['ink2'], lw=3.2, alpha=.55, solid_capstyle='butt', zorder=3); ax.plot(mu, len(rows), 'o', mfc='white', mec=PAL['ink2'], ms=4.2, zorder=4)
ax.axvline(0, color=PAL['ink'], lw=.8, zorder=2); ax.axvline(.01, color=PAL['ink2'], lw=.7, ls=':', zorder=2)
ax.set_yticks(range(nrow)); ax.set_yticklabels([f"{nm} ({D['n_params'][k]})" for k, nm, _ in rows] + ['5-d, outcome shuffled']); ax.set_ylim(nrow - .4, -.7); ax.tick_params(axis='y', length=0)
ax.set_xlabel('change in AUROC vs. baseline\n(added dimensions in brackets)'); ax.set_xlim(-.21, .07); ax.set_xticks([-.2, -.1, 0, .05]); ax.set_xticklabels(['−0.2', '−0.1', '0', '+0.05']); panel(ax, 'B')

# ------------------------------------------------------------------ C  few-shot label efficiency (EHRSHOT protocol)
ax = fig.add_subplot(gs[1, 0]); ygrid(ax); FS = R['fewshot']; ks = [8, 16, 32, 64]
styles = [('static + time + known state', 'baseline covariates', PAL['ink2'], '-'), ('+ 5-d label-free summary (PCA)', '+ 5-d summary (PCA)', PAL['aqua'], '-'),
          ('+ 5-d label-free summary (pretrained encoder)', '+ 5-d summary (pretrained encoder)', PAL['blue'], '--'), ('+ raw recent window', '+ whole window (615-d)', PAL['bad'], '-')]
for k, nm, c, ls in styles:
    if k not in FS: continue
    y = [FS[k][str(q)]['mean'] for q in ks if str(q) in FS[k]]; ax.plot(ks[:len(y)], y, ls, color=c, marker='o', ms=3.2, label=nm)
ax.set_xscale('log', base=2); ax.set_xticks(ks); ax.set_xticklabels([str(q) for q in ks]); ax.minorticks_off(); ax.set_xlabel('severe-onset events used to fit the head (and as many non-events)'); ax.set_ylabel('AUROC, same-day pairs'); ax.set_ylim(.50, .68)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.30), ncol=1, handlelength=1.6, labelspacing=.25); panel(ax, 'C')

# ------------------------------------------------------------------ D  what one centre can resolve
ax = fig.add_subplot(gs[1, 1]); ygrid(ax); rs = R['resolve']; dg = np.linspace(0, .06, 200); n0 = rs['patients']
for n, c, nm in ((n0, PAL['seq'][1], f'this cohort ({n0})'), (1500, PAL['seq'][2], 'a 1,500-patient centre'), (6000, PAL['seq'][3], 'four such centres (6,000)')):
    se = rs['se_observed'] * np.sqrt(n0 / n); ax.plot(dg, norm.cdf((dg - rs['margin']) / se - 1.96), color=c, lw=1.6, label=nm)
ax.axhline(.8, color=PAL['bad'], ls=':', lw=.8); ax.axvline(rs['gain'], color=PAL['orange'], ls='--', lw=1.0)
ax.set_xlim(0, .06); ax.set_ylim(-.02, 1.03); ax.set_xlabel('true gain in AUROC over the baseline'); ax.set_ylabel('chance the gate is passed\n(lower bound of the interval ≥ +0.01)')
ax.annotate(f"observed +{rs['gain']:.3f}", (rs['gain'], 1.04), xytext=(3, 0), textcoords='offset points', fontsize=6.4, color=PAL['orange'], ha='left', va='bottom', fontweight='bold', annotation_clip=False)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.30), ncol=1, handlelength=1.6, labelspacing=.25); panel(ax, 'D')

c = R['cohort']; w = A['+ raw recent window']; p5 = A['+ 5-d label-free summary']; co = R['control_permuted_outcome']; rs = R['resolve']
finish(fig, 1, 'perspective',
       f"Is a foundation model for GvHD plausible? A check on the one public longitudinal cohort (one centre, {c['n']} adults, {c['severe']} with severe aGvHD; daily labs, vitals and fluid loss), with every block predicted from earlier patients only. "
       f"(A) Severe aGvHD starts between day {c['severe_onset_min']:.0f} and day 90 (shaded: middle half); blood counts, chemistry, vitals and fluid loss are dense through that window, flow cytometry and cytokines are not. "
       f"(B) Change in AUROC (same-day pairs; dotted line: the +0.01 margin of the gate) for forecasting severe onset in the next 14 days, against baseline covariates plus time and known aGvHD: a head that takes the whole recent window (615 inputs) loses {abs(w['delta']):.2f}; "
       f"single modalities add nothing detectable; a 5-dimensional label-free summary gains {p5['delta']:+.3f} (95% CI {p5['dlo']:+.3f} to {p5['dhi']:+.3f}), which sits inside the spread of the same arm when the outcome is shuffled (grey bar: central 95% of {co['n_permutations']} shuffles; P = {co['p_value_ge_observed']['+ 5-d label-free summary']:.2f}). "
       f"(C) EHRSHOT-style few-shot curves (mean of three blocks and 25 draws): label-free summaries make the timeline harmless, not helpful; the whole window is harmful. "
       f"(D) Chance that a gate fixed in advance (interval lower bound at least +0.01) is passed at the observed standard error, by cohort size: the observed gain would pass {rs['pass_probability_observed_gain'][str(rs['patients'])]:.0%} of the time at this cohort size and {rs['pass_probability_observed_gain']['1500']:.0%} at 1,500 patients.")
open(os.path.join(figsys.FIGDIR, 'Fig01_perspective_caption.txt'), 'w').write(figsys.FIGS['Fig 1']['caption'] + '\n')
