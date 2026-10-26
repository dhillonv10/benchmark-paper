from part1 import CELLS, md, code

md(r'''
---
# 3 · What a GvHD foundation model can and cannot be expected to predict

This section states the paper's **capability claims** before any model exists, so that they are falsifiable. It does three things: it derives the ceiling on discrimination that any model faces at a given decision point, it places each task against that ceiling, and it answers the specific question of what a **transcriptomic profile taken at diagnosis** can support.

## 3.1 The ceiling is set by how much of the outcome is already determined

Acute GvHD is not a fixed property of a donor–recipient pair. It is the result of a chain of events that unfolds *after* day 0: how much mucosal injury the conditioning caused, whether the gut microbiome collapsed, what the calcineurin-inhibitor exposure actually was, which alloreactive clones expanded, whether an infection intervened. A model standing at day −7 cannot see any of that, because none of it has happened yet.

Formally, let a patient's latent liability be $\eta = s_L + u_L$, where $s_L$ is the part determined *and observable* by landmark $L$ and $u_L$ is everything still to come, with $\mathrm{Var}(\eta)=1$ and the outcome $y=\mathbb 1\{\eta > z_{1-\pi}\}$ for prevalence $\pi$. Write $f_L = \mathrm{Var}(s_L)$: the **fraction of outcome-determining variance available at the landmark**. Then the best achievable AUROC at that landmark is the AUROC of the Bayes-optimal predictor $s_L$, which depends only on $f_L$ (and very weakly on $\pi$). The cell below computes that function, inverts it, and plots it.

This is a modelling identity, not an empirical claim: what is assumed is $f_L$, and those assumptions are labelled. The value of the exercise is that it converts "how good could this get?" into "how much of the causal chain can we observe by then?" — which is a **data question**, and therefore actionable.
''')

code(r'''
# ----------------------------------------------------------------------------- the discrimination ceiling as a function of information available at the landmark
def ceiling_auroc(f, prev=.35, n=400_000, seed=0):
    """Best achievable AUROC for a binary outcome when only a fraction f of its latent variance is
    determined-and-observable at prediction time (the rest happens afterwards). Bayes-optimal predictor = s."""
    f = float(np.clip(f, 1e-6, 1 - 1e-9)); r = np.random.default_rng(seed)
    s = r.normal(0, math.sqrt(f), n); u = r.normal(0, math.sqrt(1 - f), n)
    y = ((s + u) > norm.ppf(1 - prev)).astype(int)
    return auroc(y, s)

def f_for_auroc(target, prev=.35):
    """Invert the ceiling: what fraction of the causal chain must be observable to reach a target AUROC?"""
    lo, hi = 1e-4, 1 - 1e-4
    for _ in range(40):
        mid = (lo + hi) / 2
        if ceiling_auroc(mid, prev) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2

FS = np.linspace(.02, .98, 25)
ceil_curve = {pv: np.array([ceiling_auroc(f, pv) for f in FS]) for pv in (.15, .35)}
inv = pd.DataFrame([dict(**{'target AUROC': t, 'fraction of causal chain that must be observable (prevalence 0.35)':
                            round(f_for_auroc(t), 3)}) for t in (.65, .70, .75, .80, .85, .90)])

# Landmarks: assumed share of the causal chain observable at each (ASSUMPTIONS, stated as such), and what
# published models actually achieve there (see the reference list; ranges are approximate).
LAND = [  # id, landmark, assumed f, published-performance range, source note
    ('A1/A2', 'day −7\n(pre-conditioning)',   .22, (.60, .66), 'registry + HLA models'),
    ('B1',    'day +7\n(biomarkers, labs)',   .42, (.68, .75), 'MAGIC MAP + clinical'),
    ('B2/B4', 'day +14\n(+ reconstitution)',  .52, (.70, .76), 'biomarker + clinical'),
    ('C4/C5', 'GvHD onset\n(staging, biopsy)', .70, (.65, .75), 'grade + biomarkers at onset'),
    ('C1/C7', 'onset, diagnosis\n(tissue in hand)', .88, (.80, .90), 'tissue transcriptome / histology')]
fig = plt.figure(figsize=(FULL, 3.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], left=.075, right=.975, bottom=.165, top=.90, wspace=.34)
ax = fig.add_subplot(gs[0, 0]); ygrid(ax)
ax.plot(FS, ceil_curve[.35], color=PAL['ink'], lw=1.6, label='ceiling, prevalence 0.35')
ax.plot(FS, ceil_curve[.15], color=PAL['grey'], lw=1.2, ls='--', label='ceiling, prevalence 0.15')
LC = (PAL['seq'][0], PAL['seq'][1], PAL['seq'][2], PAL['orange'], PAL['aqua'])
for (tid, lab, f, (plo, phi), _), c in zip(LAND, LC):
    cl = ceiling_auroc(f, .35)
    ax.plot([f, f], [plo, phi], color=c, lw=3.6, solid_capstyle='butt', zorder=3)          # what is published today
    ax.plot([f, f], [phi, cl], color=PAL['ink'], lw=.8, zorder=3)                           # the headroom
    ax.plot(f, cl, 'o', color=PAL['ink'], ms=4.0, zorder=4)
    ax.text(f, .505, tid, fontsize=6.4, ha='center', va='bottom', color=PAL['ink2'], fontweight='bold', rotation=90)
ax.set_xlabel('fraction of the causal chain observable at the landmark, $f_L$'); ax.set_ylabel('AUROC')
ax.set_ylim(.49, 1.0); ax.set_xlim(0, 1.02)
hs, ls_ = ax.get_legend_handles_labels()
hs += [plt.Line2D([], [], color=PAL['grey'], lw=3.6), plt.Line2D([], [], color=PAL['ink'], lw=.8, marker='o', ms=3.6)]
ls_ += ['published models today', 'headroom to the ceiling']
ax.legend(hs, ls_, loc='upper left', labelspacing=.3); panel(ax, 'a')
ax = fig.add_subplot(gs[0, 1]); ygrid(ax, 'x')
ax.barh(np.arange(len(inv)), inv.iloc[:, 1], color=PAL['seq'][2], height=.62, zorder=3)
for i, f in enumerate(inv.iloc[:, 1]): ax.text(f + .02, i, f'{f:.2f}', va='center', fontsize=6.5)
ax.set_yticks(np.arange(len(inv))); ax.set_yticklabels([f'{t:.2f}' for t in inv.iloc[:, 0]])
ax.set_ylabel('target AUROC'); ax.set_xlabel('required $f_L$'); ax.set_xlim(0, 1.0); ax.invert_yaxis(); panel(ax, 'b')
finish(fig, 3, 'predictability_ceiling',
       'What is predictable, and when. (a) The best achievable AUROC rises with the fraction of the outcome-determining '
       'causal chain that is observable at prediction time. Coloured bars: the AUROC range of published models at each '
       'landmark; thin line and dot: headroom to the ceiling at the assumed fraction [SIM assumption]. Landmarks: '
       'A1/A2 day -7; B1 day +7; B2/B4 day +14; C4/C5 GvHD onset; C1/C7 diagnosis with tissue in hand. (b) The inverse: the '
       'share of the causal chain a model must observe to reach a given AUROC.')
display(md_table(inv))
''')

code(r'''
c22, c42, c70, c88 = (ceiling_auroc(f, .35) for f in (.22, .42, .70, .88))
display(Markdown(f"""**Result.** With prevalence 0.35, a model that can observe a fraction $f_L$ of the causal chain cannot exceed AUROC **{c22:.2f}** at $f_L$ = 0.22 (our assumption for pre-transplant), **{c42:.2f}** at 0.42 (day +7, with biomarkers), **{c70:.2f}** at 0.70 (at GvHD onset) and **{c88:.2f}** at 0.88 (tissue in hand). Inverting it: AUROC 0.75 needs **{f_for_auroc(.75):.2f}** of the chain observable, 0.80 needs **{f_for_auroc(.80):.2f}**, and 0.90 needs **{f_for_auroc(.90):.2f}**.

**Three consequences for this paper's claims.**
1. **Pre-transplant prediction has a low ceiling, and published models are already close to it.** Registry + HLA models reach roughly 0.60–0.66 against an assumed ceiling near {c22:.2f}. The remaining headroom is of the order of 0.05 AUROC — which is *exactly* the regime where §9 shows you need thousands of patients to demonstrate an improvement. A foundation model should therefore not be sold on pre-transplant discrimination.
2. **The largest headroom is early post-transplant** (day +7 to +14), where multimodal information is accumulating fast but current models use only two proteins and a handful of clinical variables. This is where ALLO-FM's primary task sits (B1), and it is the only place where a gain of 0.05–0.10 AUROC is both physically possible and clinically consequential.
3. **At onset, the ceiling is high and the binding constraint is the measurement, not the model.** Once tissue is in hand the question is nearly determined; this is why the pilot's tissue result (§10) replicated across laboratories while its blood result did not.

*What would falsify the framing:* a pre-transplant model that reproducibly exceeds AUROC ≈ 0.72 out-of-centre would show that $f_{-7}$ is much larger than assumed here — that pair-level biology determines more of the outcome than the post-transplant chain does. That is a measurable claim, and GvHD-Bench task A1 is where it would be tested."""))
''')

md(r'''
## 3.2 Capability claims, task by task

The table below is the paper's commitment. For each task family it states what we claim is achievable, the evidence the claim rests on, and what would have to be true for it to fail. **"Achievable"** means the out-of-centre IPCW time-dependent AUROC we expect a well-built multimodal model to reach with the data volumes in §9 — not a target to be reached by trying harder on the same data.

| Task | Claim | Achievable (out-of-centre) | Evidence basis | Status of the claim |
|---|---|---|---|---|
| **A1/A2** grade II–IV / III–IV aGvHD from pre-transplant data | Modest, near-ceiling discrimination; the real value is **calibrated counterfactual comparison between donors and prophylaxis platforms**, not ranking patients | **0.62–0.70** | registry + HLA models ≈ 0.60–0.66 †; adding immunogenetics (DPB1 TCE, B-leader, mHA load) is the plausible increment | *modest claim, high confidence* |
| **A6/A7** donor choice and prophylaxis choice | A calibrated model of the *difference* between options is attainable and is where decision value lies; it must be validated inside randomised arms | not an AUROC claim: **calibration of predicted arm difference** | BMT CTN 1703 gives a randomised anchor (n = 431) ✓ | *decision-useful, needs trial data* |
| **B1/B2** grade II–IV / III–IV aGvHD from day +7 to +14 | **The flagship claim.** A multimodal model beats MAP + clinical covariates by a margin that matters | **0.74–0.80** (vs ≈ 0.70–0.75 today) | MAGIC day-7 MAP validated at n = 1,287 ✓; headroom to the day-7 ceiling ≈ 0.08–0.12 | *falsifiable primary endpoint* |
| **B3** 7-day rolling onset alarm | Useful at fixed alert budgets; reported as alerts per 100 patient-days at fixed sensitivity, never as a bare AUROC | **0.72–0.80** per alarm-day | daily staging inputs plus labs are highly informative short-horizon †; no published equivalent | *plausible, unproven* |
| **B5/C6** non-relapse mortality | Already the strongest validated biomarker use; a model should match and then add | **0.72–0.80** | Ann Arbor and MAGIC algorithms were derived for exactly this ✓ | *high confidence, incremental* |
| **C1/C2** differential diagnosis of post-HCT diarrhoea / rash | Genuinely useful and under-served: separating GvHD from infection, drug effect and engraftment syndrome | **0.78–0.88** | the discriminating information (timing, microbiology, histology, tissue expression) is in hand at that moment | *high confidence, high value* |
| **C7** histological severity from the biopsy | A pathology-adapter task with a clear label; pathology foundation models transfer well across stains and sites | **κ 0.6–0.8** vs central read | pathology FM literature †; our pilot's tissue transcriptome analogue reached LOSO 0.80 | *high confidence* |
| **C4/C5** steroid refractoriness / day-28 response | Moderate, and worth it: this is the decision that currently has no model at all | **0.68–0.76** | onset grade + organ pattern + MAP is the baseline; our pilot at n = 45 was uninformative (0.56 [0.37, 0.77]) | *plausible, needs ~1,000 treated patients* |
| **D1/D2** second-line agent response and choice | Not claimable from observational data; specify now, validate inside trials | no claim | confounded by indication; REACH-style randomised data needed | *explicitly deferred* |
| **E1** NIH moderate/severe cGvHD by 1 year from day +100 | Moderate; chronic GvHD has a long, partly unobserved causal chain of its own | **0.65–0.72** | prior aGvHD + graft source + donor sex clinical models; day-100 biomarker panels † | *modest claim* |
| **E2** bronchiolitis obliterans | Detection of decline is tractable; prediction of onset is harder | **0.70–0.78** for FEV₁ decline | serial PFT/CT trajectories are strong predictors of their own continuation † | *plausible* |
| **F1** donor-reactive clone identification | A real but small sequence signal; no model class has beaten k-mers | **0.60–0.66** within abundance strata | our pilot: k-mers 0.645, transformer 0.646, ESM2 0.642 on an alloreactivity proxy | *low ceiling, mechanistic value only* |

**What we do not claim, anywhere in this paper.** That a foundation model improves patient outcomes (that needs a trial, §11); that pre-onset blood transcriptomes predict GvHD (the pilot found no signal); that a model can replace histological diagnosis; that any number in §3 has been measured rather than projected.
''')

md(r'''
## 3.3 The specific question: what can a transcriptomic profile **at diagnosis** support?

This deserves its own answer, because it is the modality most often proposed for GvHD and the one our pilot tested hardest. "At diagnosis" means a sample taken when GvHD is first suspected or confirmed — a GI or skin biopsy, or blood drawn at the same visit. The honest answer splits three ways.

**(1) It is well suited to confirming and refuting the diagnosis in tissue.** This is the strongest evidence we have. In the pilot, a plain normalized-gene model transferred across laboratories and across organs (colon → skin) at leave-one-study-out AUROC ≈ 0.80, and a **zero-parameter** score — the mean expression of the Hallmark interferon-γ and allograft-rejection gene sets, written down before any cohort was examined — reached 0.80–0.88 in colon and skin. Nothing was fitted. That is the signature of a signal large relative to between-laboratory variation, which is precisely the condition (high ρ, §2.1) under which a shared model is worth building. The clinically valuable form of this is not "is it GvHD" in isolation but the **differential against its mimics** (CMV colitis, C. difficile, drug effect, conditioning toxicity) — task C1/C2 — because that is the question a clinician actually faces at the bedside, and a transcriptome contains information that histology alone does not.

**(2) It is plausibly useful, but unproven, for stratifying the course once GvHD has started.** Steroid refractoriness (C4) is the decision with the least support today. Our pilot's attempt on the only public labelled set — 45 skin biopsies, 10 non-responders — returned AUROC 0.56 with a 95 % interval of [0.37, 0.77]: an interval that spans "useless" to "clinically useful" and therefore establishes nothing. The cell below computes what it would take to settle it.

**(3) It does not support pre-onset prognosis, and the pilot is the reason.** A profile taken at diagnosis is, by construction, taken after the outcome has begun; using it to "predict" GvHD is a tautology. And when the pilot looked at blood *before* onset in the only large public longitudinal cohort (plasma cfRNA, 92 patients), pre-GvHD samples gave AUROC 0.32 [0.14, 0.48] — below chance — with no reproducible signal at any timepoint, and a forecasting objective that barely beat predicting the population mean. Sorted blood T cells were separable *within* their own study (≈ 0.80) but did not transfer to any other study. Under the §3.1 framing this is unsurprising: at the time blood is drawn, most of the causal chain has not happened.

**Therefore the role of transcriptomics in ALLO-FM is as a high-value, low-coverage adapter**: it enters the model where it is informative (tissue at onset; task C1, C2, C7, and as an input to C4/C5), it is never required at inference, and its contribution is measured by modality knockout (§8.6) rather than assumed.
''')

code(r'''
# ----------------------------------------------------------------------------- what would settle the open transcriptomic questions? (n needed, and what the pilot had)
def n_for_ci(target_auroc, prev, half_width=.07):
    """Patients needed so that one AUROC's 95% CI half-width is <= half_width (Hanley-McNeil)."""
    def hm(A, n1, n0):
        Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
        return math.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n0 - 1) * (Q2 - A * A)) / (n1 * n0))
    for n in range(20, 20001, 10):
        n1 = max(int(round(n * prev)), 2)
        if 1.96 * hm(target_auroc, n1, n - n1) <= half_width: return n
    return None

TX = [  # question, the sample type, what the pilot had, pilot result, assumed prevalence, target AUROC
 ('Confirm GvHD vs no GvHD in tissue', 'GI / skin biopsy', '104 samples, 3 studies', 'LOSO 0.80; zero-parameter score 0.80-0.88', .50, .85),
 ('Differential: GvHD vs infection vs drug', 'GI biopsy + microbiology', 'none (no public adjudicated set)', 'not testable', .45, .85),
 ('Histological severity (grade >= 3)', 'biopsy WSI + RNA', 'none public', 'not testable', .30, .80),
 ('Steroid refractoriness at day 28', 'onset biopsy or blood', '45 samples, 10 non-responders', '0.56 [0.37, 0.77] - uninformative', .25, .72),
 ('NRM at 6 months from onset', 'onset blood', 'none with dates', 'not testable', .30, .75),
 ('Pre-onset prognosis from blood', 'plasma cfRNA / sorted T cells', '92 patients longitudinal', '0.32 [0.14, 0.48] - no signal', .35, .72),
]
tx = pd.DataFrame(TX, columns=['question', 'sample at diagnosis', 'what public data gave the pilot', 'pilot result',
                               'prevalence', 'target AUROC'])
tx['n for ±0.07 CI'] = [n_for_ci(a, p) for a, p in zip(tx['target AUROC'], tx['prevalence'])]
tx['n for ±0.03 CI'] = [n_for_ci(a, p, .03) for a, p in zip(tx['target AUROC'], tx['prevalence'])]
tx['n to beat a 0.65 baseline (80% power, paired)'] = [
    int(np.ceil(2 * (1.96 + 0.84) ** 2 * .055 ** 2 / max(a - .65, .02) ** 2)) for a in tx['target AUROC']]
display(md_table(tx))
display(Markdown(f"""**Result.** To *measure* a tissue-diagnosis AUROC of 0.85 to within ±0.07 needs **{n_for_ci(.85, .5)} biopsies**; to within ±0.03, **{n_for_ci(.85, .5, .03):,}**. The pilot had 104. To settle steroid refractoriness at a target of 0.72 to within ±0.07 needs **{n_for_ci(.72, .25)} treated patients** — against the 45 samples and 10 events that were publicly available, which is why that question is open rather than answered. (The last column is an approximate paired comparison against a 0.65 clinical baseline and is indicative only; §9 does this properly with the DeLong test.)

**The actionable reading.** The tissue-diagnosis claim is within reach of a single multi-centre biopsy collection in the low hundreds — the cheapest decisive experiment in this entire programme. The response-prediction claim needs about an order of magnitude more, and only a consortium can supply it."""))
''')
