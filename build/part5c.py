from part1 import CELLS, md, code

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
| Label efficiency | tissue curve still rising at k = 256; blood flat at 0.52–0.59 whatever k | blood's problem was **domain shift, not sample count** — the ρ = 0 regime of Figure 2 (and, quantitatively, the plateau of Figure 13c) |
| Data quality | 4 of 54 skin libraries were empty, including 2 of 9 controls | QC audits belong in the harness, not in the appendix |

**Three design decisions follow directly.**
1. **Change the backbone.** The modality with high ρ and registry-scale n is the clinical timeline. Omics becomes an adapter with modality dropout, so a Tier A patient still gets a prediction (§5.1).
2. **Stop relying on retrospective public specimens for prognosis.** The sampling-time confound and the absence of onset dates make pre-onset tasks unanswerable on GEO. Hence the protocolised schedule (§4.4) and `t_known` discipline (§4.7).
3. **Keep the controls.** Random-weight twins, confounder-only models, shuffled-prior controls and pre-stated gates did their job: they turned an exciting four-study result into a correctly-reported replication failure. They are mandatory in GvHD-Bench (§8.3, §8.6).

**What the pilot does not show.** It does not show that transcriptomes are useless for GvHD — tissue biopsies transferred well, and the IFN-γ programme is real. It shows that *blood transcriptome alone, from heterogeneous public studies, at n ≈ 10²* cannot support a foundation-model claim.
''')

code(r'''
# ----------------------------------------------------------------------------- Figure 18: the pilot results that this paper is built on (numbers from the earlier notebook)
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
finish(fig, 18, 'pilot_results',
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
| Backbone input | **the day-0-anchored clinical event timeline** | it is the only modality every patient has, and the one with high cross-centre shared structure (§2.1, §10); if a prospective study showed blood transcriptomes transferring across centres at ρ ≳ 0.8, an omics-first backbone would become defensible |
| Sequence model | decoder-style transformer with explicit continuous-time encoding; **not** a fixed-grid RNN | visit times are irregular and clinically meaningful; a fixed grid discards exactly the information that distinguishes day +7 from day +14 |
| What is trained at the centre | **a tiny model from scratch (≈ 0.5 M parameters) or, preferably, a released pretrained timeline model kept frozen, with a head of 5–25 parameters** (LoRA only if the data allow) | 1,500 patients support a head of 15–25 parameters and compute-optimal pretraining of well under a million (§7.2, §9.1); pretraining at tens of millions of parameters needs a network and is not part of this plan |
| Pretraining objectives (for whoever pretrains) | next-event + time-to-event + masked-value (O-T1/T2/T3) | the time-to-event objective is the closest to every downstream question, and is the one with published label-efficiency gains ✓; a single centre inherits them from a released model rather than training them |
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
| **1 · One institution, S0 (EHR timelines, ≤ 1,500 patients)** | Build the OMOP + HCT-extension extract for one large centre; implement the label functions and the harness; run the EHR-based tasks (§9.8 lists them) with the model ladder L0–L2 under the few-shot protocol (§8.15) and a temporal split | single-centre IRB; a data-use agreement for a released model if one is used | Proceed only if **Level 1C** is reached against the tuned simple model. **Also measure the quantities the plan rests on**: label reliability κ (§9.7), the shared fraction ρ and the between-centre SD τ (needs a second centre; §9.6) |
| **2 · One institution, S1/S2 (banked specimens, 420 patients; weekly cohort of 300)** | Plasma biomarkers, blood RNA and tissue RNA + slide at the schedule of §4.4; B1 against the MAGIC algorithm; the tissue-transcriptome claims of §9.4; the trajectory cohort of §9.5 | prospective consent; the S1/S2 rows of §9.9 | Proceed if a frozen adapter with a ≤ 10-parameter head adds ≥ 0.02 AUROC over Tier A+B on the temporal test set, or report the interval |
| **3 · A network, federated** | The same code at ≥ 20 centres, each running ingest, labels, splits and metrics locally and sharing summary statistics; the benchmark's Level 2 (§8.12); only now a model may be pretrained at scale | a consortium agreement; not part of this plan | Level 2 only if the adopted gate (§8.12) is passed |
| **4 · Tier D and mechanism** | Repertoire, single-cell, pathology adapters on nested sub-cohorts; tasks C7, F1–F3 | prospective protocol (§4.4), with the nested weekly trajectory cohort of §9.5 | Research output regardless; no deployment claim |
| **5 · Prospective silent mode** | Model runs without influencing care; calibration drift monitored; subgroup audit (needs the pooled cohort of stage 3) | prospective approval; about 3 years of one centre's transplants for the validation of §9.7 | Level 3 only if calibration and net benefit hold prospectively |
| **6 · Interventional trial** | Randomise the *decision* (e.g. model-guided pre-emptive therapy vs standard) | trial | Only this licenses "improves outcomes" |

**Publish the negative.** If stage 1 fails Level 1C, or stage 2 shows no gain, the result is a paper: *"adapting a released clinical foundation model does not beat tuned registry baselines for GvHD prediction at one institution, under these splits."* That is the output our pilot produced in the transcriptomic setting, and it was worth reporting.

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

1. **Nothing here is trained.** This is a specification with executed statistics and simulations, not a model. Figures 2, 8, 9 and 10 and the learning-curve, longitudinal and gate calculations of §9 rest on **[SIM]** or on stated closed-form models: their effect sizes and assumptions are ours, chosen to exercise the machinery, and they say nothing about patients.
2. **The clinical-timeline premise is itself untested for GvHD.** Label efficiency from EHR foundation models is published for other endpoints (✓ for MOTOR's reported numbers), not for GvHD. Stage 1 exists to test it, and may refute it.
3. **The data dictionary is a target, not an inventory.** No single existing cohort holds all 25 domains. The tier structure is the honest statement of what is available where.
4. **Some definitions are from domain knowledge (†)** and must be checked line-by-line against the primary consensus documents before submission: the MAGIC grade table, the 2018 steroid-refractoriness criteria, NIH 2014 organ scoring, refined DRI, HCT-CI weights, the DPB1 TCE algorithm, and the EBMT SOS criteria. The *code* in §4.6 is the authoritative version of what this benchmark means by each label, and it is testable; if it disagrees with the consensus table, the consensus table wins and the code must change.
5. **Prevalences in the registry are planning assumptions**, so every "minimum n" moves with the real prevalence of the cohort.
6. **Competing-risk discrimination is still contested.** IPCW time-dependent AUROC depends on a correct censoring model; informative censoring (transfer of care, which correlates with outcome) will bias it. We report Brier/IBS and calibration alongside, and the censoring model is part of the audit.
7. **Simulation cannot validate the premises.** Figure 2 shows *that* pretraining depends on shared structure; only real multi-centre data can show *whether* GvHD data has it.
8. **The sample-size calculators are models, with stated assumptions.** The learning curve of §9.4 assumes a linear head, an equal-covariance binormal signal and a retained-signal fraction κ that is an assumption until tasks F2/G4 measure it; the paired-power and validation sizes assume equal-variance binormal scores; the longitudinal results assume a prodrome length and a single-draw reliability that are properties of each assay, to be measured with bridging and replicate samples. Each was checked against simulation where the model is exact and stress-tested where it is not, but none was checked against real GvHD data, because that data does not yet exist.
9. **The three quantities that set the plan are unmeasured**: the between-centre SD τ of the gain, the cross-centre shared fraction ρ, and label reliability. We plan at τ = 0.02/0.03/0.04 and ρ = 0.8, and stage 1 measures them; if they are worse, §9.4, §9.6 and §9.8 give the revised requirement.
10. **The plan is one institution, and its claims are about that institution.** The ceiling cohort (1,500), the specimen cohort (420), the eligible fractions of each task, the subgroup shares and the onset-time distribution are planning assumptions (†). The result cannot speak to other centres, to subgroups, or to gains below about +0.03; those need a network (§9.6, stage 3).
11. **The foundation-model claim depends on a released model.** The plan assumes a pretrained timeline model that can be obtained under a data-use agreement and whose vocabulary can be mapped from the centre's records (OMOP). If none can be obtained, the plan falls back to the tuned simple model (L0) and the tiny model from scratch (L1), and the foundation-model claim is simply not tested; the benchmark, the data plan and the single-centre claim at Level 1C are unaffected.
12. **The prior-art search of §1.1 was not systematic.** It supports "we found no few-shot or foundation-model benchmark for GvHD", not "none exists".

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
         f"At one institution (up to {N_CAP:,} patients, {N_SPEC} with banked specimens): runnable {(reg_full.one_inst_verdict == 'RUNNABLE').sum()}; "
         f"estimable only: {(reg_full.one_inst_verdict == 'ESTIMABLE').sum()}; out of reach: {(reg_full.one_inst_verdict == 'OUT OF REACH').sum()}; "
         f"no single-AUROC verdict: {(reg_full.one_inst_verdict == '-').sum()}.", '',
         'Full machine-readable definitions: `task_registry.csv` (all fields) and `task_cards.csv`',
         '(readiness, eligible patients at one centre, requirements, verdict). Columns are documented in notebook 8.3-8.4.', '']
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
                  f'- eligible patients at one centre: {r["one_inst_n"]:,} · n for a 10-parameter head: '
                  f'{r["n to develop (10-param head)"]} · n to show +0.03: {r["n to show +0.03"]}',
                  f'- **verdict at one centre: {r.one_inst_verdict}**', '']
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
          '- treatment-proxy-only model, run deliberately as a leakage canary',
          '- presence-only model (whether and when each lab/drug/assay was recorded, no values); the audit warns if it beats chance by > 0.02',
          '- permuted-outcome and negative-exposure controls (task G7): the false-pass rate of the whole pipeline', '',
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
                     ('2', 'better out-of-centre discrimination (>= 20 held-out centres; centre-level 95% t-interval of the paired AUROC difference has a lower bound >= +0.01 against the best simple model AND the clinical baseline; improves in >= 70% of held-out centres; survives BH; beats random-weight twin and no-pretraining)'),
                     ('3', 'clinically useful (Level 2 plus calibration slope 0.8-1.25, positive net benefit 10-40%, leave-centre-out AND temporal validation, n >= the Riley minimum, prospective silent-mode evaluation)'),
                     ('4', 'improves outcomes — requires a prospective trial of the model-guided decision; nothing less licenses the word')):
        L.append(f'- **Level {lvl}**: {txt}')
    L += ['', '## 9 Pre-registered gate for the v1 attempt', '', '```json', json.dumps(GATE, indent=2), '```', '',
          'Operating characteristics of this gate (notebook 8.12): the centre-level t-interval keeps its 95% coverage at every number of centres, whereas a percentile bootstrap over centres does not;',
          f'at 20 centres of ~300 patients and a between-centre SD of 0.02 the gate passes a true gain of +0.03 with probability {g20(.03, "pass_adopted"):.2f} and a true gain of +0.01 with probability {g20(.01, "pass_adopted"):.2f}.', '',
          '## 10 Benchmark-validity requirements', '',
          '- **Issue day**: a prediction is issued on the landmark day and may use a specimen only once its result exists (draw day + assay turnaround).',
          '- **Label reliability**: at least 10% of outcomes double-read; kappa >= 0.8, else the attenuation of notebook 9.7 is applied to every required n.',
          '- **Subgroups (G3)**: a subgroup is assessed only with the events notebook 9.7 requires (about 1,000 patients for an equal-opportunity gap of 0.10); otherwise it is reported as not assessable.',
          '- **Drift (G1)**: report observed/expected incidence per calendar year and the recalibration plan.',
          '- **Utility**: report trial-size saving under enrichment and alerts per 100 patient-days alongside AUROC (notebook 8.13).',
          '- **Target-trial emulation (G6)**: reproduce the BMT CTN 1703 GRFS effect (HR 0.64, 95% CI 0.49-0.83) from registry data.', '',
          '## 11 Required artefacts for a published result', '',
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
# ----------------------------------------------------------------------------- export the data requirements as a standalone document + CSV
T = lambda df, **kw: md_table(df, **kw).data
def _req_md():
    L = ['# Data requirements for ALLO-FM and GvHD-Bench v1.0 — one institution', '',
         f'Generated by `ALLO_FM_GvHD_foundation_model.ipynb` on {time.strftime("%Y-%m-%d", time.gmtime())}. Every number is computed in the notebook (section 9). '
         f'The plan assumes **one large transplant centre**: about {TRANSPLANTS_PER_YEAR} transplants a year, **at most {N_CAP:,} patients** with an EHR timeline and {N_SPEC} with banked specimens. '
         'Items marked † are planning assumptions. No costs are estimated.', '',
         '## 1 Headline numbers', '',
         f'- **Show a gain**: +0.03 AUROC over a correlated baseline needs about {n80(.03, .85):,} patients (score correlation 0.85); +0.05 needs {n80(.05, .85):,}; +0.02 needs {n80(.02, .85):,}, beyond one centre.',
         f'- **Head size**: {N_CAP:,} patients support a head of about {max_params(N_CAP, .72, .35)} parameters (AUROC 0.72, prevalence 0.35); the {N_DEV} patients of a 60/40 temporal development set about {max_params(N_DEV, .72, .35)}; the {N_SPEC} with specimens about {max_params(N_SPEC, .72, .35)}.',
         f'- **Model**: a tiny timeline model from scratch is about {fmt_n(N_tiny)} parameters ({fmt_t(H_tiny)} on one A100); the preferred route is a released pretrained model kept frozen with a head of 5–25 parameters (embeddings for the cohort in {fmt_t(H_emb)}).',
         f'- **Tasks runnable at one institution**: {len(vs_ids["RUNNABLE"])} of {int((cards_df.one_inst_verdict != "-").sum())} numeric-target tasks ({", ".join(vs_ids["RUNNABLE"])}); {len(vs_ids["ESTIMABLE"])} estimable only ({", ".join(vs_ids["ESTIMABLE"])}); {len(vs_ids["OUT OF REACH"])} out of reach ({", ".join(vs_ids["OUT OF REACH"])}).',
         f'- **Transcriptomes at diagnosis**: a pre-specified tissue score is validated on {n_for_ci(.85, .5, .10)}–{n_for_ci(.85, .5, .07)} biopsies ({N_BIOPSY} are collected in three years); a 10-parameter adapter head *learns* tissue differential diagnosis from {TC_NUM[0][0]:,.0f} profiles but showing +0.05 over the clinical baseline takes {TC_NUM[0][1]:,.0f} ({ONE("C1")} eligible, marginal); steroid refractoriness, the day-7 blood transcriptome and any whole-transcriptome head are not feasible at one centre.',
         f'- **Longitudinal**: {k_needed(.80, .5, .02)} draws per window at single-draw reliability 0.5; draws 2–3.5 days apart for a 10-day prodrome; about {WEEKLY_N} patients with weekly draws for 100 evaluable cases.',
         f'- **Level-1C gate**: passed with probability {level1c_power(N_TEST, .05):.0%} for a true gain of +0.05 on {N_TEST} temporal-test patients; about {level1c_power(N_TEST, .01):.0%} for +0.01.', '',
         '## 2 Capacity of the cohort sizes one centre has', '', T(cap_si), '',
         '## 3 Task feasibility at one institution', '', T(feas), '',
         '## 4 Transcriptome claims', '', T(plan), '', T(tc), '',
         '## 5 Longitudinal design', '', T(kn), '', T(prod), '', T(disp_sched), '', T(lg), '', T(icc_tab), '',
         '## 6 Validation, subgroups, label noise', '', T(v_show[['AUROC', 'prevalence', 'patients', 'events', 'binding criterion', 'years of silent-mode at one centre']]), '', T(sg), '', T(ltab), '',
         '## 7 The data plan', '', T(disp_si), '',
         '## 8 Assumptions (†)', '',
         f'- {TRANSPLANTS_PER_YEAR} transplants a year; {EHR_YEARS:g} years of EHR timelines; {SPECIMEN_YEARS} years of banked specimens at {USABLE_FRACTION:.0%} usable; about 20 % of transplants biopsied for suspected GvHD.',
         '- Eligible fraction of transplants per task (§8.4); onset of grade II–IV aGvHD log-normal with median day 28; pre-onset window 14 days; subgroup shares as listed in section 6.',
         '- Retained-signal fraction κ of a frozen adapter = 0.9 (to be measured by tasks F2 and G4); the effective dimension of a raw-gene head is inflated by the nuisance multiplier measured in §9.4.']
    return '\n'.join(L)
bench_req_path = os.path.join(SPEC_DIR, 'DATA_REQUIREMENTS_v1.0.md')
open(bench_req_path, 'w').write(_req_md())
plan_si.to_csv(os.path.join(SPEC_DIR, 'data_requirements.csv'), index=False)
print(f'wrote {bench_req_path} ({os.path.getsize(bench_req_path) / 1024:.0f} kB) and data_requirements.csv')
''')

code(r'''
# ----------------------------------------------------------------------------- export the specification bundle and verify it
manifest = {}
for f, desc in [('data_dictionary.csv', f'{len(dd)} data elements in {dd.domain.nunique()} domains'),
                ('sampling_schedule.csv', f'{len(sched)} specimen types x {len(SCHED_DAYS)} visits'),
                ('task_registry.csv', f'{len(reg_full)} benchmark tasks, full fields'),
                ('task_cards.csv', 'readiness, accessible n, requirements, powered verdict'),
                ('GVHD_BENCH_v1.0.md', 'standalone benchmark specification'),
                ('gate.json', 'pre-stated gate for the ALLO-FM v1 attempt'),
                ('DATA_REQUIREMENTS_v1.0.md', 'standalone data-requirement specification: what to collect, how much, how often'),
                ('data_requirements.csv', f'{len(plan_si)} rows: the S0/S1/S2 data plan, patients, specimens, storage, time')]:
    p = os.path.join(SPEC_DIR, f); manifest[f] = dict(description=desc, bytes=os.path.getsize(p),
        sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16])
json.dump(manifest, open(os.path.join(SPEC_DIR, 'MANIFEST.json'), 'w'), indent=2)
display(md_table(pd.DataFrame(manifest).T.reset_index().rename(columns={'index': 'file'})))

# ----------------------------------------------------------------------------- consistency check: every number quoted in the abstract/summary must match what was computed
checks = [
 ('data elements in the dictionary', len(dd), 129, 0),
 ('domains in the dictionary', dd.domain.nunique(), 25, 0),
 ('elements required for ALLO-FM v1', int((dd.priority == 'R').sum()), 74, 0),
 ('benchmark tasks', len(reg_full), 40, 0),
 ('figures written (PDF + PNG each)', len(FIGS), 18, 0),
 ('tokens per patient, Tier A+B', tok_by_tier['B'], 7580, 100),
 ('task families', reg.family.nunique(), 7, 0),
 ('Riley n, P=10, AUROC .72, prev .35', riley_n(10, .72, .35)['n'], 621, 30),
 ('tasks: runnable at one centre', int((cards_df.one_inst_verdict == 'RUNNABLE').sum()), 12, 0), ('tasks: estimable only', int((cards_df.one_inst_verdict == 'ESTIMABLE').sum()), 9, 0),
 ('tasks: out of reach', int((cards_df.one_inst_verdict == 'OUT OF REACH').sum()), 4, 0),
 ('tasks: no single-AUROC verdict', int((cards_df.one_inst_verdict == '-').sum()), 15, 0),
 # ---- the one-institution plan: every number the abstract and section 9 quote
 ('ceiling cohort: patients with an EHR timeline', N_CAP, 1500, 0),
 ('patients with banked specimens', N_SPEC, 420, 0),
 ('biopsies in three years', N_BIOPSY, 120, 0),
 ('largest package in the data plan (patients; must not exceed the ceiling)', int(plan_si.patients.max()), 1500, 0),
 ('largest head, 1,500 patients (AUROC .72, prevalence .35)', max_params(N_CAP, .72, .35), 24, 0),
 ('largest head, 900-patient development set', max_params(N_DEV, .72, .35), 14, 0),
 ('largest head, 420 patients with specimens', max_params(N_SPEC, .72, .35), 6, 0),
 ('patients to show +0.03 over a correlated baseline (r = .85)', n80(.03, .85), 836, 15),
 ('patients to show +0.05', n80(.05, .85), 299, 10),
 ('patients to show +0.02 (beyond one centre)', n80(.02, .85), 1893, 40),
 ('tiny timeline model from scratch, thousand parameters', N_tiny / 1e3, 568, 6),
 ('whole one-institution programme, A100 minutes', PROG_H * 60, 19, 2),
 ('Level-1C pass probability, true gain +0.05, 600 test patients', level1c_power(N_TEST, .05), .89, .02),
 ('Level-1C pass probability, true gain +0.03', level1c_power(N_TEST, .03), .35, .03),
 ('Level-1C pass probability, true gain +0.01', level1c_power(N_TEST, .01), .02, .02),
 ('few-shot: patients per class, 5-parameter head within 0.02 of 0.75', kw(5), 52, 3),
 ('few-shot: 10-parameter head', kw(10), 104, 5),
 ('few-shot: 25-parameter head', kw(25), 260, 10),
 ('biopsies to validate a pre-specified tissue score to +-0.07', n_for_ci(.85, .5, .07), 120, 0),
 ('tissue differential diagnosis: patients to show +0.05 with a 10-parameter head', TC_NUM[0][1], 241, 6),
 ('eligible patients for C1/C2 at one centre', ONE('C1'), 210, 0),
 ('weekly cohort of 420: cases with two or more pre-onset draws', int(lg['cases with ≥ 2 pre-onset draws'].iloc[-1]), 141, 4),
 ('draws per window to come within 0.02 of the oracle (reliability .5)', k_needed(.80, .5, .02), 6, 0),
]
if not QUICK:                                          # the benchmark's own operating characteristics are simulated: compare them to the abstract only at full replicate counts
    checks += [
     ('percentile bootstrap coverage, 8 centres', cb(8, 'cov_boot'), .88, .03), ('percentile bootstrap coverage, 20 centres', cb(20, 'cov_boot'), .93, .03),
     ('obvious gate: pass probability at a true gain of exactly +0.03 (20 centres)', g20(.03, 'pass_obvious'), .51, .06),
     ('adopted gate: pass probability at +0.03 (20 centres)', pw_(20, .03), .85, .05), ('adopted gate: pass probability at +0.01', pw_(20, .01), .02, .03)]
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
3. Wornow M, Thapa R, Steinberg E, Fries JA, Shah NH. **EHRSHOT: an EHR benchmark for few-shot evaluation of foundation models.** NeurIPS Datasets & Benchmarks 2023; arXiv:2307.02028. One institution (Stanford); de-identified structured records of 6,739 patients; 15 few-shot tasks; the released CLMBR-T-base (141 M parameters, pretrained on 2.57 M patients' records) used frozen with a logistic-regression head; access by data-use agreement. ✓ (the template of this paper, §1.1)
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
37. Raudys S, Duin RPW. **Expected classification error of the Fisher linear classifier with pseudo-inverse covariance matrix.** *Pattern Recognit Lett* 19:385–392 (1998). † (the finite-sample learning curve of a plug-in linear classifier used in §9.4)
38. Riley RD, Debray TPA, Collins GS, et al. **Minimum sample size for external validation of a clinical prediction model with a binary outcome.** *Stat Med* 40:4230–4251 (2021). † (the calibration-slope and O/E precision targets of §9.7)
39. Hoffmann J, et al. **Training compute-optimal large language models.** arXiv:2203.15556 (2022). † (≈ 20 tokens per parameter) · Muennighoff N, et al. **Scaling data-constrained language models.** *NeurIPS* 2023. † (repeating data up to ≈ 4 epochs)
40. Spearman C. *Br J Psychol* 3:271–295 (1910) and Brown W. *Br J Psychol* 3:296–322 (1910). † (reliability of an average of *k* measurements, §9.5)
41. Hernán MA, Robins JM. **Using big data to emulate a target trial when a randomized trial is not available.** *Am J Epidemiol* 183:758–764 (2016). † (task G6)
42. Agniel D, Kohane IS, Weber GM. **Biases in electronic health record data due to processes within the healthcare system: retrospective observational study.** *BMJ* 361:k1479 (2018). † (when and whether a test is ordered carries information about the patient; the mechanism behind the presence-only comparator of §8.12)

**Prior GvHD prediction work (§1.1)**
43. Arai Y, et al. **Using a machine learning algorithm to predict acute graft-versus-host disease following allogeneic transplantation.** *Blood Adv* 3(22):3626–3634 (2019). 26,695 Japanese recipients; alternating decision tree; 15 of > 40 variables; validation incidence 29.0 % vs 58.7 %, HR 2.57. ✓
44. Jo et al. **A convolutional neural network-based model that predicts acute graft-versus-host disease after allogeneic hematopoietic stem cell transplantation.** *Commun Med* 3:67 (2023); doi:10.1038/s43856-023-00299-5. 18,763 adults from the Japanese registry. ✓
45. Chandra et al. Machine-learning models for acute GvHD after allogeneic transplantation for T-cell prolymphocytic leukaemia, CIBMTR data. *BMC Med Res Methodol* 24:112 (2024). Best test balanced accuracy 0.58. ✓ (article checked; exact title †)
46. **MatchGraft.AI** — donor–recipient-pair aGvHD risk for donor selection. ASH 2023 abstract (not peer-reviewed): 1,035 donor–recipient sets; AUC 0.70; integrated calibration index 0.028. ✓
47. **Early GVHD prediction in liver transplantation via multi-modal deep learning on imbalanced EHR data.** arXiv:2511.11623 (2025). ✓ (title as retrieved; methods not checked †)
48. **Machine learning applications and challenges in graft-versus-host disease: a scoping review.** ✓ (title as retrieved; authors and venue †)

---
## Appendix A — Reproducibility

* **Runtime.** CPU only. Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib. End-to-end run ≈ 6–10 minutes on a laptop CPU; set the environment variable `ALLOFM_QUICK=1` to cut simulation replicates (numbers get noisier, the harness uses 8 held-out centres so the Level-2 gate reports *not evaluable*, and the consistency tolerances still apply).
* **Seeds.** Global seed in the setup cell; every simulation takes an explicit seed. The [SIM] cohort generator is `simulate_cohort(...)` and is deterministic given its seed.
* **Exported specification** (written to `spec/`, hashes in `spec/MANIFEST.json`): `data_dictionary.csv`, `sampling_schedule.csv`, `task_registry.csv`, `task_cards.csv`, `GVHD_BENCH_v1.0.md`, `gate.json`, and the data-requirement bundle `DATA_REQUIREMENTS_v1.0.md`, `data_requirements.csv`.
* **Figure layout gate.** Every figure is measured before it is saved: the build fails if any two pieces of rendered text overlap, if text sits on a data marker, line or bar, or if any font is below 6 pt (set `ALLOFM_STRICT_FIGS=1` to make a failure stop the run; otherwise it is printed as a warning). The whole suite passes under both Liberation Sans and the wider DejaVu Sans that Colab falls back to (`ALLOFM_FONT="DejaVu Sans"`).
* **Tests executed in-notebook.** MAGIC grading vignettes; steroid-response and GRFS vignettes; fast-vs-textbook DeLong agreement; Hanley–McNeil versus simulated standard error; IPCW td-AUROC against the uncensored AUROC; IPCW recovery under censoring; Aalen–Johansen against empirical incidence; net-benefit bounds for perfect and random models; Riley criterion-1 arithmetic; the leakage audit on a cohort with a planted leak; closed-form paired power against a Monte-Carlo DeLong test; the transcriptome learning-curve formula against simulated heads, with a stress test; Spearman–Brown against simulation; closed-form gate power against the gate simulation; calibration-slope Fisher information against simulation; the label-noise attenuation formula against simulation; interval coverage and gate operating characteristics; planted-effect recovery, a permuted-outcome control and an informative-presence test through the real harness; and a final consistency check of every number quoted in the narrative.
* **What is *not* in this notebook.** Any trained foundation model, any real patient data, and any claim about clinical performance. The pilot results in §10 come from the executed notebook `GvHD_FM_benchmark_blueprint.ipynb` in this repository.

## Appendix B — How to use this specification
1. **To build a cohort:** take `spec/data_dictionary.csv` as the extraction checklist; implement `t_valid` / `t_known` for every element; implement the §4.6 label functions; run the §8.7 audit before any modelling.
2. **To run the benchmark:** take `spec/task_registry.csv`; for each task fix the landmark (the *issue* day), horizon and competing events as given; use the §8.6 metrics; report every mandatory baseline from §8.5 in the same table, including the presence-only and permuted-outcome controls.
3. **To make a claim:** check it against the ladder in §8.10 and against `spec/gate.json`, whose hash should be reported with the results.
4. **To plan a data collection:** read `spec/DATA_REQUIREMENTS_v1.0.md` for the package that matches the claim (S0, S1 or S2), replace the planning values (τ, ρ, κ, transplants per year, usable fraction) with measured ones as stage 1 produces them, and re-run §9.
''')
