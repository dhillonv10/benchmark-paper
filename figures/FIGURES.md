# Figures

Vector PDF and 600-dpi PNG of each figure in the paper.

## Fig 1 — decision_timeline

The allo-HCT timeline. Bars: the protocolised exposures and the risk windows they open. Numbered markers: the clinical decisions, each of which is a family of GvHD-Bench tasks (task ids in brackets; section 8). The time axis is signed-log, so the first 100 days, where nearly every decision sits, are legible beside two years of follow-up. A model standing at any decision may use only data whose results were known then.

Files: `Fig01_decision_timeline.pdf`, `Fig01_decision_timeline.png`

## Fig 2 — pretraining_vs_shared_structure

[SIM] Label-free pretraining pays in proportion to the structure shared across centres. (a) AUROC at a new centre against the number of labelled patients there, at four levels of shared structure rho. (b) The same result as the gain over the tuned raw-feature baseline: positive at few labels and high rho, absent at rho = 0, and shrinking everywhere as labels accumulate. Simulated data; effect sizes follow from the stated model.

Files: `Fig02_pretraining_vs_shared_structure.pdf`, `Fig02_pretraining_vs_shared_structure.png`

## Fig 3 — predictability_ceiling

What is predictable, and when. (a) The best achievable AUROC rises with the fraction of the outcome-determining causal chain that is observable at prediction time. Coloured bars: the AUROC range of published models at each landmark; thin line and dot: headroom to the ceiling at the assumed fraction [SIM assumption]. Landmarks: A1/A2 day -7; B1 day +7; B2/B4 day +14; C4/C5 GvHD onset; C1/C7 diagnosis with tissue in hand. (b) The inverse: the share of the causal chain a model must observe to reach a given AUROC.

Files: `Fig03_predictability_ceiling.pdf`, `Fig03_predictability_ceiling.png`

## Fig 4 — architecture

ALLO-FM. The backbone is the day-0-anchored clinical event timeline, which every transplant patient has. Biology enters through modality adapters that each contribute one summary token at the time its result became known; they are trained with modality dropout and are therefore optional at inference. Every head reads only tokens whose results were available at its landmark L; the mask is enforced inside the attention and tested in section 8.11.

Files: `Fig04_architecture.pdf`, `Fig04_architecture.png`

## Fig 5 — knowledge_time_and_time_encoding

Knowledge time and time encoding. (a) One patient timeline. Filled marker: when an event happened; open marker: when its result became known. An event is a legal input at landmark L only if its result was known by then, which is not the same as having happened by then: the day-20 biopsy signs out on day +23 and is inadmissible, and the day-21 steroid order is banned outright as a proxy for the outcome. (b) The signed-log time encoding of the model keeps early days distinguishable (day +7 and +14 are different decisions) and compresses late follow-up.

Files: `Fig05_knowledge_time_and_time_encoding.pdf`, `Fig05_knowledge_time_and_time_encoding.png`

## Fig 6 — training_requirements

What one institution has to train. (a) Event tokens per patient, cumulative across data tiers: the EHR timeline (Tier B) supplies nearly all of them, and the deep assays add almost none. (b) The compute-optimal size of a model trained from scratch on a centre's own corpus (up to 1,500 patients) is a fraction of a million parameters, two to three orders below a released 141 M-parameter model (dashed), which is why the plan adapts a released model with a small head rather than pretraining. (c) GPU time on one A100 at 30% utilisation: a tiny model from scratch, one LoRA fine-tune of the released model, the whole enumerated programme, and a general 7B language model for scale. Compute is not the binding constraint.

Files: `Fig06_training_requirements.pdf`, `Fig06_training_requirements.png`

## Fig 7 — gvhd_bench_structure

GvHD-Bench v1.0. (a) The 40 tasks, grouped by family and by the landmark at which the prediction is made. Marker shape is data readiness and colour is the feasibility verdict at one institution (up to 1,500 patients, about 420 with banked specimens); tasks that are not a single-AUROC comparison (donor ranking, counterfactual contrasts, representation and robustness audits) and research-only tasks are grey. (b) For the 25 tasks with a numeric requirement: the smallest AUROC gain over the task baseline that the eligible patients of one centre can show with 80% power. The dotted line at +0.05 separates runnable from estimable; tasks to its right are labelled, not omitted.

Files: `Fig07_gvhd_bench_structure.pdf`, `Fig07_gvhd_bench_structure.png`

## Fig 8 — harness_task_B1

[SIM] The GvHD-Bench harness executing task B1 (grade II-IV aGvHD by day +100 from the day +7 state) on a simulated 24-centre cohort. (a) Leave-centre-out IPCW time-dependent AUROC, one point per held-out centre, with the mean printed above; the pooled mean alone would hide this spread. (b) Difference from the clinical baseline with a centre-level t-interval. (c) Calibration against the Aalen-Johansen cumulative incidence, the competing-risk analogue of a calibration plot. (d) Decision-curve net benefit. (e) The leakage canary: one planted treatment-proxy feature inflates the headline number, and the audit flags it before any model is fitted. Effect sizes belong to the simulation.

Files: `Fig08_harness_task_B1.pdf`, `Fig08_harness_task_B1.png`

## Fig 9 — benchmark_self_validation

[SIM] The benchmark tested like a model. (a) Coverage of the nominal 95% interval for the true gain: the centre-level t-interval holds its level at every number of centres; the percentile bootstrap over centres does not. (b) Probability of passing the gate against the true gain. The obvious gate (point estimate at least +0.03, bootstrap bound above 0) passes a threshold-sized gain about half the time; the adopted gate (t-interval lower bound at least +0.01) is powered for +0.03 at 20 centres and almost never passes +0.01 or less. Dotted lines mark 80% power, the 5% false-pass level and the designed-for gain. (c) Effects of known size planted into the simulated cohort and recovered through the real harness code; the planted null is covered by its interval and does not pass. (d) Informative presence: a model that sees only whether the biomarker panel was ordered scores above chance as soon as clinicians order it selectively, which is why presence-only is a mandatory baseline.

Files: `Fig09_benchmark_self_validation.pdf`, `Fig09_benchmark_self_validation.png`

## Fig 10 — clinical_translation

[SIM / closed form] Three translations of the same score. (a) Calibration decays under practice drift while discrimination barely moves: observed/expected incidence of a static model against calendar year, and after re-calibration on the previous year. (b) Prognostic enrichment: patients enrolled (solid) and pool that must be ranked to find them (dotted) for a two-arm trial of a therapy that cuts risk by 30% in a population with a 13% event rate, against the fraction enrolled from the top of the risk ranking. Even AUROC 0.65 shrinks the trial substantially, at the price of a pool about twice as large; the better the score, the smaller the pool needed. (c) Alert burden of a rolling 7-day onset alarm at an onset rate of 0.04 per alarm-day, which decides whether anyone will use it.

Files: `Fig10_clinical_translation.pdf`, `Fig10_clinical_translation.png`

## Fig 11 — sample_size_and_power

What one institution's cohort can resolve. (a) Precision: the 95% confidence interval of a single AUROC against the number of patients; at the ceiling cohort of 1,500 it is about ±0.03. (b) Power: the paired DeLong test for an improvement of 0.02 to 0.05 AUROC over a correlated baseline, up to the ceiling cohort; lines are the closed form, circles are a Monte-Carlo DeLong test; solid lines are scores correlated at 0.85, dashed at 0.60. A gain of 0.03 is within reach at score correlation 0.85; 0.02 is not. (c) Development: the Riley minimum sample size against the number of candidate parameters of the head, with the ceiling cohort dotted: it supports a head of roughly 15 to 25 parameters, which is why this plan adapts a model and fits a small head rather than training a large one.

Files: `Fig11_sample_size_and_power.pdf`, `Fig11_sample_size_and_power.png`

## Fig 12 — one_institution_accrual

What a single institution can reach, and how fast. (a) The largest head the cohort supports (Riley criteria, AUROC 0.72, prevalence 0.35) against years of accrual, at 100, 200 and 300 transplants a year, for EHR timelines (solid) and for prospectively banked specimens at 70% usable (dashed). A head of 10 parameters is reached within about 3 years of EHR history at 200 a year; specimens take longer. (b) Years of accrual each task needs for a head of 10 parameters and a +0.05 gain over its clinical baseline, by annual volume: green within three years, blue within the 7.5-year ceiling cohort, orange longer.

Files: `Fig12_one_institution_accrual.pdf`, `Fig12_one_institution_accrual.png`

## Fig 13 — transcriptome_requirements

Labelled transcriptomes at diagnosis, inside the ceiling cohort (closed form, validated against simulation in §9.4). (a) Learning curves at a ceiling AUROC of 0.80: a head on the whole transcriptome is still at chance over the whole range a centre can reach, a 64-dimensional adapter head is slow, and a head of 10 parameters on a frozen adapter is close to its plateau within a few hundred profiles; the dotted line marks the 420 patients with banked specimens. (b) Labelled profiles a 10-parameter head needs to add +0.03 AUROC to a baseline, against the gain the transcriptome could add at its ceiling, with the 1,500-patient ceiling and the 420 specimen patients marked: small ceiling gains cannot be realised at one centre. (c) What a second centre would see: the plateau falls as the shared fraction rho falls, and more training centres push it toward its limit; this is the reason a single-centre result is a claim about that centre.

Files: `Fig13_transcriptome_requirements.pdf`, `Fig13_transcriptome_requirements.png`

## Fig 14 — longitudinal_design

Longitudinal sampling design. (a) AUROC against the number of draws averaged in a window, for four single-draw reliabilities (oracle AUROC 0.80); lines are the Spearman-Brown closed form, circles are simulation. (b) Expected AUROC of a pre-onset alarm against the interval between draws, for prodromes of 5, 10 and 21 days: sampling less often than the prodrome is long loses the trajectory. (c) The fraction of cases that have at least two specimens in the 14 days before onset under the minimum protocol of section 4.4 and under a weekly schedule to day +56, for each specimen type, against a realistic onset distribution.

Files: `Fig14_longitudinal_design.pdf`, `Fig14_longitudinal_design.png`

## Fig 15 — centres_validation_subgroups

Centres, validation, and subgroups. (a) Context for the benchmark's Level 2, not the plan of this paper: centres needed for the gate to have 80% power for a true gain of +0.03, against the between-centre SD of the gain, for three centre sizes; once heterogeneity dominates, more centres help and more patients per centre do not. (b) Patients for a Level-3 silent-mode validation (calibration slope SE 0.10, O/E SE 0.10, AUROC half-width 0.05) against prevalence, with the 1,500-patient ceiling and three years of one centre's transplants marked. (c) The size of a subgroup at one centre against the roughly 1,000 patients that detecting a 0.10 sensitivity gap needs: no subgroup of a single centre reaches it, so a single-centre equity audit is descriptive.

Files: `Fig15_centres_validation_subgroups.pdf`, `Fig15_centres_validation_subgroups.png`

## Fig 16 — one_institution_capacity

What one institution can resolve. (a) The smallest paired AUROC gain over a correlated baseline shown with 80% power (solid) and the half-width of one AUROC's 95% interval (dashed), at the cohort sizes a centre actually has: 120 biopsies, 420 patients with banked specimens, a 600-patient temporal test set, a 900-patient development set, and the 1,500-patient ceiling. (b) The largest head, in candidate parameters, that those cohorts support: about 6 on the 420 patients with specimens and about 14 on the 900-patient development set (AUROC 0.72, prevalence 0.35). (c) Operating characteristics of the single-centre gate (lower bound of the paired interval at least +0.01 on a temporal test set): at 600 test patients it passes a true gain of +0.05 89% of the time, +0.03 35% and +0.01 or less about 2%. Dotted lines mark 80% and 5%.

Files: `Fig16_one_institution_capacity.pdf`, `Fig16_one_institution_capacity.png`

## Fig 17 — one_institution_data_plan

The one-institution data plan. (a) Three nested packages, none above the 1,500-patient ceiling: S0 uses the EHR timeline only; S1 adds specimens banked on routine draws for three years, sequenced as a case-cohort; S2 adds a weekly trajectory cohort inside S1. Counts are computed from the sampling schedule of section 4.4 and the onset distribution of section 9.5. (b) Accrual at 200 transplants a year: EHR timelines reach the ceiling in 7.5 years (the first 4.5 years are retrospective), and patients with specimens (70% usable) reach 420 in three.

Files: `Fig17_one_institution_data_plan.pdf`, `Fig17_one_institution_data_plan.png`

## Fig 18 — pilot_results

The pilot this paper is built on (9 public studies, 838 samples; numbers from the earlier notebook in this repository). (a) Leave-one-study-out transfer with plain normalised genes: tissue biopsies transfer across laboratories and organs, blood and sorted T cells do not. (b) Frozen Geneformer (GF) at three sizes against its own random-weight twins and against plain genes: one size beats its twin, none beats the simple baseline, and the largest model is not the best. (c) A zero-parameter score written down in advance, the mean expression of the interferon-gamma and allograft-rejection gene sets, matches fitted models in inflamed tissue and fails in sorted blood T cells.

Files: `Fig18_pilot_results.pdf`, `Fig18_pilot_results.png`

