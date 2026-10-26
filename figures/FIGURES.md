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

Training requirements, computed from the data specification. (a) Event tokens per patient, cumulative across data tiers: the EHR timeline (Tier B) supplies nearly all of them, and the deep assays add almost none while costing the most to obtain. (b) Compute-optimal model size at 20 tokens per parameter: a realistic corpus implies tens to low hundreds of millions of parameters, not billions (Tier A+B+C lies on the A+B line). (c) Pretraining cost at $2 per A100-hour, 30% utilisation and 4 epochs: one run (100,000 patients, Tier A+B), the whole enumerated programme (§7.2 table), and a general 7B language model. Compute is not the binding constraint; assay and collection cost is (§9.8).

Files: `Fig06_training_requirements.pdf`, `Fig06_training_requirements.png`

## Fig 7 — gvhd_bench_structure

GvHD-Bench v1.0. (a) The 40 tasks, grouped by family and by the landmark at which the prediction is made. Marker shape is data readiness and colour is the powered verdict; tasks that are not a single-AUROC comparison (donor ranking, counterfactual contrasts, representation and robustness audits) are grey. (b) For the 26 tasks with a numeric requirement: patients accessible divided by patients needed to demonstrate a +0.03 AUROC gain over the task baseline. Bars left of 1 are tasks the field cannot currently settle; they are labelled, not omitted.

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

Three different sample-size questions. (a) Precision: the 95% confidence interval of a single AUROC against the number of patients. (b) Power: the paired DeLong test for an improvement of 0.02 to 0.05 AUROC over a correlated baseline, the regime any foundation-model claim lives in; lines are the closed form, circles are a Monte-Carlo DeLong test; solid lines are scores correlated at 0.85, dashed at 0.60. (c) Development: the Riley minimum sample size against the number of candidate parameters, which is why head size sets the data requirement.

Files: `Fig11_sample_size_and_power.pdf`, `Fig11_sample_size_and_power.png`

## Fig 12 — data_landscape_and_accrual

What exists against what is needed, and how long it would take. (a) Patients available from each real source on a log scale, against the Riley minimum for two head sizes: public transcriptomic resources are orders of magnitude short, registry extracts are not. (b) Projected time to reach the patients required for each task, by route. Tasks that need protocolised biospecimens cannot be served by a registry extract at all, which is why the staged plan buys the registry tasks first and the biobank tasks second.

Files: `Fig12_data_landscape_and_accrual.pdf`, `Fig12_data_landscape_and_accrual.png`

## Fig 13 — transcriptome_requirements

Labelled transcriptomes needed at diagnosis (closed form, validated against simulation in §9.4). (a) Learning curve of a head on the whole transcriptome against heads on a 64-dimensional adapter embedding that retains a fraction kappa of the signal: the embedding wins at the label counts a consortium can reach, and the whole-transcriptome head approaches its ceiling only at several hundred thousand profiles. Dashed: the within-centre ceiling. (b) Labelled profiles an adapter needs to add +0.03 AUROC to a baseline, against the gain the transcriptome could add at its ceiling; a weak ceiling gain cannot be realised at any n once centres only partly share the signal. (c) The plateau at a new centre falls as the shared fraction rho falls, and more training centres push it toward its limit.

Files: `Fig13_transcriptome_requirements.pdf`, `Fig13_transcriptome_requirements.png`

## Fig 14 — longitudinal_design

Longitudinal sampling design. (a) AUROC against the number of draws averaged in a window, for four single-draw reliabilities (oracle AUROC 0.80); lines are the Spearman-Brown closed form, circles are simulation. (b) Expected AUROC of a pre-onset alarm against the interval between draws, for prodromes of 5, 10 and 21 days: sampling less often than the prodrome is long loses the trajectory. (c) The fraction of cases that have at least two specimens in the 14 days before onset under the minimum protocol of section 4.4 and under a weekly schedule to day +56, for each specimen type, against a realistic onset distribution.

Files: `Fig14_longitudinal_design.pdf`, `Fig14_longitudinal_design.png`

## Fig 15 — centres_validation_subgroups

How many centres, how many validation patients, how large a subgroup. (a) Centres needed for the adopted gate to have 80% power for a true gain of +0.03, against the between-centre SD of the gain, for three centre sizes: once heterogeneity dominates, more centres help and more patients per centre do not. The dotted line is the gate's minimum. (b) Patients for a Level-3 validation (calibration slope SE 0.10, O/E SE 0.10, AUROC half-width 0.05) against prevalence, set against what one centre collects in three years and what a 20-centre consortium collects in one. (c) Total cohort at which a subgroup of a given share has enough patients to detect an equal-opportunity gap in sensitivity of 0.10 or 0.05, against two registry-extract sizes.

Files: `Fig15_centres_validation_subgroups.pdf`, `Fig15_centres_validation_subgroups.png`

## Fig 16 — data_bill_of_materials

The data bill of materials. (a) Patients required in each data layer for three packages, with the number of centres at which the gate of section 8.12 has 80% power; the packages differ in what they must demonstrate (a single honest test of the primary claim; several claims with a subgroup audit; a Level-3 claim with prospective validation) and in the between-centre heterogeneity planned for (tau = 0.02, 0.03, 0.04). (b) Programme cost by package, split into specimen collection and coordination, assays, centre set-up and data engineering, and compute; unit costs are planning figures. Compute is the smallest bar by orders of magnitude.

Files: `Fig16_data_bill_of_materials.pdf`, `Fig16_data_bill_of_materials.png`

## Fig 17 — pilot_results

The pilot this paper is built on (9 public studies, 838 samples; numbers from the earlier notebook in this repository). (a) Leave-one-study-out transfer with plain normalised genes: tissue biopsies transfer across laboratories and organs, blood and sorted T cells do not. (b) Frozen Geneformer (GF) at three sizes against its own random-weight twins and against plain genes: one size beats its twin, none beats the simple baseline, and the largest model is not the best. (c) A zero-parameter score written down in advance, the mean expression of the interferon-gamma and allograft-rejection gene sets, matches fitted models in inflamed tissue and fails in sorted blood T cells.

Files: `Fig17_pilot_results.pdf`, `Fig17_pilot_results.png`

