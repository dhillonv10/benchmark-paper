from part1 import CELLS, md, code

md(r"""
---
# 8 · GvHD-Bench v1.0

> **This section is the benchmark.** It is written to be used without the rest of the paper, and it is exported in full to `spec/GVHD_BENCH_v1.0.md` together with machine-readable task and cohort definitions. A benchmark earns trust by being *specific about the things people get wrong*: who is in the cohort, what the label function is, which inputs are admissible at which time, how the splits are drawn, what must be beaten, how many submissions are allowed, and what counts as a win. Each of those is pinned below.

## 8.0 Scope, version and governance

| | |
|---|---|
| **Version** | v1.0 (this notebook). Semantic versioning: a **patch** fixes a bug in a label function or metric; a **minor** adds tasks or baselines; a **major** changes a split, a cohort definition or a gate. Results are only comparable within a major version, and every reported number carries the version and the artefact hashes. |
| **Primary question** | Does a pretrained multimodal model improve *out-of-centre* risk prediction in allo-HCT over the models clinicians already use? |
| **Unit of analysis** | the patient; a donor–recipient pair where both contribute data. Never the sample, never the specimen. |
| **Clock** | all times in days relative to the first graft infusion (day 0), fractional days allowed. |
| **Label authority** | the executable functions in §4.6. Where they disagree with a consensus document, the consensus document wins and the functions are patched (a patch release). |
| **Admissibility** | an event may be read at landmark *L* only if `t_known ≤ L` (§4.1, §4.7). This is enforced by the harness, not left to the modeller. |
| **Frozen artefacts** | split assignment files, seeds, the gate, and the hash of each are released with the benchmark; a result computed against unreleased splits is not a GvHD-Bench result. |
| **Governance** | tasks are added by pull request with a task card (§8.2) and a power calculation; a task whose eligible n at the running centre cannot support a comparison is published as **estimable** or **out of reach** rather than quietly scored. |

## 8.1 Cohort definition (the part that is usually left vague)

**Included.** First allogeneic HCT for any indication; any donor type, graft source and conditioning intensity; age ≥ 0; transplant year within the stated era window; a recorded day 0 and at least one follow-up contact after day +7.

**Excluded.** Autologous and syngeneic transplants; second and subsequent transplants (they open a new `pair_id` and are analysed separately); patients who withdrew consent for research use; records whose day 0 or outcome date is missing or internally inconsistent (these are counted and reported, never silently dropped).

**Landmark eligibility, applied per task.** A patient enters a task at landmark *L* only if they are alive, in follow-up and **have not already had the outcome event** at *L*. Patients who died, relapsed or were censored before *L* are excluded from that task and counted in its attrition table. For response tasks the clock starts at treatment initiation, not at day 0.

**Landmarks are issue days, not draw days.** A prediction is *issued* on the landmark day and may use a specimen only once its result exists: a day-7 plasma draw with a 48-hour assay turnaround supports a prediction issued on **day +9**, a sequenced sample a week or more later (§4.7, Figure 5). Task cards therefore state the issue day, and the audit tests `t_known ≤ L` against it.

**Competing events.** Death and relapse are competing events for every GvHD endpoint, never censoring. Cumulative incidence uses Aalen–Johansen; discrimination uses inverse-probability-of-censoring weighting (§8.4).

**Mandatory attrition reporting.** Every task reports: patients in cohort → meeting landmark eligibility → with the required input tier → with a non-missing outcome, with the count lost at each step. A benchmark that cannot show its attrition cannot be audited.

"""
)

md(r"""
## 8.2 Design principles

1. **Tasks are decisions.** Every task sits at a real decision point on the Figure-1 timeline, with exactly the inputs a clinician would have had at that moment.
2. **Every task names the model it must beat** — not chance, but what is used today: refined DRI, HCT-CI, EASIX, the MAGIC algorithm probability, onset grade and organ pattern, NIH organ scores, or a tuned gradient-boosted tree on registry variables.
3. **Competing risks are first-class.** Death and relapse are events, not censoring.
4. **Splits are hard by default**: leave-centre-out whenever the data come from several centres, and temporal for any deployment claim; one institution uses the temporal split (Level 1C, §8.10). Random splits are reported only as an upper bound and labelled as such.
5. **Minimum n is computed, not assumed** (§8.3), and each task carries a *runnable / estimable / out-of-reach* verdict for the running centre that travels with its result.
6. **A negative result is a result.** The benchmark is designed so that "the foundation model did not beat the baseline" is a reportable, publishable outcome with a clear interpretation.
""")

md(r'''
## 8.3 The task registry
Families: **A** pre-transplant planning · **B** early post-transplant prognosis · **C** diagnosis and acute course · **D** treatment response · **E** chronic GvHD and late effects · **F** mechanism and representation · **G** robustness and governance.
''')

code(r'''
# ----------------------------------------------------------------------------- Riley et al. sample-size criteria (needed by the registry; validated below)
def r2cs_from_auc(A, prev, n_mc=400_000, seed=0):
    return _r2cs(round(float(A), 4), round(float(prev), 4), n_mc, seed)
@functools.lru_cache(maxsize=None)
def _r2cs(A, prev, n_mc, seed):
    """Model-implied Cox-Snell R^2 for a calibrated model with given AUROC and prevalence (equal-variance binormal)."""
    rng = np.random.default_rng(seed); d = math.sqrt(2) * norm.ppf(A)
    y = rng.random(n_mc) < prev; x = np.where(y, rng.normal(d, 1, n_mc), rng.normal(0, 1, n_mc))
    lp = math.log(prev / (1 - prev)) + d * x - d * d / 2; p = 1 / (1 + np.exp(-lp))
    ll1 = np.mean(np.where(y, np.log(p), np.log(1 - p))); ll0 = prev * math.log(prev) + (1 - prev) * math.log(1 - prev)
    return 1 - math.exp(-2 * (ll1 - ll0))

def riley_n(P, A, prev, S=.9, delta=.05, margin=.05):
    """Minimum n to DEVELOP a model with P candidate parameters (Riley 2020): shrinkage, optimism, prevalence precision."""
    r2 = r2cs_from_auc(A, prev); max_r2 = 1 - math.exp(2 * (prev * math.log(prev) + (1 - prev) * math.log(1 - prev)))
    n1 = P / ((S - 1) * math.log(1 - r2 / S))
    S2 = r2 / (r2 + delta * max_r2); n2 = P / ((S2 - 1) * math.log(1 - r2 / S2))
    n3 = (1.96 / margin) ** 2 * prev * (1 - prev)
    n = max(n1, n2, n3)
    return dict(n=int(math.ceil(n)), events=int(math.ceil(n * prev)), EPP=round(n * prev / P, 1),
                binding=['shrinkage', 'optimism', 'prevalence'][int(np.argmax([n1, n2, n3]))])
def max_params(n, A, prev, S=.9, delta=.05, margin=.05):
    """The LARGEST number of candidate parameters a development set of n patients can support (Riley criteria 1-3 inverted); 0 if n is below the prevalence-precision floor."""
    if n < (1.96 / margin) ** 2 * prev * (1 - prev): return 0
    r2 = r2cs_from_auc(A, prev); max_r2 = 1 - math.exp(2 * (prev * math.log(prev) + (1 - prev) * math.log(1 - prev))); S2 = r2 / (r2 + delta * max_r2)
    return int(math.floor(min(n * (S - 1) * math.log(1 - r2 / S), n * (S2 - 1) * math.log(1 - r2 / S2))))
def min_detectable_gain(n, A_base, prev, rho=.85, power=.8):
    """Smallest paired AUROC gain over a baseline of AUROC A_base that n patients detect with the given power (closed-form DeLong)."""
    for d in np.arange(.005, .30, .0025):
        if power_delta(n, prev, A_base, A_base + d, rho) >= power: return float(d)
    return float('nan')
assert all(max_params(riley_n(P, .72, .35)['n'], .72, .35) in (P, P - 1) for P in (5, 10, 20, 50)), 'max_params must invert riley_n'
assert abs(24 / ((0.9 - 1) * math.log(1 - 0.288 / 0.9)) - 623) < 1.5     # arithmetic check of criterion 1 on a published worked example
print('Riley criterion-1 arithmetic check passed; R2_CS(AUROC .75, prev .4) =', round(r2cs_from_auc(.75, .4), 3))
''')

code(r'''
# ----------------------------------------------------------------------------- GVHD-BENCH TASK REGISTRY (exported to spec/task_registry.csv)
# id, family, task, landmark, horizon, inputs allowed (tiers), outcome (label fn), competing events, primary metric, clinical baseline to beat, prev, target AUROC
TASKS = [
('A1','A pre-HCT','Grade II–IV aGvHD by day +100 from pre-transplant data only','day −7 (before conditioning)','day +100','A (+C germline/KIR)','MAGIC grade ≥ II (§4.6)','death, relapse','IPCW td-AUROC @100 + calibration','donor-type + DRI + HCT-CI + prophylaxis logistic (registry standard)',.38,.68),
('A2','A pre-HCT','Grade III–IV aGvHD by day +100','day −7','day +100','A (+C)','MAGIC grade ≥ III','death, relapse','IPCW td-AUROC, AUPRC','same registry model',.13,.70),
('A3','A pre-HCT','cGvHD requiring systemic therapy by 2 years','day −7','2 years','A (+C)','NIH 2014 moderate/severe or systemic tx','death, relapse','IPCW td-AUROC @730','graft source + donor age/sex + ATG model',.33,.65),
('A4','A pre-HCT','GRFS at 1 year','day −7','1 year','A','GRFS first event (§4.6)','—(composite)','C-index, IBS, calibration','DRI + HCT-CI + donor type',.52,.68),
('A5','A pre-HCT','Non-relapse mortality at 1 year','day −7','1 year','A','death without relapse','relapse','IPCW td-AUROC, Aalen–Johansen calibration','HCT-CI + age + DRI',.18,.70),
('A6','A pre-HCT','Donor ranking: order candidate donors by predicted GRFS','donor search','1 year','A + D06 immunogenetics','counterfactual GRFS per donor','—','pairwise concordance among realised choices; policy value','current algorithm (match grade → age → CMV → sex)',None,None),
('A7','A pre-HCT','Prophylaxis-platform benefit (PtCy/Tac/MMF vs Tac/MTX) for this pair','day −3','1 year','A','GRFS; grade III–IV aGvHD','death, relapse','calibration of predicted arm difference within RANDOMISED arms (BMT CTN 1703)','no-effect-modification (constant ATE) model',None,None),
('B1','B early','Grade II–IV aGvHD by day +100 from the day +7 state','day +9 (day-7 draw, resulted)','day +100','A,B (+C biomarkers)','MAGIC grade ≥ II','death, relapse','IPCW td-AUROC; net benefit at 10–40 %','**MAP (ST2+REG3α day-7 algorithm)** + clinical',.35,.72),
('B2','B early','Severe (grade III–IV) aGvHD by day +100 from day +14','day +16 (day-14 draw, resulted)','day +100','A,B,C','MAGIC grade ≥ III','death, relapse','IPCW td-AUROC, AUPRC','MAP + clinical',.13,.75),
('B3','B early','7-day-ahead rolling onset alarm (continuous early warning)','every day, +7 to +60','7 days','A,B (+C,D)','first day of grade ≥ II within 7 d','death, relapse, discharge','AUROC per alarm-day; alerts per 100 patient-days at 50 % sensitivity','clinician-trigger proxy (CRP + stool volume thresholds)',.04,.75),
('B4','B early','Lower-GI aGvHD specifically by day +100 from day +14','day +16 (day-14 draw, resulted)','day +100','A,B,C,D','lower-GI stage ≥ 1','death, relapse','IPCW td-AUROC','REG3α + clinical',.22,.72),
('B5','B early','NRM at 1 year from day +7','day +9 (day-7 draw, resulted)','1 year','A,B,C','death without relapse','relapse','IPCW td-AUROC, calibration','MAP (its validated use) + HCT-CI',.18,.75),
('B6','B early','Relapse by 1 year from day +30 (GvL trade-off partner)','day +30','1 year','A,B (+C chimerism/MRD)','relapse/progression','death','IPCW td-AUROC','DRI + day-30 MRD + chimerism',.24,.72),
('C1','C diagnosis & acute course','Differential diagnosis of early post-HCT diarrhoea: GvHD vs infection vs drug/other','day of symptom onset','7 days (adjudicated)','A,B (+C,D; biopsy-independent)','adjudicated cause (biopsy + microbiology + course)','—(multiclass)','macro-AUROC; GvHD-vs-rest AUPRC','clinical gestalt proxy: stool PCR + timing + CRP',.45,.80),
('C2','C diagnosis & acute course','Is this rash GvHD, drug eruption, or engraftment syndrome?','day of rash','7 days','A,B (+D photo/WSI)','adjudicated cause','—','macro-AUROC','timing + eosinophils + drug-start model',.5,.75),
('C3','C diagnosis & acute course','Will this patient need a GI biopsy (triage) / can biopsy be avoided?','day of symptom onset','14 days','A,B','biopsy-confirmed GvHD among those biopsied; decision-analytic','—','net benefit vs biopsy-all / biopsy-none','biopsy-all',.45,.78),
('C4','C diagnosis & acute course','Steroid-refractory aGvHD by day +28 of therapy','day systemic steroids start','28 days','A,B,C (+D onset biopsy/transcriptome)','SR phenotype (§4.6)','death','AUROC + calibration','MAGIC grade at onset + MAP + organ pattern',.30,.72),
('C5','C diagnosis & acute course','Day +28 overall response (CR/PR vs NR) to first-line steroids','steroid start','28 days','A,B,C,D','MAGIC response criteria','death','AUROC','grade + organ stage at onset',.35,.70),
('C6','C diagnosis & acute course','NRM at 6 months from aGvHD onset','onset day','6 months','A,B,C','death without relapse','relapse','IPCW td-AUROC','Ann Arbor score (TNFR1/ST2/REG3α) / MAP at onset',.30,.75),
('C7','C diagnosis & acute course','Histological severity / crypt-apoptosis grade from the biopsy itself','biopsy sign-out','—','D19 WSI (+D17)','pathologist Lerner/Sale grade (central read)','—','quadratic-weighted κ; AUROC for grade ≥ 3','ResNet/pathology-FM linear probe on WSI tiles',.3,.80),
('C8','C diagnosis & acute course','Organ-involvement multi-label at onset (skin/upper GI/lower GI/liver)','onset day','—','A,B,C,D','MAGIC organ stages','—','macro-AUROC, per-organ calibration','overall grade only',None,None),
('D1','D response','Second-line response at day +28 (ruxolitinib era)','second-line start','28 days','A,B,C,D','overall response (MAGIC)','death','AUROC + calibration','grade + prior response + organ pattern',.40,.70),
('D2','D response','Which second-line agent? (policy, validated on REACH-style randomised data where available)','second-line start','28 days','A,B,C,D','response under received agent','death','policy value; calibration of predicted contrast within randomised data','single-agent default policy',None,None),
('D3','D response','Steroid taper safety: flare within 28 days of a taper step','each taper decision','28 days','A,B,C','grade increase or treatment intensification','death, relapse','AUROC; net benefit','fixed-schedule taper',.20,.70),
('D4','D response','Infection within 30 days under intensified immunosuppression','treatment start','30 days','A,B','blood-culture-positive bacteraemia / IFD / CMV disease','death','AUROC','ALC + neutropenia + prior infection',.30,.72),
('E1','E chronic','NIH moderate/severe cGvHD by 1 year, from day +100','day +100','1 year','A,B (+C day-100 biomarkers, D14 subsets)','NIH 2014 global ≥ moderate','death, relapse','IPCW td-AUROC','prior aGvHD + graft source + ATG + donor sex (clinical model) ± biomarker panel',.30,.68),
('E2','E chronic','Bronchiolitis-obliterans syndrome / FEV1 decline ≥ 10 % by 2 years','day +100','2 years','A,B,D19 (PFT/CT)','NIH BOS criteria','death, relapse','IPCW td-AUROC','baseline + day-100 FEV1 slope',.08,.72),
('E3','E chronic','6-month response to cGvHD therapy (NIH response criteria)','therapy start','6 months','A,B,C,D,D22','NIH 2014 CR/PR vs not','death','AUROC','organ scores at start + clinician global',.35,.68),
('E4','E chronic','Immunosuppression-free survival at 3 years','day +365','3 years','A,B','alive, relapse-free, off systemic IS','death, relapse','IPCW td-AUROC','cGvHD severity + taper status',.40,.68),
('E5','E chronic','Safe immunosuppression taper: relapse vs flare trade-off','each taper decision after +180','6 months','A,B (+MRD/chimerism)','relapse and flare (joint)','death','bivariate net benefit; decision curves','clinician practice (fixed taper)',None,None),
('F1','F mechanism','Donor-reactive clone identification (clone-level)','pre-HCT / onset','—','D16 (+D17)','MLR-defined reactive clones','—','AUROC within abundance deciles','V/J+CDR3-length and k-mer models (the pilot ceiling)',.17,.65),
('F2','F mechanism','Zero-shot representation quality: biology kept, centre/batch discarded','any','—','all','compartment / cell-type / organ identity vs centre ID','—','two-axis score (scIB-style): biology kNN accuracy vs centre decodability','HVG / PCA / random-weight twin',None,None),
('F3','F mechanism','Counterfactual direction under JAK inhibition in primary T cells','perturbation experiment','—','D17 (+external screens)','post-perturbation profile','—','direction accuracy, rank correlation','additive/linear baseline',None,None),
('G1','G robustness','Temporal validation: train ≤ era k, test era k+1 (letermovir, PtCy expansion, ruxolitinib approvals)','as parent task','as parent','as parent','as parent','—','AUROC and calibration drift vs the random-split number','same model, random split',None,None),
('G2','G robustness','Leave-centre-out and leave-registry-out transfer (CIBMTR → EBMT)','as parent','as parent','as parent','as parent','—','per-centre AUROC spread; calibration slope per centre','pooled random split',None,None),
('G3','G robustness','Subgroup equity: paediatric vs adult, donor type, race/ethnicity, centre volume','as parent','as parent','as parent','as parent','—','worst-subgroup AUROC and calibration; equal-opportunity gap','overall model',None,None),
('G4','G robustness','Tier ablation: A only → A+B → A+B+C → A+B+C+D','as parent','as parent','progressively','as parent','—','ΔAUROC per added tier, with the data elements each tier adds per patient','Tier A model',None,None),
('G5','G robustness','Missingness robustness: performance when a modality or a lab stream is absent at inference','as parent','as parent','as parent','as parent','—','AUROC under simulated and natural missingness','complete-case model',None,None),
('G6','G robustness','Target-trial emulation: reproduce a published randomised effect (BMT CTN 1703: PTCy/Tac/MMF vs Tac/MTX, GRFS HR 0.64, 95% CI 0.49–0.83 ✓) from registry data','trial eligibility + randomisation-equivalent day','1 year','A (trial-eligibility variables) + model-based adjustment','GRFS (§4.6)','death, relapse (composite)','|log HR(emulated) − log HR(trial)| and coverage of the trial interval by the emulation interval','unadjusted registry comparison; propensity-weighted tuned-tree adjustment',None,None),
('G7','G robustness','Negative-control suite: every submission is also run on permuted outcomes (within centre) and on a negative-control exposure (e.g. donor birth month)','as parent','as parent','as parent','permuted / null outcome','—','false-pass rate across the whole suite; must not exceed the nominal level','—',None,None),
]
COLS = ['id','family','task','landmark','horizon','inputs allowed (tiers)','outcome (label fn)','competing events',
        'primary metric','clinical baseline to beat','assumed prevalence','target AUROC']
reg = pd.DataFrame(TASKS, columns=COLS)
ok_n = [bool(pd.notna(a) and pd.notna(p)) for a, p in zip(reg['target AUROC'], reg['assumed prevalence'])]
reg['min n (10-param head)'] = [riley_n(10, a, p)['n'] if o else '' for o, a, p in zip(ok_n, reg['target AUROC'], reg['assumed prevalence'])]
reg['min n (25-param head)'] = [riley_n(25, a, p)['n'] if o else '' for o, a, p in zip(ok_n, reg['target AUROC'], reg['assumed prevalence'])]
reg.to_csv(os.path.join(SPEC_DIR, 'task_registry.csv'), index=False)
print(f'{len(reg)} tasks in {reg.family.nunique()} families; '
      f"{int((reg['target AUROC'].notna()).sum())} have a numeric target and a computed minimum n")
display(md_table(reg.groupby('family').agg(tasks=('id', 'count'), ids=('id', lambda s: ', '.join(s))), index=True))
''')

code(r'''
# ----------------------------------------------------------------------------- render the registry family by family
for fam in reg.family.unique():
    display(Markdown(f'#### Family {fam}'))
    display(md_table(reg[reg.family == fam].drop(columns=['family']).fillna(''), max_w=170))
''')

md(r"""
## 8.4 Task cards: readiness, size at one institution, and the feasibility verdict

The registry above says what each task *is*. A benchmark also has to say, for each task, **whether it can be run at all, with which data, on how many patients, and whether that number is enough.** The cell below attaches those fields and computes the verdict. `readiness` is the honest status:

* **R1 — runnable now** on the centre's own EHR extract (Tier A/B) with no new collection.
* **R2 — needs banked specimens** (Tier C), which the plan collects prospectively on the routine blood draws (§9.9).
* **R3 — needs prospective collection** on the 4.4 schedule, including biopsies.
* **R4 — research-grade only**: a nested sub-cohort, reported as mechanism, never as a deployment claim.

`patients at one centre` is the number of the plan's patients who are eligible for that task — the plan being **up to 1,500 patients with an EHR timeline and about 420 of them with banked specimens** (§9.8) — after the task's own landmark eligibility (alive and outcome-free at the landmark, symptomatic, treated, biopsied …). Three numbers follow from it: the **largest head** the Riley criteria allow at that n, the **smallest paired AUROC gain** over the task's clinical baseline that n can show with 80 % power, and the **half-width of one AUROC's 95 % interval**. The verdict is **RUNNABLE** if the head can have at least 5 parameters and the smallest gain shown is at most +0.05; **ESTIMABLE** if one AUROC can still be estimated to within ±0.10 but no comparison is possible; **OUT OF REACH** otherwise. Tasks that are not a single-AUROC comparison, and R4 research tasks, carry their own criteria instead. The `split protocol` column is the benchmark's protocol; **one institution uses its temporal variant** (earliest 60 % to develop, latest 40 % to test; Level 1C of §8.10), because leave-centre-out needs more than one centre.
""")

code(r'''
# ----------------------------------------------------------------------------- task cards: readiness, size at one institution, feasibility verdict (exported with the registry)
CARDS = {   # task id -> (readiness, tier needed, split protocol, achievable AUROC range from section 3.2)
 'A1': ('R1', 'A', 'leave-centre-out + temporal', (0.62, 0.7)), 'A2': ('R1', 'A', 'leave-centre-out + temporal', (0.64, 0.72)),
 'A3': ('R1', 'A', 'leave-centre-out + temporal', (0.62, 0.68)), 'A4': ('R1', 'A', 'leave-centre-out + temporal', (0.64, 0.7)),
 'A5': ('R1', 'A', 'leave-centre-out + temporal', (0.68, 0.76)), 'A6': ('R1', 'A+C', 'leave-centre-out; counterfactual', (None, None)),
 'A7': ('R2', 'A', 'randomised arms (BMT CTN 1703)', (None, None)), 'B1': ('R2', 'A+B+C', 'leave-centre-out + temporal', (0.74, 0.8)),
 'B2': ('R2', 'A+B+C', 'leave-centre-out + temporal', (0.74, 0.82)), 'B3': ('R1', 'A+B', 'leave-centre-out; per-day alarms', (0.72, 0.8)),
 'B4': ('R2', 'A+B+C', 'leave-centre-out', (0.7, 0.78)), 'B5': ('R2', 'A+B+C', 'leave-centre-out + temporal', (0.72, 0.8)),
 'B6': ('R1', 'A+B', 'leave-centre-out', (0.68, 0.76)), 'C1': ('R3', 'A+B(+C)', 'leave-centre-out; adjudicated', (0.78, 0.88)),
 'C2': ('R3', 'A+B', 'leave-centre-out; adjudicated', (0.72, 0.82)), 'C3': ('R1', 'A+B', 'leave-centre-out; decision-analytic', (0.72, 0.8)),
 'C4': ('R2', 'A+B+C', 'leave-centre-out', (0.68, 0.76)), 'C5': ('R2', 'A+B+C', 'leave-centre-out', (0.66, 0.74)),
 'C6': ('R2', 'A+B+C', 'leave-centre-out + temporal', (0.72, 0.8)), 'C7': ('R3', 'D', 'leave-centre-out; central read', (None, None)),
 'C8': ('R1', 'A+B', 'leave-centre-out', (None, None)), 'D1': ('R2', 'A+B', 'leave-centre-out', (0.66, 0.74)),
 'D2': ('R2', 'A+B', 'randomised arms only', (None, None)), 'D3': ('R1', 'A+B', 'leave-centre-out', (0.64, 0.72)),
 'D4': ('R1', 'A+B', 'leave-centre-out', (0.68, 0.76)), 'E1': ('R1', 'A+B', 'leave-centre-out + temporal', (0.65, 0.72)),
 'E2': ('R1', 'A+B', 'leave-centre-out', (0.7, 0.78)), 'E3': ('R2', 'A+B', 'leave-centre-out', (0.64, 0.72)),
 'E4': ('R1', 'A+B', 'leave-centre-out + temporal', (0.64, 0.72)), 'E5': ('R1', 'A+B', 'leave-centre-out; bivariate', (None, None)),
 'F1': ('R4', 'D', 'leave-recipient-out', (0.6, 0.66)), 'F2': ('R4', 'C+D', 'no head trained', (None, None)),
 'F3': ('R4', 'D', 'held-out perturbations', (None, None)), 'G1': ('R1', 'as parent', 'temporal', (None, None)),
 'G2': ('R1', 'as parent', 'leave-registry-out', (None, None)), 'G3': ('R1', 'as parent', 'subgroup within parent split', (None, None)),
 'G4': ('R2', 'A->D', 'nested tiers, same patients', (None, None)), 'G6': ('R1', 'A', 'registry vs trial effect (BMT CTN 1703)', (None, None)),
 'G7': ('R1', 'as parent', 'permutation + negative exposure', (None, None)), 'G5': ('R1', 'as parent', 'simulated + natural missingness', (None, None)),
}

# ---- what ONE institution could run: ~200 allo-HCTs a year, up to 1,500 patients with an EHR timeline (Tier A/B), 420 with prospectively banked specimens (Tier C/D)   (planning values, †)
SI = dict(transplants_per_year=TRANSPLANTS_PER_YEAR, ehr_years=EHR_YEARS, specimen_years=SPECIMEN_YEARS, usable=USABLE_FRACTION)
N_RETRO, N_PRO = N_CAP, N_SPEC               # EHR timelines for N_CAP patients; banked specimens for N_SPEC of them
ELIG = dict(A1=1, A2=1, A3=.9, A4=1, A5=1, B1=.95, B2=.95, B3=1, B4=.95, B5=.95, B6=.85, C1=.5, C2=.3, C3=.45, C4=.3, C5=.3, C6=.35, C7=.2,
            D1=.12, D3=.25, D4=.25, E1=.8, E2=.8, E3=.3, E4=.75)       # fraction of transplants that are eligible for the task (alive at the landmark, symptomatic, treated, biopsied ...)
ELIG_OTHER = dict(A6=.9, C8=.5, E5=.8, G6=.85)                          # the same, for tasks that are not a single-AUROC comparison
N_ABS = dict(A7=431, D2=400, F1=80, F2=80, F3=40)                       # public trial data (BMT CTN 1703) or the nested research subset of section 7: not the centre's own cohort size
card_rows = []
for _, r in reg.iterrows():
    rd, tier, split, ach = CARDS[r.id]
    A_, pv_ = r['target AUROC'], r['assumed prevalence']
    n_ = N_ABS.get(r.id, int((N_RETRO if rd == 'R1' else N_PRO) * {**ELIG, **ELIG_OTHER}.get(r.id, 1)))
    row = {'id': r.id, 'family': r.family.split()[0], 'readiness': rd, 'tier needed': tier, 'split protocol': split, 'one_inst_n': n_,
           'achievable AUROC': '' if ach[0] is None else f'{ach[0]:.2f}-{ach[1]:.2f}'}
    if pd.notna(A_) and pd.notna(pv_) and r.id in ELIG:
        n1_ = max(int(n_ * pv_), 2)
        pm_ = max_params(n_, A_, pv_); mg_ = min_detectable_gain(n_, A_ - .03, pv_); ci_ = 1.96 * hm_se(A_, n1_, n_ - n1_)
        row.update({'n to develop (10-param head)': riley_n(10, A_, pv_)['n'], 'n to show +0.03': n_for_delta(A_ - .03, .03, pv_),
                    'max_head_params': pm_, 'min_detectable_gain': round(mg_, 3), 'ci_halfwidth': round(ci_, 3),
                    'one_inst_verdict': 'RUNNABLE' if (pm_ >= 5 and mg_ <= .05) else ('ESTIMABLE' if ci_ <= .10 else 'OUT OF REACH')})
    else:
        row.update({'one_inst_verdict': '-'})
    card_rows.append(row)
cards_df = pd.DataFrame(card_rows)
reg_full = reg.merge(cards_df.drop(columns=['family']), on='id', how='left')
reg_full.to_csv(os.path.join(SPEC_DIR, 'task_registry.csv'), index=False)
cards_df.to_csv(os.path.join(SPEC_DIR, 'task_cards.csv'), index=False)
vs = cards_df.one_inst_verdict.value_counts().to_dict()
print(f"{len(cards_df)} task cards | ONE INSTITUTION ({SI['transplants_per_year']} transplants a year; {N_RETRO:,} EHR timelines, {N_PRO} with specimens): "
      f"RUNNABLE {vs.get('RUNNABLE', 0)} | ESTIMABLE {vs.get('ESTIMABLE', 0)} | OUT OF REACH {vs.get('OUT OF REACH', 0)} | no single-AUROC verdict {vs.get('-', 0)}")
print('readiness:', cards_df.readiness.value_counts().sort_index().to_dict())
display(md_table(cards_df))
''')

code(r'''
vi = {v: cards_df[cards_df.one_inst_verdict == v].id.tolist() for v in ('RUNNABLE', 'ESTIMABLE', 'OUT OF REACH')}
n_none = int((cards_df.one_inst_verdict == '-').sum())
display(Markdown(f"""**Result.** Of {len(cards_df)} tasks, **{len(vi['RUNNABLE'])} are runnable at one institution** ({', '.join(vi['RUNNABLE'])}), **{len(vi['ESTIMABLE'])} are estimable only** ({', '.join(vi['ESTIMABLE'])}) and **{len(vi['OUT OF REACH'])} are out of reach** ({', '.join(vi['OUT OF REACH'])}); the other {n_none} are not single-AUROC comparisons (donor ranking, counterfactual prophylaxis contrasts, multi-label organ involvement, representation quality, robustness audits) or are research-only, and carry their own criteria.

**The runnable tasks are the benchmark's spine at one centre**: they need the centre's own EHR extract (and, for B1, its banked plasma), not a consortium. Section 9.8 repeats the table with the head size and the smallest gain each task can show.

**The estimable and out-of-reach tasks are named, not hidden.** A single centre can *describe* the estimable ones, with an honest interval, and cannot compare models on them; pooling with other centres would turn them into comparisons (Level 2, §8.10). Listing them as ordinary leaderboard rows would manufacture precision that does not exist, so GvHD-Bench publishes them with the verdict attached, and a positive result on a task that cannot support a comparison is reported as a hypothesis, never as evidence."""))
''')


md(r'''
## 8.5 Baselines that every model must beat

| Baseline class | Concretely | Why it is mandatory |
|---|---|---|
| **Clinical standard of care** | DRI, HCT-CI, EASIX, refined donor-selection rules, MAGIC MAP, Ann Arbor score, NIH organ scores | A model that does not beat what clinicians already use has no path to deployment |
| **Tuned simple model** | penalised logistic / Cox or gradient-boosted trees on the 20–40 registry variables, with the same splits | The published audits of foundation models repeatedly found simple models competitive; trees on tabular registry data are strong |
| **Confounder-only** | centre, era, assay batch, library size, pre-analytics, sampling day | If this matches the model, the signal is technical |
| **Treatment-proxy-only** | the banned features, run deliberately as a *leakage canary* | Quantifies how much a leaking pipeline would have "gained" |
| **Presence-only** | indicators of *whether and when* each lab, drug or assay was recorded, with no values | In health records the act of ordering a test carries clinical suspicion; a model that approaches the full model on presence alone has learned practice, not biology (§8.12) |
| **Permuted-outcome and negative-exposure controls** (task G7) | the same pipeline run on outcomes shuffled within centre, and on a negative-control exposure | Measures the false-pass rate of the whole pipeline, not just of one metric (§8.12) |
| **Random-weight twin** | identical architecture, untrained weights, same adapters | Isolates what pretraining contributed |
| **No-pretraining** | stage 3 only, same labels | Isolates what pretraining contributed, the other way |

## 8.6 Metrics, implemented and validated here

Discrimination under competing risks is **IPCW cumulative/dynamic time-dependent AUROC**: at horizon $h$, cases are patients with the event of interest by $h$, controls are patients event-free at $h$, and each is weighted by the inverse of its estimated censoring probability. Competing events are *not* censoring — a patient who dies of relapse before $h$ is not a control for GvHD. Calibration is assessed against **Aalen–Johansen** cumulative incidence inside risk deciles, which is the competing-risk analogue of a calibration plot. Clinical utility is **net benefit** over a range of threshold probabilities.

The cell below implements these three and validates them on simulated data where the truth is known.
''')

code(r'''
# ----------------------------------------------------------------------------- competing-risk metrics: IPCW td-AUROC, Aalen-Johansen, net benefit (+ validation)
def km_censoring(time, event_any):
    """Kaplan-Meier estimate of the CENSORING survival function G(t) = P(C > t); event_any = 1 if any event (not censored)."""
    t = np.asarray(time, float); c = (np.asarray(event_any) == 0).astype(int)
    order = np.argsort(t); t, c = t[order], c[order]
    uniq = np.unique(t); G = np.ones(len(uniq)); surv = 1.0
    at_risk = len(t)
    for i, u in enumerate(uniq):
        d = int(((t == u) & (c == 1)).sum()); n = int((t >= u).sum())
        if n > 0 and d > 0: surv *= (1 - d / n)
        G[i] = surv
    return uniq, G

def G_at(uniq, G, q):
    """Left-continuous lookup of the censoring survival at times q."""
    idx = np.searchsorted(uniq, q, side='right') - 1
    out = np.where(idx >= 0, G[np.clip(idx, 0, len(G) - 1)], 1.0)
    return np.clip(out, 1e-3, 1.0)

def ipcw_td_auroc(time, cause, score, horizon, cause_of_interest=1):
    """Cumulative/dynamic time-dependent AUROC at `horizon` with IPCW; competing events are NOT controls.
    cause: 0 = censored, 1 = event of interest, 2.. = competing events."""
    time = np.asarray(time, float); cause = np.asarray(cause, int); score = np.asarray(score, float)
    uniq, G = km_censoring(time, (cause > 0).astype(int))
    case = (time <= horizon) & (cause == cause_of_interest)
    ctrl = time > horizon                                              # event-free at the horizon
    wc = 1.0 / G_at(uniq, G, np.minimum(time, horizon))                # weight cases at their event time
    wk = 1.0 / G_at(uniq, G, np.full(time.shape, horizon))             # weight controls at the horizon
    si, sj = score[case], score[ctrl]; wi, wj = wc[case], wk[ctrl]
    if len(si) < 2 or len(sj) < 2: return float('nan')
    W = wi[:, None] * wj[None, :]
    num = ((si[:, None] > sj[None, :]) * W).sum() + .5 * ((si[:, None] == sj[None, :]) * W).sum()
    return float(num / W.sum())

def aalen_johansen(time, cause, horizon, cause_of_interest=1):
    """Aalen-Johansen cumulative incidence of `cause_of_interest` at `horizon` (competing risks)."""
    time = np.asarray(time, float); cause = np.asarray(cause, int)
    order = np.argsort(time); time, cause = time[order], cause[order]
    surv, cif = 1.0, 0.0
    for u in np.unique(time):
        n = int((time >= u).sum())
        if n == 0: continue
        d_all = int(((time == u) & (cause > 0)).sum()); d_k = int(((time == u) & (cause == cause_of_interest)).sum())
        if u <= horizon: cif += surv * d_k / n
        surv *= (1 - d_all / n)
        if u > horizon: break
    return float(cif)

def net_benefit(y, p, thresholds):
    """Decision-curve net benefit for a binary outcome at each threshold, plus treat-all and treat-none."""
    y = np.asarray(y); p = np.asarray(p); n = len(y); rows = []
    for t in thresholds:
        tp = int(((p >= t) & (y == 1)).sum()); fp = int(((p >= t) & (y == 0)).sum())
        rows.append(dict(threshold=t, model=tp / n - fp / n * t / (1 - t),
                         treat_all=y.mean() - (1 - y.mean()) * t / (1 - t), treat_none=0.0))
    return pd.DataFrame(rows)

# ---------- validation 1: with no censoring, IPCW td-AUROC must equal the plain AUROC on the same case/control sets
r = np.random.default_rng(0); n = 1500
x = r.normal(size=n); t_ev = np.exp(1.2 - .8 * x + .3 * r.normal(size=n)) * 30
t_comp = np.exp(1.4 + .2 * r.normal(size=n)) * 40
cause = np.where(t_ev < t_comp, 1, 2); tt_obs = np.minimum(t_ev, t_comp)
H = float(np.quantile(tt_obs, .5))
a_ipcw = ipcw_td_auroc(tt_obs, cause, x, H)
case = (tt_obs <= H) & (cause == 1); ctrl = tt_obs > H
a_plain = auroc(np.r_[np.ones(case.sum()), np.zeros(ctrl.sum())], np.r_[x[case], x[ctrl]])
# ---------- validation 2: add independent censoring; IPCW must recover the uncensored value better than naive deletion
C = np.exp(1.5 + .5 * r.normal(size=n)) * 35
cause_c = np.where(C < tt_obs, 0, cause); time_c = np.minimum(tt_obs, C)
a_ipcw_c = ipcw_td_auroc(time_c, cause_c, x, H)
case_c = (time_c <= H) & (cause_c == 1); ctrl_c = time_c > H
a_naive_c = auroc(np.r_[np.ones(case_c.sum()), np.zeros(ctrl_c.sum())], np.r_[x[case_c], x[ctrl_c]])
# ---------- validation 3: Aalen-Johansen vs the empirical (uncensored) incidence
aj = aalen_johansen(time_c, cause_c, H); emp = float(((tt_obs <= H) & (cause == 1)).mean())
# ---------- validation 4: net benefit of a perfect model >= treat-all; of a random model <= treat-all
yb = (r.random(4000) < .3).astype(int); nb = net_benefit(yb, yb * .99 + .005, [.1, .2, .3, .4])
nb_rand = net_benefit(yb, r.random(4000), [.1, .2, .3, .4])
val = pd.DataFrame([
 dict(check='IPCW td-AUROC == plain AUROC when nothing is censored', value=f'{a_ipcw:.4f} vs {a_plain:.4f}', pass_=abs(a_ipcw - a_plain) < 2e-3),
 dict(check='IPCW recovers the uncensored AUROC under censoring (vs naive)', value=f'IPCW {a_ipcw_c:.3f}, naive {a_naive_c:.3f}, target {a_ipcw:.3f}',
      pass_=abs(a_ipcw_c - a_ipcw) <= abs(a_naive_c - a_ipcw) + 1e-9),
 dict(check='Aalen-Johansen CIF ≈ empirical incidence (competing risks)', value=f'{aj:.3f} vs {emp:.3f}', pass_=abs(aj - emp) < .03),
 dict(check='net benefit: perfect model ≥ treat-all at every threshold', value=f"min margin {np.min(nb.model - nb.treat_all):+.3f}", pass_=bool((nb.model >= nb.treat_all - 1e-9).all())),
 dict(check='net benefit: random model never beats the better default strategy',
      value=f"max excess over max(treat-all, treat-none) {np.max(nb_rand.model - np.maximum(nb_rand.treat_all, 0)):+.3f}",
      pass_=bool((nb_rand.model <= np.maximum(nb_rand.treat_all, 0) + .02).all())),
])
assert val.pass_.all(), val[~val.pass_]
print('metric implementations validated'); display(md_table(val.rename(columns={'pass_': 'pass'})))
''')

md(r'''
## 8.7 Splits, and the leakage audit as code

**Split hierarchy** (strictest first, all grouped by patient and by donor–recipient pair):
1. **Leave-registry-out** (CIBMTR → EBMT): the hardest test of premise P3.
2. **Leave-centre-out**, with per-centre performance reported, not just the mean.
3. **Temporal**: train on transplants up to year *k*, test on year *k*+1 onward. Mandatory for deployment claims, since practice shifts (letermovir, PtCy expansion, ruxolitinib).
4. **Grouped random split**: reported as an upper bound and labelled as such.

**Audit checks run before any number is reported** (implemented below): patient overlap between folds; donor shared across split sides; centre bleeding into both sides for a leave-centre-out claim; any feature whose `t_known` exceeds the landmark; any banned treatment-proxy feature present; a single feature whose univariate AUROC is implausibly high for the landmark; label-vs-centre association strong enough that a centre-only model scores well; a presence-only model that beats chance by more than 0.02; and duplicate patient records across sources.

## 8.8 Mandatory ablations
| Ablation | Question it answers |
|---|---|
| random-weight twin of the backbone | did pretraining contribute anything at all? |
| no-pretraining (supervised only) | same question, from the other side |
| drop each stage-1 objective | which objective carries the transfer? |
| modality knockout (each of D07/D14/D15/D16/D17/D18/D19) | **premise P2**: is the signal really spread across modalities? |
| tier ablation A → A+B → A+B+C → A+B+C+D (task G4) | what does each extra data tier buy, per element collected? |
| pretraining-corpus scaling (10 / 30 / 100 %) | is scale binding, or does it plateau as in the published scaling study? |
| shared-structure audit (centre decodability, subspace overlap) | **premise P3**: measure ρ before blaming the model |
| values-masked (every measured value replaced by its training mean, presence kept) | how much of the gain is *what was ordered* rather than *what it showed* (§8.12) |
| prediction-issue-day sweep (issue on day +9, +11, +14, with the matching inputs) | does the claimed gain survive the assay turnaround that real use imposes (§4.7) |

''')

md(r"""
## 8.9 Submission, scoring and anti-gaming

A benchmark without submission rules becomes a record of who tried the most configurations. GvHD-Bench is specified so that the number reported is the number a new centre would get.

**Split construction (normative).** For leave-centre-out, centres are the folds; a centre is never split. For temporal validation, the cut is a calendar year chosen in advance, training on transplants up to and including year *k* and testing from year *k*+1. For leave-registry-out, the entire second registry is the test set. Assignments are written once to `spec/splits_*.json`, hashed, and released; the harness refuses to score a run whose split hash does not match.

```
assign_splits(cohort, protocol, seed):
    groups  = cohort.patient_id                    # never the sample, never the specimen
    strata  = (centre, era, donor_type)            # for reporting, not for leakage
    if protocol == "leave-centre-out":   folds = partition_by(cohort.centre)
    if protocol == "temporal":           folds = [(year <= k), (year > k)]
    if protocol == "leave-registry-out": folds = partition_by(cohort.registry)
    assert no_patient_in_two_folds(folds) and no_donor_spanning_folds(folds)
    freeze(folds); publish(hash(folds))
```

**The three-way data split.** *Development* (visible: features and labels), *validation* (visible features, visible labels, intended for model selection), and *evaluation* (visible features, **withheld labels**). Primary numbers come from the evaluation split only.

**Submission protocol.**
1. A submission is a container that takes the admissible inputs for one landmark and emits a risk or a hazard vector. It never sees the evaluation labels.
2. **At most five scored submissions per team per major version**, each with a written description of what changed. This is the single most effective anti-overfitting device a benchmark has.
3. Every submission declares its data tier, its pretraining corpus, its parameter count and whether the backbone was frozen or fine-tuned. Fine-tuned entries report **≥ 5 seeds** with the spread, because few-label fine-tuning is seed-sensitive.
4. The required ablations (§8.8) are submitted *with* the model, not on request. An entry without its random-weight twin and no-pretraining arms is logged as incomplete and is not eligible for a Level 2 claim.
5. Code and a model card are deposited; the model card states the preprocessing a deployment site must reproduce, including the block normalisation and the units of every input.

**Scoring.** Each task reports its primary metric with a **patient-cluster bootstrap** CI and, for leave-centre-out, the *per-centre* values and their spread — never the pooled number alone. Across a family, p-values are Benjamini–Hochberg adjusted. There is a composite score, but it is deliberately secondary: the mean rank across the powered tasks in a family, reported alongside the per-task table, so that a single flattering task cannot carry an entry.

**Anti-gaming provisions, stated explicitly.**

| Failure mode | Provision |
|---|---|
| Tuning on the test centres | evaluation labels withheld; five-submission cap; split hashes verified |
| Treatment-proxy leakage | banned-feature list enforced by the harness; a leakage-canary arm quantifies the inflation (§8.11) |
| Result-availability leakage | `t_known ≤ L` filter inside the harness, with the audit re-checking every feature |
| Cherry-picking a favourable subgroup | pre-declared subgroups only (task G3); worst-subgroup performance is a reported primary quantity |
| Reporting the best of many seeds | fine-tuned entries must report ≥ 5 seeds and their spread |
| Picking the flattering metric | the primary metric is fixed per task card before submission |
| Silently dropping hard patients | mandatory attrition table (§8.1); missingness is an input, not an exclusion criterion |
| A win on a task that cannot support a comparison | the feasibility verdict (§8.4) travels with every result |

**Required artefacts for a published GvHD-Bench result.** Benchmark version and artefact hashes; the attrition table; per-task primary metric with CI; per-centre values; calibration slope and intercept; decision-curve net benefit over 10–40 %; all mandatory baselines in the same table; the ablation suite; the leakage audit output; the seed list; and the negative-results table of everything that did not beat its baseline.
"""
)

code(r'''
# ----------------------------------------------------------------------------- Figure 7: the shape of GvHD-Bench (a map of the tasks, and which ones the field can actually settle)
BINS = ['pre-HCT\n(before day 0)', 'day +7\nto +30', 'onset /\ndiagnosis', 'treatment\nstart', 'day +100\nonward', 'any /\nresearch']
FAMN = {'A': 'Pre-HCT planning', 'B': 'Early prognosis', 'C': 'Diagnosis and\nacute course', 'D': 'Treatment response',
        'E': 'Chronic GvHD', 'F': 'Mechanism', 'G': 'Robustness'}
def _bin(r):
    f = r.family[0]
    if f in 'FG': return 5
    if r.id in ('C4', 'C5', 'D1', 'D2', 'D4'): return 3
    if r.id in ('D3', 'E1', 'E2', 'E3', 'E4', 'E5'): return 4
    return {'A': 0, 'B': 1, 'C': 2}[f]
reg_full['bin'] = [_bin(r) for _, r in reg_full.iterrows()]
reg_full['fam'] = [r.family[0] for _, r in reg_full.iterrows()]
MARK = {'R1': 'o', 'R2': 's', 'R3': '^', 'R4': 'D'}
ROW, FPAD, SLOTS = .185, .13, 2                                             # inches per slot row; padding between families
fam_rows = {f: max(int(np.ceil(len(reg_full[(reg_full.fam == f) & (reg_full.bin == b)]) / SLOTS)) for b in range(6)) for f in FAMN}
Hin = sum(fam_rows.values()) * ROW + FPAD * (len(FAMN) - 1)
FWIN, FHIN = FULL, Hin + 1.50
fig = plt.figure(figsize=(FWIN, FHIN))
L1, W1, L2, W2 = 1.42, 4.05, 6.05, 1.10
axm = fig.add_axes([L1 / FWIN, .98 / FHIN, W1 / FWIN, Hin / FHIN]); axr = fig.add_axes([L2 / FWIN, .98 / FHIN, W2 / FWIN, Hin / FHIN])
axm.set_xlim(0, 6); axm.set_ylim(Hin, 0)
ymid, y0 = {}, 0.0
for f in FAMN:
    h = fam_rows[f] * ROW; ymid[f] = y0 + h / 2
    for b in range(6):
        sub = reg_full[(reg_full.fam == f) & (reg_full.bin == b)].sort_values('id')
        for k, (_, r) in enumerate(sub.iterrows()):
            x = b + .075 + (k % SLOTS) * .5; y = y0 + (k // SLOTS + .5) * ROW
            v = r.one_inst_verdict
            kw = {'RUNNABLE': dict(mfc=PAL['aqua'], mec=PAL['aqua']), 'ESTIMABLE': dict(mfc='none', mec=PAL['orange']), 'OUT OF REACH': dict(mfc='none', mec=PAL['bad'])}.get(v, dict(mfc=PAL['grey2'], mec=PAL['grey']))
            axm.plot(x, y, MARK[r.readiness], ms=4.4, mew=1.0, **kw, zorder=3)
            axm.text(x + .085, y, r.id, fontsize=6.4, va='center', ha='left', color=PAL['ink'])
    y0 += h + FPAD
    if f != 'G': axm.axhline(y0 - FPAD / 2, color=PAL['grid'], lw=.5)
for b in range(1, 6): axm.axvline(b, color=PAL['grid'], lw=.5)
axm.set_yticks([ymid[f] for f in FAMN]); axm.set_yticklabels([f'{f}  {FAMN[f]}' for f in FAMN]); axm.tick_params(axis='y', length=0)
axm.xaxis.tick_top(); axm.set_xticks(np.arange(6) + .5); axm.set_xticklabels(BINS); axm.tick_params(axis='x', length=0)
for sp in ('left', 'bottom', 'right', 'top'): axm.spines[sp].set_visible(False)
axm.annotate('a', (0, 1), xycoords='axes fraction', xytext=(-1.3 * 72, 32), textcoords='offset points', fontsize=9, fontweight='bold')
# (b) smallest gain one centre can show, for the tasks that have a numeric requirement
VCOL = {'RUNNABLE': PAL['aqua'], 'ESTIMABLE': PAL['orange'], 'OUT OF REACH': PAL['bad']}
need = reg_full[reg_full.one_inst_verdict.isin(VCOL)].copy().sort_values('min_detectable_gain').reset_index(drop=True)
axr.barh(np.arange(len(need)), need.min_detectable_gain, height=.68, zorder=3, color=[VCOL[v] for v in need.one_inst_verdict])
axr.set_xlim(0, .16); axr.axvline(.05, color=PAL['ink'], lw=.8, ls=':', zorder=4)
axr.set_yticks(np.arange(len(need))); axr.set_yticklabels(need.id); axr.set_ylim(len(need) - .4, -.6)
axr.set_xticks([0, .05, .10, .15]); axr.set_xticklabels(['0', '.05', '.10', '.15']); axr.minorticks_off()
axr.tick_params(axis='y', length=0); ygrid(axr, 'x')
axr.set_xlabel('smallest AUROC gain\nshown (80% power)')
axr.annotate('b', (0, 1), xycoords='axes fraction', xytext=(-.2 * 72, 32), textcoords='offset points', fontsize=9, fontweight='bold')
fig.legend([plt.Line2D([], [], marker=m, ls='none', ms=4.6, mfc=PAL['ink2'], mec=PAL['ink2']) for m in MARK.values()] +
           [plt.Line2D([], [], marker='o', ls='none', ms=4.6, mfc=PAL['aqua'], mec=PAL['aqua']),
            plt.Line2D([], [], marker='o', ls='none', ms=4.6, mfc='none', mec=PAL['orange'], mew=1.0),
            plt.Line2D([], [], marker='o', ls='none', ms=4.6, mfc='none', mec=PAL['bad'], mew=1.0),
            plt.Line2D([], [], marker='o', ls='none', ms=4.6, mfc=PAL['grey2'], mec=PAL['grey'])],
           ['R1 EHR timeline only', 'R2 needs banked specimens', 'R3 needs prospective collection', 'R4 research only',
            'runnable at one centre', 'estimable only', 'out of reach', 'no single-AUROC verdict'], ncol=4, loc='lower center', bbox_to_anchor=(.5, 0.0),
           columnspacing=1.5, handletextpad=.3, labelspacing=.45)
finish(fig, 7, 'gvhd_bench_structure',
       'GvHD-Bench v1.0. (a) The 40 tasks, grouped by family and by the landmark at which the prediction is made. Marker '
       'shape is data readiness and colour is the feasibility verdict at one institution (up to 1,500 patients, about 420 with banked specimens); tasks that are not a single-AUROC comparison '
       '(donor ranking, counterfactual contrasts, representation and robustness audits) and research-only tasks are grey. (b) For the 25 tasks with a numeric '
       'requirement: the smallest AUROC gain over the task baseline that the eligible patients of one centre can show with 80% power. The dotted line at +0.05 separates runnable from estimable; '
       'tasks to its right are labelled, not omitted.')
''')


md(r'''
## 8.10 Claim ladder, with gates fixed in advance
| Level | What may be claimed | Requirements (all of them) |
|---|---|---|
| **0** | nothing | — |
| **1** | "more label-efficient than a tuned simple model" | equal or better at full labels, **and** better at k ≤ 100 labelled patients, paired across centres; beats its random-weight twin |
| **1C** | "works at this centre" (one institution, **no out-of-centre claim**) | temporal split (earliest 60 % development, latest 40 % test; ≥ 600 test patients at 1,500); the paired DeLong 95 % interval of ΔIPCW-td-AUROC against the tuned simple model has a **lower bound ≥ +0.01** (designed for a true gain of +0.05); a head no larger than the development set supports; beats random-weight twin and no-pretraining; calibration slope and O/E reported |
| **2** | "better discrimination out-of-centre" | ≥ 20 held-out centres; the **centre-level 95 % t-interval** of ΔIPCW-td-AUROC against the best simple model **and** against the clinical baseline has a **lower bound ≥ +0.01** (the design is powered for a true gain of +0.03; §8.12); improves in ≥ 70 % of held-out centres; survives BH correction across the task family; beats random-weight twin and no-pretraining |
| **3** | "clinically useful" | Level 2 **plus** calibration slope within 0.8–1.25 and positive net benefit across 10–40 % thresholds in leave-centre-out **and** temporal validation, at n ≥ the Riley minimum, with a prospective silent-mode evaluation and a decision-impact plan |
| **4** | "improves outcomes" | a prospective trial of the model-guided decision. Nothing short of this licenses the word *improves* |

**Pre-registration rule.** Gate thresholds, splits, seeds and the primary task are written to `spec/gate.json` before any model is run, and the file hash is reported with the results. The gate below is the one this specification proposes for the ALLO-FM v1 attempt. Its interval method, margin and minimum centre count are not arbitrary: §8.12 simulates the gate and shows that the obvious alternative (a point estimate of +0.03 and a percentile-bootstrap interval) passes a model whose true gain equals the threshold only about half the time and under-covers when centres are few.
''')

code(r'''
# ----------------------------------------------------------------------------- the pre-stated gate for the ALLO-FM v1 attempt (written to spec/gate.json)
GATE = dict(
  version='ALLO-FM v1',
  primary_task='B1 — grade II–IV aGvHD by day +100 from the day +7 draw (issued day +9); Level 2: leave-centre-out across >= 20 centres; Level 1C: temporal split at one institution',
  primary_metric='IPCW time-dependent AUROC at day 100, competing risks = death and relapse',
  baselines_to_beat=['MAGIC MAP (ST2+REG3a day-7 algorithm) + clinical covariates',
                     'gradient-boosted trees on 20-40 registry variables (tuned, same splits)',
                     'random-weight twin of the backbone', 'no-pretraining (supervised only)'],
  level2_gate=dict(delta_auroc_designed_for=0.03, ci_method='centre-level t-interval, K-1 d.f. (not a percentile bootstrap)',
                   ci_lower_bound_min=0.01, fraction_centres_improved_min=0.70, min_centres=20,
                   beats_random_twin=True, beats_no_pretraining=True, bh_across_family=True,
                   rationale='interval method, margin and centre count are fixed by the operating characteristics computed in section 8.12'),
  level1c_single_centre=dict(applies_to='one institution; no out-of-centre claim', split='temporal: earliest 60 % of transplants for development, latest 40 % for test (600 test patients at 1,500)',
                             interval='paired DeLong 95 % interval of the AUROC difference on the temporal test set', ci_lower_bound_min=0.01, designed_for_gain=0.05, min_test_patients=600,
                             must_beat=['tuned simple model (L0)', 'random-weight twin', 'no-pretraining'], max_head_parameters='from the Riley criteria for the development set (about 14 at 900 patients, AUROC 0.72, prevalence 0.35)',
                             claims_not_allowed=['generalisation to other centres', 'subgroup equity', 'a gain smaller than +0.03']),
  level3_additional=dict(calibration_slope=[0.8, 1.25], net_benefit_positive_thresholds=[0.10, 0.40],
                         temporal_validation_required=True, min_n='computed per task by riley_n(P, target AUROC, prevalence) and validation_n (section 9.7)'),
  stopping_rule='if Level 1 is not reached on B1 at stage-1 scale, publish the negative result and do not proceed to stage 3',
  frozen=dict(seed=SEED, splits='spec/splits_*.json', label_functions='section 4.6 of this notebook'))
def level2_verdict(lo, frac, K):
    """Apply the Level-2 gate to a centre-level interval lower bound, fraction of centres improved, and centre count."""
    g = GATE['level2_gate']
    if K < g['min_centres']: return f'not evaluable (K={K} < {g["min_centres"]})'
    return 'PASS' if (lo >= g['ci_lower_bound_min'] and frac >= g['fraction_centres_improved_min']) else 'fail'
json.dump(GATE, open(os.path.join(SPEC_DIR, 'gate.json'), 'w'), indent=2)
import hashlib
h = hashlib.sha256(open(os.path.join(SPEC_DIR, 'gate.json'), 'rb').read()).hexdigest()[:16]
print(f'gate written to spec/gate.json  (sha256:{h})')
for k, v in GATE.items(): print(f'  {k:22s} {v if not isinstance(v, dict) else json.dumps(v)}'[:200])
''')

md(r'''
## 8.11 [SIM] The harness, executed end-to-end

To show the benchmark *runs*, we simulate a multi-centre cohort with the structure of a real registry extract: 24 centres, differing case mix and prophylaxis practice, competing risks (relapse and NRM), administrative censoring, a day-7 biomarker available in a subset, and — deliberately — **a planted leakage feature** (an early steroid order that follows the outcome). We then run task B1 through the harness: leave-centre-out, clinical baseline versus a timeline model, IPCW metrics, the leakage audit, calibration and net benefit.

This demonstrates the machinery and the audit. **It is not evidence about patients**: every effect size is one we chose.
''')

code(r'''
# ----------------------------------------------------------------------------- [SIM] multi-centre registry-like cohort generator
def simulate_cohort(n=6000, n_centres=24, seed=3, leak=True, drift=0.0, presence_gamma=0.0):
    r = np.random.default_rng(seed)
    era = np.random.default_rng(seed + 1).integers(2014, 2024, n)       # own stream: adding drift leaves every other column unchanged
    centre = r.integers(0, n_centres, n)
    c_off = r.normal(0, .35, n_centres)                                  # centre-level risk offset (case mix + practice)
    c_ptcy = r.beta(2, 2, n_centres)                                     # centre-level PtCy adoption rate
    age = np.clip(r.normal(52, 14, n), 1, 78)
    donor_type = r.choice(['MSD', 'MUD 8/8', 'MMUD 7/8', 'haplo'], n, p=[.22, .42, .14, .22])
    hla_mm = np.select([donor_type == 'MSD', donor_type == 'MUD 8/8', donor_type == 'MMUD 7/8'], [0, 0, 1], default=2)
    source = np.where(r.random(n) < .78, 'PBSC', 'BM')
    fem_to_male = (r.random(n) < .18).astype(int)
    hctci = r.poisson(2.2, n); dri = r.choice([0, 1, 2, 3], n, p=[.18, .46, .26, .10])
    mac = (r.random(n) < .55).astype(int)
    ptcy = (r.random(n) < c_ptcy[centre]).astype(int)
    atg = ((1 - ptcy) * (r.random(n) < .25)).astype(int)
    cd34 = np.clip(r.normal(6.2, 2.1, n), 1, 15)
    # day-7 state (Tier B/C): CRP, ALC, tacrolimus trough, and the MAGIC-like biomarker (measured in 55%)
    base = (-1.15 + .55 * hla_mm + .30 * fem_to_male + .22 * mac + .10 * (source == 'PBSC')
            + .11 * dri * .3 + .018 * (age - 52) / 10 - .62 * ptcy - .25 * atg + c_off[centre] + drift * (era - 2014))
    lat = base + r.normal(0, .55, n)                                      # latent alloreactivity
    crp7 = np.exp(np.clip(1.9 + .55 * lat + r.normal(0, .55, n), -2, 6))
    alc7 = np.exp(np.clip(-1.4 - .25 * lat + r.normal(0, .5, n), -5, 3))
    tac7 = np.clip(r.normal(9.5 - .6 * lat, 2.6, n), 1, 22)
    st2 = np.exp(np.clip(3.3 + .62 * lat + r.normal(0, .5, n), 0, 9))
    reg3a = np.exp(np.clip(3.0 + .70 * lat + r.normal(0, .6, n), 0, 9))
    u_meas = r.random(n)
    zl = (lat - lat.mean()) / lat.std()          # presence_gamma > 0: clinicians order the biomarker more often when they suspect GvHD
    measured = u_meas < 1 / (1 + np.exp(-(math.log(.55 / .45) + presence_gamma * zl)))
    p_ag = 1 / (1 + np.exp(-(lat + .45 * np.log(st2 / 27) + .40 * np.log(reg3a / 20) - .15)))
    agvhd = (r.random(n) < p_ag).astype(int)
    t_ag = np.clip(r.gamma(3.2, 9, n) + 10, 8, 170)
    t_rel = np.exp(np.clip(5.6 - .35 * dri + r.normal(0, .8, n), 2, 9))
    t_nrm = np.exp(np.clip(6.1 - .20 * hctci - .5 * agvhd * p_ag + r.normal(0, .9, n), 2, 9))
    t_cens = np.clip(r.uniform(90, 1200, n), 30, 1200)
    T = np.vstack([np.where(agvhd == 1, t_ag, 1e6), t_rel, t_nrm, t_cens])
    which = T.argmin(0); time = T.min(0)
    cause = np.where(which == 0, 1, np.where(which == 1, 2, np.where(which == 2, 3, 0)))   # 1 aGvHD, 2 relapse, 3 NRM, 0 censored
    # the planted leak: an "early steroid order" recorded at day +5 that is really driven by the outcome
    steroid_d5 = ((cause == 1) & (time <= 100)).astype(int) * (r.random(n) < .8) + (r.random(n) < .03)
    df = pd.DataFrame(dict(pid=np.arange(n), centre=centre, era=era, age=age, donor_type=donor_type,
                           hla_mm=hla_mm, source=source, fem_to_male=fem_to_male, hctci=hctci, dri=dri, mac=mac,
                           ptcy=ptcy, atg=atg, cd34=cd34, crp7=crp7, alc7=alc7, tac7=tac7,
                           st2=np.where(measured, st2, np.nan), reg3a=np.where(measured, reg3a, np.nan),
                           biomarker_measured=measured.astype(int), time=time, cause=cause,
                           _latent_event=((agvhd == 1) & (t_ag <= 100)).astype(int)))   # oracle column for planted-effect tests; never a feature
    if leak: df['steroid_order_d5'] = steroid_d5.astype(int)
    return df

LANDMARK = 9                                            # task B1: day-7 draw, 48-hour turnaround, prediction issued on day +9
COH_ALL = simulate_cohort()
keep_ = (COH_ALL.time > LANDMARK).values
attr = pd.DataFrame([
    ('patients in the cohort (day 0 recorded)', len(COH_ALL)),
    (f'  died without relapse before day +{LANDMARK}', int(((COH_ALL.cause == 3) & ~keep_).sum())),
    (f'  relapsed before day +{LANDMARK}', int(((COH_ALL.cause == 2) & ~keep_).sum())),
    (f'  censored before day +{LANDMARK}', int(((COH_ALL.cause == 0) & ~keep_).sum())),
    (f'  already had grade II-IV aGvHD before day +{LANDMARK}', int(((COH_ALL.cause == 1) & ~keep_).sum())),
    ('meeting landmark eligibility (analysed)', int(keep_.sum())),
    ('  of whom with the day-7 biomarker panel', int(COH_ALL.loc[keep_, 'biomarker_measured'].sum()))], columns=['attrition step', 'patients'])
print('MANDATORY ATTRITION TABLE (§8.1):'); display(md_table(attr))
COH = COH_ALL[keep_].reset_index(drop=True)
inc100 = aalen_johansen(COH.time.values, COH.cause.values, 100, 1)
print(f'[SIM] cohort: {len(COH)} patients, {COH.centre.nunique()} centres | '
      f'Aalen-Johansen incidence of grade II-IV aGvHD by d+100, given alive and event-free at the landmark = {inc100:.3f}')
print('cause distribution (0 censored, 1 aGvHD, 2 relapse, 3 NRM):', dict(COH.cause.value_counts().sort_index()))
display(md_table(COH.groupby(COH.centre).agg(n=('pid', 'count'), ptcy_rate=('ptcy', 'mean'),
        aGvHD_by_d100=('cause', lambda s: float(((COH.loc[s.index, 'cause'] == 1) & (COH.loc[s.index, 'time'] <= 100)).mean()))).round(3).head(8), index=True))
''')

code(r'''
# ----------------------------------------------------------------------------- the leakage audit, as code (run on the [SIM] cohort for task B1, issued on day +9)
BANNED_PREFIXES = ('steroid', 'ruxolitinib', 'ecp_', 'budesonide', 'biopsy_order', 'infliximab')
PLAUSIBLE_SINGLE_FEATURE_AUROC = .80      # no single day-7 variable has been reported to discriminate aGvHD better than ~0.75; above 0.80 is a red flag
def leakage_audit(df, feature_cols, landmark, t_known_map, group_col='pid', centre_col='centre',
                  outcome=lambda d: ((d.cause == 1) & (d.time <= 100)).astype(int), folds=None):
    """Runs the §4.7 checks and returns a findings table. t_known_map: {feature: day the value becomes known}."""
    y = outcome(df).values; out = []
    banned = [c for c in feature_cols if c.lower().startswith(BANNED_PREFIXES)]
    out.append(dict(check='treatment-proxy features present in the input set', finding=', '.join(banned) or 'none', status='FAIL' if banned else 'pass'))
    late = [c for c in feature_cols if t_known_map.get(c, -np.inf) > landmark]
    out.append(dict(check=f't_known > landmark ({landmark:+g}d)', finding=', '.join(late) or 'none', status='FAIL' if late else 'pass'))
    dup = int(df[group_col].duplicated().sum())
    out.append(dict(check='duplicate patient records', finding=f'{dup} duplicated ids', status='FAIL' if dup else 'pass'))
    if folds is not None:
        bleed = sum(len(set(df[group_col].values[tr]) & set(df[group_col].values[te])) for tr, te in folds)
        cbleed = sum(len(set(df[centre_col].values[tr]) & set(df[centre_col].values[te])) for tr, te in folds)
        out.append(dict(check='patient overlap between train and test folds', finding=f'{bleed} patients', status='FAIL' if bleed else 'pass'))
        out.append(dict(check='centre appears on both sides (leave-centre-out claim)', finding=f'{cbleed} centres', status='FAIL' if cbleed else 'pass'))
    ce = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(pd.get_dummies(df[centre_col].astype(str)).values, y)
    a_centre = auroc(y, ce.decision_function(pd.get_dummies(df[centre_col].astype(str)).values))
    out.append(dict(check='centre-ID-only model AUROC (case-mix/practice confounding)', finding=f'{a_centre:.3f}',
                    status='WARN' if a_centre > .6 else 'pass'))
    for c in [c for c in feature_cols if c in df.columns and df[c].notna().sum() > 50 and np.issubdtype(df[c].dtype, np.number)]:
        a = auroc(y[df[c].notna().values], df[c].dropna().values)
        if a == a and (a > PLAUSIBLE_SINGLE_FEATURE_AUROC or a < 1 - PLAUSIBLE_SINGLE_FEATURE_AUROC):
            out.append(dict(check=f'single feature with implausible AUROC: {c}', finding=f'{a:.3f}', status='FAIL'))
    return pd.DataFrame(out)

CLIN = ['age', 'hla_mm', 'fem_to_male', 'hctci', 'dri', 'mac', 'ptcy', 'atg', 'cd34']
D7 = ['crp7', 'alc7', 'tac7']
BIO = ['st2', 'reg3a']
T_KNOWN = {**{c: -7 for c in CLIN}, **{c: 7 for c in D7}, 'st2': 9, 'reg3a': 9, 'steroid_order_d5': 5, 'biomarker_measured': 9}
audit_bad = leakage_audit(COH, CLIN + D7 + BIO + ['biomarker_measured', 'steroid_order_d5'], landmark=LANDMARK, t_known_map=T_KNOWN)
print('AUDIT — candidate input set that includes the planted leak:')
display(md_table(audit_bad))
audit_good = leakage_audit(COH, CLIN + D7 + BIO + ['biomarker_measured'], landmark=LANDMARK, t_known_map=T_KNOWN)
print('AUDIT — the cleaned input set actually used for B1 (prediction issued on day +9):')
display(md_table(audit_good))
''')

code(r'''
# ----------------------------------------------------------------------------- [SIM] run task B1 through the harness: leave-centre-out, 6 arms, IPCW metrics
def prep(df, cols):
    X = df[cols].copy()
    for c in cols:
        if X[c].dtype == object: X[c] = pd.factorize(X[c])[0]
    med = X.median(); X = X.fillna(med)
    for c in ('st2', 'reg3a', 'crp7', 'alc7'):
        if c in X.columns: X[c] = np.log(np.clip(X[c], 1e-3, None))
    return X.values.astype(float)

def y_b1(df): return ((df.cause == 1) & (df.time <= 100)).astype(int).values

ARMS = {
 'clinical baseline (registry covariates)':      CLIN,
 'clinical + MAP-like biomarkers':               CLIN + BIO + ['biomarker_measured'],
 'timeline model (clinical + day-7 state)':      CLIN + D7,
 'full (clinical + day-7 + biomarkers)':         CLIN + D7 + BIO + ['biomarker_measured'],
 'confounder-only (centre + era)':               ['centre', 'era'],
 'LEAKING pipeline (adds steroid order d+5)':    CLIN + D7 + BIO + ['steroid_order_d5'],
}
centres = sorted(COH.centre.unique())
held = centres if not QUICK else centres[:8]
rows = []
for arm, cols in ARMS.items():
    X = prep(COH, cols); y = y_b1(COH)
    for c in held:
        tr, te = (COH.centre != c).values, (COH.centre == c).values
        if y[tr].sum() < 10 or te.sum() < 40: continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000, class_weight='balanced')).fit(X[tr], y[tr])
        s = m.decision_function(X[te]); p = m.predict_proba(X[te])[:, 1]
        rows.append(dict(arm=arm, centre=c, n=int(te.sum()),
                         ipcw_auroc=ipcw_td_auroc(COH.time.values[te], COH.cause.values[te], s, 100),
                         naive_auroc=auroc(y[te], s), brier=float(np.mean((p - y[te]) ** 2))))
LCO = pd.DataFrame(rows)
summ = LCO.groupby('arm').agg(centres=('centre', 'count'), mean_ipcw_auroc=('ipcw_auroc', 'mean'),
                              sd_across_centres=('ipcw_auroc', 'std'), min_centre=('ipcw_auroc', 'min'),
                              max_centre=('ipcw_auroc', 'max'), mean_brier=('brier', 'mean')).round(3)
summ = summ.reindex(list(ARMS)); display(md_table(summ, index=True))

from scipy.stats import t as student_t
def centre_interval(d, level=.95):
    """Centre-level t-interval for the mean paired difference across K held-out centres (K-1 d.f.): the gate's interval."""
    d = np.asarray(d, float); K = len(d); m = d.mean(); se = d.std(ddof=1) / math.sqrt(K); q = student_t.ppf(1 - (1 - level) / 2, K - 1)
    return float(m), float(m - q * se), float(m + q * se)
def boot_delta(a, b, n=4000, seed=0):
    """Percentile bootstrap over centres, kept for comparison: section 8.12 shows it under-covers when centres are few."""
    d = np.asarray(a) - np.asarray(b); r = np.random.default_rng(seed)
    bs = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
piv = LCO.pivot_table(index='centre', columns='arm', values='ipcw_auroc')
base = 'clinical baseline (registry covariates)'
def _row(a):
    m, lo, hi = centre_interval(piv[a].values - piv[base].values); bl, bh = boot_delta(piv[a], piv[base]); frac = float((piv[a] > piv[base]).mean())
    return dict(arm=a, delta_vs_clinical=m, lo=lo, hi=hi, boot_lo=bl, boot_hi=bh, centres_improved=f"{int((piv[a] > piv[base]).sum())}/{len(piv)}",
                level2_gate='excluded: leakage audit FAIL' if 'LEAKING' in a else level2_verdict(lo, frac, len(piv)))
delta = pd.DataFrame([_row(a) for a in piv.columns if a != base]).round(3)
display(md_table(delta))
''')

code(r'''
# ----------------------------------------------------------------------------- out-of-centre predictions of the full arm -> calibration and decision curve
X_full = prep(COH, ARMS['full (clinical + day-7 + biomarkers)']); y_full = y_b1(COH)
P = np.full(len(COH), np.nan)
for c in centres:
    tr, te = (COH.centre != c).values, (COH.centre == c).values
    m = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000)).fit(X_full[tr], y_full[tr])
    P[te] = m.predict_proba(X_full[te])[:, 1]
dec = pd.qcut(P, 10, labels=False, duplicates='drop')
cal = pd.DataFrame([dict(decile=int(d), predicted=P[dec == d].mean(),
                         observed_AJ=aalen_johansen(COH.time.values[dec == d], COH.cause.values[dec == d], 100, 1),
                         n=int((dec == d).sum())) for d in sorted(set(dec[~pd.isna(dec)]))])
slope = np.polyfit(cal.predicted, cal.observed_AJ, 1)
nbc = net_benefit(y_full, P, np.linspace(.05, .6, 23))

# ----------------------------------------------------------------------------- Figure 8: the harness executing task B1 on the [SIM] cohort
ARM_STYLE = {   # key -> (colour, two-line axis label, one-line label)
 'clinical baseline (registry covariates)':   (PAL['seq'][0], 'clinical',            'clinical'),
 'clinical + MAP-like biomarkers':            (PAL['seq'][1], 'clinical\n+ MAP',      'clinical + MAP'),
 'timeline model (clinical + day-7 state)':   (PAL['seq'][2], 'clinical\n+ day 7',    'clinical + day 7'),
 'full (clinical + day-7 + biomarkers)':      (PAL['seq'][3], 'full\nmodel',          'full model'),
 'confounder-only (centre + era)':            (PAL['grey'],   'centre + era\nonly',   'centre + era only'),
 'LEAKING pipeline (adds steroid order d+5)': (PAL['bad'],    'leaking\n(excluded)',  'leaking (excluded)'),
}
order = [a for a in ARM_STYLE if a in piv.columns]
FWIN, FHIN = FULL, 5.0
fig = plt.figure(figsize=(FWIN, FHIN))
def A(l, b, w, h): return fig.add_axes([l / FWIN, b / FHIN, w / FWIN, h / FHIN])
axa = A(1.05, 2.95, 3.25, 1.70); axb = A(5.55, 2.95, 1.55, 1.70)
axc = A(0.95, 0.60, 1.55, 1.50); axd = A(3.35, 0.60, 1.55, 1.50); axe = A(5.75, 0.60, 1.35, 1.50)
ax = axa; ygrid(ax)                                                    # (a) per-centre spread
for i, a in enumerate(order):
    c = ARM_STYLE[a][0]; v = piv[a].dropna().values
    ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.15, .15, len(v)), v, s=8, color=c, alpha=.8, lw=0, zorder=3)
    ax.plot([i - .30, i + .30], [v.mean()] * 2, color=PAL['ink'], lw=1.4, zorder=4)
    ax.text(i, 1.03, f'{v.mean():.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
ax.axhline(.5, color=PAL['ink2'], ls='--', lw=.6)
ax.set_xticks(range(len(order))); ax.set_xticklabels([ARM_STYLE[a][1] for a in order]); ax.set_xlim(-.55, len(order) - .45)
ax.set_ylim(.42, 1.08); ax.set_yticks([.5, .6, .7, .8, .9, 1.0]); ax.set_ylabel('IPCW td-AUROC at day +100\n(one point per held-out centre)'); panel(ax, 'a')
ax = axb; ygrid(ax, 'x')                                               # (b) difference from the clinical baseline
dd_ = delta.set_index('arm').reindex([a for a in order if a in delta.arm.values]); ypos = np.arange(len(dd_))
for y, (a, r) in zip(ypos, dd_.iterrows()):
    c = ARM_STYLE[a][0]; ax.plot([r.lo, r.hi], [y, y], color=c, lw=1.8, solid_capstyle='round'); ax.plot(r.delta_vs_clinical, y, 'o', color=c, ms=4.2, zorder=4)
ax.axvline(0, color=PAL['ink'], lw=.8)
ax.set_yticks(ypos); ax.set_yticklabels([ARM_STYLE[a][2] for a in dd_.index]); ax.invert_yaxis(); ax.tick_params(axis='y', length=0)
ax.set_xlim(-.30, .32); ax.set_xlabel('Δ IPCW td-AUROC vs clinical\n(centre-level 95% t-interval)'); panel(ax, 'b')
ax = axc; ygrid(ax)                                                    # (c) calibration against Aalen-Johansen
mx = float(max(cal.predicted.max(), cal.observed_AJ.max())) * 1.12
ax.plot([0, mx], [0, mx], color=PAL['ink2'], ls='--', lw=.7)
ax.plot(cal.predicted, cal.observed_AJ, 'o-', color=PAL['seq'][3], ms=3.8, lw=1.3, zorder=3)
ax.set_xlabel('predicted risk (decile mean)'); ax.set_ylabel('observed Aalen–Johansen\nincidence'); ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.text(.04, .96, f'slope {slope[0]:.2f}\nintercept {slope[1]:+.3f}', transform=ax.transAxes, fontsize=6.5, va='top'); panel(ax, 'c')
ax = axd; ygrid(ax)                                                    # (d) decision curve
ax.plot(nbc.threshold, nbc.model, color=PAL['seq'][3], lw=1.6, label='full model')
ax.plot(nbc.threshold, nbc.treat_all, color=PAL['grey'], lw=1.2, ls='--', label='treat all')
ax.axhline(0, color=PAL['ink2'], lw=.7, label='treat none')
ax.set_xlabel('threshold probability'); ax.set_ylabel('net benefit')
ax.set_ylim(min(-.03, nbc.model.min() * 1.1), max(.02, nbc.model.max() * 1.15)); ax.legend(loc='upper right', labelspacing=.3); panel(ax, 'd')
ax = axe; ygrid(ax)                                                    # (e) the leakage canary
lk = ['full (clinical + day-7 + biomarkers)', 'LEAKING pipeline (adds steroid order d+5)']; vals = [summ.loc[a, 'mean_ipcw_auroc'] for a in lk]
ax.bar([0, 1], vals, color=[ARM_STYLE[a][0] for a in lk], width=.62, zorder=3)
for i, v in enumerate(vals): ax.text(i, v + .012, f'{v:.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
ax.annotate('', xy=(1.42, vals[1]), xytext=(1.42, vals[0]), arrowprops=dict(arrowstyle='<->', color=PAL['bad'], lw=.9, shrinkA=0, shrinkB=0))
ax.text(1.52, (vals[0] + vals[1]) / 2, f'+{vals[1] - vals[0]:.3f}\nfrom one\nleaking\nfeature', fontsize=6.4, color=PAL['bad'], va='center', ha='left', fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(['audited\ninputs', 'leak\nincluded']); ax.set_xlim(-.5, 2.6)
ax.set_ylabel('mean IPCW td-AUROC'); ax.set_ylim(.5, 1.06); panel(ax, 'e')
finish(fig, 8, 'harness_task_B1',
       '[SIM] The GvHD-Bench harness executing task B1 (grade II-IV aGvHD by day +100 from the day +7 state) on a simulated '
       '24-centre cohort. (a) Leave-centre-out IPCW time-dependent AUROC, one point per held-out centre, with the mean '
       'printed above; the pooled mean alone would hide this spread. (b) Difference from the clinical baseline with a '
       'centre-level t-interval. (c) Calibration against the Aalen-Johansen cumulative incidence, the competing-risk '
       'analogue of a calibration plot. (d) Decision-curve net benefit. (e) The leakage canary: one planted treatment-proxy '
       'feature inflates the headline number, and the audit flags it before any model is fitted. Effect sizes belong to the simulation.')
display(md_table(cal.round(3)))
''')

code(r'''
g = lambda a: summ.loc[a, 'mean_ipcw_auroc']
leak_gain = g('LEAKING pipeline (adds steroid order d+5)') - g('full (clinical + day-7 + biomarkers)')
d_full = delta.set_index('arm').loc['full (clinical + day-7 + biomarkers)']
fail_rows = audit_bad[audit_bad.status == 'FAIL']; n_fail = len(fail_rows)
fail_list = '; '.join(f"{r.check}: {r.finding}" for r in fail_rows.itertuples())
nb_pos = float(nbc.threshold[(nbc.model > nbc.treat_all) & (nbc.model > 0)].min()) if ((nbc.model > nbc.treat_all) & (nbc.model > 0)).any() else float('nan')
display(Markdown(f"""**Result [SIM].** The harness runs. Mean IPCW td-AUROC over held-out centres: clinical baseline **{g('clinical baseline (registry covariates)'):.3f}**, clinical + MAP-like biomarkers **{g('clinical + MAP-like biomarkers'):.3f}**, clinical + day-7 state **{g('timeline model (clinical + day-7 state)'):.3f}**, full **{g('full (clinical + day-7 + biomarkers)'):.3f}** (Δ vs clinical **{d_full.delta_vs_clinical:+.3f}**, centre-level 95 % t-interval [{d_full.lo:+.3f}, {d_full.hi:+.3f}], improved in {d_full.centres_improved} centres; Level-2 gate: **{d_full.level2_gate}**). Confounder-only (centre + era) reaches **{g('confounder-only (centre + era)'):.3f}**, which is the number any real result must be read against.

**The leakage canary worked.** Adding the planted day +5 steroid order lifts the mean to **{g('LEAKING pipeline (adds steroid order d+5)'):.3f}** ({leak_gain:+.3f}), and the audit failed that input set on **{n_fail} counts before any model was fitted** ({fail_list}). **A gain of this size is exactly what a leaking pipeline produces**, which is why the audit runs first and is reported alongside the result.

Calibration out-of-centre has slope {slope[0]:.2f} against the Aalen–Johansen incidence, and the decision curve is above both default strategies from a threshold of about {nb_pos:.2f}. All of these numbers are properties of a cohort we generated; their only purpose is to show that the metrics, splits, audit and gate logic execute and behave as specified."""))
''')
