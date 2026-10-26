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
| **Governance** | tasks are added by pull request with a task card (§8.2) and a power calculation; a task whose accessible n falls below its computed minimum is published as **under-powered** rather than quietly scored. |

## 8.1 Cohort definition (the part that is usually left vague)

**Included.** First allogeneic HCT for any indication; any donor type, graft source and conditioning intensity; age ≥ 0; transplant year within the stated era window; a recorded day 0 and at least one follow-up contact after day +7.

**Excluded.** Autologous and syngeneic transplants; second and subsequent transplants (they open a new `pair_id` and are analysed separately); patients who withdrew consent for research use; records whose day 0 or outcome date is missing or internally inconsistent (these are counted and reported, never silently dropped).

**Landmark eligibility, applied per task.** A patient enters a task at landmark *L* only if they are alive, in follow-up and **have not already had the outcome event** at *L*. Patients who died, relapsed or were censored before *L* are excluded from that task and counted in its attrition table. For response tasks the clock starts at treatment initiation, not at day 0.

**Competing events.** Death and relapse are competing events for every GvHD endpoint, never censoring. Cumulative incidence uses Aalen–Johansen; discrimination uses inverse-probability-of-censoring weighting (§8.4).

**Mandatory attrition reporting.** Every task reports: patients in cohort → meeting landmark eligibility → with the required input tier → with a non-missing outcome, with the count lost at each step. A benchmark that cannot show its attrition cannot be audited.

"""
)

md(r"""
## 8.2 Design principles

1. **Tasks are decisions.** Every task sits at a real decision point on the Figure-1 timeline, with exactly the inputs a clinician would have had at that moment.
2. **Every task names the model it must beat** — not chance, but what is used today: refined DRI, HCT-CI, EASIX, the MAGIC algorithm probability, onset grade and organ pattern, NIH organ scores, or a tuned gradient-boosted tree on registry variables.
3. **Competing risks are first-class.** Death and relapse are events, not censoring.
4. **Splits are hard by default**: leave-centre-out, and temporal for any deployment claim. Random splits are reported only as an upper bound and labelled as such.
5. **Minimum n is computed, not assumed** (§8.3), and each task carries a *powered / under-powered* verdict that travels with its result.
6. **A negative result is a result.** The benchmark is designed so that "the foundation model did not beat the baseline" is a reportable, publishable outcome with a clear interpretation.
""")

md(r'''
## 8.3 The task registry
Families: **A** pre-transplant planning · **B** early post-transplant prognosis · **C** diagnosis and acute course · **D** treatment response · **E** chronic GvHD and late effects · **F** mechanism and representation · **G** robustness and governance.
''')

code(r'''
# ----------------------------------------------------------------------------- Riley et al. sample-size criteria (needed by the registry; validated below)
def r2cs_from_auc(A, prev, n_mc=400_000, seed=0):
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
('B1','B early','Grade II–IV aGvHD by day +100 from the day +7 state','day +7','day +100','A,B (+C biomarkers)','MAGIC grade ≥ II','death, relapse','IPCW td-AUROC; net benefit at 10–40 %','**MAP (ST2+REG3α day-7 algorithm)** + clinical',.35,.72),
('B2','B early','Severe (grade III–IV) aGvHD by day +100 from day +14','day +14','day +100','A,B,C','MAGIC grade ≥ III','death, relapse','IPCW td-AUROC, AUPRC','MAP + clinical',.13,.75),
('B3','B early','7-day-ahead rolling onset alarm (continuous early warning)','every day, +7 to +60','7 days','A,B (+C,D)','first day of grade ≥ II within 7 d','death, relapse, discharge','AUROC per alarm-day; alerts per 100 patient-days at 50 % sensitivity','clinician-trigger proxy (CRP + stool volume thresholds)',.04,.75),
('B4','B early','Lower-GI aGvHD specifically by day +100 from day +14','day +14','day +100','A,B,C,D','lower-GI stage ≥ 1','death, relapse','IPCW td-AUROC','REG3α + clinical',.22,.72),
('B5','B early','NRM at 1 year from day +7','day +7','1 year','A,B,C','death without relapse','relapse','IPCW td-AUROC, calibration','MAP (its validated use) + HCT-CI',.18,.75),
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
('G4','G robustness','Tier ablation: A only → A+B → A+B+C → A+B+C+D','as parent','as parent','progressively','as parent','—','ΔAUROC per added tier, with cost per patient','Tier A model',None,None),
('G5','G robustness','Missingness robustness: performance when a modality or a lab stream is absent at inference','as parent','as parent','as parent','as parent','—','AUROC under simulated and natural missingness','complete-case model',None,None),
]
COLS = ['id','family','task','landmark','horizon','inputs allowed (tiers)','outcome (label fn)','competing events',
        'primary metric','clinical baseline to beat','assumed prevalence','target AUROC']
reg = pd.DataFrame(TASKS, columns=COLS)
ok_n = [bool(pd.notna(a) and pd.notna(p)) for a, p in zip(reg['target AUROC'], reg['assumed prevalence'])]
reg['min n (50-param head)'] = [riley_n(50, a, p)['n'] if o else '' for o, a, p in zip(ok_n, reg['target AUROC'], reg['assumed prevalence'])]
reg['min n (768-param head)'] = [riley_n(768, a, p)['n'] if o else '' for o, a, p in zip(ok_n, reg['target AUROC'], reg['assumed prevalence'])]
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
## 8.4 Task cards: readiness, accessible n, and the powered / under-powered verdict

The registry above says what each task *is*. A benchmark also has to say, for each task, **whether it can be run at all today, with which data, on how many patients, and whether that number is enough.** The cell below attaches those fields and computes the verdict. `readiness` is the honest status:

* **R1 — runnable now** on a registry extract (Tier A/B) with no new collection.
* **R2 — runnable on an existing biobank** (Tier C: MAGIC / BMT CTN specimens) with a collaboration.
* **R3 — needs prospective collection** on the 4.4 schedule.
* **R4 — research-grade only**: a nested sub-cohort, reported as mechanism, never as a deployment claim.

`n accessible` is a deliberately conservative estimate of the patients a serious 2026-era effort could assemble for that specific task, given its landmark eligibility and the data tier it needs. The verdict compares it against two requirements: the Riley minimum to *develop* a 50-parameter head, and the n needed to *demonstrate* a +0.03 AUROC improvement over the task's clinical baseline on a paired test. A task passes only if the accessible n clears both.
""")

code(r'''
# ----------------------------------------------------------------------------- task cards: readiness, accessible n, powered verdict (exported with the registry)
# task id -> (readiness, tier needed, conservative accessible n, split protocol, achievable AUROC range from section 3.2)
CARDS = {
 'A1': ('R1', 'A', 120000, 'leave-centre-out + temporal', (.62, .70)), 'A2': ('R1', 'A', 120000, 'leave-centre-out + temporal', (.64, .72)),
 'A3': ('R1', 'A', 90000, 'leave-centre-out + temporal', (.62, .68)),  'A4': ('R1', 'A', 120000, 'leave-centre-out + temporal', (.64, .70)),
 'A5': ('R1', 'A', 120000, 'leave-centre-out + temporal', (.68, .76)), 'A6': ('R1', 'A+C', 40000, 'leave-centre-out; counterfactual', (None, None)),
 'A7': ('R2', 'A', 431, 'randomised arms (BMT CTN 1703)', (None, None)),
 'B1': ('R2', 'A+B+C', 4000, 'leave-centre-out + temporal', (.74, .80)), 'B2': ('R2', 'A+B+C', 4000, 'leave-centre-out + temporal', (.74, .82)),
 'B3': ('R1', 'A+B', 15000, 'leave-centre-out; per-day alarms', (.72, .80)), 'B4': ('R2', 'A+B+C', 3000, 'leave-centre-out', (.70, .78)),
 'B5': ('R2', 'A+B+C', 4000, 'leave-centre-out + temporal', (.72, .80)), 'B6': ('R1', 'A+B', 25000, 'leave-centre-out', (.68, .76)),
 'C1': ('R3', 'A+B(+C)', 800, 'leave-centre-out; adjudicated', (.78, .88)), 'C2': ('R3', 'A+B', 600, 'leave-centre-out; adjudicated', (.72, .82)),
 'C3': ('R1', 'A+B', 6000, 'leave-centre-out; decision-analytic', (.72, .80)),
 'C4': ('R2', 'A+B+C', 2500, 'leave-centre-out', (.68, .76)), 'C5': ('R2', 'A+B+C', 2500, 'leave-centre-out', (.66, .74)),
 'C6': ('R2', 'A+B+C', 2500, 'leave-centre-out + temporal', (.72, .80)), 'C7': ('R3', 'D', 500, 'leave-centre-out; central read', (None, None)),
 'C8': ('R1', 'A+B', 8000, 'leave-centre-out', (None, None)),
 'D1': ('R2', 'A+B', 1200, 'leave-centre-out', (.66, .74)), 'D2': ('R2', 'A+B', 400, 'randomised arms only', (None, None)),
 'D3': ('R1', 'A+B', 9000, 'leave-centre-out', (.64, .72)), 'D4': ('R1', 'A+B', 20000, 'leave-centre-out', (.68, .76)),
 'E1': ('R1', 'A+B', 40000, 'leave-centre-out + temporal', (.65, .72)), 'E2': ('R1', 'A+B', 12000, 'leave-centre-out', (.70, .78)),
 'E3': ('R2', 'A+B', 3000, 'leave-centre-out', (.64, .72)), 'E4': ('R1', 'A+B', 30000, 'leave-centre-out + temporal', (.64, .72)),
 'E5': ('R1', 'A+B', 20000, 'leave-centre-out; bivariate', (None, None)),
 'F1': ('R4', 'D', 80, 'leave-recipient-out', (.60, .66)), 'F2': ('R4', 'C+D', 1000, 'no head trained', (None, None)),
 'F3': ('R4', 'D', 40, 'held-out perturbations', (None, None)),
 'G1': ('R1', 'as parent', 120000, 'temporal', (None, None)), 'G2': ('R1', 'as parent', 120000, 'leave-registry-out', (None, None)),
 'G3': ('R1', 'as parent', 120000, 'subgroup within parent split', (None, None)), 'G4': ('R2', 'A->D', 2500, 'nested tiers, same patients', (None, None)),
 'G5': ('R1', 'as parent', 120000, 'simulated + natural missingness', (None, None)),
}
def hm_se_local(A, n1, n0):
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    return math.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n0 - 1) * (Q2 - A * A)) / (n1 * n0))

def n_for_delta(auroc_base, delta, prev, rho=.85, power=.8, alpha=.05):
    """Approximate n for 80% power on a paired comparison of two correlated AUROCs
    (score correlation rho); cross-checked against the DeLong simulation in section 9."""
    need = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    for n in range(100, 80001, 50):
        se_d = math.sqrt(2 * (1 - rho)) * hm_se_local(auroc_base, max(int(n * prev), 2), max(int(n * (1 - prev)), 2))
        if delta / max(se_d, 1e-12) >= need: return n
    return None

card_rows = []
for _, r in reg.iterrows():
    rd, tier, n_acc, split, ach = CARDS[r.id]
    a, pv_ = r['target AUROC'], r['assumed prevalence']
    has = bool(pd.notna(a) and pd.notna(pv_))
    n_dev = riley_n(50, a, pv_)['n'] if has else None
    n_del = n_for_delta(a - .03, .03, pv_) if has else None
    need = max([x for x in (n_dev, n_del) if x]) if has else None
    verdict = '-' if need is None else ('POWERED' if n_acc >= need else 'UNDER-POWERED')
    card_rows.append({'id': r.id, 'family': r.family.split()[0], 'readiness': rd, 'tier needed': tier,
                      'split protocol': split, 'n accessible (est.)': n_acc,
                      'n to develop (50-param)': n_dev, 'n to show +0.03': n_del,
                      'achievable AUROC': '' if ach[0] is None else f'{ach[0]:.2f}-{ach[1]:.2f}',
                      'verdict': verdict})
cards_df = pd.DataFrame(card_rows)
reg_full = reg.merge(cards_df.drop(columns=['family']), on='id', how='left')
reg_full.to_csv(os.path.join(SPEC_DIR, 'task_registry.csv'), index=False)
cards_df.to_csv(os.path.join(SPEC_DIR, 'task_cards.csv'), index=False)
vc = cards_df.verdict.value_counts().to_dict()
print(f"{len(cards_df)} task cards | POWERED: {vc.get('POWERED', 0)} | UNDER-POWERED: {vc.get('UNDER-POWERED', 0)} | "
      f"not a single-AUROC task: {vc.get('-', 0)}")
print('readiness:', cards_df.readiness.value_counts().sort_index().to_dict())
display(md_table(cards_df))
''')

code(r'''
pw_now = cards_df[(cards_df.readiness == 'R1') & (cards_df.verdict == 'POWERED')].id.tolist()
up = cards_df[cards_df.verdict == 'UNDER-POWERED']
display(Markdown(f"""**Result.** Of {len(cards_df)} tasks, **{(cards_df.verdict == 'POWERED').sum()} are powered** at the accessible sample sizes and **{len(up)} are not**; {(cards_df.verdict == '-').sum()} are not single-AUROC tasks (donor ranking, counterfactual prophylaxis contrasts, multi-label organ involvement, representation quality, robustness audits) and carry their own criteria instead.

**{len(pw_now)} tasks are both runnable today on a registry extract and adequately powered**: {', '.join(pw_now)}. These are the benchmark's spine: they need a data-use agreement, not a new collection.

**The under-powered tasks are named, not hidden**: {', '.join(up.id)}. For each, the accessible n and the n required to demonstrate a +0.03 improvement are printed above, so a reader can see exactly how far short the field currently is. Listing these as ordinary leaderboard rows would manufacture precision that does not exist; GvHD-Bench publishes them with the verdict attached, and a positive result on an under-powered task is reported as a hypothesis, never as evidence."""))
''')


md(r'''
## 8.5 Baselines that every model must beat

| Baseline class | Concretely | Why it is mandatory |
|---|---|---|
| **Clinical standard of care** | DRI, HCT-CI, EASIX, refined donor-selection rules, MAGIC MAP, Ann Arbor score, NIH organ scores | A model that does not beat what clinicians already use has no path to deployment |
| **Tuned simple model** | penalised logistic / Cox or gradient-boosted trees on the 20–40 registry variables, with the same splits | The published audits of foundation models repeatedly found simple models competitive; trees on tabular registry data are strong |
| **Confounder-only** | centre, era, assay batch, library size, pre-analytics, sampling day | If this matches the model, the signal is technical |
| **Treatment-proxy-only** | the banned features, run deliberately as a *leakage canary* | Quantifies how much a leaking pipeline would have "gained" |
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

**Audit checks run before any number is reported** (implemented below): patient overlap between folds; donor shared across split sides; centre bleeding into both sides for a leave-centre-out claim; any feature whose `t_known` exceeds the landmark; any banned treatment-proxy feature present; label-vs-centre association strong enough that a centre-only model scores well; and duplicate patient records across sources.

## 8.8 Mandatory ablations
| Ablation | Question it answers |
|---|---|
| random-weight twin of the backbone | did pretraining contribute anything at all? |
| no-pretraining (supervised only) | same question, from the other side |
| drop each stage-1 objective | which objective carries the transfer? |
| modality knockout (each of D07/D14/D15/D16/D17/D18/D19) | **premise P2**: is the signal really spread across modalities? |
| tier ablation A → A+B → A+B+C → A+B+C+D (task G4) | what does each extra data tier buy, per patient cost? |
| pretraining-corpus scaling (10 / 30 / 100 %) | is scale binding, or does it plateau as in the published scaling study? |
| shared-structure audit (centre decodability, subspace overlap) | **premise P3**: measure ρ before blaming the model |

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
| An under-powered win | the powered/under-powered verdict (§8.4) travels with every result |

**Required artefacts for a published GvHD-Bench result.** Benchmark version and artefact hashes; the attrition table; per-task primary metric with CI; per-centre values; calibration slope and intercept; decision-curve net benefit over 10–40 %; all mandatory baselines in the same table; the ablation suite; the leakage audit output; the seed list; and the negative-results table of everything that did not beat its baseline.
"""
)

code(r'''
# ----------------------------------------------------------------------------- Figure 7: the shape of GvHD-Bench (tasks x landmark, readiness, powered verdict)
LAND_X = {'day \u22127 (before conditioning)': -7, 'day \u22127': -7, 'donor search': -45, 'day \u22123': -3,
          'day +7': 7, 'day +14': 14, 'day +30': 30, 'day +100': 100, 'day +365': 365,
          'every day, +7 to +60': 18, 'day of symptom onset': 21, 'day of rash': 23,
          'onset day': 26, 'biopsy sign-out': 29, 'steroid start': 33, 'day systemic steroids start': 33,
          'second-line start': 45, 'treatment start': 45, 'each taper decision': 120,
          'each taper decision after +180': 200, 'therapy start': 130,
          'pre-HCT / onset': 1, 'any': 60, 'perturbation experiment': 60, 'as parent task': 60, 'as parent': 60}
missing = sorted({l for l in reg_full.landmark if l not in LAND_X})
assert not missing, f'unmapped landmarks: {missing}'          # the figure must never silently default
FAMS = list(reg.family.unique())
FAM_Y = {f.split()[0]: len(FAMS) - 1 - i for i, f in enumerate(FAMS)}
MARK = {'R1': 'o', 'R2': 's', 'R3': '^', 'R4': 'D'}
fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.1), gridspec_kw=dict(width_ratios=[2.0, 1], wspace=.26))
ax = axes[0]
def sx2(d): return np.sign(d) * np.log1p(abs(d))
for fam_key in FAM_Y:
    sub_f = reg_full[reg_full.family.str.startswith(fam_key)].copy()
    sub_f['x'] = [sx2(LAND_X[l]) for l in sub_f.landmark]
    sub_f = sub_f.sort_values('x').reset_index(drop=True)
    # deterministic vertical offsets so co-located tasks never print on top of one another
    offs, used = [], {}
    for x in sub_f.x:
        k = round(float(x), 2); n = used.get(k, 0); used[k] = n + 1
        offs.append(.145 * n - .145 * (list(sub_f.x).count(x) - 1) / 2)
    for (_, r), off in zip(sub_f.iterrows(), offs):
        y = FAM_Y[fam_key] + off
        powered = r.verdict == 'POWERED'
        ax.plot(r.x, y, MARK[r.readiness], ms=3.9, mfc=PAL['aqua'] if powered else 'none',
                mec=PAL['aqua'] if powered else PAL['orange'], mew=.9, zorder=3)
        ax.annotate(r.id, (r.x, y), textcoords='offset points', xytext=(4.2, 0), fontsize=4.7,
                    ha='left', va='center', color=PAL['ink2'])
ax.axvline(0, color=PAL['ink'], lw=.8)
ticks = [-45, -7, 0, 7, 14, 28, 100, 365]
ax.set_xticks([sx2(t) for t in ticks]); ax.set_xticklabels([f'{t:+d}' if t else '0' for t in ticks])
ax.set_yticks(list(FAM_Y.values()))
ax.set_yticklabels([f.split(' ', 1)[0] + '  ' + f.split(' ', 1)[1] for f in FAMS], fontsize=5.6)
ax.set_xlabel('landmark (day relative to graft infusion, signed-log)')
ax.set_xlim(sx2(-82), sx2(900)); ax.set_ylim(-.5, len(FAM_Y) - .0); panel(ax, 'a', dx=-.165)
hx = sx2(-66)
for i, (k, lab) in enumerate([('R1', 'registry now'), ('R2', 'biobank'), ('R3', 'prospective'), ('R4', 'research')]):
    ax.plot(hx, 2.00 - i * .22, MARK[k], ms=3.6, mfc=PAL['ink2'], mec=PAL['ink2'])
    ax.text(hx + .20, 2.00 - i * .22, lab, fontsize=5.0, va='center', color=PAL['ink2'])
ax.plot(hx, .92, 'o', ms=3.6, mfc=PAL['aqua'], mec=PAL['aqua'])
ax.text(hx + .20, .92, 'powered', fontsize=5.0, va='center', color=PAL['aqua'], fontweight='bold')
ax.plot(hx, .70, 'o', ms=3.6, mfc='none', mec=PAL['orange'], mew=.9)
ax.text(hx + .20, .70, 'under-powered', fontsize=5.0, va='center', color=PAL['orange'], fontweight='bold')
ax = axes[1]; ygrid(ax)
need = reg_full.dropna(subset=['n to show +0.03'])
ax.scatter(need['n to show +0.03'], need['n accessible (est.)'],
           c=[PAL['aqua'] if v == 'POWERED' else PAL['orange'] for v in need.verdict], s=14, zorder=3)
lim = [2e2, 3e5]
ax.plot(lim, lim, color=PAL['ink'], lw=.8, ls='--')
ax.text(4e3, 2.6e3, 'powered', fontsize=5.8, color=PAL['aqua'], fontweight='bold', rotation=45)
ax.text(1.6e3, 6e3, 'under-powered', fontsize=5.8, color=PAL['orange'], fontweight='bold', rotation=45)
seen = set()
for _, r in need.sort_values('n accessible (est.)').iterrows():
    if r.verdict == 'POWERED': continue
    key = (round(math.log10(r['n to show +0.03']), 1), round(math.log10(r['n accessible (est.)']), 1))
    dy = -1 - 5 * len([k for k in seen if k == key]); seen.add(key)
    ax.annotate(r.id, (r['n to show +0.03'], r['n accessible (est.)']), textcoords='offset points',
                xytext=(4, dy), fontsize=4.8, color=PAL['ink2'])
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel('patients needed to show +0.03'); ax.set_ylabel('patients accessible'); panel(ax, 'b', dx=-.30)
finish(fig, 7, 'gvhd_bench_structure',
       'GvHD-Bench v1.0. (a) The 38 tasks by family and by the landmark at which the prediction is made; '
       'marker shape is data readiness and fill is the powered verdict. Most of the benchmark sits in the '
       'first 100 days, where the decisions are. (b) Accessible patients against the number needed to '
       'demonstrate a +0.03 AUROC gain over the task baseline: points below the diagonal are tasks the field '
       'cannot currently settle, and they are labelled rather than omitted.')
''')


md(r'''
## 8.10 Claim ladder, with gates fixed in advance
| Level | What may be claimed | Requirements (all of them) |
|---|---|---|
| **0** | nothing | — |
| **1** | "more label-efficient than a tuned simple model" | equal or better at full labels, **and** better at k ≤ 100 labelled patients, paired across centres; beats its random-weight twin |
| **2** | "better discrimination out-of-centre" | ΔIPCW-td-AUROC ≥ +0.03 vs the best simple model **and** vs the clinical baseline; bootstrap CI over centres excludes 0; improves in ≥ 70 % of held-out centres; survives BH correction across the task family; beats random-weight twin and no-pretraining |
| **3** | "clinically useful" | Level 2 **plus** calibration slope within 0.8–1.25 and positive net benefit across 10–40 % thresholds in leave-centre-out **and** temporal validation, at n ≥ the Riley minimum, with a prospective silent-mode evaluation and a decision-impact plan |
| **4** | "improves outcomes" | a prospective trial of the model-guided decision. Nothing short of this licenses the word *improves* |

**Pre-registration rule.** Gate thresholds, splits, seeds and the primary task are written to `spec/gate.json` before any model is run, and the file hash is reported with the results. The gate below is the one this specification proposes for the ALLO-FM v1 attempt.
''')

code(r'''
# ----------------------------------------------------------------------------- the pre-stated gate for the ALLO-FM v1 attempt (written to spec/gate.json)
GATE = dict(
  version='ALLO-FM v1',
  primary_task='B1 — grade II–IV aGvHD by day +100 from the day +7 state, leave-centre-out across >= 20 centres',
  primary_metric='IPCW time-dependent AUROC at day 100, competing risks = death and relapse',
  baselines_to_beat=['MAGIC MAP (ST2+REG3a day-7 algorithm) + clinical covariates',
                     'gradient-boosted trees on 20-40 registry variables (tuned, same splits)',
                     'random-weight twin of the backbone', 'no-pretraining (supervised only)'],
  level2_gate=dict(delta_auroc_min=0.03, ci_over_centres_excludes_zero=True, fraction_centres_improved_min=0.70,
                   beats_random_twin=True, beats_no_pretraining=True, bh_across_family=True),
  level3_additional=dict(calibration_slope=[0.8, 1.25], net_benefit_positive_thresholds=[0.10, 0.40],
                         temporal_validation_required=True, min_n='computed per task by riley_n(P, target AUROC, prevalence)'),
  stopping_rule='if Level 1 is not reached on B1 at stage-1 scale, publish the negative result and do not proceed to stage 3',
  frozen=dict(seed=SEED, splits='spec/splits_*.json', label_functions='section 3.6 of this notebook'))
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
def simulate_cohort(n=6000, n_centres=24, seed=3, leak=True):
    r = np.random.default_rng(seed)
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
            + .11 * dri * .3 + .018 * (age - 52) / 10 - .62 * ptcy - .25 * atg + c_off[centre])
    lat = base + r.normal(0, .55, n)                                      # latent alloreactivity
    crp7 = np.exp(np.clip(1.9 + .55 * lat + r.normal(0, .55, n), -2, 6))
    alc7 = np.exp(np.clip(-1.4 - .25 * lat + r.normal(0, .5, n), -5, 3))
    tac7 = np.clip(r.normal(9.5 - .6 * lat, 2.6, n), 1, 22)
    st2 = np.exp(np.clip(3.3 + .62 * lat + r.normal(0, .5, n), 0, 9))
    reg3a = np.exp(np.clip(3.0 + .70 * lat + r.normal(0, .6, n), 0, 9))
    measured = r.random(n) < .55
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
    df = pd.DataFrame(dict(pid=np.arange(n), centre=centre, era=r.integers(2014, 2024, n), age=age, donor_type=donor_type,
                           hla_mm=hla_mm, source=source, fem_to_male=fem_to_male, hctci=hctci, dri=dri, mac=mac,
                           ptcy=ptcy, atg=atg, cd34=cd34, crp7=crp7, alc7=alc7, tac7=tac7,
                           st2=np.where(measured, st2, np.nan), reg3a=np.where(measured, reg3a, np.nan),
                           biomarker_measured=measured.astype(int), time=time, cause=cause))
    if leak: df['steroid_order_d5'] = steroid_d5.astype(int)
    return df

COH = simulate_cohort()
inc100 = aalen_johansen(COH.time.values, COH.cause.values, 100, 1)
print(f'[SIM] cohort: {len(COH)} patients, {COH.centre.nunique()} centres | '
      f'Aalen-Johansen incidence of grade II-IV aGvHD by d+100 = {inc100:.3f}')
print('cause distribution (0 censored, 1 aGvHD, 2 relapse, 3 NRM):', dict(COH.cause.value_counts().sort_index()))
display(md_table(COH.groupby(COH.centre).agg(n=('pid', 'count'), ptcy_rate=('ptcy', 'mean'),
        aGvHD_by_d100=('cause', lambda s: float(((COH.loc[s.index, 'cause'] == 1) & (COH.loc[s.index, 'time'] <= 100)).mean()))).round(3).head(8), index=True))
''')

code(r'''
# ----------------------------------------------------------------------------- the leakage audit, as code (run on the [SIM] cohort for task B1, landmark day +7)
BANNED_PREFIXES = ('steroid', 'ruxolitinib', 'ecp_', 'budesonide', 'biopsy_order', 'infliximab')
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
        if a == a and (a > .9 or a < .1):
            out.append(dict(check=f'single feature with implausible AUROC: {c}', finding=f'{a:.3f}', status='FAIL'))
    return pd.DataFrame(out)

CLIN = ['age', 'hla_mm', 'fem_to_male', 'hctci', 'dri', 'mac', 'ptcy', 'atg', 'cd34']
D7 = ['crp7', 'alc7', 'tac7']
BIO = ['st2', 'reg3a']
T_KNOWN = {**{c: -7 for c in CLIN}, **{c: 7 for c in D7}, 'st2': 9, 'reg3a': 9, 'steroid_order_d5': 5, 'biomarker_measured': 7}
audit_bad = leakage_audit(COH, CLIN + D7 + BIO + ['steroid_order_d5'], landmark=7, t_known_map=T_KNOWN)
print('AUDIT — candidate input set that includes the planted leak and a late-resulting assay:')
display(md_table(audit_bad))
audit_good = leakage_audit(COH, CLIN + D7, landmark=7, t_known_map=T_KNOWN)
print('AUDIT — the cleaned input set actually used for B1 (day +7 landmark):')
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

def boot_delta(a, b, n=4000, seed=0):
    d = np.asarray(a) - np.asarray(b); r = np.random.default_rng(seed)
    bs = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
piv = LCO.pivot_table(index='centre', columns='arm', values='ipcw_auroc')
base = 'clinical baseline (registry covariates)'
delta = pd.DataFrame([dict(arm=a, **dict(zip(['delta_vs_clinical', 'lo', 'hi'], boot_delta(piv[a], piv[base]))),
                           centres_improved=f"{int((piv[a] > piv[base]).sum())}/{len(piv)}")
                      for a in piv.columns if a != base]).round(3)
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
ARM_STYLE = {
 'clinical baseline (registry covariates)':   (PAL['seq'][0], 'clinical'),
 'clinical + MAP-like biomarkers':            (PAL['seq'][1], '+ biomarkers'),
 'timeline model (clinical + day-7 state)':   (PAL['seq'][2], '+ day-7 state'),
 'full (clinical + day-7 + biomarkers)':      (PAL['seq'][3], 'full model'),
 'confounder-only (centre + era)':            (PAL['grey'],   'confounder\nonly'),
 'LEAKING pipeline (adds steroid order d+5)': (PAL['bad'],    'LEAKING\n(excluded)'),
}
order = [a for a in ARM_STYLE if a in piv.columns]
fig = plt.figure(figsize=(FULL, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=.70, wspace=.52)
# (a) per-centre spread
ax = fig.add_subplot(gs[0, :2]); ygrid(ax)
for i, a in enumerate(order):
    c, lab = ARM_STYLE[a]; v = piv[a].dropna().values
    ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.14, .14, len(v)), v, s=7,
               color=c, alpha=.75, lw=0, zorder=3)
    ax.plot([i - .3, i + .3], [v.mean()] * 2, color=PAL['ink'], lw=1.3, zorder=4)
    ax.text(i, 1.005, f'{v.mean():.3f}', ha='center', fontsize=6.0, fontweight='bold', color=PAL['ink'])
ax.axhline(.5, color=PAL['ink2'], ls='--', lw=.6)
ax.set_xticks(range(len(order))); ax.set_xticklabels([ARM_STYLE[a][1] for a in order], fontsize=5.9)
ax.set_ylabel('IPCW td-AUROC at day +100\n(one point per held-out centre)'); ax.set_ylim(.42, 1.04)
panel(ax, 'a', dx=-.085)
# (b) delta vs clinical baseline, bootstrapped over centres
ax = fig.add_subplot(gs[0, 2]); ygrid(ax, 'x')
dd_ = delta.set_index('arm').reindex([a for a in order if a in delta.arm.values])
ypos = np.arange(len(dd_))
for y, (a, r) in zip(ypos, dd_.iterrows()):
    c = ARM_STYLE[a][0]
    ax.plot([r.lo, r.hi], [y, y], color=c, lw=1.6, solid_capstyle='round')
    ax.plot(r.delta_vs_clinical, y, 'o', color=c, ms=4.0)
ax.axvline(0, color=PAL['ink'], lw=.7)
ax.set_yticks(ypos); ax.set_yticklabels([ARM_STYLE[a][1].replace('\n', ' ') for a in dd_.index], fontsize=5.9)
ax.set_xlabel('Δ IPCW td-AUROC vs clinical\n[centre bootstrap 95% CI]'); ax.invert_yaxis(); panel(ax, 'b', dx=-.62)
# (c) calibration against Aalen-Johansen
ax = fig.add_subplot(gs[1, 0]); ygrid(ax)
mx = float(max(cal.predicted.max(), cal.observed_AJ.max())) * 1.12
ax.plot([0, mx], [0, mx], color=PAL['ink2'], ls='--', lw=.7)
ax.plot(cal.predicted, cal.observed_AJ, 'o-', color=PAL['seq'][3], ms=3.6, lw=1.3)
ax.set_xlabel('predicted risk (decile mean)'); ax.set_ylabel('observed Aalen–Johansen CIF')
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.text(.03, .97, f'slope {slope[0]:.2f}\nintercept {slope[1]:+.3f}', transform=ax.transAxes,
        fontsize=6.0, va='top', color=PAL['ink'])
panel(ax, 'c', dx=-.34)
# (d) decision curve
ax = fig.add_subplot(gs[1, 1]); ygrid(ax)
ax.plot(nbc.threshold, nbc.model, color=PAL['seq'][3], lw=1.5)
ax.plot(nbc.threshold, nbc.treat_all, color=PAL['grey'], lw=1.2, ls='--')
ax.axhline(0, color=PAL['ink2'], lw=.7)
dlabel(ax, nbc.threshold.iloc[len(nbc) // 3], nbc.model.iloc[len(nbc) // 3], 'full model', PAL['seq'][3], dy=6, size=6.0)
dlabel(ax, nbc.threshold.iloc[2], nbc.treat_all.iloc[2], 'treat all', PAL['grey'], dy=-7, size=6.0)
ax.text(.97, .06, 'treat none', transform=ax.transAxes, fontsize=6.0, ha='right', color=PAL['ink2'])
ax.set_xlabel('threshold probability'); ax.set_ylabel('net benefit')
ax.set_ylim(min(-.03, nbc.model.min() * 1.1), max(.02, nbc.model.max() * 1.3)); panel(ax, 'd', dx=-.34)
# (e) the leakage canary
ax = fig.add_subplot(gs[1, 2]); ygrid(ax)
lk = ['full (clinical + day-7 + biomarkers)', 'LEAKING pipeline (adds steroid order d+5)']
vals = [summ.loc[a, 'mean_ipcw_auroc'] for a in lk]
ax.bar([0, 1], vals, color=[ARM_STYLE[a][0] for a in lk], width=.56)
for i, v in enumerate(vals): ax.text(i, v + .012, f'{v:.3f}', ha='center', fontsize=6.2, fontweight='bold')
ax.annotate('', xy=(.5, vals[1]), xytext=(.5, vals[0]),
            arrowprops=dict(arrowstyle='<->', color=PAL['bad'], lw=.9))
ax.text(.5, .555, f'+{vals[1]-vals[0]:.3f} from one\nleaking feature', fontsize=5.8,
        color=PAL['bad'], va='bottom', ha='center', fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(['audited\ninput set', 'leak\nincluded'], fontsize=6.0)
ax.set_ylabel('mean IPCW td-AUROC'); ax.set_ylim(.5, 1.10); ax.set_xlim(-.55, 1.55); panel(ax, 'e', dx=-.36)
finish(fig, 8, 'harness_task_B1',
       '[SIM] The GvHD-Bench harness executing task B1 (grade II-IV aGvHD by day +100 from the day +7 state) '
       'on a simulated 24-centre cohort. (a) Leave-centre-out IPCW time-dependent AUROC, one point per '
       'held-out centre; the pooled mean alone would hide this spread. (b) Difference from the clinical '
       'baseline with a centre-level bootstrap interval. (c) Calibration against the Aalen-Johansen '
       'cumulative incidence, which is the competing-risk analogue of a calibration plot. (d) Decision-curve '
       'net benefit against treat-all and treat-none. (e) The leakage canary: adding one planted '
       'treatment-proxy feature inflates the headline number, and the audit flags it before any model is fitted. '
       'Effect sizes are properties of the simulation, not of patients.')
display(md_table(cal.round(3)))
''')

code(r'''
g = lambda a: summ.loc[a, 'mean_ipcw_auroc']
leak_gain = g('LEAKING pipeline (adds steroid order d+5)') - g('full (clinical + day-7 + biomarkers)')
d_full = delta.set_index('arm').loc['full (clinical + day-7 + biomarkers)']
nb_pos = float(nbc.threshold[(nbc.model > nbc.treat_all) & (nbc.model > 0)].min()) if ((nbc.model > nbc.treat_all) & (nbc.model > 0)).any() else float('nan')
display(Markdown(f"""**Result [SIM].** The harness runs. Mean IPCW td-AUROC over held-out centres: clinical baseline **{g('clinical baseline (registry covariates)'):.3f}**, clinical + MAP-like biomarkers **{g('clinical + MAP-like biomarkers'):.3f}**, clinical + day-7 state **{g('timeline model (clinical + day-7 state)'):.3f}**, full **{g('full (clinical + day-7 + biomarkers)'):.3f}** (Δ vs clinical **{d_full.delta_vs_clinical:+.3f}**, centre-bootstrap CI [{d_full.lo:+.3f}, {d_full.hi:+.3f}], improved in {d_full.centres_improved} centres). Confounder-only (centre + era) reaches **{g('confounder-only (centre + era)'):.3f}**, which is the number any real result must be read against.

**The leakage canary worked.** Adding the planted day +5 steroid order lifts the mean to **{g('LEAKING pipeline (adds steroid order d+5)'):.3f}** ({leak_gain:+.3f}), and the audit flagged it twice before any model was fitted — once as a treatment-proxy feature and once as a single feature with an implausible univariate AUROC. **A gain of this size is exactly what a leaking pipeline produces**, which is why the audit runs first and is reported alongside the result.

Calibration out-of-centre has slope {slope[0]:.2f} against the Aalen–Johansen incidence, and the decision curve is above both default strategies from a threshold of about {nb_pos:.2f}. All of these numbers are properties of a cohort we generated; their only purpose is to show that the metrics, splits, audit and gate logic execute and behave as specified."""))
''')
