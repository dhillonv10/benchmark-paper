from part1 import CELLS, md, code

md(r'''
---
# 4 · Data specification

This section is the core of the paper. A foundation model is defined by its data far more than by its architecture. Everything below is machine-readable and is exported to `spec/` at the end of the notebook.

## 4.1 Design rules for every datum

1. **One event schema.** Every measurement is stored as one event:
   `(patient_id, pair_id, centre_id, source, t_rel_day, t_valid, t_known, code_system, code, value, unit, specimen_id, assay_batch, provenance)`.
   *Static* facts (HLA typing, donor sex) are events with `t_rel_day` at the time they were determined.
2. **Day-0 anchor.** `t_rel_day` = days from the first graft infusion (fractional days are allowed). For second transplants a new `pair_id` and day 0 are created, and the earlier transplant becomes history.
3. **Two times per event.** `t_valid` is when the biological state existed (biopsy taken, blood drawn). `t_known` is when the result became available to the care team (pathology signed out, assay run). **A model at landmark L may use an event only if `t_known ≤ L`.** This one rule removes most temporal leakage in HCT data. The classic example: a biopsy performed on day +20 whose GvHD-positive report is signed on day +23 may not be used by a day +21 prediction.
4. **Standards first.** LOINC for laboratory tests, RxNorm/ATC for drugs (doses in mg, mg/kg or mg/m² with the weight/BSA used), ICD-10/ICD-O-3 and WHO-HAEM5 / ICC 2022 for disease, SNOMED CT for findings, HLA alleles as WHO-nomenclature GL strings at ≥2-field resolution, HGVS for variants, and CIBMTR form / EBMT MED-A field identifiers kept as provenance. The storage target is the **OMOP CDM** (with the Oncology extension) plus an HCT-specific extension table for the elements OMOP lacks (GvHD organ stages, chimerism, graft cell doses).
5. **Label-proxy flags.** Some inputs are *consequences* of the outcome being predicted (systemic steroids, a GI biopsy order, budesonide, ruxolitinib, extracorporeal photopheresis). Each element carries a `leakage` flag. The benchmark removes flagged elements for any task whose outcome they could reveal (§4.7).
6. **Tiers of availability.** **A** = registry core (collected for essentially all patients on CIBMTR/EBMT forms). **B** = EHR longitudinal (available in any centre's EHR). **C** = biospecimen assays on a protocolised schedule (MAGIC/BMT CTN-like). **D** = deep research phenotyping (single-cell, repertoire, spatial). The model must work with Tier A alone and *improve* as tiers are added; this is tested in G4.

## 4.2 The data dictionary
Twenty-five domains. *Priority*: **R** = required for ALLO-FM v1; **Rec** = recommended; **Opt** = optional or research. *Leakage*: `proxy` = can reveal the outcome; `lag` = has a material t_known − t_valid delay; `post` = only meaningful after the outcome occurs, so it is a label or a post-onset input.
''')

code(r'''
# ----------------------------------------------------------------------------- THE DATA DICTIONARY (machine-readable; exported to spec/data_dictionary.csv)
# columns: domain, element, specification (encoding / units / categories), timing relative to day 0, standard / coding, tier, priority, leakage, used by (task families)
DD = [
# ---------------- D01 Recipient demographics & anthropometrics
('D01 Recipient baseline', 'Age at HCT', 'years (decimal); paediatric flag <18', 'day 0', 'OMOP person + HCT date', 'A', 'R', '', 'all'),
('D01 Recipient baseline', 'Sex (biological) and gender identity', 'female / male (+ intersex/unknown); gender identity separate', 'baseline', 'OMOP gender concept', 'A', 'R', '', 'all; female→male pairing'),
('D01 Recipient baseline', 'Race, ethnicity, genetic ancestry', 'self-report (OMB categories) + genetic ancestry PCs if genotyped', 'baseline', 'OMOP race/ethnicity; PCs from D20', 'A', 'R', '', 'fairness audit (G3); donor availability'),
('D01 Recipient baseline', 'Weight, height, BSA, BMI', 'kg, cm, m² (Mosteller); actual and adjusted weight used for dosing', 'day −10 and daily inpatient', 'LOINC 29463-7, 8302-2', 'A', 'R', '', 'dose normalisation; fluid overload'),
('D01 Recipient baseline', 'Performance status', 'Karnofsky (adults, 10–100) / Lansky (<16 y)', 'pre-HCT work-up', 'LOINC 89243-0 (Karnofsky)', 'A', 'R', '', 'A, NRM'),
('D01 Recipient baseline', 'Socioeconomic and access proxies', 'area deprivation index, insurance type, distance to centre', 'baseline', 'derived (geocode)', 'B', 'Opt', '', 'fairness; NRM'),
# ---------------- D02 Disease & disease status
('D02 Disease', 'Primary diagnosis', 'AML, ALL (B/T), MDS, MDS/MPN, MPN, CML, CLL, NHL subtypes, HL, MM, AA/SAA, PNH, haemoglobinopathy, PID, IEM …', 'pre-HCT', 'WHO-HAEM5 / ICC 2022; ICD-O-3; CIBMTR 2402', 'A', 'R', '', 'all; relapse competing risk'),
('D02 Disease', 'Disease status at HCT', 'CR1/CR2/≥CR3, PIF, relapse, untreated; MDS blast %; lymphoma CR/PR/chemoresistant', 'day −30 to 0', 'CIBMTR 2400', 'A', 'R', '', 'relapse/NRM; GvL trade-off'),
('D02 Disease', 'Refined Disease Risk Index (DRI)', 'low / intermediate / high / very high', 'derived at HCT', 'Armand 2014 refined DRI †', 'A', 'R', '', 'baseline score for A-family'),
('D02 Disease', 'Cytogenetics & molecular risk', 'karyotype (ISCN); ELN 2022 risk; FLT3-ITD/TKD, NPM1, TP53, ASXL1, RUNX1 … (HGVS, VAF)', 'diagnosis and pre-HCT', 'ISCN; HGVS; ELN 2022', 'A', 'R', '', 'relapse; maintenance therapy'),
('D02 Disease', 'Measurable residual disease pre-HCT', 'flow (% of CD45+, LOD) / NGS-MRD (VAF) / qPCR (NPM1, BCR::ABL1 IS %)', 'day −30 to 0', 'LOINC; ELN-MRD', 'A', 'Rec', '', 'relapse; conditioning choice'),
('D02 Disease', 'Prior therapy lines', 'regimens, number of lines, prior auto-HCT/allo-HCT, prior CAR-T, prior checkpoint inhibitor (raises GvHD risk)', 'history', 'RxNorm; CIBMTR 2400', 'A', 'R', '', 'A; GvHD after PD-1 blockade'),
('D02 Disease', 'Time from diagnosis to HCT', 'days', 'derived', '—', 'A', 'R', '', 'A'),
('D02 Disease', 'Post-HCT maintenance', 'sorafenib/gilteritinib, azacitidine, TKI, lenalidomide; start day, dose', 'day +30 onward', 'RxNorm', 'B', 'Rec', '', 'relapse; GvHD (immunomodulatory maintenance)'),
# ---------------- D03 Comorbidity
('D03 Comorbidity', 'HCT-CI (Sorror) components', '17 comorbidities with weights (arrhythmia, cardiac, IBD, diabetes, CVD, psychiatric, hepatic mild/mod-severe, obesity, infection, rheumatologic, peptic ulcer, renal, pulmonary mod/severe, prior solid tumour, valve)', 'pre-HCT work-up', 'Sorror 2005 †; CIBMTR 2400', 'A', 'R', '', 'NRM; A'),
('D03 Comorbidity', 'Pulmonary function pre-HCT', 'FEV1, FVC (% predicted, GLI-2012), DLCO corrected for Hb (% predicted)', 'day −30', 'LOINC 20150-9 etc.', 'A', 'R', '', 'NRM; BOS baseline (E2)'),
('D03 Comorbidity', 'Cardiac function', 'LVEF % (echo / MUGA)', 'day −30', 'LOINC 10230-1', 'A', 'R', '', 'NRM; cyclophosphamide toxicity'),
('D03 Comorbidity', 'Iron overload', 'ferritin µg/L; liver iron concentration (MRI R2*) if done', 'day −30', 'LOINC 2276-4', 'B', 'Rec', '', 'NRM, SOS/VOD, infection'),
('D03 Comorbidity', 'Frailty / geriatric assessment', 'gait speed, grip strength, IADL, CGA domains (≥60 y)', 'pre-HCT', 'LOINC panels', 'D', 'Opt', '', 'NRM in older recipients'),
# ---------------- D04 Infection serology
('D04 Serology & infection', 'CMV serostatus (D/R)', 'R+/R− × D+/D− (IgG)', 'pre-HCT', 'LOINC 22244-8', 'A', 'R', '', 'CMV tasks; GvHD (CMV↔GvHD interplay)'),
('D04 Serology & infection', 'EBV, HSV, VZV, HBV (HBsAg, anti-HBc), HCV, HIV, toxoplasma serology', 'pos/neg; HBV DNA if anti-HBc+', 'pre-HCT', 'LOINC', 'A', 'R', '', 'infection auxiliary tasks'),
('D04 Serology & infection', 'Pre-HCT colonisation', 'VRE, CRE, ESBL rectal swabs; MRSA', 'admission', 'SNOMED organism + LOINC culture', 'B', 'Rec', '', 'microbiome; antibiotic exposure; NRM'),
('D04 Serology & infection', 'Active / recent infection at HCT', 'invasive fungal disease (EORTC/MSG category), bacteraemia within 30 d', 'day −30 to 0', 'EORTC/MSG; ICD-10', 'A', 'R', '', 'NRM'),
# ---------------- D05 Donor
('D05 Donor', 'Donor type', 'MSD, syngeneic, MUD 8/8, MMUD 7/8, haploidentical related, mismatched related, UCB (single/double)', 'donor selection', 'CIBMTR 2400; EBMT MED-A', 'A', 'R', '', 'all A-family; donor ranking (A6)'),
('D05 Donor', 'Donor age, sex, relationship', 'years; sex; sibling/parent/child/other', 'donor selection', '—', 'A', 'R', '', 'aGvHD/cGvHD (older donor, female→male)'),
('D05 Donor', 'Donor parity / prior pregnancies; transfusions', 'number of pregnancies (allo-immunisation, H-Y)', 'donor work-up', 'NMDP donor form', 'A', 'Rec', '', 'cGvHD; female donor effect'),
('D05 Donor', 'Donor CMV/EBV serostatus, ABO/Rh', 'IgG pos/neg; ABO major/minor/bidirectional mismatch with recipient', 'donor work-up', 'LOINC', 'A', 'R', '', 'CMV; PRCA/haemolysis'),
('D05 Donor', 'Donor-specific anti-HLA antibodies in recipient (DSA)', 'single-antigen bead MFI per specificity; C1q; desensitisation given', 'donor selection (haplo/MMUD/UCB)', 'LOINC; HLA nomenclature', 'A', 'R', '', 'graft failure; donor ranking'),
('D05 Donor', 'Donor G-CSF mobilisation / BM harvest details', 'G-CSF dose, apheresis sessions, plerixafor', 'collection', '—', 'B', 'Opt', '', 'graft composition'),
('D05 Donor', 'Donor clonal haematopoiesis (CHIP)', 'DNMT3A/TET2/ASXL1 … variants with VAF ≥ 2 % (error-corrected NGS of donor sample)', 'banked donor sample', 'HGVS', 'C', 'Rec', '', 'cGvHD and relapse associations (Frick 2019 †; Gibson 2022 †)'),
# ---------------- D06 Histocompatibility
('D06 HLA & immunogenetics', 'HLA typing, recipient and donor', 'HLA-A, -B, -C, -DRB1, -DRB3/4/5, -DQA1, -DQB1, -DPA1, -DPB1; ≥2-field (high-resolution) by NGS; GL string', 'donor selection', 'WHO HLA nomenclature; GL string; HML', 'A', 'R', '', 'all; pair encoder input'),
('D06 HLA & immunogenetics', 'Allele vs antigen mismatch, direction', 'per locus: match / allele / antigen mismatch; GvH-vector vs HvG-vector (homozygosity)', 'derived', 'derived from typing', 'A', 'R', '', 'aGvHD (GvH vector) vs graft failure (HvG)'),
('D06 HLA & immunogenetics', 'HLA-DPB1 T-cell-epitope (TCE) group', 'permissive / non-permissive (GvH / HvG)', 'derived', 'TCE3/TCE4 algorithm †', 'A', 'R', '', 'aGvHD, relapse (Fleischhauer 2012 †)'),
('D06 HLA & immunogenetics', 'HLA-B leader (−21 M/T) and HLA-C expression (rs9264942 / 3′UTR)', 'MM/MT/TT; leader matched vs mismatched; HLA-C expression level', 'derived', 'Petersdorf 2015 †, 2020 †', 'A', 'Rec', '', 'aGvHD, especially in haplo/MMUD'),
('D06 HLA & immunogenetics', 'HLA evolutionary divergence (HED) and mismatched-epitope load', 'Grantham-distance HED per locus; PIRCHE-II / HLAMatchmaker eplet counts', 'derived', 'published algorithms †', 'A', 'Opt', '', 'GvL vs GvHD; pair encoder features'),
('D06 HLA & immunogenetics', 'Minor histocompatibility antigens', 'H-Y (female→male); HA-1, HA-2, ACC-1 … disparities from germline genotypes; predicted mHA load (genome-wide)', 'derived (D20)', 'dbMinor / published lists †', 'C', 'Rec', '', 'aGvHD in HLA-matched; GvL'),
('D06 HLA & immunogenetics', 'KIR genotype and KIR-ligand mismatch', 'donor KIR A/B haplotype, B-content score, centromeric/telomeric; KIR-ligand mismatch (missing self)', 'donor work-up', 'IPD-KIR; KIR B-content †', 'C', 'Rec', '', 'relapse (AML), NK alloreactivity'),
('D06 HLA & immunogenetics', 'Non-HLA germline immune variants', 'NOD2/CARD15, IL10, TNF, IL23R, CTLA4 … (candidate); or polygenic score', 'derived (D20)', 'HGVS / rsID', 'D', 'Opt', '', 'exploratory'),
# ---------------- D07 Graft
('D07 Graft', 'Stem cell source', 'PBSC / BM / UCB (single, double); fresh vs cryopreserved', 'day 0', 'CIBMTR 2006 †', 'A', 'R', '', 'cGvHD (PBSC), engraftment'),
('D07 Graft', 'Cell doses infused', 'TNC ×10⁸/kg; CD34⁺ ×10⁶/kg; CD3⁺ ×10⁸/kg (×10⁷/kg UCB); viability %', 'day 0', 'ISCT/ISHAGE CD34 enumeration', 'A', 'R', '', 'engraftment; aGvHD/cGvHD (CD3 dose)'),
('D07 Graft', 'Graft T-cell composition (flow on graft aliquot)', 'CD4, CD8, CD4:CD8, naive (CD45RA⁺CCR7⁺), Tscm (CD45RA⁺CCR7⁺CD95⁺), Tcm, Tem, TEMRA, Treg (CD4⁺CD25^hi CD127^lo FOXP3⁺), γδ, iNKT (Vα24-Jα18), MAIT (Vα7.2⁺CD161^hi), NK (CD56^bright/dim), B, pDC, monocytes; absolute ×10⁶/kg', 'day 0', 'EuroFlow-style panels †; FCS files + gating', 'C', 'Rec', '', 'aGvHD (Treg:Tcon, naive dose), graft encoder'),
('D07 Graft', 'Ex vivo graft manipulation', 'none; CD34 selection; αβ-TCR/CD19 depletion; CD45RA (naive) depletion; Treg/Tcon add-back (e.g. Orca-T); residual αβ T cells/kg', 'day 0', 'free text → controlled list', 'A', 'R', '', 'aGvHD (strong effect modifier)'),
('D07 Graft', 'Graft handling', 'collection-to-infusion hours; cryopreservation; ABO processing (RBC/plasma depletion); DMSO volume', 'day 0', '—', 'B', 'Opt', '', 'engraftment; infusion reactions'),
# ---------------- D08 Conditioning
('D08 Conditioning', 'Regimen (agent-level, not just name)', 'per agent: drug, total dose (busulfan mg/kg and target exposure; cyclophosphamide mg/kg; fludarabine mg/m²; melphalan mg/m²; treosulfan g/m²; thiotepa mg/kg; carmustine/etoposide/cytarabine for BEAM; clofarabine) and the day of each dose', 'day −10 to −1', 'RxNorm + dose + day; CIBMTR 2400', 'A', 'R', '', 'all; conditioning-injury pathway'),
('D08 Conditioning', 'Total body irradiation / TMLI', 'dose (Gy), fractions, dose rate (cGy/min), lung/kidney shielding; TMLI target doses', 'day −7 to −1', 'CIBMTR 2400', 'A', 'R', '', 'aGvHD (GI injury), IPS, NRM'),
('D08 Conditioning', 'Busulfan pharmacokinetics', 'measured cumulative AUC (mg·h/L or µM·min) and target; test-dose PK; dose adjustments', 'day −6 to −2', 'LOINC (busulfan level)', 'B', 'Rec', '', 'SOS/VOD, NRM, relapse (exposure-response †)'),
('D08 Conditioning', 'Intensity classification', 'MAC / RIC / NMA (CIBMTR operational definitions) and EBMT Transplant Conditioning Intensity (TCI) score', 'derived', 'Bacigalupo 2009 †; Spyridonidis 2020 †', 'A', 'R', '', 'A-family baseline'),
('D08 Conditioning', 'Conditioning toxicity', 'mucositis (WHO grade 0–4 daily), SOS/VOD (EBMT 2023 criteria †), engraftment syndrome, CRS after haplo (ASTCT grade)', 'day −7 to +30', 'CTCAE v5; EBMT SOS', 'B', 'R', '', 'GI barrier injury → aGvHD; differential diagnosis (C1)'),
# ---------------- D09 Serotherapy
('D09 Serotherapy', 'Anti-thymocyte globulin', 'product (rabbit thymoglobulin vs ATG-Grafalon/Fresenius — not interchangeable), total mg/kg, days given', 'day −5 to −1', 'RxNorm', 'A', 'R', '', 'aGvHD/cGvHD, immune reconstitution'),
('D09 Serotherapy', 'ATG exposure covariates', 'absolute lymphocyte count before first dose (×10⁹/L); active-ATG levels if measured; modelled post-HCT exposure', 'day −5; day 0 to +30', 'LOINC; Admiraal 2015 PK model †', 'C', 'Rec', '', 'reconstitution, GvHD (exposure-response)'),
('D09 Serotherapy', 'Alemtuzumab', 'total mg, schedule, in vivo vs "in the bag"', 'day −8 to −1', 'RxNorm', 'A', 'R', '', 'aGvHD, viral reactivation'),
# ---------------- D10 Prophylaxis
('D10 GvHD prophylaxis', 'Prophylaxis platform', 'PtCy/Tac/MMF; Tac/MTX; CsA/MTX; Tac/sirolimus; CsA/MMF; ATG-based; abatacept add-on; vedolizumab; ruxolitinib (trial); ex vivo TCD (no pharmacological)', 'day −3 to +180', 'controlled list; CIBMTR 2400', 'A', 'R', '', 'all; treatment-effect tasks (A7)'),
('D10 GvHD prophylaxis', 'Post-transplant cyclophosphamide', 'dose (50 or 25 mg/kg/day), days (+3/+4 standard; alternative schedules), mesna', 'day +3 to +5', 'RxNorm + day', 'A', 'R', '', 'A7; infection; cardiac toxicity'),
('D10 GvHD prophylaxis', 'Methotrexate', 'dose per day (15 mg/m² d+1; 10 mg/m² d+3, +6, +11 or "mini" 5 mg/m²), doses omitted, leucovorin rescue', 'day +1 to +11', 'RxNorm + day', 'A', 'R', '', 'mucositis; engraftment; A7'),
('D10 GvHD prophylaxis', 'Calcineurin inhibitor exposure', 'tacrolimus / cyclosporine daily dose (route) and **every trough level** (ng/mL), target range, time in therapeutic range; CYP3A5 genotype (D20)', 'day −3 to taper end (daily → weekly)', 'LOINC 35152-8 (tacrolimus) †; RxNorm', 'B', 'R', '', 'aGvHD (low exposure), TMA/renal toxicity; PK auxiliary objective'),
('D10 GvHD prophylaxis', 'MMF / sirolimus exposure', 'MMF mg/day (and MPA AUC if measured); sirolimus troughs (ng/mL)', 'day +5 to +35 (MMF); to taper (siro)', 'RxNorm; LOINC', 'B', 'R', '', 'A7; TMA (siro + CNI)'),
('D10 GvHD prophylaxis', 'Immunosuppression taper schedule', 'planned vs actual taper start/end days; reason for early taper (relapse, MRD) or for continuing', 'day +60 to +365', 'derived', 'B', 'R', 'proxy', 'cGvHD (E1); GvL; *taper reason can leak relapse*'),
# ---------------- D11 Supportive care
('D11 Supportive care & antimicrobials', 'Antibacterial exposure', 'agent, start/stop day; flag anaerobe-active broad-spectrum (piperacillin-tazobactam, carbapenems), associated with dysbiosis and GvHD-related mortality †', 'day −10 to +100', 'RxNorm/ATC J01', 'B', 'R', '', 'microbiome pathway; aGvHD'),
('D11 Supportive care & antimicrobials', 'Antiviral / antifungal prophylaxis', 'letermovir (CMV), acyclovir, posaconazole/voriconazole (CNI interaction), TMP-SMX', 'day 0 to +100+', 'RxNorm', 'B', 'R', '', 'CMV task; CNI level variability'),
('D11 Supportive care & antimicrobials', 'Nutrition', 'parenteral nutrition days, enteral feeds, oral intake kcal/day, albumin', 'daily inpatient', 'LOINC; flowsheet', 'B', 'Rec', '', 'GI GvHD; NRM'),
('D11 Supportive care & antimicrobials', 'Growth factors and transfusions', 'G-CSF start day; RBC / platelet units per day', 'daily', 'RxNorm; ISBT 128', 'B', 'Rec', '', 'engraftment definitions'),
('D11 Supportive care & antimicrobials', 'Other GI-active drugs', 'PPIs, laxatives, opioids, magnesium supplements (diarrhoea confounders)', 'daily', 'RxNorm', 'B', 'Rec', '', 'C1 differential diagnosis'),
# ---------------- D12 Labs
('D12 Laboratory timeline', 'Complete blood count with differential', 'WBC, ANC, **ALC**, monocytes, **eosinophils**, Hb, platelets (×10⁹/L)', 'daily inpatient; ≥ weekly to d+100; monthly to 1 y', 'LOINC 751-8 (ANC), 731-0 (ALC), 711-2 (eos) …', 'B', 'R', '', 'engraftment; ALC recovery; eosinophilia in cGvHD'),
('D12 Laboratory timeline', 'Liver panel', 'total and conjugated bilirubin (mg/dL — **MAGIC liver staging input**), ALT, AST, ALP, GGT', 'daily inpatient; ≥ weekly to d+100', 'LOINC 1975-2, 1968-7 …', 'B', 'R', '', 'liver aGvHD staging; SOS/VOD; drug toxicity'),
('D12 Laboratory timeline', 'Renal and electrolytes', 'creatinine (eGFR CKD-EPI 2021), urea, K, Mg (CNI), phosphate', 'daily inpatient', 'LOINC 2160-0 …', 'B', 'R', '', 'CNI toxicity; TMA'),
('D12 Laboratory timeline', 'Inflammation / tissue injury', 'CRP, ferritin, LDH, albumin, triglycerides, IL-6 (if available), procalcitonin', '≥ twice weekly to d+30', 'LOINC', 'B', 'R', '', 'early warning (B3); EASIX (LDH×creatinine/platelets) baseline †'),
('D12 Laboratory timeline', 'TMA markers', 'haptoglobin, schistocytes, LDH, urine protein:creatinine, sC5b-9', 'when triggered', 'LOINC', 'B', 'Rec', 'proxy', 'TA-TMA (GvHD-associated); triggered testing is a proxy'),
('D12 Laboratory timeline', 'Immunoglobulins', 'IgG, IgA, IgM (g/L); IVIG given', 'monthly to 1 y', 'LOINC', 'B', 'Rec', '', 'B-cell reconstitution; cGvHD'),
# ---------------- D13 Staging inputs
('D13 GvHD staging inputs', 'Skin rash body-surface area', '% BSA of maculopapular rash (rule of nines / Lund-Browder), erythroderma, bullae and desquamation % BSA', 'daily inpatient; each outpatient visit', 'MAGIC 2016 ✓ (staging rules); flowsheet', 'B', 'R', 'post', 'skin stage → grade (label construction)'),
('D13 GvHD staging inputs', 'Stool output', 'mL/day (adults); mL/kg/day (paediatric); episodes/day when not measured; bloody stool; abdominal pain / ileus', 'daily inpatient', 'MAGIC 2016 ✓; flowsheet I/O', 'B', 'R', 'post', 'lower-GI stage; C1'),
('D13 GvHD staging inputs', 'Upper GI symptoms', 'persistent nausea, vomiting or anorexia (yes/no, days)', 'daily', 'MAGIC 2016', 'B', 'R', 'post', 'upper-GI stage'),
('D13 GvHD staging inputs', 'Vitals', 'temperature (fever curves), heart rate, BP, SpO₂, daily weight (fluid balance)', 'q4–8h inpatient → daily summary', 'LOINC 8310-5 …', 'B', 'R', '', 'early warning; CRS / engraftment syndrome'),
('D13 GvHD staging inputs', 'Stool infectious work-up', 'C. difficile toxin/PCR, GI multiplex PCR (norovirus, adenovirus …), CMV in tissue', 'when diarrhoea', 'LOINC; SNOMED organism', 'B', 'R', 'proxy', 'C1 (negative work-up enriches for GvHD)'),
('D13 GvHD staging inputs', 'Endoscopy findings', 'site, mucosal appearance (oedema, erythema, ulceration, denudation), biopsy sites', 'at work-up', 'SNOMED; structured report', 'B', 'Rec', 'post', 'C1, C7'),
# ---------------- D14 Engraftment & immune reconstitution
('D14 Engraftment & T-cell reconstitution', 'Neutrophil and platelet engraftment', 'ANC ≥ 0.5×10⁹/L on 3 consecutive days (day of first); platelets ≥ 20×10⁹/L unsupported for 7 days', 'derived', 'CIBMTR definitions', 'A', 'R', '', 'landmark eligibility; graft failure'),
('D14 Engraftment & T-cell reconstitution', 'Lineage-specific donor chimerism', '% donor in CD3⁺ (T) and CD33⁺/CD15⁺ (myeloid) fractions; whole blood; marrow; method (STR, qPCR-indel, NGS) and LOD', 'day +30, +60, +100, +180, +365', 'LOINC; method code', 'A', 'R', '', 'graft failure; relapse; mixed T chimerism ↔ less GvHD'),
('D14 Engraftment & T-cell reconstitution', 'T-cell subset counts (flow)', 'cells/µL: CD3, CD4, CD8, CD4 naive (CD45RA⁺CCR7⁺ / CD31⁺ recent thymic emigrants), Tcm, Tem, TEMRA, Tscm; Treg (CD4⁺CD25^hi CD127^lo, FOXP3⁺) and Treg:Tcon ratio', 'day +14, +30, +60, +100, +180, +365', 'FCS 3.1 + gating ML; harmonised panel', 'C', 'R', '', 'B-family, E-family; reconstitution objective'),
('D14 Engraftment & T-cell reconstitution', 'Activated and homing T cells', 'CD38⁺HLA-DR⁺ CD8; Ki-67⁺; gut-homing α4β7⁺ (± CCR9⁺); skin-homing CLA⁺ CCR4⁺ CCR10⁺; Th1 (CXCR3⁺), Th17 (CCR6⁺ CD161⁺), Th22; circulating Tfh (CXCR5⁺ PD-1⁺)', 'day +7, +14, +21, +28; at onset', 'harmonised flow / CyTOF panel', 'C', 'Rec', '', 'B1–B3 early warning; organ involvement (gut vs skin)'),
('D14 Engraftment & T-cell reconstitution', 'Innate-like and NK/B compartments', 'MAIT, iNKT, Vδ2 γδ T; NK CD56^bright/dim, NKG2A, KIR; B cells: transitional, CD21^lo, plasmablasts; BAFF:B-cell ratio', 'day +30 to +365', 'flow; LOINC (BAFF)', 'C', 'Rec', '', 'cGvHD (B-cell axis), relapse (NK)'),
('D14 Engraftment & T-cell reconstitution', 'Virus-specific T cells', 'CMV/EBV/ADV-specific CD8 (multimer or IFN-γ ELISpot/QuantiFERON-CMV)', 'day +30 to +180', 'LOINC', 'C', 'Opt', '', 'CMV task; alloreactive vs virus-reactive clone annotation'),
# ---------------- D15 Protein biomarkers
('D15 Circulating protein biomarkers', 'MAGIC panel', 'ST2 (sIL1RL1), REG3α (ng/mL) — **MAGIC algorithm probability (MAP)**; TNFR1, IL-2Rα, elafin, HGF, IL-8 (Ann Arbor / earlier panels)', 'day 0, +7, +14, +28; at onset; day +7 after treatment start', 'ELISA, single lab (assay drift!); MAGIC SOP', 'C', 'R', '', 'B1 (MAP baseline), C4/C6; NRM'),
('D15 Circulating protein biomarkers', 'Endothelial / cGvHD markers', 'Ang-2, sTNFR, CXCL9, CXCL10, MMP3, DKK3, OPN, CD163, BAFF, soluble CD13, ST2 at day +100', 'day +100; at cGvHD onset', 'ELISA / Luminex', 'C', 'Rec', '', 'E1 (cGvHD), BIOPREVENT-type baseline †'),
('D15 Circulating protein biomarkers', 'High-plex proteomics', 'Olink Explore (≈3,000) or SomaScan (≈7,000) NPX/RFU; plate and batch IDs', 'day 0, +7, +28, +100', 'UniProt IDs; vendor panels', 'D', 'Opt', 'lag', 'adapter pretraining (masked protein modelling)'),
('D15 Circulating protein biomarkers', 'Pre-analytics', 'tube (EDTA / serum), time to freeze (h), freeze–thaw count, haemolysis index', 'each sample', 'ISBER / BRISQ', 'C', 'R', '', 'confounder-only model (mandatory comparator)'),
# ---------------- D16 Repertoire
('D16 Immune repertoire', 'TCRβ (± TCRα) bulk repertoire', 'productive CDR3 amino-acid + nucleotide, TRBV/TRBJ, template count; ≥ 10⁵ templates per sample; sequencing depth', 'donor graft or pre-donation blood; recipient day +30, +60, +100, at onset; affected tissue', 'AIRR-C standard (MiAIRR) ✓†', 'D', 'Rec', '', 'TCR adapter (O-M2); clonal expansion tasks'),
('D16 Immune repertoire', 'Paired single-cell TCR + transcriptome', '10x 5′ GEX + VDJ (+ CITE-seq ADT); clonotype → cell state', 'day +30, at onset; tissue at biopsy', 'AIRR; CELLxGENE schema', 'D', 'Opt', '', 'O-C2 TCR↔state alignment'),
('D16 Immune repertoire', 'Donor-reactive clone reference', 'pre-transplant MLR: donor T cells × irradiated recipient cells; CFSE-low or CD137⁺ sort; TCR-seq of reactive fraction (sets the alloreactive "dictionary")', 'pre-HCT (donor + recipient samples)', 'AIRR', 'D', 'Opt', '', 'label for clone-level tasks; mechanism'),
('D16 Immune repertoire', 'Repertoire summary metrics', 'clonality (1 − Pielou), Shannon, Simpson, Gini, top-100 clone fraction, richness (Chao1), CMV/EBV-annotated fraction (VDJdb)', 'derived', 'AIRR / VDJdb', 'D', 'Rec', '', 'baseline features for repertoire tasks'),
('D16 Immune repertoire', 'BCR / IGH repertoire', 'IGH CDR3, isotype, SHM rate', 'day +100, at cGvHD onset', 'AIRR', 'D', 'Opt', '', 'cGvHD (B-cell pathology)'),
# ---------------- D17 Transcriptomics
('D17 Transcriptomics', 'Sorted T-cell bulk RNA-seq', 'FACS CD4 Tconv / CD8 / Treg (± MAIT, γδ); raw counts (not FPKM); RIN, sort purity, cell number', 'day +14, +28, at onset', 'GENCODE gene IDs; raw FASTQ in controlled repository', 'D', 'Rec', '', 'T-cell state adapter; pilot tasks (§10)'),
('D17 Transcriptomics', 'PBMC scRNA-seq / CITE-seq', 'UMI counts, ADT counts, cell-hashing; ≥ 5,000 cells per sample; doublet / ambient metrics', 'day +30, at onset; donor graft aliquot', 'CELLxGENE schema', 'D', 'Opt', '', 'cell-level adapter (single-cell FM, fine-tuned)'),
('D17 Transcriptomics', 'Tissue biopsy RNA-seq', 'colon/rectosigmoid, upper GI, skin, liver; fresh-frozen or FFPE (DV200); raw counts', 'at biopsy (t_known = run date)', 'GENCODE', 'C', 'Rec', 'lag', 'C7 tissue diagnosis (pilot: transfers, LOSO ≈ 0.80)'),
('D17 Transcriptomics', 'Spatial transcriptomics', 'Visium HD / Xenium / CosMx on biopsy FFPE: spot or cell × gene; registered to the H&E slide', 'at biopsy', 'vendor + OME-TIFF', 'D', 'Opt', 'lag', 'crypt / epidermal niche mechanisms'),
('D17 Transcriptomics', 'Whole-blood / plasma cfRNA', 'PAXgene whole-blood RNA-seq; plasma cfRNA (log CPM); library-size and batch recorded', 'day 0, +7, +14, +28, monthly', 'GENCODE', 'C', 'Opt', '', 'B-family (pilot: no pre-onset signal at n = 92)'),
('D17 Transcriptomics', 'Assay metadata', 'platform (array vs RNA-seq), library kit, read length, depth, batch, run date', 'each sample', 'MINSEQE', 'C', 'R', '', 'confounder-only comparator; leave-platform-out'),
# ---------------- D18 Microbiome & metabolome
('D18 Microbiome & metabolome', 'Stool microbiome', '16S rRNA V4 (ASVs) and/or shotgun metagenomics (species, strains, ARGs); relative and (qPCR-scaled) absolute abundance', 'day −7 to 0; weekly to +35; at GI onset; +100', 'NCBI taxonomy; MIxS', 'C', 'Rec', '', 'microbiome adapter; aGvHD/NRM (Peled 2020 ✓)'),
('D18 Microbiome & metabolome', 'Derived ecology', 'inverse Simpson diversity at engraftment; domination (single taxon ≥ 30 %); Enterococcus / Enterobacterales domination; butyrate-producer (Lachnospiraceae, Blautia) abundance', 'derived', '—', 'C', 'Rec', '', 'B-family; interpretable baseline'),
('D18 Microbiome & metabolome', 'Microbial metabolites', 'faecal SCFAs (butyrate, propionate, acetate; µmol/g), secondary bile acids, indole-3-carboxaldehyde; urinary 3-indoxyl sulfate', 'day −7, +7, +14, +21', 'LC-MS; HMDB IDs', 'D', 'Opt', 'lag', 'mechanism; metabolite adapter'),
('D18 Microbiome & metabolome', 'Oral / skin microbiome', 'oral rinse 16S; skin swabs', 'weekly to +28', 'MIxS', 'D', 'Opt', '', 'mucositis; oral cGvHD'),
# ---------------- D19 Pathology & imaging
('D19 Pathology & imaging', 'GI / skin / liver biopsy histology', 'whole-slide images (H&E, 40×, SVS/OME-TIFF); structured report: crypt apoptosis count per 10 crypts, crypt loss, Lerner/Sale grade, CMV IHC, eosinophils', 'biopsy (t_valid) → sign-out (t_known)', 'DICOM WSI; SNOMED', 'C', 'Rec', 'lag', 'C1, C7; pathology adapter (UNI / Virchow / CONCH-class FMs †)'),
('D19 Pathology & imaging', 'Immunohistochemistry / multiplex IF', 'CD3, CD8, FOXP3, Ki-67, Paneth-cell lysozyme, REG3α', 'biopsy', 'OME-TIFF', 'D', 'Opt', 'lag', 'tissue mechanism'),
('D19 Pathology & imaging', 'Chest CT and PFT trajectory', 'inspiratory/expiratory CT (air trapping; parametric response mapping, PRM^fSAD %); FEV1, FVC, FEV1/FVC, RV, DLCO serially (home spirometry if available)', 'day −30, +100, +180, +365, then 6-monthly', 'DICOM; LOINC', 'B', 'Rec', '', 'E2 bronchiolitis obliterans (BOS) early detection'),
('D19 Pathology & imaging', 'Clinical photographs', 'standardised skin photographs (rash, sclerosis), oral mucosa', 'at visits', 'DICOM VL photography', 'D', 'Opt', 'post', 'skin cGvHD scoring (adapter)'),
('D19 Pathology & imaging', 'Ophthalmic and oral tests', 'Schirmer test (mm/5 min), OSDI; oral mucosal score', 'day +100, then 6-monthly', 'LOINC', 'B', 'Rec', 'post', 'NIH eye / mouth score (label construction)'),
# ---------------- D20 Genomics
('D20 Germline & somatic genomics', 'Recipient and donor germline genotype', 'SNP array (imputed) or WGS ≥ 30×; used for mHA disparity, HLA imputation QC, ancestry, PGx', 'banked pre-HCT samples (NMDP repository)', 'VCF (GRCh38)', 'D', 'Rec', '', 'D06 mHA load; pair encoder'),
('D20 Germline & somatic genomics', 'Pharmacogenomics', 'CYP3A5 *1/*3 (tacrolimus), CYP2C19 (voriconazole), TPMT/NUDT15, GSTA1 (busulfan)', 'pre-HCT', 'PharmVar star alleles', 'C', 'Rec', '', 'PK auxiliary objective; CNI exposure'),
('D20 Germline & somatic genomics', 'Post-HCT disease genomics', 'NGS-MRD at day +30/+100; HLA loss of heterozygosity at relapse', 'day +30, +100; at relapse', 'HGVS', 'C', 'Rec', 'post', 'relapse competing risk; HLA-loss relapse'),
# ---------------- D21 Text
('D21 Clinical text', 'Notes', 'admission H&P, daily progress notes, transplant-physician GvHD assessments, discharge summaries, consult notes (GI, dermatology)', 'continuous', 'note type (LOINC document ontology)', 'B', 'Rec', 'proxy', 'text adapter (clinical LLM); weak labels; *notes mention suspected GvHD before the formal diagnosis*'),
('D21 Clinical text', 'Pathology and radiology reports', 'free text + structured synoptic fields', 'at sign-out', 'LOINC document ontology', 'B', 'Rec', 'lag', 'label extraction (onset date, biopsy confirmation)'),
# ---------------- D22 PROs
('D22 Patient-reported & function', 'cGvHD symptom burden', 'Lee cGvHD Symptom Scale (30 items, 7 subscales); PROMIS-29; FACT-BMT', 'day +100, then 3-monthly', 'LOINC PRO panels', 'B', 'Rec', 'post', 'cGvHD response (E3) secondary endpoint'),
('D22 Patient-reported & function', 'Wearables', 'step count, resting heart rate, sleep (consumer devices)', 'continuous (opt-in)', 'Open mHealth', 'D', 'Opt', '', 'early warning (exploratory)'),
# ---------------- D23 Outcome labels
('D23 Outcome labels', 'aGvHD organ stages (daily) and overall grade', 'skin 0–4, liver 0–4, upper GI 0–1, lower GI 0–4 → MAGIC grade 0–IV (function in §4.6); onset date = first day of grade ≥ I (and ≥ II)', 'daily d0–d+180 (late aGvHD beyond)', 'MAGIC 2016 ✓', 'A', 'R', 'post', 'labels for A, B, C families'),
('D23 Outcome labels', 'Biopsy confirmation & adjudication', 'biopsy-proven / clinically diagnosed / biopsy-negative; central adjudication flag', 'onset ± 7 d', 'MAGIC; trial CRF', 'B', 'R', 'post', 'C1/C2 diagnostic labels'),
('D23 Outcome labels', 'Treatment response (aGvHD)', 'CR / VGPR / PR / mixed / NR / progression at day +7, +14, +28, +56 from start of systemic steroids', 'from steroid start', 'MAGIC / CIBMTR response criteria', 'A', 'R', 'post', 'C4, C5, D1'),
('D23 Outcome labels', 'Steroid-refractory / -dependent / -intolerant', '2018 EBMT–NIH–CIBMTR definitions (function in §4.6)', 'from steroid start', 'Schoemans 2018 †', 'A', 'R', 'post', 'C4 label; D-family eligibility'),
('D23 Outcome labels', 'cGvHD (NIH 2014)', 'diagnostic/distinctive signs per organ; organ scores 0–3 (skin, mouth, eyes, GI, liver, lungs, joints/fascia, genital); global mild / moderate / severe; onset date; overlap syndrome flag', 'from day +60 onward, each visit', 'Jagasia 2015 (NIH 2014) †', 'A', 'R', 'post', 'E-family labels'),
('D23 Outcome labels', 'cGvHD response (NIH 2014)', 'CR / PR / unchanged / progression (organ-specific + global) at 3, 6, 12 months of therapy', 'from therapy start', 'Lee 2015 NIH response †', 'A', 'R', 'post', 'E3'),
('D23 Outcome labels', 'Relapse / progression', 'date; morphological / molecular / extramedullary; HLA-loss flag', 'continuous', 'CIBMTR 2100/2200 †', 'A', 'R', 'post', 'competing event in every GvHD task'),
('D23 Outcome labels', 'Death and cause of death', 'date; primary cause (GvHD, infection, relapse, organ failure, SOS, TMA …) by CIBMTR hierarchy', 'continuous', 'CIBMTR cause-of-death algorithm', 'A', 'R', 'post', 'NRM, OS, GRFS'),
('D23 Outcome labels', 'Composite endpoints', 'GRFS (grade III–IV aGvHD, cGvHD requiring systemic therapy, relapse, death; Holtan 2015 †); CRFS; NRM; OS; immunosuppression-free survival', 'derived', 'function in §4.6', 'A', 'R', 'post', 'A4, E4'),
('D23 Outcome labels', 'Last follow-up / censoring', 'date of last contact; reason (administrative, lost, transferred)', 'continuous', '—', 'A', 'R', '', 'IPCW censoring model (§8.4)'),
# ---------------- D24 GvHD treatment
('D24 GvHD treatment exposures', 'Systemic corticosteroids', 'agent, mg/kg/day **prednisone-equivalent**, start day, taper; topical / non-absorbable (budesonide, beclomethasone)', 'from first use', 'RxNorm; conversion table', 'B', 'R', 'proxy', '**label proxy for aGvHD onset**: banned as input before onset tasks; defines C-family landmark'),
('D24 GvHD treatment exposures', 'Second-line and later agents', 'ruxolitinib (dose), ECP (sessions), vedolizumab, α1-antitrypsin, etanercept, infliximab, MSC, FMT; cGvHD: belumosudil, ruxolitinib, axatilimab, ibrutinib, sirolimus, imatinib, rituximab', 'from start', 'RxNorm', 'B', 'R', 'proxy', 'D-family exposure; E3; banned before outcome'),
('D24 GvHD treatment exposures', 'Clinical-trial enrolment', 'trial ID (NCT), arm (blinded where applicable)', 'enrolment day', 'ClinicalTrials.gov NCT', 'A', 'R', '', 'randomised anchor for treatment-effect validation (A7)'),
# ---------------- D25 Provenance
('D25 Provenance & operations', 'Centre, country, registry', 'centre ID (pseudonymised), country, CIBMTR/EBMT/JSHCT, centre volume (allo-HCT/year), JACIE/FACT accreditation', 'static', '—', 'A', 'R', '', 'leave-centre-out splits; centre-adversarial objective'),
('D25 Provenance & operations', 'Transplant era', 'calendar year of HCT; era flags: PtCy beyond haplo (≈2019+ †), letermovir (≈2017+), ruxolitinib for SR-aGvHD (2019 FDA)', 'static', '—', 'A', 'R', '', 'temporal split (G1)'),
('D25 Provenance & operations', 'Assay batch & laboratory', 'batch ID, run date, operator, reagent lot, instrument; central vs local lab', 'each sample', '—', 'C', 'R', '', 'batch-only comparator; harmonisation objective'),
('D25 Provenance & operations', 'Missingness indicators', 'per element: observed / not-done / not-applicable / unknown, plus the reason where recorded', 'continuous', '—', 'A', 'R', '', 'missingness is informative; masked-value objective must not be fooled by it'),
('D25 Provenance & operations', 'Consent and data-use terms', 'consent scope (research use, re-identification, international transfer), DUA ID, embargo', 'static', 'GA4GH DUO codes', 'A', 'R', '', 'governance (§11.3)'),
]
DD_COLS = ['domain', 'element', 'specification (encoding / units / categories)', 'timing relative to day 0',
           'standard / coding', 'tier', 'priority', 'leakage', 'used by']
dd = pd.DataFrame(DD, columns=DD_COLS)
dd.to_csv(os.path.join(SPEC_DIR, 'data_dictionary.csv'), index=False)
summary = (dd.assign(n=1).groupby('domain')
             .agg(elements=('n', 'sum'), required=('priority', lambda s: int((s == 'R').sum())),
                  tiers=('tier', lambda s: '/'.join(sorted(set(s)))),
                  leakage_flags=('leakage', lambda s: int((s != '').sum()))))
print(f'{len(dd)} elements in {dd.domain.nunique()} domains | '
      f"required for v1: {(dd.priority == 'R').sum()} | leakage-flagged: {(dd.leakage != '').sum()}")
print(f"tier counts: {dict(dd.tier.value_counts().sort_index())}")
display(summary)
''')

md(r'''
The full dictionary is rendered below, grouped by domain. It is also written to `spec/data_dictionary.csv` so it can be used as a CRF checklist or as the schema for an extraction pipeline.
''')

code(r'''
# ----------------------------------------------------------------------------- render the dictionary, domain by domain
for dom in dd.domain.unique():
    display(Markdown(f'#### {dom}'))
    display(md_table(dd[dd.domain == dom].drop(columns=['domain']), max_w=240))
''')

md(r'''
## 4.3 Why these elements and not others — the mechanistic skeleton

The dictionary is not a shopping list. It follows the accepted three-step model of acute GvHD, plus the feedback loops that modern data can actually measure. Each step names the domains that observe it.

| Step | Biology | Domains that measure it |
|---|---|---|
| **0 · Predisposition** | Genetic disparity between donor and recipient: HLA (major), minor antigens, HLA-DP epitopes, B-leader, KIR; plus host factors (age, comorbidity) and GvHD-permissive history (prior checkpoint inhibitor) | D01, D03, D06, D20 |
| **1 · Tissue injury & innate activation** | Conditioning and TBI damage gut epithelium and liver; DAMPs (ATP, uric acid, HMGB1) and PAMPs from a leaky, dysbiotic gut activate antigen-presenting cells; Paneth-cell and goblet-cell loss | D08 (agents, exposures, mucositis), D11 (antibiotics, nutrition), D18 (microbiome, SCFAs), D12 (CRP, albumin), D19 (histology) |
| **2 · Donor T-cell activation & expansion** | Host and donor APCs prime donor T cells; the infused graft's naive-T, Treg and innate-like content and the prophylaxis platform decide whether alloreactive clones expand; PtCy preferentially spares Tregs and depletes proliferating alloreactive cells | D07 (graft composition, manipulation), D09/D10 (serotherapy, PtCy, CNI exposure), D14 (reconstitution, Treg:Tcon), D16 (repertoire clonality), D17 (T-cell transcriptome) |
| **3 · Effector migration & target damage** | Activated T cells acquire homing receptors (α4β7/CCR9 → gut, CLA/CCR4 → skin), traffic to targets, and kill epithelium via cytotoxicity and IFN-γ; crypt apoptosis and Reg3α/ST2 release follow | D14 (homing subsets), D15 (ST2, REG3α, CXCL9), D13 (stool volume, rash BSA), D19 (crypt apoptosis), D17 (tissue transcriptome) |
| **4 · Feedback and chronicity** | Thymic damage impairs tolerance; B-cell dysregulation (BAFF excess), TGF-β-driven fibrosis and Tfh/Th17 skewing drive chronic GvHD; the microbiome never fully recovers | D14 (thymic output, B subsets), D15 (BAFF, CXCL9, endothelial markers), D22 (symptom burden), D19 (PFT/CT) |

**Design consequence.** Steps 0–2 are observable **before onset**, which is what makes prognosis conceivable. Steps 3–4 are largely observable **at or after onset**, which is why those measurements are inputs for diagnosis and response tasks and are *banned* as inputs for pre-onset prognosis (§4.7). The model must be told which step a feature belongs to, because that determines in which tasks it is legal.

## 4.4 Protocolised sampling schedule

Prospective collection matters more than retrospective harmonisation, because a pre-onset landmark can only exist if specimens were taken on a fixed schedule rather than when someone fell ill. Taking samples only "when clinically indicated" creates a sampling-time confound that no analysis can undo. The schedule below is the minimum for ALLO-FM v1; it is deliberately close to MAGIC and BMT CTN biorepository practice so that existing banks qualify.
''')

code(r'''
# ----------------------------------------------------------------------------- the prospective sampling schedule (exported to spec/sampling_schedule.csv)
SCHED_DAYS = ['d-30/pre', 'd0 graft', 'd+7', 'd+14', 'd+21', 'd+28', 'd+60', 'd+100', 'd+180', 'd+365', 'at onset', 'onset+7d', 'at 2nd line']
SCHEDULE = [
 ('Plasma / serum (EDTA + serum, 2×1 mL aliquots)', 'D15', ['X','X','X','X','X','X','X','X','X','X','X','X','X']),
 ('Whole blood PAXgene RNA (2.5 mL)',               'D17', ['X','X','X','X','','X','','X','','','X','X','']),
 ('PBMC viable cryopreserve (≥10×10⁶)',             'D14/D16/D17', ['X','X','','X','','X','X','X','X','X','X','','X']),
 ('Flow immune phenotyping (fresh, harmonised panel)', 'D14', ['X','X','X','X','X','X','X','X','X','X','X','','X']),
 ('Graft aliquot (cells + cryopreserve)',           'D07', ['','X','','','','','','','','','','',''] ),
 ('Stool (−80 °C, no preservative or OMNIgene)',    'D18', ['X','X','X','X','X','X','X','X','','','X','X','']),
 ('Lineage-specific chimerism (CD3 / myeloid)',     'D14', ['','','','','','X','X','X','X','X','','','']),
 ('Sorted T-cell subsets → RNA (CD4 Tconv/CD8/Treg)', 'D17', ['','','','X','','X','','X','','','X','','']),
 ('TCRβ repertoire (bulk)',                         'D16', ['X','X','','','','X','X','X','','X','X','','X']),
 ('Tissue biopsy (GI / skin / liver) + WSI + RNA',  'D17/D19', ['','','','','','','','','','','X','','X']),
 ('Germline DNA (recipient + donor)',               'D20', ['X','','','','','','','','','','','','']),
]
sched = pd.DataFrame([dict(specimen=s, domains=d, **dict(zip(SCHED_DAYS, m))) for s, d, m in SCHEDULE])
sched.to_csv(os.path.join(SPEC_DIR, 'sampling_schedule.csv'), index=False)
n_draws = {c: int((sched[c] == 'X').sum()) for c in SCHED_DAYS}
print('specimen draws per visit:', n_draws)
print(f'total specimen-visit cells = {sum(n_draws.values())}; '
      f"scheduled (non-event-driven) = {sum(v for k, v in n_draws.items() if not k.startswith(('at', 'onset')))}")
display(md_table(sched))
''')

md(r'''
**Non-negotiables of the schedule.** (i) The scheduled visits (`d+7` … `d+365`) must be drawn **whether or not the patient is unwell**, otherwise "sample exists" becomes a predictor of GvHD. (ii) Pre-analytic variables (time to freeze, tube type, freeze–thaw count) are recorded for every aliquot and enter the mandatory confounder-only comparator. (iii) Biomarker assays are run in one laboratory, in batches that are **randomised with respect to outcome and centre**, with bridging samples across batches; batch ID is always available to the audit. (iv) Event-driven samples (at onset, onset+7 days, at second line) are kept for diagnosis and response tasks and may not be used as pre-onset inputs.

## 4.5 Where the data actually are

| Source | What it holds | Scale | Access | Role for ALLO-FM |
|---|---|---|---|---|
| **CIBMTR Research Database** | Registry core on structured forms (TED and comprehensive-report-form subsets): demographics, disease, donor/HLA, graft, conditioning, prophylaxis, aGvHD/cGvHD grades, relapse, death | >470,000 recipients, >500 centres (2017 figure) ✓; 18,000+ collected/derived variables, 148 published-study datasets downloadable ✓ | Controlled; study proposal and DUA; some final datasets are public | The *network-stage* corpus and evaluation cohort (§11); **not needed** for the one-institution plan |
| **EBMT Registry** | Comparable core (MED-A/MED-B), European centres; promotion/validation pipeline | order 10⁵–10⁶ registrations † | Controlled; via study proposal and Working Party (e.g. Transplant Complications WP) | External-registry validation (tests the shared-structure premise P3 across continents) |
| **BMT CTN trials** (e.g. 1703 ✓, 1301, 1203, 0402) | Randomised arms with protocol-fixed prophylaxis, central adjudication, and banked specimens | 1703: 431 patients ✓; others 10²–10³ | BioLINCC / dbGaP controlled | **Treatment-effect validation anchor** (A7); sample-backed Tier C |
| **MAGIC consortium** | Protocolised day-7/day-14 serum biomarkers (ST2, REG3α …) with harmonised aGvHD staging across >20 centres | day-7 algorithm derived and validated in 1,287 patients ✓ | Collaboration | Tier C biomarker adapter; MAP is the baseline to beat in B1 (the centre re-implements it on its own banked plasma) |
| **NMDP/CIBMTR Repository** | Paired pre-HCT recipient and donor biospecimens linked to registry outcomes | order 10⁵ pairs † | Controlled, assay-result return required ✓ | Germline, mHA, KIR, HLA-expression work (D06, D20) |
| **MSK HCT "hospitalome"** | Longitudinal stool microbiota + daily drug exposures + clinical metadata | >10,000 stool samples, >1,000 patients ✓ | **Public** (figshare / Scientific Data 2021) ✓ | Microbiome adapter pretraining; the one large *public* allo-HCT longitudinal resource |
| **Single-centre EHR (OMOP)** | Full Tier B timeline: every lab, drug, vital, note | 10³–10⁴ transplants per large centre | Local IRB | **The data this plan is built on**: up to 1,500 timelines (≈ 200 transplants a year over 7.5 years †), of which ≈ 420 with banked specimens (§9.8); the only place Tier B exists today |
| **GEO / ArrayExpress / CELLxGENE** | Public transcriptomes; immune-cell atlases | GvHD series: 78 human series, 2,954 samples, median series very small (pilot census); atlases 10⁶–10⁷ cells | Public | Omics-adapter pretraining; **not** sufficient for clinical labels (§10) |
| **dbGaP / EGA** | Controlled-access sequencing (including BMT CTN companion studies) | varies | Controlled | Tier C/D |

**The honest summary of supply.** Tier A exists at 10⁵–10⁶ scale but only in registries, behind proposals and DUAs. Tier B exists at 10³–10⁴ transplants per large centre and is not pooled anywhere. Tier C exists for 10³–10⁴ patients inside consortium biobanks. Tier D exists for 10¹–10² patients per study. **The plan therefore starts from what one institution already holds** — a Tier A/B timeline for up to 1,500 recipients and, from prospective banking, Tier C specimens for a few hundred of them — and uses a released model for everything that needs millions of patients. Tier C/D data are adapters that are optional at inference; any design that requires single-cell data for every patient is not trainable in this decade.
''')

md(r'''
## 4.6 Outcome labels as code

A benchmark is only reproducible if its labels are functions, not prose. Below are executable implementations of the three label definitions every GvHD task depends on, with unit tests. The staging-to-grade mapping follows the MAGIC 2016 consensus (✓ for the existence of the criteria; the thresholds below are the standard Glucksberg/MAGIC formulation and should be checked line-by-line against the consensus table before publication †). Deliberately, nothing here looks at a model: these are definitions.
''')

code(r'''
# ----------------------------------------------------------------------------- aGvHD organ staging -> overall grade (MAGIC/Glucksberg formulation)
def stage_skin(bsa_rash_pct, bullae_or_desquamation_pct=0.0):
    """Skin stage 0-4 from % body-surface-area of maculopapular rash; stage 4 = generalised erythroderma with bullae/desquamation."""
    if bullae_or_desquamation_pct and bullae_or_desquamation_pct > 0: return 4
    b = float(bsa_rash_pct)
    return 0 if b < 1 else 1 if b <= 25 else 2 if b <= 50 else 3

def stage_liver(bilirubin_mg_dl):
    """Liver stage 0-4 from total bilirubin (mg/dL). Alternative causes (SOS, drug, sepsis, haemolysis) must be adjudicated separately."""
    b = float(bilirubin_mg_dl)
    return 0 if b < 2 else 1 if b < 3.1 else 2 if b < 6.1 else 3 if b < 15.1 else 4

def stage_upper_gi(persistent_nausea_vomiting_anorexia):
    """Upper-GI stage 0-1 (MAGIC collapses upper GI to present/absent)."""
    return 1 if bool(persistent_nausea_vomiting_anorexia) else 0

def stage_lower_gi(stool_ml_day=None, stool_ml_kg_day=None, severe_pain_or_ileus=False, grossly_bloody=False):
    """Lower-GI stage 0-4 from daily stool volume (adult mL/day, or paediatric mL/kg/day); stage 4 = severe pain/ileus or grossly bloody stool."""
    if severe_pain_or_ileus or grossly_bloody: return 4
    if stool_ml_kg_day is not None:                      # paediatric thresholds
        v = float(stool_ml_kg_day)
        return 0 if v < 10 else 1 if v < 20 else 2 if v < 30 else 3
    v = float(stool_ml_day or 0)                         # adult thresholds
    return 0 if v < 500 else 1 if v < 1000 else 2 if v < 1500 else 3

def magic_grade(skin=0, liver=0, ugi=0, lgi=0):
    """Overall MAGIC/Glucksberg grade 0-IV from organ stages. Returns 0..4 (4 == grade IV).
    IV: skin 4, liver 4 or lower-GI 4.  III: liver 2-3 or lower-GI 2-3.
    II: skin 3, liver 1, upper GI, or lower-GI 1.  I: skin 1-2 only."""
    s, l, u, g = int(skin), int(liver), int(ugi), int(lgi)
    if s == 4 or l == 4 or g == 4: return 4
    if l in (2, 3) or g in (2, 3): return 3
    if s == 3 or l == 1 or u == 1 or g == 1: return 2
    if s in (1, 2): return 1
    return 0

# ---- unit tests against hand-worked vignettes
T = [dict(case='no GvHD', st=dict(skin=0, liver=0, ugi=0, lgi=0), want=0),
     dict(case='rash 20% BSA only', st=dict(skin=stage_skin(20)), want=1),
     dict(case='rash 60% BSA only', st=dict(skin=stage_skin(60)), want=2),
     dict(case='persistent nausea only', st=dict(ugi=stage_upper_gi(True)), want=2),
     dict(case='stool 800 mL/d', st=dict(lgi=stage_lower_gi(stool_ml_day=800)), want=2),
     dict(case='stool 1400 mL/d (lower-GI stage 2)', st=dict(lgi=stage_lower_gi(stool_ml_day=1400)), want=3),
     dict(case='stool 1800 mL/d (lower-GI stage 3)', st=dict(lgi=stage_lower_gi(stool_ml_day=1800)), want=3),
     dict(case='bilirubin 2.5 mg/dL (liver stage 1) only', st=dict(liver=stage_liver(2.5)), want=2),
     dict(case='bilirubin 7 mg/dL', st=dict(liver=stage_liver(7)), want=3),
     dict(case='bilirubin 20 mg/dL', st=dict(liver=stage_liver(20)), want=4),
     dict(case='bloody stool + ileus', st=dict(lgi=stage_lower_gi(stool_ml_day=600, grossly_bloody=True)), want=4),
     dict(case='paediatric 25 mL/kg/d (stage 2)', st=dict(lgi=stage_lower_gi(stool_ml_kg_day=25)), want=3),
     dict(case='paediatric 15 mL/kg/d (stage 1)', st=dict(lgi=stage_lower_gi(stool_ml_kg_day=15)), want=2),
     dict(case='rash 30% + stool 600 mL/d', st=dict(skin=stage_skin(30), lgi=stage_lower_gi(stool_ml_day=600)), want=2)]
rows = []
for t in T:
    got = magic_grade(**t['st']); rows.append(dict(vignette=t['case'], stages=str(t['st']), grade=got, expected=t['want'], ok=got == t['want']))
tt = pd.DataFrame(rows)
assert tt.ok.all(), tt[~tt.ok]
print(f'MAGIC grading: {len(tt)}/{len(tt)} vignettes pass'); display(md_table(tt.drop(columns=['ok'])))
''')

code(r'''
# ----------------------------------------------------------------------------- steroid-refractory / -dependent / -intolerant aGvHD (2018 EBMT-NIH-CIBMTR formulation)
def sr_agvhd(day_of_steroid, grade_series, pred_eq_mg_kg_series, taper_attempted=False,
             progression_day=None, toxicity_requiring_stop=False):
    """Classify the steroid-response phenotype from the day-by-day course after systemic steroids start.
    grade_series / pred_eq_mg_kg_series: dict {day_from_steroid_start: value}. Returns one of
    'refractory' | 'dependent' | 'intolerant' | 'responsive'."""
    g0 = grade_series.get(0)
    if toxicity_requiring_stop: return 'intolerant'
    if progression_day is not None and progression_day <= 3: return 'refractory'        # progression after >=3 days of >=2 mg/kg
    g3, g7 = grade_series.get(3, g0), grade_series.get(7, g0)
    dose_ok = max([v for d, v in pred_eq_mg_kg_series.items() if d <= 7] or [0]) >= 2.0
    if dose_ok and g3 is not None and g3 > g0: return 'refractory'                      # worse by day 3
    if dose_ok and g7 is not None and g7 >= g0 and g0 >= 1: return 'refractory'         # no improvement by day 7
    if taper_attempted:
        late = [v for d, v in pred_eq_mg_kg_series.items() if d >= 28]
        if late and min(late) >= 0.25 and (grade_series.get(28, 0) or 0) >= 1: return 'dependent'
    return 'responsive'

def grfs_event(day_grade34_agvhd=None, day_cgvhd_systemic=None, day_relapse=None, day_death=None,
               day_last_followup=None):
    """GvHD-free, relapse-free survival (Holtan-style): first of grade III-IV aGvHD, cGvHD needing systemic therapy,
    relapse, or death; else censored at last follow-up. Returns (time_days, event 0/1, cause)."""
    cand = [(d, c) for d, c in ((day_grade34_agvhd, 'aGvHD III-IV'), (day_cgvhd_systemic, 'cGvHD systemic'),
                                (day_relapse, 'relapse'), (day_death, 'death')) if d is not None]
    if not cand: return float(day_last_followup), 0, 'censored'
    t, c = min(cand); return float(t), 1, c

cases = [
 dict(case='grade II -> grade III by d3 on 2 mg/kg', f=lambda: sr_agvhd(0, {0: 2, 3: 3}, {0: 2.0, 3: 2.0}), want='refractory'),
 dict(case='grade III unchanged at d7 on 2 mg/kg',   f=lambda: sr_agvhd(0, {0: 3, 3: 3, 7: 3}, {0: 2.0, 7: 2.0}), want='refractory'),
 dict(case='grade II -> I by d7',                    f=lambda: sr_agvhd(0, {0: 2, 3: 2, 7: 1}, {0: 2.0, 7: 2.0}), want='responsive'),
 dict(case='flare on taper, still 0.3 mg/kg at d28', f=lambda: sr_agvhd(0, {0: 2, 7: 1, 28: 1}, {0: 2.0, 28: 0.3}, taper_attempted=True), want='dependent'),
 dict(case='steroid psychosis -> stop',              f=lambda: sr_agvhd(0, {0: 2}, {0: 2.0}, toxicity_requiring_stop=True), want='intolerant'),
 dict(case='GRFS: relapse d200 before death d400',   f=lambda: grfs_event(day_relapse=200, day_death=400)[2], want='relapse'),
 dict(case='GRFS: alive, no events, f/u d500',       f=lambda: grfs_event(day_last_followup=500)[1], want=0),
 dict(case='GRFS: cGvHD on systemic tx d150',        f=lambda: grfs_event(day_cgvhd_systemic=150, day_last_followup=600)[2], want='cGvHD systemic'),
]
lt = pd.DataFrame([dict(case=c['case'], got=c['f'](), expected=c['want'], ok=c['f']() == c['want']) for c in cases])
assert lt.ok.all(), lt[~lt.ok]
print(f'steroid-response and GRFS label functions: {len(lt)}/{len(lt)} vignettes pass'); display(md_table(lt.drop(columns=['ok'])))
''')

md(r'''
**What the label code makes explicit.** Three things that prose definitions usually hide. (1) aGvHD grade is a **daily** quantity, so "grade II–IV aGvHD by day 100" needs a daily series, not a single field; a registry that stores only the maximum grade cannot support a landmark task at day +14. (2) Steroid refractoriness is defined **relative to the steroid start day and the dose actually given**, so the exposure timeline (D24) is part of the label, not an input. (3) Composite endpoints such as GRFS are **first-event** definitions, and the cause matters for competing-risk analysis. Each of these is a place where two centres can silently disagree, which is exactly why the benchmark ships the functions.

## 4.7 Leakage: the four ways an HCT model cheats, and the rule that stops each

| Leak | How it happens | Rule in the specification |
|---|---|---|
| **Treatment-as-feature** | Systemic steroids, budesonide, ruxolitinib or an ECP order are the strongest "predictors" of GvHD, because they are its treatment | Elements flagged `proxy` in D24 (and the steroid-driven parts of D10 taper) are **removed from the input window** of any task whose outcome they could reveal. The removal is implemented in the harness, and the audit re-checks it (§8.5) |
| **Result-availability** | A biopsy taken on day +20 is signed out on day +23, but a naive pipeline attaches it to day +20 | Every event carries `t_known`; the input filter uses `t_known ≤ L`, never `t_valid ≤ L` |
| **Sampling-time** | Research samples are drawn *because* a patient is unwell, so "a sample exists at day +21" predicts GvHD | Scheduled visits are protocolised and drawn regardless of state (§4.4); event-driven samples are labelled and excluded from pre-onset tasks; missingness indicators are inputs so the model cannot be tricked by them silently |
| **Grouping** | The same patient appears as several samples; a donor serves two recipients; a centre's batch aligns with its outcome rate | The unit is the **patient** (and the donor–recipient **pair**); splits are grouped by patient, then by centre; bootstrap resamples patients; donor-sharing is checked |

The harness in §8.8 runs this audit on a cohort that contains a **planted leakage feature**, and reports whether the audit catches it.
''')
