# Build sources

The paper notebook `ALLO_FM_GvHD_foundation_model.ipynb` is generated, not hand-edited, so that
the narrative and the computed numbers cannot drift apart.

- `part1.py`   front matter, abstract, setup, statistical toolkit, publication figure system, 1 Introduction, 2 Plausibility
- `part1b.py`  3 What can be predicted (ceilings, capability claims, transcriptomics at diagnosis)
- `part2.py`   4 Data specification (dictionary, schedule, sources, label functions, leakage)
- `part3.py`   5 Architecture, 6 Training objectives
- `part3b.py`  7 Training requirements (tokens, parameters, compute with attention, programme enumerated run by run)
- `part4.py`   8.0-8.11 GvHD-Bench v1.0: task registry, task cards, metrics, audit, gate, executed harness
- `part4b.py`  8.12-8.14 the benchmark's own operating characteristics, clinical translation (drift, enrichment, alert burden), added rules
- `part5a.py`  9.1-9.3 precision, paired power (closed form, validated), development size, sources, accrual
- `part5b.py`  9.4-9.8 transcriptome learning curves, longitudinal design, centres, validation / subgroups / label noise, the data bill
- `part5c.py`  10 Pilot evidence, 11 Roadmap, 12-13, exports, consistency checks, references, appendices
- `assemble.py` builds the notebook from the parts; `check_figures.py` / `run_cells.py` execute it cell by cell for development

Rebuild:

    python3 assemble.py                       # writes the .ipynb
    ALLOFM_STRICT_FIGS=1 jupyter nbconvert --to notebook --execute --inplace ALLO_FM_GvHD_foundation_model.ipynb

Running the notebook writes `figures/` (vector PDF + 600-dpi PNG per figure, 17 in all) and `spec/`
(data dictionary, sampling schedule, task registry and cards, gate, standalone benchmark specification,
and the data-requirement bundle `DATA_REQUIREMENTS_v1.0.md`, `data_requirements.csv`, `programme_cost.csv`).

Environment variables: `ALLOFM_QUICK=1` (fewer simulation replicates, 8 held-out centres), `ALLOFM_STRICT_FIGS=1`
(a layout failure stops the run), `ALLOFM_FONT="DejaVu Sans"` (test the wider fallback font).
