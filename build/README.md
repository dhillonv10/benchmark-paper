# Build sources

The paper notebook `ALLO_FM_GvHD_foundation_model.ipynb` is generated, not hand-edited, so that
the narrative and the computed numbers cannot drift apart.

- `part1.py`   front matter, abstract, setup, publication figure system, 1 Introduction, 2 Plausibility
- `part1b.py`  3 What can be predicted (ceilings, capability claims, transcriptomics at diagnosis)
- `part2.py`   4 Data specification (dictionary, schedule, sources, label functions, leakage)
- `part3.py`   5 Architecture, 6 Training objectives
- `part3b.py`  7 Training requirements (tokens, parameters, compute, cost)
- `part4.py`   8 GvHD-Bench v1.0
- `part5.py`   9 Sample size and accrual, 10 Pilot evidence, 11 Engineering path, 12-13, references, exports

Rebuild:

    python3 assemble.py                       # writes the .ipynb
    jupyter nbconvert --to notebook --execute --inplace ALLO_FM_GvHD_foundation_model.ipynb

Running the notebook writes `figures/` (vector PDF + 600-dpi PNG per figure) and `spec/`
(data dictionary, sampling schedule, task registry, task cards, gate, standalone benchmark spec).
