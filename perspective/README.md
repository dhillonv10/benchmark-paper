# Perspective package

A short opinion piece (about 1,000 words, one four-panel figure) and the analysis behind it.

| File | What it is |
|---|---|
| `perspective_draft.md` | the draft text; every number is filled in from `results.json` by `make_text.py` |
| `figures/Fig01_perspective.pdf` / `.png` | the four-panel figure (vector PDF + 600-dpi PNG) and its legend (`..._caption.txt`) |
| `analysis.py` | downloads the public cohort and computes every number (about 14 minutes on 4 CPU cores; `PERSPECTIVE_QUICK=1` for a smoke test) |
| `make_figure.py` | draws the figure from `results.json`; uses the notebook's layout gate (`figsys.py`), which fails on any text overlap or font below 6 pt |
| `results.json` | all numbers behind the text and the figure |

## Data
The aGOAT cohort (Liu et al., *Nat Comput Sci* 2022; OMIX001095, https://ngdc.cncb.ac.cn/omix/release/OMIX001095/): one centre in Tianjin, 584 adults, 194 variables on days -14 to 100, severe aGvHD with onset day.
It is open access, but its sharing is governed by PRC human-genetic-resources rules, so `analysis.py` downloads it from the source into `perspective/data/` (git-ignored) and nothing here redistributes it. The download host resets connections now and then; the script retries.

## What was done, and what was not
* Temporal validation throughout: patients are ordered by transplant date; each of four later blocks is predicted by models fitted on earlier patients only (training sizes 142 / 224 / 351 / 519).
* Every arm uses the same regularisation grid and patient-grouped inner cross-validation. (An early draft gave one arm a wider grid and found a spurious +0.03; the script now shares one grid.)
* The headline metric for the 14-day forecast compares only patients at risk on the same landmark day and predicted by the same model, because a pooled AUROC also rewards anything that tracks days since transplant.
* A shuffled-outcome null (20 shuffles, within landmark) gives the distribution of the gain that the pipeline produces by chance.
* Not done: the cohort authors' own model (R code at https://github.com/chenjunren-ihcams/daGOAT) was not run; the 45-patient paediatric cohort was not analysed (about 11 events); no claim is made about foundation models, only about heads on label-free summaries of the timeline.
* Exploratory: one cohort, one centre, mostly haploidentical transplants; several arms were compared, all are reported.
