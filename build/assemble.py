import json, importlib
import part1
for m in ('part1b', 'part2', 'part3', 'part3b', 'part4', 'part4b', 'part5a', 'part5b', 'part5c'):
    importlib.import_module(m)
from part1 import CELLS
nb = dict(cells=[], metadata=dict(
    kernelspec=dict(display_name='Python 3', language='python', name='python3'),
    language_info=dict(name='python', version='3.13'),
    colab=dict(name='ALLO_FM_GvHD_foundation_model.ipynb', provenance=[])), nbformat=4, nbformat_minor=5)
for kind, src in CELLS:
    c = dict(cell_type=kind, metadata={}, source=src.splitlines(keepends=True))
    if kind == 'code': c.update(outputs=[], execution_count=None)
    nb['cells'].append(c)
json.dump(nb, open('ALLO_FM_GvHD_foundation_model.ipynb', 'w'), indent=1)
print(len(nb['cells']), 'cells;', sum(1 for k, _ in CELLS if k == 'code'), 'code cells')
