"""Run every code cell in order (QUICK mode for speed), keep going past non-figure failures, print only layout results."""
import json, sys, time, os, io, contextlib
import matplotlib; matplotlib.use('Agg')
nb = json.load(open('ALLO_FM_GvHD_foundation_model.ipynb'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
g = {}; t00 = time.time()
for i, src in enumerate(cells):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): exec(src, g)
    except Exception as e:
        print(f'cell {i} raised {type(e).__name__}: {str(e)[:160]}')
    out = buf.getvalue()
    for line in out.splitlines():
        if line.startswith(('[layout', '    text/', '    too small')): print(line)
print(f'done in {time.time()-t00:.0f}s')
