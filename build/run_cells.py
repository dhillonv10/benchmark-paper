"""Execute the notebook's code cells in one namespace; print everything (Markdown rendered as text); keep going past errors.
usage: python3 devrun.py [first_code_cell [last_code_cell]]   env: ALLOFM_QUICK, ALLOFM_STRICT_FIGS"""
import json, sys, time, traceback, io, contextlib
import matplotlib; matplotlib.use('Agg')
import IPython.display as ipd
nb = json.load(open('ALLO_FM_GvHD_foundation_model.ipynb'))
cells = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0; hi = int(sys.argv[2]) if len(sys.argv) > 2 else len(cells)
def my_display(*objs, **kw):
    for o in objs:
        if isinstance(o, ipd.Markdown): print(o.data)
        else: print(repr(o) if not hasattr(o, 'to_string') else o.to_string())
g = {'__name__': '__main__'}; t00 = time.time()
for k, (i, src) in enumerate(cells):
    if k >= hi: break
    t0 = time.time(); print(f'\n######## code cell {k} (nb cell {i})'); sys.stdout.flush()
    try:
        exec(src, g); g['display'] = my_display
    except Exception:
        g['display'] = my_display
        print(f'!!! cell {k} raised:'); traceback.print_exc(limit=4)
    print(f'-- {time.time() - t0:.0f}s'); sys.stdout.flush()
print(f'done in {time.time() - t00:.0f}s')
