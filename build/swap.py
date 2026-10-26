def replace_span(path, start_marker, finish_num, new_text):
    s = open(path).read()
    assert s.count(start_marker) == 1, f'{path}: marker count {s.count(start_marker)} for {start_marker[:60]!r}'
    a = s.index(start_marker); f = s.index(f"finish(fig, {finish_num},", a); e = s.index("')\n", f) + 3
    s = s[:a] + new_text.rstrip('\n') + '\n' + s[e:]
    open(path, 'w').write(s)
    import ast; ast.parse(s)
