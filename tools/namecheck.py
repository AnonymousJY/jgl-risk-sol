"""Static undefined-name check for a module's functions.

py_compile catches syntax, not a name that is never bound - which is how
SYS_PARAMS reached a committed file it was never defined in. Walks each
function body, collects every Name it LOADS, and subtracts module-level
bindings, imports (including ones inside the function), parameters, locals,
comprehension targets and builtins.
"""
import ast, builtins, sys

def bound_by(node):
    """Every name a node binds: assignment, import, for, with, except, args."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            out.update(n.names)
    return out

def check(path):
    tree = ast.parse(open(path).read())
    top = bound_by(ast.Module(
        [n for n in tree.body if not isinstance(n, ast.FunctionDef)], []))
    top |= {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    bi = set(dir(builtins))
    bad = []
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        local = bound_by(fn)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id not in top and n.id not in local and n.id not in bi:
                    bad.append((fn.name, n.lineno, n.id))
    return bad

rc = 0
for path in sys.argv[1:]:
    bad = check(path)
    if bad:
        rc = 1
        print("%s:" % path)
        for fn, ln, name in sorted(set(bad)):
            print("   line %-5d in %-18s undefined name: %s" % (ln, fn, name))
    else:
        print("%s: ok" % path)
sys.exit(rc)
