#!/usr/bin/env python
"""Verify every cross-module import in this repo binds to a real definition.

    python tools/check_imports.py

Exists because of a specific failure. estimate_idiosyncratic.py did
`from poc.estimate_systematic import _drain`, an edit deleted the `def _drain`
in that module while leaving a CALL to it behind, and the check I used -
`"_drain(" in source` - matched the call site and reported success. The run
then died at import.

Grep on a bare name cannot distinguish a definition from a use. This parses
instead: for every `from X import Y` where X is a module in this repo, confirm
Y is bound at X's top level. Needs no third-party imports, so it runs anywhere
- including where the pymc stack is missing or broken.
"""
import ast
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKGS = ("poc", "Library", "Scripts", "tools")


def toplevel_names(path):
    """Every name bound at module level, including inside if/try blocks."""
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    out = set()

    def add_targets(node):
        for tg in getattr(node, "targets", []):
            if isinstance(tg, ast.Name):
                out.add(tg.id)
            elif isinstance(tg, (ast.Tuple, ast.List)):
                for e in tg.elts:
                    if isinstance(e, ast.Name):
                        out.add(e.id)

    def walk_body(body):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
            elif isinstance(n, ast.Assign):
                add_targets(n)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                out.add(n.target.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    out.add(a.asname or a.name.split(".")[0])
            elif isinstance(n, (ast.If, ast.Try, ast.With)):
                for attr in ("body", "orelse", "finalbody"):
                    walk_body(getattr(n, attr, []) or [])
                for h in getattr(n, "handlers", []):
                    walk_body(h.body)
    walk_body(tree.body)
    return out


def main():
    mods, files = {}, []
    for pkg in PKGS:
        d = os.path.join(ROOT, pkg)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                path = os.path.join(d, f)
                mods["%s.%s" % (pkg, f[:-3])] = toplevel_names(path)
                files.append(path)

    bad = []
    for path in files:
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module in mods:
                for a in n.names:
                    if a.name != "*" and a.name not in mods[n.module]:
                        bad.append((os.path.relpath(path, ROOT), a.name, n.module))

    for path, nm, mod in bad:
        print("  BROKEN  %s imports '%s' from %s - no top-level definition"
              % (path, nm, mod))
    print("%d module(s), %d cross-module import(s) checked: %s"
          % (len(mods), sum(1 for _ in files),
             "ALL RESOLVE" if not bad else "%d BROKEN" % len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
