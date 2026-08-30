"""
TableHeatmap.py — ANSI heat-shaded numeric tables for the terminal.

Colour encodes MAGNITUDE, so the ramp is sequential and single-hue, light to
dark (ColorBrewer Blues). Not a rainbow: a multi-hue ramp implies categories
that are not there, and a single hue is colourblind-safe by construction.

Each column is normalised on its OWN min/max. The parameters in these tables
differ by orders of magnitude - dSIGMA near 0.1, dETA1 near 50 - so a shared
scale would paint every column flat except one.

Truecolor (24-bit) escapes, which VS Code's terminal and every modern emulator
support. Honours NO_COLOR, and turns itself off when stdout is not a tty so
piping to a file or a pager stays clean.
"""

import os
import sys

# ColorBrewer 9-class Blues, light -> dark.
_RAMP = [(247, 251, 255), (222, 235, 247), (198, 219, 239), (158, 202, 225),
         (107, 174, 214), (66, 146, 198), (33, 113, 181), (8, 81, 156),
         (8, 48, 107)]


def enabled(force=None):
    if force is not None:
        return force
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _rgb(t):
    """Interpolate the ramp at t in [0, 1]."""
    if t != t:                                   # NaN
        return None
    t = min(max(t, 0.0), 1.0)
    x = t * (len(_RAMP) - 1)
    i = int(x)
    if i >= len(_RAMP) - 1:
        return _RAMP[-1]
    f = x - i
    a, b = _RAMP[i], _RAMP[i + 1]
    return tuple(int(round(a[k] + f * (b[k] - a[k]))) for k in range(3))


def _ink(bg):
    """Black or white text, whichever has contrast against this cell."""
    r, g, b = bg
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def _cell(text, t, on):
    if not on or t is None or t != t:
        return text
    bg = _rgb(t)
    fg = _ink(bg)
    return ("\033[48;2;%d;%d;%dm\033[38;2;%d;%d;%dm%s\033[0m"
            % (bg[0], bg[1], bg[2], fg[0], fg[1], fg[2], text))


def render(df, decimals=4, index_label="", color=None, index_width=6):
    """Heat-shaded table. Colour is per-column; the layout is unchanged."""
    on = enabled(color)
    cols = list(df.columns)
    width = {c: max(len(str(c)), decimals + 6) + 2 for c in cols}

    lo = {c: float(df[c].min()) for c in cols}
    hi = {c: float(df[c].max()) for c in cols}

    out = [" " * index_width + "".join(str(c).rjust(width[c]) for c in cols)]
    for idx, row in df.iterrows():
        line = ("%-*s" % (index_width, idx))
        for c in cols:
            v = float(row[c])
            span = hi[c] - lo[c]
            t = 0.5 if span == 0 else (v - lo[c]) / span
            line += _cell(("%.*f" % (decimals, v)).rjust(width[c]), t, on)
        out.append(line)
    return "\n".join(out)


def legend(color=None, label="low -> high, shaded within each column"):
    if not enabled(color):
        return "   (%s; colour off)" % label
    bar = "".join(_cell("  ", i / 11.0, True) for i in range(12))
    return "   %s  %s" % (bar, label)
