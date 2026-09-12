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

render(abs_shade=True) shades on |v| while printing the signed value, for
tables whose columns are negative by construction - a down-side expected
shortfall, say. Default off: a signed parameter shades by its position on the
number line, which is what dMUI and dRHOIX want.

to_html() and to_eml() turn a captured run - the same ANSI, read back out of
the escapes - into HTML that survives Outlook, which renders mail through Word
rather than a browser.
"""

import io
import logging
import os
import re
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


def render(df, decimals=4, index_label="", color=None, index_width=6,
           abs_shade=False):
    """Heat-shaded table. Colour is per-column; the layout is unchanged.

    abs_shade shades on |v| while still PRINTING the signed value. Off by
    default, because a column like dMUI or dRHOIX carries real information in
    its sign and should shade by position on the number line. Turn it on for
    a column whose sign is fixed by what it measures rather than by the data -
    a down-side expected shortfall is negative by construction, so shading it
    signed would paint the worst year lightest and the mildest year darkest,
    exactly backwards.
    """
    on = enabled(color)
    cols = list(df.columns)
    width = {c: max(len(str(c)), decimals + 6) + 2 for c in cols}

    src = df.abs() if abs_shade else df
    lo = {c: float(src[c].min()) for c in cols}
    hi = {c: float(src[c].max()) for c in cols}

    out = [" " * index_width + "".join(str(c).rjust(width[c]) for c in cols)]
    for idx, row in df.iterrows():
        line = ("%-*s" % (index_width, idx))
        for c in cols:
            v = float(row[c])
            span = hi[c] - lo[c]
            t = 0.5 if span == 0 else ((abs(v) if abs_shade else v) - lo[c]) / span
            line += _cell(("%.*f" % (decimals, v)).rjust(width[c]), t, on)
        out.append(line)
    return "\n".join(out)


def legend(color=None, label="low -> high, shaded within each column"):
    if not enabled(color):
        return "   (%s; colour off)" % label
    bar = "".join(_cell("  ", i / 11.0, True) for i in range(12))
    return "   %s  %s" % (bar, label)


# ---------------------------------------------------------------------------
# The same tables, as HTML that survives Outlook.
# ---------------------------------------------------------------------------
# Outlook on Windows renders mail through WORD, not a browser, and Word throws
# away most of what a page would rely on: CSS classes, stylesheets, float,
# position, background images. What it does honour is an inline style on an
# inline element, and a table for layout. So the ANSI a terminal run already
# produced is translated span by span - background-color and color written
# straight onto each run - and the whole thing dropped into a one-cell table.
#
# Spaces become &nbsp; and newlines become <br> rather than relying on
# white-space:pre, which Word applies inconsistently. That costs bytes and
# buys alignment that does not collapse when the message is replied to,
# forwarded, or opened on a phone.
#
# The colours are the ones the terminal printed, because they are read back
# out of the escapes rather than recomputed - so what lands in the mail is
# what the run showed, including any table this module did not render.

_ANSI_RE = re.compile(r"\033\[([0-9;]*)m")


def _esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace(" ", "&nbsp;"))


def _apply(codes, bg, fg):
    """Fold one escape's parameters into the running (bg, fg)."""
    parts = [p for p in codes.split(";") if p != ""] or ["0"]
    i = 0
    while i < len(parts):
        p = parts[i]
        if p == "0":
            bg = fg = None
            i += 1
        elif p in ("38", "48") and i + 4 < len(parts) and parts[i + 1] == "2":
            c = "#%02x%02x%02x" % (int(parts[i + 2]), int(parts[i + 3]),
                                   int(parts[i + 4]))
            if p == "38":
                fg = c
            else:
                bg = c
            i += 5
        else:
            i += 1                                   # bold, dim, 256-colour...
    return bg, fg


def _span(txt, bg, fg):
    t = _esc(txt)
    if bg is None and fg is None:
        return t
    st = []
    if bg:
        st.append("background-color:%s" % bg)
    if fg:
        st.append("color:%s" % fg)
    return '<span style="%s">%s</span>' % (";".join(st), t)


def _line_html(ln):
    out, pos, bg, fg = [], 0, None, None
    for m in _ANSI_RE.finditer(ln):
        if m.start() > pos:
            out.append(_span(ln[pos:m.start()], bg, fg))
        bg, fg = _apply(m.group(1), bg, fg)
        pos = m.end()
    if pos < len(ln):
        out.append(_span(ln[pos:], bg, fg))
    return "".join(out) or "&nbsp;"


def to_html(lines, title="", font_pt=9,
            mono="Consolas, 'Courier New', monospace"):
    """ANSI-coloured terminal output -> one Outlook-safe HTML document.

    `lines` is a list of strings or one string with newlines; anything this
    module's render() produced with color=True comes through with its shading
    intact. Plain lines pass through unchanged apart from escaping.
    """
    if isinstance(lines, str):
        lines = lines.split("\n")
    body = []
    for raw in lines:
        for ln in str(raw).split("\n"):
            body.append(_line_html(ln))
    head = ('<div style="font-family:Calibri,Arial,sans-serif;font-size:11pt;'
            'margin:0 0 10px 0"><b>%s</b></div>' % _esc(title)) if title else ""
    return (
        '<html><head><meta http-equiv="Content-Type" '
        'content="text/html; charset=utf-8"></head>'
        '<body style="margin:0;padding:0">'
        '<table cellpadding="10" cellspacing="0" border="0"><tr><td '
        'style="font-family:%s;font-size:%dpt;line-height:115%%">'
        '%s%s</td></tr></table></body></html>'
        % (mono, font_pt, head, "<br>".join(body)))


def to_eml(lines, subject="", title=None, to="", sender="", **kw):
    """The same document wrapped as a message file Outlook will open.

    Double-clicking the .eml opens it as a received message; Forward or Reply
    then gives an editable draft with the formatting intact. Outlook cannot
    open a ready-to-send DRAFT from a standard file - that needs the
    proprietary .msg - so forwarding is the route.
    """
    from email.message import EmailMessage
    m = EmailMessage()
    m["Subject"] = subject
    if sender:
        m["From"] = sender
    if to:
        m["To"] = to
    m.set_content("This message is formatted in HTML.")
    m.add_alternative(to_html(lines, title=subject if title is None else title,
                              **kw), subtype="html")
    return m.as_bytes()


# ---------------------------------------------------------------------------
# Capturing a whole run, without editing the run.
# ---------------------------------------------------------------------------
# estimate_systematic.py has 158 _LOG.info calls and estimate_idiosyncratic.py
# has 100. Neither should acquire an HTML export by being rewritten 258 times.
# A logging handler collects every record as the terminal received it, escapes
# and all, so to_html reproduces the run exactly and the scripts need four
# lines each.
#
# ONE CONSEQUENCE. The shading has to be present in the RECORD, which means
# colour has to be on for the run and not only for the export - the handler
# sees what render() already produced and cannot re-render it. A caller that
# asks for --html should therefore force colour on. The terminal then carries
# escapes too; redirect it if that matters.


class _Capture(logging.Handler):

    def __init__(self):
        logging.Handler.__init__(self)
        self.lines = []

    def emit(self, record):
        try:
            self.lines.append(record.getMessage())
        except Exception:                    # a report must never die of its
            pass                             # own bookkeeping


def capture(logger):
    """Attach a collector to `logger` and return it. Read `.lines` after."""
    h = _Capture()
    logger.addHandler(h)
    return h


def write_capture(handler, html_path=None, eml_path=None, title=""):
    """Write the collected run out. Returns the paths written, in order."""
    done = []
    if html_path:
        with io.open(html_path, "w", encoding="utf-8") as fh:
            fh.write(to_html(handler.lines, title=title))
        done.append(html_path)
    if eml_path:
        with open(eml_path, "wb") as fh:
            fh.write(to_eml(handler.lines, subject=title))
        done.append(eml_path)
    return done
