"""Shared figure style — copied from fewer_bits (annstudy.analysis.paper_figures) so both papers
look alike. Figures are drawn at printed size (ACM two-column: full width 7.0 in, column
3.33 in; 6-7.5 pt text), saved as 300-dpi PNG plus PDF with TrueType fonts, and every figure
gets ``figures/<name>.md`` with each plotted value (``write_data``), generated from the same
frames as the figure. Identity is never colour-alone (marker shape + legend or direct label).
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SHADE = "#f2f1ed"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (validated all-pairs)
NEUTRAL = "#8a8983"
FULL_W = 7.0   # ACM \textwidth, inches
LABEL_PT = 5.8  # direct labels
MODEL_NAME = {"A": "Model A: text-embedding-3-large", "B": "Model B: Qwen3-Embedding-0.6B"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.6, "axes.labelcolor": INK_2,
        "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.titleweight": "bold",
        "axes.titlepad": 4, "legend.frameon": False, "lines.linewidth": 1.3,
        "lines.markersize": 4.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 100, "savefig.dpi": 300,
    })
    return plt


def _save(fig, name: str, out_dir="figures") -> Path:
    out = Path(out_dir) / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


#: Labels sit on a near-opaque surface-coloured box so error bars and grid lines behind them
#: do not cut through the text.
_LABEL_BOX = {"boxstyle": "round,pad=0.12", "fc": SURFACE, "ec": "none", "alpha": 0.85}


def _write_data(name: str, title: str, intro: str, sections, sources, out_dir="figures") -> Path:
    """``figures/<name>.md``: every value the figure draws, as Markdown tables. ``sections`` is
    a list of (heading, note, DataFrame); written from the same frames the figure is drawn
    from, so the data file and the figure cannot disagree."""
    lines = [f"# {title}: data", "", intro, "",
             f"Figure files: `figures/{name}.png`, `figures/{name}.pdf`. Generated with the "
             "figure by `nestbench.analysis` (`make analysis`); do not edit by "
             "hand. Sources: " + ", ".join(f"`{x}`" for x in sources) + ".", ""]
    for heading, note, df in sections:
        lines += [f"## {heading}", ""]
        if note:
            lines += [note, ""]
        # missing values (e.g. k′ for arms without a re-rank) print as an em dash
        t = df.astype(object).where(df.notna(), None)
        lines += [t.to_markdown(index=False, floatfmt=".6g", missingval="—") if len(df)
                  else "*(none)*", ""]
    out = Path(out_dir) / f"{name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def _add_leader(ax, x, y, dx: float, dy: float) -> None:
    """A thin leader line for a label pushed away from its point (|dy| >= 8 pt), so a label in
    a crowded spot is never read as belonging to a neighbour. Drawn after placement, as its
    own artist, so the collision test measures the text alone."""
    if abs(dy) < 8:
        return
    ax.annotate("", (x, y), xytext=(dx, dy + (2 if dy > 0 else 3)), textcoords="offset points",
                zorder=3.5, arrowprops={"arrowstyle": "-", "color": NEUTRAL, "lw": 0.5,
                                        "shrinkA": 0, "shrinkB": 3})


def place_labels(ax, pts, fontsize=LABEL_PT, extra_marks=(), mark_radius_pt: float = 3.0):
    """Direct-label points without overlaps: try offsets around each point in order and keep
    the first whose text box clears every box already placed and every marker. A marker is a
    square of ``mark_radius_pt`` around each labelled point and each ``extra_marks`` (x, y),
    so a label box never covers a plotted point, labelled or not."""
    from matplotlib.transforms import Bbox

    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # Text already on the axes (reference-line labels, notes) and the legend are obstacles too.
    placed = [t.get_window_extent(r).expanded(1.05, 1.15) for t in ax.texts if t.get_text()]
    if ax.get_legend() is not None:
        placed.append(ax.get_legend().get_window_extent(r))
    rad = mark_radius_pt * fig.dpi / 72
    marks = [ax.transData.transform((x, y)) for x, y, _ in pts]
    marks += [ax.transData.transform(m) for m in extra_marks]
    offsets = [(4, -2), (4, 4), (4, -9), (-4, 4), (-4, -9), (-4, -2), (4, 12), (4, -16),
               (-4, 12), (-4, -16), (4, 20), (4, -24), (-4, 20), (-4, -24), (4, 28), (-4, 28)]
    boxes = [Bbox.from_extents(mx - rad, my - rad, mx + rad, my + rad) for mx, my in marks]
    frame = ax.get_window_extent(r)
    for i, (x, y, text) in enumerate(pts):
        best = None  # (collisions, offset) of the least-bad position, if none is clear
        for dx, dy in offsets:
            ha = "left" if dx > 0 else "right"
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, color=INK_2, ha=ha, zorder=4, bbox=_LABEL_BOX)
            bb = t.get_window_extent(r).expanded(1.05, 1.15)
            # covering a marker is worse than touching a label; a label's own marker is exempt
            n = sum(bb.overlaps(b) for b in placed) + 2 * sum(
                bb.overlaps(m) for j, m in enumerate(boxes) if j != i)
            # a label must stay inside its panel
            n += 3 * (bb.x0 < frame.x0 or bb.x1 > frame.x1 or bb.y0 < frame.y0
                      or bb.y1 > frame.y1)
            if n == 0:
                placed.append(bb)
                best = None
                _add_leader(ax, x, y, dx, dy)
                break
            if best is None or n < best[0]:
                best = (n, (dx, dy))
            t.remove()
        else:  # nothing clear: use the least-bad position rather than drop the label
            dx, dy = best[1]
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, color=INK_2, ha="left" if dx > 0 else "right",
                            zorder=4, bbox=_LABEL_BOX)
            placed.append(t.get_window_extent(r))
            _add_leader(ax, x, y, dx, dy)


def byte_axis(ax, lo: float = 420, hi: float = 9500):
    ticks = [t for t in (256, 512, 1024, 2048, 4096, 8192) if lo <= t <= hi] or [512, 8192]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t / 1024:g}K" for t in ticks])
    ax.minorticks_off()


# ------------------------------------------------------------------ Stage E

# public names
style = _style
save = _save
write_data = _write_data
