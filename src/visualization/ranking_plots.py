"""
Ranking visualization module for the Diversity Decoding Arena.

Provides visualizations for algorithm ranking analysis including:
- Bradley-Terry ranking with confidence intervals
- Rank stability plots (across bootstrap resamples)
- Win-rate matrices
- Tournament bracket visualization

All visualizations produce SVG strings using a lightweight SVG builder,
with numpy for numerical computation.
"""

from __future__ import annotations

import math
import html as html_module
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RankingPlotConfig:
    """Configuration for ranking plots."""
    figsize: Tuple[int, int] = (800, 500)
    font_size: int = 13
    title: str = "Ranking Plot"
    grid: bool = True
    margin_left: int = 140
    margin_right: int = 40
    margin_top: int = 50
    margin_bottom: int = 55
    bar_height: int = 22
    bar_gap: int = 8

    @property
    def plot_width(self) -> int:
        return self.figsize[0] - self.margin_left - self.margin_right

    @property
    def plot_height(self) -> int:
        return self.figsize[1] - self.margin_top - self.margin_bottom


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_COLORS: List[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_MEDAL_COLORS = ["#FFD700", "#C0C0C0", "#CD7F32"]  # gold, silver, bronze


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

_SVG_HEADER = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'width="{w}" height="{h}" '
    'viewBox="0 0 {w} {h}" style="background:#ffffff;">'
)
_SVG_FOOTER = "</svg>"


def _svg_text(
    x: float, y: float, text: str, *,
    size: int = 12, anchor: str = "start",
    weight: str = "normal", rotate: Optional[float] = None,
    fill: str = "#333",
) -> str:
    escaped = html_module.escape(str(text))
    rot = f' transform="rotate({rotate},{x},{y})"' if rotate else ""
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}"{rot}>'
        f'{escaped}</text>'
    )


def _empty_svg(message: str) -> str:
    escaped = html_module.escape(message)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="80">'
        f'<text x="200" y="45" text-anchor="middle" font-size="14" '
        f'fill="#888">{escaped}</text></svg>'
    )


# ---------------------------------------------------------------------------
# Bradley-Terry model helpers
# ---------------------------------------------------------------------------

def _fit_bradley_terry(
    win_matrix: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Fit Bradley-Terry model parameters using iterative procedure.

    Parameters
    ----------
    win_matrix : n×n matrix where w[i,j] = number of times i beat j

    Returns
    -------
    Strength parameters (log-scale), normalised so sum = 0.
    """
    n = win_matrix.shape[0]
    pi = np.ones(n)

    for _ in range(max_iter):
        pi_old = pi.copy()
        for i in range(n):
            num = 0.0
            den = 0.0
            for j in range(n):
                if i == j:
                    continue
                n_ij = win_matrix[i, j] + win_matrix[j, i]
                if n_ij == 0:
                    continue
                num += win_matrix[i, j]
                den += n_ij / (pi[i] + pi[j])
            if den > 0:
                pi[i] = num / den
            else:
                pi[i] = 1.0

        # normalise
        pi = pi / pi.sum() * n
        if np.max(np.abs(pi - pi_old)) < tol:
            break

    # return log strengths centered at 0
    log_pi = np.log(np.maximum(pi, 1e-10))
    return log_pi - log_pi.mean()


def _bootstrap_bradley_terry(
    win_matrix: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap confidence intervals for Bradley-Terry parameters.

    Returns (mean_strengths, std_strengths).
    """
    rng = np.random.RandomState(seed)
    n = win_matrix.shape[0]
    strengths = np.zeros((n_bootstrap, n))

    for b in range(n_bootstrap):
        # resample matches
        boot_w = np.zeros_like(win_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                total = int(win_matrix[i, j] + win_matrix[j, i])
                if total == 0:
                    continue
                p = win_matrix[i, j] / total if total > 0 else 0.5
                wins_i = rng.binomial(total, p)
                boot_w[i, j] = wins_i
                boot_w[j, i] = total - wins_i
        strengths[b] = _fit_bradley_terry(boot_w)

    return np.mean(strengths, axis=0), np.std(strengths, axis=0)


# ---------------------------------------------------------------------------
# RankingPlotter
# ---------------------------------------------------------------------------


class RankingPlotter:
    """Generate ranking visualizations as SVG strings."""

    def __init__(self, config: Optional[RankingPlotConfig] = None) -> None:
        self.config = config or RankingPlotConfig()

    def plot_bradley_terry_ranking(
        self,
        win_matrix: np.ndarray,
        algorithm_names: List[str],
        title: str = "Bradley-Terry Algorithm Ranking",
        n_bootstrap: int = 200,
    ) -> str:
        """Horizontal bar chart of Bradley-Terry strengths with CIs.

        Parameters
        ----------
        win_matrix : n×n matrix where w[i,j] = number of times i beat j
        algorithm_names : list of algorithm names
        """
        cfg = self.config
        n = win_matrix.shape[0]
        if n == 0:
            return _empty_svg("No algorithms")

        mean_s, std_s = _bootstrap_bradley_terry(win_matrix, n_bootstrap)
        order = np.argsort(mean_s)[::-1]  # descending

        total_bar_h = n * (cfg.bar_height + cfg.bar_gap)
        h = max(cfg.figsize[1], cfg.margin_top + total_bar_h + cfg.margin_bottom)
        w = cfg.figsize[0]

        parts: List[str] = [_SVG_HEADER.format(w=w, h=h)]
        parts.append(_svg_text(w / 2, 26, title,
                               size=cfg.font_size + 2, anchor="middle", weight="bold"))

        s_min = float(np.min(mean_s - 2 * std_s))
        s_max = float(np.max(mean_s + 2 * std_s))
        pad = (s_max - s_min) * 0.1 if s_max != s_min else 0.5
        s_min -= pad
        s_max += pad
        s_rng = s_max - s_min if s_max != s_min else 1.0

        ml = cfg.margin_left
        pw = cfg.plot_width

        # draw zero line
        zero_x = ml + (0 - s_min) / s_rng * pw
        parts.append(
            f'<line x1="{zero_x}" y1="{cfg.margin_top}" '
            f'x2="{zero_x}" y2="{cfg.margin_top + total_bar_h}" '
            f'stroke="#aaa" stroke-width="1" stroke-dasharray="4,3"/>'
        )

        for rank, idx in enumerate(order):
            by = cfg.margin_top + rank * (cfg.bar_height + cfg.bar_gap)
            val = mean_s[idx]
            sd = std_s[idx]
            name = algorithm_names[idx]
            color = _COLORS[rank % len(_COLORS)]

            # CI line
            ci_lo_x = ml + (val - 2 * sd - s_min) / s_rng * pw
            ci_hi_x = ml + (val + 2 * sd - s_min) / s_rng * pw
            bar_cy = by + cfg.bar_height / 2
            parts.append(
                f'<line x1="{ci_lo_x}" y1="{bar_cy}" x2="{ci_hi_x}" y2="{bar_cy}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
            # CI caps
            cap_h = cfg.bar_height * 0.4
            for cap_x in [ci_lo_x, ci_hi_x]:
                parts.append(
                    f'<line x1="{cap_x}" y1="{bar_cy - cap_h}" '
                    f'x2="{cap_x}" y2="{bar_cy + cap_h}" '
                    f'stroke="{color}" stroke-width="2"/>'
                )

            # point estimate
            pt_x = ml + (val - s_min) / s_rng * pw
            parts.append(
                f'<circle cx="{pt_x}" cy="{bar_cy}" r="5" '
                f'fill="{color}" stroke="white" stroke-width="1.5"/>'
            )

            # label
            parts.append(_svg_text(
                ml - 8, bar_cy + 4, f"#{rank+1} {name}",
                size=cfg.font_size - 1, anchor="end",
                weight="bold" if rank < 3 else "normal",
            ))

            # value
            parts.append(_svg_text(
                ci_hi_x + 8, bar_cy + 4, f"{val:.3f}",
                size=cfg.font_size - 2, anchor="start",
            ))

            # medal for top 3
            if rank < 3:
                parts.append(
                    f'<circle cx="{ml - 40}" cy="{bar_cy}" r="8" '
                    f'fill="{_MEDAL_COLORS[rank]}" stroke="#666" stroke-width="0.5"/>'
                )

        # x-axis ticks
        for i in range(5):
            v = s_min + s_rng * i / 4
            tx = ml + pw * i / 4
            parts.append(_svg_text(tx, cfg.margin_top + total_bar_h + 18,
                                   f"{v:.2f}", size=cfg.font_size - 3,
                                   anchor="middle"))

        parts.append(_svg_text(
            ml + pw / 2, h - 10,
            "Bradley-Terry Strength (log scale)",
            size=cfg.font_size - 1, anchor="middle",
        ))

        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def plot_rank_stability(
        self,
        win_matrix: np.ndarray,
        algorithm_names: List[str],
        n_bootstrap: int = 200,
        title: str = "Rank Stability Across Bootstrap Samples",
        seed: int = 42,
    ) -> str:
        """Bump chart showing how ranks vary across bootstrap samples."""
        cfg = self.config
        n = win_matrix.shape[0]
        if n == 0:
            return _empty_svg("No algorithms")

        rng_gen = np.random.RandomState(seed)
        n_show = min(n_bootstrap, 50)  # show up to 50 bootstrap samples

        all_ranks = np.zeros((n_show, n), dtype=int)
        for b in range(n_show):
            boot_w = np.zeros_like(win_matrix)
            for i in range(n):
                for j in range(i + 1, n):
                    total = int(win_matrix[i, j] + win_matrix[j, i])
                    if total == 0:
                        continue
                    p = win_matrix[i, j] / total if total > 0 else 0.5
                    wins_i = rng_gen.binomial(total, p)
                    boot_w[i, j] = wins_i
                    boot_w[j, i] = total - wins_i
            strengths = _fit_bradley_terry(boot_w)
            ranks = np.argsort(np.argsort(-strengths))  # 0 = best
            all_ranks[b] = ranks

        total_h = max(cfg.figsize[1], cfg.margin_top + n * 30 + cfg.margin_bottom)
        w = cfg.figsize[0]
        parts: List[str] = [_SVG_HEADER.format(w=w, h=total_h)]
        parts.append(_svg_text(w / 2, 26, title,
                               size=cfg.font_size + 2, anchor="middle", weight="bold"))

        ml = cfg.margin_left
        pw = cfg.plot_width
        plot_h = total_h - cfg.margin_top - cfg.margin_bottom

        # draw grid
        for r in range(n):
            gy = cfg.margin_top + (r + 0.5) / n * plot_h
            parts.append(
                f'<line x1="{ml}" y1="{gy}" x2="{ml+pw}" y2="{gy}" '
                f'stroke="#f0f0f0" stroke-width="0.5"/>'
            )
            parts.append(_svg_text(ml + pw + 8, gy + 4, f"#{r+1}",
                                   size=cfg.font_size - 3, anchor="start",
                                   fill="#999"))

        for ai in range(n):
            color = _COLORS[ai % len(_COLORS)]
            # draw lines between consecutive bootstrap samples
            for b in range(n_show - 1):
                r1 = all_ranks[b, ai]
                r2 = all_ranks[b + 1, ai]
                x1 = ml + b / max(n_show - 1, 1) * pw
                x2 = ml + (b + 1) / max(n_show - 1, 1) * pw
                y1 = cfg.margin_top + (r1 + 0.5) / n * plot_h
                y2 = cfg.margin_top + (r2 + 0.5) / n * plot_h
                parts.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="{color}" stroke-width="1.2" opacity="0.4"/>'
                )

            # algorithm label at left
            median_rank = int(np.median(all_ranks[:, ai]))
            ly = cfg.margin_top + (median_rank + 0.5) / n * plot_h
            parts.append(_svg_text(ml - 8, ly + 4, algorithm_names[ai],
                                   size=cfg.font_size - 2, anchor="end",
                                   fill=color, weight="bold"))

        parts.append(_svg_text(ml + pw / 2, total_h - 10,
                               "Bootstrap Sample Index",
                               size=cfg.font_size - 1, anchor="middle"))

        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def plot_win_rate_matrix(
        self,
        win_matrix: np.ndarray,
        algorithm_names: List[str],
        title: str = "Win Rate Matrix",
    ) -> str:
        """Heatmap of pairwise win rates between algorithms."""
        cfg = self.config
        n = win_matrix.shape[0]
        if n == 0:
            return _empty_svg("No algorithms")

        # compute win rates
        win_rates = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                total = win_matrix[i, j] + win_matrix[j, i]
                if total > 0:
                    win_rates[i, j] = win_matrix[i, j] / total
                else:
                    win_rates[i, j] = 0.5 if i != j else 0.0

        # order by total win rate
        total_wr = win_rates.sum(axis=1)
        order = np.argsort(-total_wr)
        win_rates = win_rates[np.ix_(order, order)]
        names_ordered = [algorithm_names[i] for i in order]

        label_margin = 130
        cell_size = min(45, (cfg.figsize[0] - label_margin - 60) / max(n, 1))
        cell_size = max(cell_size, 20)
        total_w = int(label_margin + cell_size * n + 60)
        total_h = int(label_margin + cell_size * n + 40)

        parts: List[str] = [_SVG_HEADER.format(w=total_w, h=total_h)]
        parts.append(_svg_text(total_w / 2, 24, title,
                               size=cfg.font_size + 2, anchor="middle", weight="bold"))

        top_off = label_margin
        for i in range(n):
            for j in range(n):
                if i == j:
                    color = "#f5f5f5"
                    val_str = "—"
                else:
                    val = win_rates[i, j]
                    # green for >0.5, red for <0.5
                    if val >= 0.5:
                        t = (val - 0.5) * 2
                        r = int(255 * (1 - t * 0.6))
                        g = int(180 + 75 * t)
                        b_c = int(255 * (1 - t * 0.6))
                    else:
                        t = (0.5 - val) * 2
                        r = int(180 + 75 * t)
                        g = int(255 * (1 - t * 0.6))
                        b_c = int(255 * (1 - t * 0.6))
                    color = f"#{r:02x}{min(g,255):02x}{b_c:02x}"
                    val_str = f"{val:.2f}"

                cx = label_margin + j * cell_size
                cy = top_off + i * cell_size
                parts.append(
                    f'<rect x="{cx}" y="{cy}" width="{cell_size}" '
                    f'height="{cell_size}" fill="{color}" '
                    f'stroke="#ddd" stroke-width="0.5"/>'
                )
                parts.append(_svg_text(
                    cx + cell_size / 2, cy + cell_size / 2 + 4,
                    val_str, size=max(8, cfg.font_size - 3), anchor="middle",
                    weight="bold" if i != j and win_rates[i, j] >= 0.7 else "normal",
                ))

        for i, nm in enumerate(names_ordered):
            parts.append(_svg_text(
                label_margin - 6, top_off + i * cell_size + cell_size / 2 + 4,
                nm, size=max(8, cfg.font_size - 2), anchor="end",
            ))
            parts.append(_svg_text(
                label_margin + i * cell_size + cell_size / 2,
                top_off - 6, nm, size=max(8, cfg.font_size - 2),
                anchor="end", rotate=-45,
            ))

        # color legend
        leg_y = top_off + cell_size * n + 15
        parts.append(_svg_text(label_margin, leg_y,
                               "Win rate: ",
                               size=cfg.font_size - 2, anchor="start"))
        for vi, (lbl, col) in enumerate([
            ("0.0", "#d43d3d"), ("0.5", "#ffffff"), ("1.0", "#2d8c2d"),
        ]):
            lx = label_margin + 70 + vi * 55
            parts.append(
                f'<rect x="{lx}" y="{leg_y-10}" width="14" height="14" '
                f'fill="{col}" stroke="#ccc" stroke-width="0.5"/>'
            )
            parts.append(_svg_text(lx + 18, leg_y + 2, lbl,
                                   size=cfg.font_size - 3))

        parts.append(_SVG_FOOTER)
        return "".join(parts)

    def plot_tournament_bracket(
        self,
        win_matrix: np.ndarray,
        algorithm_names: List[str],
        title: str = "Tournament Bracket",
    ) -> str:
        """Single-elimination tournament bracket based on seeded matchups.

        Seeds are determined by total win count. Each round, the highest
        seed plays the lowest remaining seed.
        """
        cfg = self.config
        n = win_matrix.shape[0]
        if n < 2:
            return _empty_svg("Need at least 2 algorithms")

        # seed by total wins
        total_wins = win_matrix.sum(axis=1)
        seeds = np.argsort(-total_wins).tolist()

        # pad to power of 2
        bracket_size = 1
        while bracket_size < n:
            bracket_size *= 2
        # fill with None for byes
        seeded: List[Optional[int]] = []
        for s in seeds:
            seeded.append(s)
        while len(seeded) < bracket_size:
            seeded.append(None)

        # run tournament rounds
        n_rounds = int(math.log2(bracket_size))
        rounds: List[List[Tuple[Optional[str], Optional[str], Optional[str]]]] = []
        current = list(seeded)

        for rd in range(n_rounds):
            matches = []
            next_round: List[Optional[int]] = []
            for m in range(0, len(current), 2):
                a = current[m]
                b = current[m + 1] if m + 1 < len(current) else None
                if a is None and b is None:
                    next_round.append(None)
                    matches.append((None, None, None))
                elif a is None:
                    next_round.append(b)
                    name_b = algorithm_names[b] if b is not None else "BYE"
                    matches.append(("BYE", name_b, name_b))
                elif b is None:
                    next_round.append(a)
                    name_a = algorithm_names[a]
                    matches.append((name_a, "BYE", name_a))
                else:
                    name_a = algorithm_names[a]
                    name_b = algorithm_names[b]
                    if win_matrix[a, b] >= win_matrix[b, a]:
                        winner = a
                    else:
                        winner = b
                    matches.append((name_a, name_b, algorithm_names[winner]))
                    next_round.append(winner)
            rounds.append(matches)
            current = next_round

        # draw bracket
        match_w = 140
        match_h = 50
        round_gap = 40
        total_w = cfg.margin_left + (n_rounds + 1) * (match_w + round_gap) + 60
        col0_h = bracket_size // 2 * (match_h + 20)
        total_h = max(cfg.figsize[1], cfg.margin_top + col0_h + cfg.margin_bottom)

        parts: List[str] = [_SVG_HEADER.format(w=total_w, h=total_h)]
        parts.append(_svg_text(total_w / 2, 26, title,
                               size=cfg.font_size + 2, anchor="middle", weight="bold"))

        for rd_idx, matches in enumerate(rounds):
            rx = cfg.margin_left + rd_idx * (match_w + round_gap)
            n_matches = len(matches)
            spacing = col0_h / max(n_matches, 1)

            for mi, (a_name, b_name, winner_name) in enumerate(matches):
                if a_name is None and b_name is None:
                    continue
                my = cfg.margin_top + mi * spacing + spacing / 2 - match_h / 2

                # match box
                parts.append(
                    f'<rect x="{rx}" y="{my}" width="{match_w}" '
                    f'height="{match_h}" fill="white" stroke="#ccc" '
                    f'stroke-width="1" rx="4"/>'
                )

                # top player
                top_name = a_name or "—"
                is_top_winner = (top_name == winner_name and winner_name is not None)
                parts.append(_svg_text(
                    rx + 8, my + 18, top_name,
                    size=cfg.font_size - 2,
                    weight="bold" if is_top_winner else "normal",
                    fill="#2e7d32" if is_top_winner else "#333",
                ))

                # divider
                parts.append(
                    f'<line x1="{rx+4}" y1="{my+match_h/2}" '
                    f'x2="{rx+match_w-4}" y2="{my+match_h/2}" '
                    f'stroke="#eee" stroke-width="1"/>'
                )

                # bottom player
                bot_name = b_name or "—"
                is_bot_winner = (bot_name == winner_name and winner_name is not None)
                parts.append(_svg_text(
                    rx + 8, my + match_h - 8, bot_name,
                    size=cfg.font_size - 2,
                    weight="bold" if is_bot_winner else "normal",
                    fill="#2e7d32" if is_bot_winner else "#333",
                ))

                # connector line to next round
                if rd_idx < n_rounds - 1:
                    nx = rx + match_w
                    ny = my + match_h / 2
                    parts.append(
                        f'<line x1="{nx}" y1="{ny}" x2="{nx+round_gap/2}" '
                        f'y2="{ny}" stroke="#aaa" stroke-width="1.5"/>'
                    )

        # champion box
        if rounds and rounds[-1]:
            _, _, champion = rounds[-1][0]
            if champion:
                champ_x = cfg.margin_left + n_rounds * (match_w + round_gap)
                champ_y = cfg.margin_top + col0_h / 2 - 20
                parts.append(
                    f'<rect x="{champ_x}" y="{champ_y}" width="{match_w}" '
                    f'height="40" fill="#FFF9C4" stroke="#FFD700" '
                    f'stroke-width="2" rx="6"/>'
                )
                parts.append(_svg_text(
                    champ_x + match_w / 2, champ_y + 15, "🏆 Champion",
                    size=cfg.font_size - 3, anchor="middle", fill="#666",
                ))
                parts.append(_svg_text(
                    champ_x + match_w / 2, champ_y + 32, champion,
                    size=cfg.font_size, anchor="middle", weight="bold",
                    fill="#1a1a2e",
                ))

        # round labels
        for rd_idx in range(n_rounds):
            rx = cfg.margin_left + rd_idx * (match_w + round_gap) + match_w / 2
            label = f"Round {rd_idx + 1}" if rd_idx < n_rounds - 1 else "Final"
            parts.append(_svg_text(rx, cfg.margin_top - 12, label,
                                   size=cfg.font_size - 2, anchor="middle",
                                   fill="#666"))

        parts.append(_SVG_FOOTER)
        return "".join(parts)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_ranking_plotter(**kwargs: Any) -> RankingPlotter:
    """Create a RankingPlotter with optional config overrides."""
    return RankingPlotter(RankingPlotConfig(**kwargs))


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------


def _demo() -> None:  # pragma: no cover
    """Generate sample ranking visualizations."""
    rng = np.random.RandomState(42)
    names = ["BeamSearch", "TopK", "Nucleus", "MCTS", "DivBeam"]
    n = len(names)

    # synthetic win matrix
    win_matrix = np.zeros((n, n))
    strengths = np.array([0.8, 0.6, 0.5, 0.7, 0.4])
    for i in range(n):
        for j in range(i + 1, n):
            p = strengths[i] / (strengths[i] + strengths[j])
            n_matches = 100
            wins = rng.binomial(n_matches, p)
            win_matrix[i, j] = wins
            win_matrix[j, i] = n_matches - wins

    plotter = RankingPlotter()

    svg1 = plotter.plot_bradley_terry_ranking(win_matrix, names)
    print(f"Bradley-Terry SVG: {len(svg1)} chars")

    svg2 = plotter.plot_rank_stability(win_matrix, names, n_bootstrap=50)
    print(f"Rank stability SVG: {len(svg2)} chars")

    svg3 = plotter.plot_win_rate_matrix(win_matrix, names)
    print(f"Win rate matrix SVG: {len(svg3)} chars")

    svg4 = plotter.plot_tournament_bracket(win_matrix, names)
    print(f"Tournament bracket SVG: {len(svg4)} chars")

    print("\nAll ranking visualizations generated successfully.")


if __name__ == "__main__":
    _demo()
