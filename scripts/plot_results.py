"""Generate evaluation plots from a results CSV."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib
import pandas as pd

try:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc


GRADE_COLUMNS = {
    "original": ["text_grade_level", "text_grade"],
    "lexical": ["text_simplified_grade_level", "text_simplified_grade"],
    "syntactic": [
        "syntactic_simplified_grade_level",
        "syntactic_simplified_grade",
        "text_syntactic_simplified_grade",
    ],
}

READING_EASE_COLUMNS = {
    "original": ["text_reading_ease"],
    "lexical": ["text_simplified_reading_ease"],
    "syntactic": ["syntactic_simplified_reading_ease"],
}

COMPLEXITY_COLUMNS = {
    "judge": ["text_vs_text_simplified_complexity_result"],
}

COMPLEXITY_PATTERN = re.compile(
    r"text_A_complexity=(?P<a>-?\d+(?:\.\d+)?)"
    r".*?text_B_complexity=(?P<b>-?\d+(?:\.\d+)?)",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grade/readability/complexity plots from a results CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Path to a results CSV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save generated plots. Defaults to results/plots/<csv_stem>/",
    )
    parser.add_argument(
        "--labels",
        choices=["index", "title"],
        default="index",
        help="How to label articles on the x-axis.",
    )
    parser.add_argument(
        "--grade-mode",
        choices=["normalized", "absolute"],
        default="normalized",
        help="Plot grade as percent change from original, or absolute scores.",
    )
    parser.add_argument(
        "--ease-mode",
        choices=["normalized", "absolute"],
        default="normalized",
        help="Plot reading ease as percent change from original, or absolute scores.",
    )
    return parser.parse_args()


def first_present_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Missing column for {label}. Expected one of: {', '.join(candidates)}")


def shorten_title(title: object, fallback: str, max_len: int = 42) -> str:
    if not isinstance(title, str) or not title.strip():
        return fallback
    clean = " ".join(title.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3].rstrip() + "..."


def article_labels(df: pd.DataFrame, mode: str) -> list[str]:
    if mode == "title" and "title" in df.columns:
        return [
            shorten_title(title, fallback=f"Article {i + 1}")
            for i, title in enumerate(df["title"].tolist())
        ]
    return [f"A{i + 1}" for i in range(len(df))]


def safe_relative_index(current: pd.Series, baseline: pd.Series) -> pd.Series:
    denominator = baseline.where(baseline != 0)
    return (current / denominator) * 100.0


def parse_complexity_scores(value: object) -> tuple[float, float]:
    if not isinstance(value, str):
        return math.nan, math.nan
    match = COMPLEXITY_PATTERN.search(value)
    if not match:
        return math.nan, math.nan
    return float(match.group("a")), float(match.group("b"))


def prepare_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = {}
    for metric_group, mapping in (
        ("grade", GRADE_COLUMNS),
        ("ease", READING_EASE_COLUMNS),
        ("complexity", COMPLEXITY_COLUMNS),
    ):
        for key, candidates in mapping.items():
            metric_columns[f"{metric_group}_{key}"] = first_present_column(
                df,
                candidates,
                f"{metric_group}_{key}",
            )

    prepared = df.copy()
    numeric_sources = {
        "grade_original": metric_columns["grade_original"],
        "grade_lexical": metric_columns["grade_lexical"],
        "grade_syntactic": metric_columns["grade_syntactic"],
        "ease_original": metric_columns["ease_original"],
        "ease_lexical": metric_columns["ease_lexical"],
        "ease_syntactic": metric_columns["ease_syntactic"],
    }
    for target, source in numeric_sources.items():
        prepared[target] = pd.to_numeric(prepared[source], errors="coerce")

    complexity_scores = prepared[metric_columns["complexity_judge"]].apply(parse_complexity_scores)
    prepared["complexity_text_a"] = complexity_scores.apply(lambda pair: pair[0])
    prepared["complexity_text_b"] = complexity_scores.apply(lambda pair: pair[1])
    return prepared


def plot_line_chart(
    labels: list[str],
    series_map: dict[str, pd.Series],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    valid_mask = pd.Series(True, index=series_map[next(iter(series_map))].index)
    for series in series_map.values():
        valid_mask &= series.notna()

    filtered_labels = [label for label, keep in zip(labels, valid_mask.tolist()) if keep]
    if not filtered_labels:
        raise ValueError(f"No complete rows available for plot: {title}")

    filtered_series = {name: series[valid_mask].reset_index(drop=True) for name, series in series_map.items()}
    positions = list(range(len(filtered_labels)))

    fig_width = max(12, len(filtered_labels) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    colors = ["#355070", "#6d597a", "#b56576"]
    markers = ["o", "s", "^"]

    for idx, (name, series) in enumerate(filtered_series.items()):
        ax.plot(
            positions,
            series.tolist(),
            label=name,
            color=colors[idx],
            marker=markers[idx],
            linewidth=2.0,
            markersize=5,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(filtered_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    csv_path = args.csv_path.resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else csv_path.parent / "plots" / csv_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    prepared = prepare_metrics(df)
    labels = article_labels(prepared, args.labels)

    if args.grade_mode == "normalized":
        grade_series = {
            "Original": pd.Series(100.0, index=prepared.index),
            "Lexical Simplified": safe_relative_index(
                prepared["grade_lexical"], prepared["grade_original"]
            ),
            "Syntactic Simplified": safe_relative_index(
                prepared["grade_syntactic"], prepared["grade_original"]
            ),
        }
        grade_title = "Normalized Grade Level per Article"
        grade_ylabel = "Normalized Grade Level (Original = 100)"
    else:
        grade_series = {
            "Original": prepared["grade_original"],
            "Lexical Simplified": prepared["grade_lexical"],
            "Syntactic Simplified": prepared["grade_syntactic"],
        }
        grade_title = "Grade Level per Article"
        grade_ylabel = "Flesch-Kincaid Grade Level"

    if args.ease_mode == "normalized":
        ease_series = {
            "Original": pd.Series(100.0, index=prepared.index),
            "Lexical Simplified": safe_relative_index(
                prepared["ease_lexical"], prepared["ease_original"]
            ),
            "Syntactic Simplified": safe_relative_index(
                prepared["ease_syntactic"], prepared["ease_original"]
            ),
        }
        ease_title = "Normalized Reading Ease per Article"
        ease_ylabel = "Normalized Reading Ease (Original = 100)"
    else:
        ease_series = {
            "Original": prepared["ease_original"],
            "Lexical Simplified": prepared["ease_lexical"],
            "Syntactic Simplified": prepared["ease_syntactic"],
        }
        ease_title = "Reading Ease per Article"
        ease_ylabel = "Flesch Reading Ease"

    complexity_series = {
        "Text A": prepared["complexity_text_a"],
        "Text B": prepared["complexity_text_b"],
    }

    plot_line_chart(
        labels,
        grade_series,
        title=grade_title,
        ylabel=grade_ylabel,
        output_path=output_dir / "01_grade_levels.png",
    )
    plot_line_chart(
        labels,
        ease_series,
        title=ease_title,
        ylabel=ease_ylabel,
        output_path=output_dir / "02_reading_ease.png",
    )
    plot_line_chart(
        labels,
        complexity_series,
        title="Text A vs Text B Complexity Scores",
        ylabel="LLM Judge Complexity Score",
        output_path=output_dir / "03_complexity_scores.png",
    )

    print(f"Saved plots to: {output_dir}")
    for path in sorted(output_dir.glob("*.png")):
        print(path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
