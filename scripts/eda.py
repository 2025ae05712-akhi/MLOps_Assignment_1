"""Exploratory data analysis: summary statistics and figures for the report."""

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CLEAN_CSV, FIGURES_DIR, NUMERIC_FEATURES, REPORTS_DIR, TARGET
from src.data import load_dataset, save_clean


def write_summary(df) -> None:
    """Write shape, missing values, class balance and descriptive statistics to text."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    counts = df[TARGET].value_counts().sort_index()
    lines = [
        f"Rows: {len(df)}  Columns: {df.shape[1]}",
        "",
        "Missing values per column:",
        df.isna().sum().to_string(),
        "",
        "Class balance (target):",
        counts.to_string(),
        f"Positive class share: {counts.get(1, 0) / len(df):.3f}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        "",
        "Descriptive statistics:",
        df.describe().T.round(2).to_string(),
    ]
    (REPORTS_DIR / "eda_summary.txt").write_text("\n".join(lines))


def plot_figures(df) -> None:
    """Save histograms by class, class balance and correlation heatmap."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.flat, NUMERIC_FEATURES, strict=False):
        sns.histplot(data=df, x=feature, hue=TARGET, kde=True, ax=ax, palette="Set1")
        ax.set_title(f"{feature} by target")
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "histograms_by_target.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET, hue=TARGET, ax=ax, palette="Set2", legend=False)
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_dataset()
    save_clean(df, CLEAN_CSV)
    write_summary(df)
    plot_figures(df)
    print(f"Wrote cleaned data to {CLEAN_CSV}, figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
