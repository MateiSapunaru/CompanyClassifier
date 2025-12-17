import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUTS_DIR = Path("outputs")

FILES = {
    "v1_bad_thresholds": OUTPUTS_DIR / "predictions_v1_bad_thresholds.csv",
    "v2_good_thresholds": OUTPUTS_DIR / "predictions_v2_good_thresholds.csv",
    "v3_enriched": OUTPUTS_DIR / "predictions_v3_enriched.csv",
    "v4_enriched_weighted": OUTPUTS_DIR / "predictions_v4_enriched_weighted.csv",
    "v4.1_enriched_weighted_stricter": OUTPUTS_DIR / "predictions_v4_1_weighted_stricter.csv",
}

def load_and_summarize(path: Path) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(path)

    # insurance_label este string cu labels separate prin "; "
    n_labels = df["insurance_label"].fillna("").astype(str).apply(
        lambda x: 0 if x.strip() == "" else len(x.split("; "))
    )

    summary = {
        "rows": len(df),
        "uncertain_pct": (df["pred_status"] == "uncertain").mean() * 100,
        "top1_mean": df["top1_score"].mean(),
        "top1_median": df["top1_score"].median(),
        "gap_mean": df["gap_top1_top2"].mean(),
        "avg_labels": n_labels.mean(),
        "pct_multilabel": (n_labels >= 2).mean() * 100,
    }
    return summary, df

def main():
    summaries = {}
    dfs = {}

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        summary, df = load_and_summarize(path)
        summaries[name] = summary
        dfs[name] = df

    # 1) tabel comparativ
    summary_df = pd.DataFrame(summaries).T
    print("\n=== COMPARISON SUMMARY (v1 vs v2 vs v3) ===")
    print(summary_df.sort_index())

    # 2) histograme Top-1
    plt.figure(figsize=(10, 6))
    for name, df in dfs.items():
        plt.hist(df["top1_score"], bins=50, alpha=0.5, label=name)
    plt.title("Top-1 Similarity Score Distribution")
    plt.xlabel("Top-1 Score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3) histograme Gap
    plt.figure(figsize=(10, 6))
    for name, df in dfs.items():
        plt.hist(df["gap_top1_top2"], bins=50, alpha=0.5, label=name)
    plt.title("Gap (Top-1 - Top-2) Distribution")
    plt.xlabel("Gap")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4) status bar (confident/uncertain)
    status_df = pd.DataFrame({
        name: df["pred_status"].value_counts(normalize=True) * 100
        for name, df in dfs.items()
    }).fillna(0).T

    status_df.plot(kind="bar", stacked=True, figsize=(9, 5), title="Prediction Status Distribution (%)")
    plt.ylabel("Percentage")
    plt.grid(axis="y")
    plt.show()

if __name__ == "__main__":
    main()
