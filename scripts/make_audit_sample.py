import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.io import save_audit_sample
from src.eval import create_audit_sample, prepare_audit_export

OUTPUTS_DIR = project_root / "outputs"
DEFAULT_PREDICTIONS_FILE = OUTPUTS_DIR / "predictions_baseline_tfidf.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Create stratified audit sample from predictions.")
    parser.add_argument(
        "--predictions_file",
        type=str,
        default=str(DEFAULT_PREDICTIONS_FILE),
        help="Calea către fișierul CSV cu predicții.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="baseline",
        help="Nume run folosit pentru numele fișierului de audit.",
    )
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--n_bins", type=int, default=3)
    parser.add_argument("--stratify_by", type=str, default="top1_score")
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def load_predictions(predictions_path: Path):
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    print(f"Loading predictions from: {predictions_path}")
    df = pd.read_csv(predictions_path, encoding="utf-8")
    print(f"✓ Loaded {len(df)} predictions")
    return df


def main():
    args = parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = Path(args.predictions_file)
    audit_output_file = OUTPUTS_DIR / f"audit_samples_{args.run_name}.csv"

    print("\n" + "=" * 70)
    print(" " * 18 + "CREATE AUDIT SAMPLE")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("STEP 1: Loading predictions")
    print("-" * 70)
    predictions_df = load_predictions(predictions_path)

    print("\n" + "-" * 70)
    print("STEP 2: Creating stratified sample")
    print("-" * 70)
    audit_sample = create_audit_sample(
        df=predictions_df,
        sample_size=args.sample_size,
        stratify_by=args.stratify_by,
        n_bins=args.n_bins,
        random_state=args.random_state,
    )

    print("\n" + "-" * 70)
    print("STEP 3: Preparing export")
    print("-" * 70)
    audit_export = prepare_audit_export(audit_sample)

    print("\n" + "-" * 70)
    print("STEP 4: Saving audit sample")
    print("-" * 70)
    save_audit_sample(audit_export, audit_output_file)

    print("\nDone.")
    print(f"Audit file: {audit_output_file}")


if __name__ == "__main__":
    main()
