import sys
import argparse
import json
from pathlib import Path


def _to_jsonable(obj):
    """Convert numpy/pandas types to plain Python types so json.dump won't crash."""
    # numpy scalars
    try:
        import numpy as _np
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    # pandas scalars (covers pd.NA etc.)
    try:
        import pandas as _pd
        if obj is _pd.NA:
            return None
    except Exception:
        pass
    # containers
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    # fallback
    return obj


# Adaugă directorul părinte la path pentru import-uri
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.io import (
    load_company_data,
    load_taxonomy,
    save_predictions,
    save_taxonomy_enrichment_template,
)
from src.preprocess import (
    preprocess_companies,
    preprocess_taxonomy,
    load_enriched_taxonomy,
)
from src.retrieval import TfidfRetriever, build_retrieval_results
from src.assign import apply_assignment, format_output_labels, get_assignment_statistics
from src.eval import compute_distribution_metrics, print_evaluation_summary, analyze_label_coverage

# Configurare paths
DATA_DIR = project_root / "data"
OUTPUTS_DIR = project_root / "outputs"
COMPANY_FILE = DATA_DIR / "ml_insurance_challenge.csv"
TAXONOMY_FILE = DATA_DIR / "insurance_taxonomy.xlsx"
ENRICHMENT_TEMPLATE_FILE = OUTPUTS_DIR / "taxonomy_enrichment_template.csv"

# Configurare parametri model
TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 200000,
}

# Default-uri (vor fi suprascrise din CLI)
ASSIGNMENT_DEFAULTS = {
    "max_labels": 3,
    "abs_keep": 0.15,
    "relative_keep": 0.75,
    "min_top1": 0.20,
}

TOP_K_CANDIDATES = 10


def parse_args():
    """Parsează argumentele CLI pentru baseline vs enriched și tuning de praguri."""
    parser = argparse.ArgumentParser(
        description="Baseline TF-IDF retrieval + multi-label assignment (opțional taxonomy enrichment)."
    )

    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Dacă este setat, folosește fișierul de enrichment pentru taxonomie (label_description).",
    )
    parser.add_argument(
        "--enrichment_file",
        type=str,
        default=str(ENRICHMENT_TEMPLATE_FILE),
        help="Calea către CSV-ul de enrichment (default: outputs/taxonomy_enrichment_template.csv).",
    )

    parser.add_argument("--min_top1", type=float, default=ASSIGNMENT_DEFAULTS["min_top1"])
    parser.add_argument("--abs_keep", type=float, default=ASSIGNMENT_DEFAULTS["abs_keep"])
    parser.add_argument("--relative_keep", type=float, default=ASSIGNMENT_DEFAULTS["relative_keep"])
    parser.add_argument("--max_labels", type=int, default=ASSIGNMENT_DEFAULTS["max_labels"])

    parser.add_argument(
        "--run_name",
        type=str,
        default="baseline",
        help="Nume pentru run (ex: v1_bad_thresholds / v2_good / v3_enriched).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUTS_DIR / f"predictions_{args.run_name}.csv"
    metrics_file = OUTPUTS_DIR / f"metrics_{args.run_name}.json"

    assignment_params = {
        "max_labels": args.max_labels,
        "abs_keep": args.abs_keep,
        "relative_keep": args.relative_keep,
        "min_top1": args.min_top1,
    }

    print("\n" + "=" * 70)
    print(" " * 15 + "BASELINE TF-IDF CLASSIFIER")
    print("=" * 70)

    print("\nRun config:")
    print(f"  - run_name: {args.run_name}")
    print(f"  - enrich: {args.enrich}")
    print(f"  - enrichment_file: {args.enrichment_file}")
    print(f"  - output_file: {output_file}")
    print(f"  - metrics_file: {metrics_file}")
    print(f"  - thresholds: {assignment_params}")

    print("\n" + "-" * 70)
    print("STEP 1: Loading data")
    print("-" * 70)

    company_df = load_company_data(COMPANY_FILE)
    taxonomy_df = load_taxonomy(TAXONOMY_FILE)

    print(f"\n✓ Loaded {len(company_df)} companies")
    print(f"✓ Loaded {len(taxonomy_df)} taxonomy labels")

    print("\n" + "-" * 70)
    print("STEP 2: Preprocessing")
    print("-" * 70)

    company_df = preprocess_companies(company_df)

    if args.enrich:
        taxonomy_df = load_enriched_taxonomy(taxonomy_df, enrichment_filepath=args.enrichment_file)
        print(f"✓ Using enriched taxonomy from: {args.enrichment_file}")
    else:
        taxonomy_df = preprocess_taxonomy(taxonomy_df)
        print("✓ Using baseline taxonomy (label only)")

    empty_companies = (company_df["company_text"] == "").sum()
    empty_labels = (taxonomy_df["label_text"] == "").sum()
    if empty_companies > 0:
        print(f"⚠ Warning: {empty_companies} companies have empty text")
    if empty_labels > 0:
        print(f"⚠ Warning: {empty_labels} labels have empty text")

    print("\n" + "-" * 70)
    print("STEP 3: Training TF-IDF retriever")
    print("-" * 70)

    print("Parameters:")
    print(f"  - ngram_range: {TFIDF_PARAMS['ngram_range']}")
    print(f"  - min_df: {TFIDF_PARAMS['min_df']}")
    print(f"  - max_features: {TFIDF_PARAMS['max_features']}")

    retriever = TfidfRetriever(**TFIDF_PARAMS)
    retriever.fit(company_texts=company_df["company_text"], label_texts=taxonomy_df["label_text"])

    print("\n" + "-" * 70)
    print("STEP 4: Computing similarity matrix")
    print("-" * 70)

    similarity_matrix = retriever.compute_similarity_matrix()

    print("\n" + "-" * 70)
    print("STEP 5: Extracting top-k candidates")
    print("-" * 70)

    results_df = build_retrieval_results(
        company_df=company_df,
        taxonomy_df=taxonomy_df,
        similarity_matrix=similarity_matrix,
        top_k=TOP_K_CANDIDATES,
    )
    print(f"✓ Extracted candidates for {len(results_df)} companies")

    print("\n" + "-" * 70)
    print("STEP 6: Multi-label assignment")
    print("-" * 70)

    results_df = apply_assignment(df=results_df, **assignment_params)

    print("\n" + "-" * 70)
    print("STEP 7: Indirect evaluation")
    print("-" * 70)

    metrics = compute_distribution_metrics(results_df)
    print_evaluation_summary(metrics)

        # Convertim metricile la tipuri JSON-safe (numpy.int64 -> int etc.)
    metrics_json = _to_jsonable(metrics)
    try:
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved metrics to: {metrics_file}")
    except TypeError as e:
        # Nu blocăm pipeline-ul dacă JSON serialization are probleme
        print(f"⚠ Could not save metrics JSON: {e}")

    coverage = analyze_label_coverage(results_df, taxonomy_df)
    assignment_stats = get_assignment_statistics(results_df)

    print("\n" + "-" * 70)
    print("STEP 8: Saving results")
    print("-" * 70)

    output_df = format_output_labels(results_df)

    save_predictions(df=output_df, filepath=output_file, include_diagnostics=True)

    # Template-ul de enrichment nu se suprascrie dacă există (logica si in src/io.py)
    save_taxonomy_enrichment_template(taxonomy_df=taxonomy_df, filepath=ENRICHMENT_TEMPLATE_FILE)

    print(f"\n✓ All results saved to {OUTPUTS_DIR}/")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    print("\nKey Statistics:")
    print(f"  - Total companies: {len(results_df)}")
    print(f"  - Uncertain predictions: {assignment_stats['uncertain_count']} ({assignment_stats['uncertain_percentage']:.1f}%)")
    print(f"  - Average labels/company: {assignment_stats['avg_labels_per_company']:.2f}")
    print(f"  - Taxonomy coverage: {coverage['coverage_pct']:.1f}%")

    print("\nNext Steps:")
    print(f"  - Audit: python run.py audit --run_name {args.run_name} --predictions_file {output_file}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
