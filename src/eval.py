import numpy as np
import pandas as pd
from typing import Optional, List


def create_score_bins(
    scores: pd.Series,
    n_bins: int = 3,
    labels: Optional[List[str]] = None
) -> pd.Series:
    """
    Împarte scorurile în bins pentru stratificare.
    
    Args:
        scores: Serie cu scoruri numerice
        n_bins: Numărul de bins (default: 3 pentru low/mid/high)
        labels: Etichetele pentru bins (default: ['low', 'mid', 'high'])
        
    Returns:
        Serie cu categoria fiecărui scor
    """
    if labels is None:
        if n_bins == 3:
            labels = ['low', 'mid', 'high']
        else:
            labels = [f'bin_{i+1}' for i in range(n_bins)]
    
    # Folosim qcut pentru bins cu dimensiuni aproximativ egale
    try:
        bins = pd.qcut(scores, q=n_bins, labels=labels, duplicates='drop')
    except ValueError:
        # Dacă qcut eșuează (prea multe duplicate), folosim cut
        bins = pd.cut(scores, bins=n_bins, labels=labels[:n_bins])
    
    return bins


def create_audit_sample(
    df: pd.DataFrame,
    sample_size: int = 100,
    stratify_by: str = 'top1_score',
    n_bins: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Creează un sample stratificat pentru audit manual.
    
    Sample-ul este stratificat pe scorurile top-1 pentru a asigura
    acoperire în toate zonele de încredere (low/mid/high scores).
    
    Args:
        df: DataFrame cu predicțiile
        sample_size: Numărul total de sample-uri de extras
        stratify_by: Coloana după care se face stratificarea (default: 'top1_score')
        n_bins: Numărul de bins pentru stratificare
        random_state: Seed pentru reproducibilitate
        
    Returns:
        DataFrame cu sample-ul pentru audit
    """
    print(f"Creating audit sample (n={sample_size})...")
    
    # Verifică dacă coloana de stratificare există
    if stratify_by not in df.columns:
        print(f"⚠ Column '{stratify_by}' not found. Random sampling instead.")
        return df.sample(n=min(sample_size, len(df)), random_state=random_state)
    
    # Creează bins pentru stratificare
    df_copy = df.copy()
    df_copy['score_bin'] = create_score_bins(df_copy[stratify_by], n_bins=n_bins)
    
    # Calculează câte sample-uri per bin
    samples_per_bin = sample_size // n_bins
    remainder = sample_size % n_bins
    
    audit_samples = []
    
    for i, bin_label in enumerate(df_copy['score_bin'].cat.categories):
        bin_df = df_copy[df_copy['score_bin'] == bin_label]
        
        # Numărul de sample-uri pentru acest bin
        n_samples = samples_per_bin + (1 if i < remainder else 0)
        n_samples = min(n_samples, len(bin_df))
        
        # Sample aleatoriu din bin
        bin_sample = bin_df.sample(n=n_samples, random_state=random_state + i)
        audit_samples.append(bin_sample)
        
        print(f"  - {bin_label} bin: {n_samples}/{len(bin_df)} samples")
    
    # Combină toate sample-urile
    audit_df = pd.concat(audit_samples, ignore_index=True)
    
    # Elimină coloana temporară
    audit_df = audit_df.drop(columns=['score_bin'])
    
    # Sortează după top1_score descrescător pentru ușurință în review
    if stratify_by in audit_df.columns:
        audit_df = audit_df.sort_values(by=stratify_by, ascending=False).reset_index(drop=True)
    
    print(f"✓ Created audit sample: {len(audit_df)} companies")
    
    return audit_df


def prepare_audit_export(
    audit_df: pd.DataFrame,
    include_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Pregătește dataframe-ul pentru export ca fișier de audit.
    
    Include doar coloanele relevante și adaugă coloane goale pentru evaluare manuală.
    
    Args:
        audit_df: DataFrame cu sample-ul de audit
        include_columns: Coloanele de inclus (None = toate cele relevante)
        
    Returns:
        DataFrame pregătit pentru export
    """
    if include_columns is None:
        # Coloanele standard pentru audit
        include_columns = [
            'description',
            'business_tags',
            'sector',
            'category',
            'niche',
            'insurance_label',
            'insurance_label_scores',
            'pred_status',
            'top1_score',
            'gap_top1_top2'
        ]
    
    # Filtrează doar coloanele care există
    available_columns = [col for col in include_columns if col in audit_df.columns]
    
    export_df = audit_df[available_columns].copy()
    
    # Adaugă coloane pentru evaluare manuală (experimental/trebuie sa sterg daca nu le folosesc)
    export_df['human_ok'] = ''  # Pentru "yes"/"no"/"partial"
    export_df['notes'] = ''  # Pentru comentarii
    
    return export_df


def compute_distribution_metrics(df: pd.DataFrame) -> dict:
    """
    Calculează metrici despre distribuția scorurilor și predicțiilor.
    
    Aceste metrici ajută la înțelegerea comportamentului clasificatorului
    fără a avea nevoie de ground truth.
    
    Args:
        df: DataFrame cu predicțiile
        
    Returns:
        Dicționar cu metrici
    """
    metrics = {}
    
    # Metrici despre top1_score
    if 'top1_score' in df.columns:
        metrics['top1_score'] = {
            'mean': df['top1_score'].mean(),
            'median': df['top1_score'].median(),
            'std': df['top1_score'].std(),
            'min': df['top1_score'].min(),
            'max': df['top1_score'].max(),
            'q25': df['top1_score'].quantile(0.25),
            'q75': df['top1_score'].quantile(0.75)
        }
    
    # Metrici despre gap_top1_top2
    if 'gap_top1_top2' in df.columns:
        metrics['gap_top1_top2'] = {
            'mean': df['gap_top1_top2'].mean(),
            'median': df['gap_top1_top2'].median(),
            'std': df['gap_top1_top2'].std(),
            'min': df['gap_top1_top2'].min(),
            'max': df['gap_top1_top2'].max()
        }
    
    # Metrici despre status
    if 'pred_status' in df.columns:
        status_counts = df['pred_status'].value_counts()
        metrics['pred_status'] = {
            'confident_count': status_counts.get('confident', 0),
            'uncertain_count': status_counts.get('uncertain', 0),
            'confident_pct': 100 * status_counts.get('confident', 0) / len(df),
            'uncertain_pct': 100 * status_counts.get('uncertain', 0) / len(df)
        }
    
    # Metrici despre numărul de etichete
    if 'insurance_label' in df.columns:
        label_counts = df['insurance_label'].apply(
            lambda x: len(x) if isinstance(x, list) else (0 if x == '' else len(str(x).split('; ')))
        )
        metrics['labels_per_company'] = {
            'mean': label_counts.mean(),
            'median': label_counts.median(),
            'mode': label_counts.mode().iloc[0] if not label_counts.mode().empty else 0,
            'distribution': label_counts.value_counts().sort_index().to_dict()
        }
    
    return metrics


def print_evaluation_summary(metrics: dict) -> None:
    """
    Afișează un sumar al metricilor de evaluare.
    
    Args:
        metrics: Dicționar cu metrici de la compute_distribution_metrics
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY (Indirect Metrics)")
    print("="*60)
    
    # Top-1 score distribution
    if 'top1_score' in metrics:
        print("\nTop-1 Score Distribution:")
        ts = metrics['top1_score']
        print(f"  Mean:   {ts['mean']:.4f}")
        print(f"  Median: {ts['median']:.4f}")
        print(f"  Std:    {ts['std']:.4f}")
        print(f"  Range:  [{ts['min']:.4f}, {ts['max']:.4f}]")
        print(f"  IQR:    [{ts['q25']:.4f}, {ts['q75']:.4f}]")
    
    # Gap distribution
    if 'gap_top1_top2' in metrics:
        print("\nGap (Top-1 vs Top-2) Distribution:")
        gap = metrics['gap_top1_top2']
        print(f"  Mean:   {gap['mean']:.4f}")
        print(f"  Median: {gap['median']:.4f}")
        print(f"  Std:    {gap['std']:.4f}")
        print(f"  Range:  [{gap['min']:.4f}, {gap['max']:.4f}]")
    
    # Prediction status
    if 'pred_status' in metrics:
        print("\nPrediction Status:")
        ps = metrics['pred_status']
        print(f"  Confident:  {ps['confident_count']} ({ps['confident_pct']:.1f}%)")
        print(f"  Uncertain:  {ps['uncertain_count']} ({ps['uncertain_pct']:.1f}%)")
    
    # Labels per company
    if 'labels_per_company' in metrics:
        print("\nLabels per Company:")
        lpc = metrics['labels_per_company']
        print(f"  Mean:   {lpc['mean']:.2f}")
        print(f"  Median: {lpc['median']:.1f}")
        print(f"  Mode:   {lpc['mode']}")
        print(f"  Distribution: {lpc['distribution']}")
    
    print("\n" + "="*60)


def analyze_label_coverage(df: pd.DataFrame, taxonomy_df: pd.DataFrame) -> dict:
    """
    Analizează acoperirea taxonomiei: câte etichete sunt folosite vs disponibile.
    
    Args:
        df: DataFrame cu predicțiile (coloana 'insurance_label')
        taxonomy_df: DataFrame cu taxonomia completă
        
    Returns:
        Dicționar cu metrici de acoperire
    """
    # Extrage toate etichetele prezise
    all_predicted_labels = set()
    
    for labels in df['insurance_label']:
        if isinstance(labels, list):
            all_predicted_labels.update(labels)
        elif isinstance(labels, str) and labels != '':
            all_predicted_labels.update(labels.split('; '))
    
    # Etichetele disponibile în taxonomie
    all_taxonomy_labels = set(taxonomy_df['label'].values)
    
    # Metrici
    coverage = {
        'total_taxonomy_labels': len(all_taxonomy_labels),
        'labels_used': len(all_predicted_labels),
        'labels_unused': len(all_taxonomy_labels - all_predicted_labels),
        'coverage_pct': 100 * len(all_predicted_labels) / len(all_taxonomy_labels),
        'unused_labels': sorted(list(all_taxonomy_labels - all_predicted_labels))
    }
    
    print(f"\nLabel Coverage:")
    print(f"  Total labels in taxonomy: {coverage['total_taxonomy_labels']}")
    print(f"  Labels used in predictions: {coverage['labels_used']}")
    print(f"  Coverage: {coverage['coverage_pct']:.1f}%")
    print(f"  Unused labels: {coverage['labels_unused']}")
    
    return coverage