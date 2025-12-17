import numpy as np
import pandas as pd
from typing import List, Tuple


def assign_multilabel(
    top_k_labels: List[str],
    top_k_scores: List[float],
    max_labels: int = 3,
    abs_keep: float = 0.15,
    relative_keep: float = 0.75,
    min_top1: float = 0.20
) -> Tuple[List[str], List[float], str, float, float]:
    """
    Atribuie etichete unei companii bazat pe scoruri și threshold-uri.
    
    Logica:
    1. Păstrează până la max_labels etichete
    2. Etichetă este păstrată dacă:
       - score >= abs_keep (threshold absolut) ȘI
       - score >= relative_keep * top1_score (threshold relativ)
    3. Dacă top1_score < min_top1, marchează ca "uncertain" dar returnează tot top-1
    
    Args:
        top_k_labels: Lista de etichete candidate (sorted descrescător după scor)
        top_k_scores: Lista de scoruri corespunzătoare
        max_labels: Număr maxim de etichete de returnat
        abs_keep: Threshold absolut minim pentru păstrarea unei etichete
        relative_keep: Threshold relativ față de top-1 (ex: 0.75 = min 75% din top-1)
        min_top1: Threshold minim pentru top-1 score (sub acesta = uncertain)
        
    Returns:
        Tuple cu:
        - insurance_label: Lista de etichete selectate
        - insurance_label_scores: Lista de scoruri corespunzătoare
        - pred_status: "confident" sau "uncertain"
        - top1_score: Scorul primei etichete
        - gap_top1_top2: Diferența între top-1 și top-2
    """
    if not top_k_labels or not top_k_scores:
        # Caz edge: nu avem candidați
        return [], [], "uncertain", 0.0, 0.0
    
    # Extragem top-1 score pentru diagnosticare
    top1_score = top_k_scores[0]
    
    # Calculăm gap-ul între top-1 și top-2
    if len(top_k_scores) > 1:
        gap_top1_top2 = top_k_scores[0] - top_k_scores[1]
    else:
        gap_top1_top2 = top1_score  # Doar o etichetă disponibilă
    
    # Determinăm status-ul predicției
    if top1_score < min_top1:
        pred_status = "uncertain"
    else:
        pred_status = "confident"
    
    # Selectăm etichetele care îndeplinesc criteriile
    selected_labels = []
    selected_scores = []
    
    for label, score in zip(top_k_labels, top_k_scores):
        # Verificăm dacă am atins limita de etichete
        if len(selected_labels) >= max_labels:
            break
        
        # Verificăm threshold-ul absolut
        if score < abs_keep:
            break  # Scorurile sunt sortate descrescător, deci putem opri
        
        # Verificăm threshold-ul relativ față de top-1
        if score < relative_keep * top1_score:
            break
        
        # Etichetă îndeplinește criteriile
        selected_labels.append(label)
        selected_scores.append(score)
    
    # Fallback: dacă nu selectăm nicio etichetă, returnăm top-1 pentru traceability
    if not selected_labels:
        selected_labels = [top_k_labels[0]]
        selected_scores = [top_k_scores[0]]
    
    return selected_labels, selected_scores, pred_status, top1_score, gap_top1_top2


def apply_assignment(
    df: pd.DataFrame,
    max_labels: int = 3,
    abs_keep: float = 0.15,
    relative_keep: float = 0.75,
    min_top1: float = 0.20
) -> pd.DataFrame:
    """
    Aplică logica de atribuire multi-label pentru toate companiile.
    
    Args:
        df: DataFrame cu coloanele 'top_k_labels' și 'top_k_scores'
        max_labels: Număr maxim de etichete per companie
        abs_keep: Threshold absolut pentru păstrare
        relative_keep: Threshold relativ pentru păstrare
        min_top1: Threshold minim pentru top-1 (sub = uncertain)
        
    Returns:
        DataFrame cu coloane adiționale:
        - insurance_label: Lista de etichete atribuite
        - insurance_label_scores: Lista de scoruri
        - pred_status: "confident" sau "uncertain"
        - top1_score: Scorul primei etichete
        - gap_top1_top2: Diferența top-1 vs top-2
    """
    print("Applying multi-label assignment logic...")
    print(f"  Parameters:")
    print(f"    - max_labels: {max_labels}")
    print(f"    - abs_keep: {abs_keep}")
    print(f"    - relative_keep: {relative_keep}")
    print(f"    - min_top1: {min_top1}")
    
    results = []
    
    for _, row in df.iterrows():
        top_k_labels = row['top_k_labels']
        top_k_scores = row['top_k_scores']
        
        insurance_label, insurance_label_scores, pred_status, top1_score, gap = assign_multilabel(
            top_k_labels=top_k_labels,
            top_k_scores=top_k_scores,
            max_labels=max_labels,
            abs_keep=abs_keep,
            relative_keep=relative_keep,
            min_top1=min_top1
        )
        
        results.append({
            'insurance_label': insurance_label,
            'insurance_label_scores': insurance_label_scores,
            'pred_status': pred_status,
            'top1_score': top1_score,
            'gap_top1_top2': gap
        })
    
    # Adaugă rezultatele la dataframe
    result_df = df.copy()
    for col in ['insurance_label', 'insurance_label_scores', 'pred_status', 'top1_score', 'gap_top1_top2']:
        result_df[col] = [r[col] for r in results]
    
    # Statistici despre atribuire
    uncertain_count = (result_df['pred_status'] == 'uncertain').sum()
    uncertain_pct = 100 * uncertain_count / len(result_df)
    
    label_counts = result_df['insurance_label'].apply(len)
    avg_labels = label_counts.mean()
    
    print(f"✓ Assignment complete:")
    print(f"  - Uncertain predictions: {uncertain_count}/{len(result_df)} ({uncertain_pct:.1f}%)")
    print(f"  - Average labels per company: {avg_labels:.2f}")
    print(f"  - Label distribution: {label_counts.value_counts().sort_index().to_dict()}")
    
    return result_df


def format_output_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formatează insurance_label pentru output final.
    
    Convertește lista de etichete într-un format string pentru CSV:
    - Lista goală [] → ""
    - Listă cu o etichetă → "Label1"
    - Listă cu multiple etichete → "Label1; Label2; Label3"
    
    Args:
        df: DataFrame cu coloana 'insurance_label' (listă)
        
    Returns:
        DataFrame cu 'insurance_label' formatat ca string
    """
    df = df.copy()
    
    # Convertește lista în string cu separator "; "
    df['insurance_label'] = df['insurance_label'].apply(
        lambda labels: '; '.join(labels) if labels else ''
    )
    
    # Similar pentru scoruri (opțional, pentru debugging)
    if 'insurance_label_scores' in df.columns:
        df['insurance_label_scores'] = df['insurance_label_scores'].apply(
            lambda scores: '; '.join([f"{s:.4f}" for s in scores]) if scores else ''
        )
    
    return df


def get_assignment_statistics(df: pd.DataFrame) -> dict:
    """
    Calculează statistici despre atribuirea etichetelor.
    
    Args:
        df: DataFrame cu rezultatele atribuirii
        
    Returns:
        Dicționar cu statistici
    """
    stats = {}
    
    # Status predictions
    stats['total_companies'] = len(df)
    stats['uncertain_count'] = (df['pred_status'] == 'uncertain').sum()
    stats['uncertain_percentage'] = 100 * stats['uncertain_count'] / stats['total_companies']
    
    # Label counts
    label_counts = df['insurance_label'].apply(
        lambda x: len(x) if isinstance(x, list) else (0 if x == '' else len(str(x).split('; ')))
    )
    stats['avg_labels_per_company'] = label_counts.mean()
    stats['label_distribution'] = label_counts.value_counts().sort_index().to_dict()
    
    # Score statistics
    if 'top1_score' in df.columns:
        stats['top1_score_mean'] = df['top1_score'].mean()
        stats['top1_score_median'] = df['top1_score'].median()
        stats['top1_score_std'] = df['top1_score'].std()
    
    if 'gap_top1_top2' in df.columns:
        stats['gap_mean'] = df['gap_top1_top2'].mean()
        stats['gap_median'] = df['gap_top1_top2'].median()
    
    return stats