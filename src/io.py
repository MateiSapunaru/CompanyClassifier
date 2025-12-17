import pandas as pd
from pathlib import Path
from typing import Optional


def load_company_data(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Fișierul {filepath} nu există.")
    
    # Citește CSV-ul
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # Verifică coloanele necesare
    required_cols = ['description', 'business_tags', 'sector', 'category', 'niche']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Coloane lipsă din fișier: {missing_cols}")
    
    print(f"✓ Loaded {len(df)} companies from {filepath.name}")
    
    return df


def load_taxonomy(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Fișierul {filepath} nu există.")
    
    # Citește Excel-ul
    df = pd.read_excel(filepath, engine='openpyxl')
    
    # Dacă nu există coloana 'label', folosește prima coloană
    if 'label' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'label'})
    
    # Elimină valorile goale și duplicatele
    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(str).str.strip()
    df = df[df['label'] != ''].drop_duplicates(subset=['label']).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} taxonomy labels from {filepath.name}")
    
    return df


def save_predictions(
    df: pd.DataFrame,
    filepath: str | Path,
    include_diagnostics: bool = True
) -> None:
    """
    Salvează predicțiile într-un fișier CSV.
    
    Args:
        df: DataFrame cu predicțiile
        filepath: Calea unde se salvează rezultatul
        include_diagnostics: Dacă True, include coloanele de diagnosticare
                            (top1_score, gap_top1_top2, pred_status)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Coloanele de bază
    base_cols = ['description', 'business_tags', 'sector', 'category', 'niche',
                 'insurance_label', 'insurance_label_scores']
    
    # Adaugă coloanele de diagnosticare dacă sunt solicitate
    if include_diagnostics:
        diag_cols = ['pred_status', 'top1_score', 'gap_top1_top2']
        cols_to_save = base_cols + [col for col in diag_cols if col in df.columns]
    else:
        cols_to_save = base_cols
    
    # Filtrează doar coloanele care există în DataFrame
    cols_to_save = [col for col in cols_to_save if col in df.columns]
    
    df[cols_to_save].to_csv(filepath, index=False, encoding='utf-8')
    print(f"✓ Saved predictions to {filepath}")


def save_audit_sample(
    df: pd.DataFrame,
    filepath: str | Path
) -> None:
    """
    Salvează un subset pentru audit manual într-un fișier CSV.
    Include coloane goale pentru evaluare umană.
    
    Args:
        df: DataFrame cu sample-ul pentru audit
        filepath: Calea unde se salvează
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Adaugă coloane goale pentru evaluare manuală
    df_audit = df.copy()
    df_audit['human_ok'] = ''
    df_audit['notes'] = ''
    
    # Salvează în CSV
    df_audit.to_csv(filepath, index=False, encoding='utf-8')
    print(f"✓ Saved audit sample ({len(df)} rows) to {filepath}")


def save_taxonomy_enrichment_template(
    taxonomy_df: pd.DataFrame,
    filepath: str | Path
) -> None:
    """
    Creează un template pentru îmbogățirea taxonomiei cu cuvinte cheie.
    
    Args:
        taxonomy_df: DataFrame cu coloana 'label'
        filepath: Calea unde se salvează template-ul
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        print(f"↪ Enrichment template already exists, skipping overwrite: {filepath}")
        return
    
    template = taxonomy_df[['label']].copy()
    template['keywords'] = '' # Coloana pentru cuvinte cheie
    template['label_description'] = template['label'] # Implicit, aceeași ca label
    
    template.to_csv(filepath, index=False, encoding='utf-8')
    print(f"✓ Saved taxonomy enrichment template to {filepath}")