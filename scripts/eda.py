import sys
from pathlib import Path

# Adaugă directorul parent la path pentru import-uri
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src import io, preprocess
from src.io import load_company_data, load_taxonomy
from src.preprocess import parse_business_tags, normalize_text

# Configurare paths
DATA_DIR = project_root / "data"
COMPANY_FILE = DATA_DIR / "ml_insurance_challenge.csv"
TAXONOMY_FILE = DATA_DIR / "insurance_taxonomy.xlsx"


def analyze_company_data(df: pd.DataFrame) -> None:
    """
    Analizează dataframe-ul cu companiile.
    
    Args:
        df: DataFrame cu datele companiilor
    """
    print("\n" + "="*60)
    print("COMPANY DATA ANALYSIS")
    print("="*60)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Total companies: {len(df)}")
    
    # Analizează missing values
    print("\nMissing values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            pct = 100 * count / len(df)
            print(f"  {col}: {count} ({pct:.1f}%)")
    
    # Analizează lungimea descrierilor
    print("\nDescription lengths:")
    desc_lengths = df['description'].fillna('').str.len()
    print(f"  Mean: {desc_lengths.mean():.1f} characters")
    print(f"  Median: {desc_lengths.median():.1f} characters")
    print(f"  Min: {desc_lengths.min()}")
    print(f"  Max: {desc_lengths.max()}")
    print(f"  Empty: {(desc_lengths == 0).sum()}")
    
    # Analizează business_tags
    print("\nBusiness tags analysis:")
    df['tags_parsed'] = df['business_tags'].apply(parse_business_tags)
    df['tags_count'] = df['tags_parsed'].apply(len)
    
    print(f"  Mean tags per company: {df['tags_count'].mean():.2f}")
    print(f"  Median tags per company: {df['tags_count'].median():.1f}")
    print(f"  Max tags: {df['tags_count'].max()}")
    print(f"  Companies with 0 tags: {(df['tags_count'] == 0).sum()}")
    
    # Top 10 cele mai frecvente tag-uri
    all_tags = []
    for tags_list in df['tags_parsed']:
        all_tags.extend(tags_list)
    
    if all_tags:
        tags_series = pd.Series(all_tags)
        print(f"\n  Total unique tags: {tags_series.nunique()}")
        print(f"  Top 10 most frequent tags:")
        for tag, count in tags_series.value_counts().head(10).items():
            print(f"    - {tag}: {count}")
    
    # Analizează sector, category, niche
    print("\nMetadata fields:")
    for col in ['sector', 'category', 'niche']:
        unique_count = df[col].nunique()
        missing_count = df[col].isnull().sum()
        print(f"  {col}:")
        print(f"    - Unique values: {unique_count}")
        print(f"    - Missing: {missing_count}")
        
        # Top 5 valori
        if unique_count > 0 and unique_count <= 50:
            print(f"    - Top values:")
            for val, count in df[col].value_counts().head(5).items():
                print(f"      · {val}: {count}")


def analyze_taxonomy(df: pd.DataFrame) -> None:
    """
    Analizează taxonomia de etichete.
    
    Args:
        df: DataFrame cu taxonomia
    """
    print("\n" + "="*60)
    print("TAXONOMY ANALYSIS")
    print("="*60)
    
    print(f"\nTotal labels: {len(df)}")
    
    # Analizează lungimea etichetelor
    label_lengths = df['label'].str.len()
    print(f"\nLabel lengths:")
    print(f"  Mean: {label_lengths.mean():.1f} characters")
    print(f"  Median: {label_lengths.median():.1f} characters")
    print(f"  Min: {label_lengths.min()}")
    print(f"  Max: {label_lengths.max()}")
    
    # Analizează structura etichetelor
    word_counts = df['label'].str.split().str.len()
    print(f"\nWords per label:")
    print(f"  Mean: {word_counts.mean():.2f}")
    print(f"  Median: {word_counts.median():.1f}")
    print(f"  Distribution: {word_counts.value_counts().sort_index().head(10).to_dict()}")
    
    # Sample de etichete
    print(f"\nSample labels (first 20):")
    for i, label in enumerate(df['label'].head(20), 1):
        print(f"  {i}. {label}")


def analyze_text_overlap(company_df: pd.DataFrame, taxonomy_df: pd.DataFrame) -> None:
    """
    Analizează overlap-ul de vocabular între companii și taxonomie.
    
    Args:
        company_df: DataFrame cu companiile
        taxonomy_df: DataFrame cu taxonomia
    """
    print("\n" + "="*60)
    print("TEXT OVERLAP ANALYSIS")
    print("="*60)
    
    # Construiește vocabularul pentru companii
    company_texts = []
    for _, row in company_df.head(1000).iterrows():  # Sample pentru viteză
        text = ' '.join([
            str(row['description']) if pd.notna(row['description']) else '',
            str(row['sector']) if pd.notna(row['sector']) else '',
            str(row['category']) if pd.notna(row['category']) else '',
            str(row['niche']) if pd.notna(row['niche']) else ''
        ])
        company_texts.append(normalize_text(text))
    
    company_vocab = set()
    for text in company_texts:
        company_vocab.update(text.split())
    
    # Construiește vocabularul pentru taxonomie
    taxonomy_vocab = set()
    for label in taxonomy_df['label']:
        taxonomy_vocab.update(normalize_text(label).split())
    
    # Calculează overlap
    overlap = company_vocab & taxonomy_vocab
    
    print(f"\nVocabulary statistics:")
    print(f"  Company vocabulary size (sample): {len(company_vocab)} words")
    print(f"  Taxonomy vocabulary size: {len(taxonomy_vocab)} words")
    print(f"  Overlap: {len(overlap)} words ({100*len(overlap)/len(taxonomy_vocab):.1f}% of taxonomy)")
    
    # Cuvintele comune cele mai frecvente
    if overlap:
        print(f"\n  Sample overlapping words: {sorted(list(overlap))[:20]}")


def main():
    """
    Rulează analiza exploratorie completă.
    """
    print("\n" + "="*70)
    print(" "*20 + "EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Încarcă datele
    print("\nLoading data...")
    company_df = load_company_data(COMPANY_FILE)
    taxonomy_df = load_taxonomy(TAXONOMY_FILE)
    
    # Analizează companiile
    analyze_company_data(company_df)
    
    # Analizează taxonomia
    analyze_taxonomy(taxonomy_df)
    
    # Analizează overlap-ul
    analyze_text_overlap(company_df, taxonomy_df)
    
    print("\n" + "="*70)
    print("EDA Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the statistics above")
    print("  2. Run: python scripts/02_train_predict_baseline.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()