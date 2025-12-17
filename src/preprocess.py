import ast
import re
from typing import List, Optional, Dict

import pandas as pd


FIELD_WEIGHTS: Dict[str, int] = {
    "description": 1,   
    "tags": 4, 
    "sector": 2,
    "category": 2,
    "niche": 2,
}


def parse_business_tags(tags_str: str) -> List[str]:
    """
    Parsează coloana business_tags din format string în listă de string-uri.

    business_tags este stocat ca string reprezentând o listă Python:
    "['Retail', 'Manufacturing', ...]"

    Args:
        tags_str: String reprezentând o listă Python

    Returns:
        Listă de tag-uri (string-uri) sau listă goală dacă parsing-ul eșuează
    """
    if pd.isna(tags_str) or tags_str == "":
        return []

    try:
        tags_list = ast.literal_eval(tags_str)

        if not isinstance(tags_list, list):
            return []

        tags_list = [str(tag).strip() for tag in tags_list if tag]
        return tags_list

    except (ValueError, SyntaxError):
        return []


def normalize_text(text: str) -> str:
    """
    Normalizează un text: lowercase, collapse whitespace, strip.
    """
    if pd.isna(text) or text == "":
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _repeat(text: str, times: int) -> str:
    """Repetă un text de N ori (cu spații), safe pentru NaN/empty."""
    if times <= 0:
        return ""
    if pd.isna(text) or str(text).strip() == "":
        return ""
    t = str(text).strip() + " "
    return t * times


def build_company_text(
    description: str,
    tags_list: List[str],
    sector: str,
    category: str,
    niche: str,
    weights: Optional[Dict[str, int]] = None,
) -> str:
    """
    Construiește textul compus pentru o companie din toate câmpurile disponibile.

    v4 (field weighting):
    - description * w_desc
    - tags_text  * w_tags
    - sector     * w_sector
    - category   * w_category
    - niche      * w_niche
    """
    w = weights or FIELD_WEIGHTS
    tags_text = " ".join(tags_list) if tags_list else ""

    parts = [
        _repeat(description, int(w.get("description", 1))),
        _repeat(tags_text, int(w.get("tags", 1))),
        _repeat(sector, int(w.get("sector", 1))),
        _repeat(category, int(w.get("category", 1))),
        _repeat(niche, int(w.get("niche", 1))),
    ]

    company_text = " ".join([p.strip() for p in parts if p and p.strip() != ""])
    return normalize_text(company_text)


def preprocess_companies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print("Preprocessing companies.")

    df["tags_list"] = df["business_tags"].apply(parse_business_tags)
    df["tags_text"] = df["tags_list"].apply(lambda tags: " ".join(tags))

    df["company_text"] = df.apply(
        lambda row: build_company_text(
            description=row["description"],
            tags_list=row["tags_list"],
            sector=row["sector"],
            category=row["category"],
            niche=row["niche"],
        ),
        axis=1,
    )

    empty_count = (df["company_text"] == "").sum()
    avg_length = df["company_text"].str.len().mean()

    print(f"Preprocessed {len(df)} companies")
    print(f"  - Empty texts: {empty_count}")
    print(f"  - Average text length: {avg_length:.1f} characters")
    print(f"  - Field weights: {FIELD_WEIGHTS}")

    return df


def preprocess_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesează dataframe-ul cu taxonomia."""
    df = df.copy()

    print("Preprocessing taxonomy...")

    df["label_text"] = df["label"].apply(normalize_text)
    df = df[df["label_text"] != ""].reset_index(drop=True)

    print(f"✓ Preprocessed {len(df)} taxonomy labels")
    return df


def load_enriched_taxonomy(
    taxonomy_df: pd.DataFrame,
    enrichment_filepath: Optional[str] = None,
) -> pd.DataFrame:
    """Încarcă taxonomia îmbogățită cu label_description dacă fișierul există."""
    if enrichment_filepath is None:
        return preprocess_taxonomy(taxonomy_df)

    from pathlib import Path as _Path

    enrichment_path = _Path(enrichment_filepath)

    if not enrichment_path.exists():
        print(f"⚠ Enrichment file not found: {enrichment_filepath}")
        print("  Using baseline taxonomy (label only)")
        return preprocess_taxonomy(taxonomy_df)

    enrichment_df = pd.read_csv(enrichment_path, encoding="utf-8")

    if "label" not in enrichment_df.columns or "label_description" not in enrichment_df.columns:
        print("⚠ Enrichment file missing required columns (label, label_description)")
        print("  Using baseline taxonomy")
        return preprocess_taxonomy(taxonomy_df)

    df = taxonomy_df.merge(
        enrichment_df[["label", "label_description"]],
        on="label",
        how="left",
    )

    df["label_description"] = df["label_description"].fillna(df["label"])
    df["label_text"] = df["label_description"].apply(normalize_text)

    enriched_count = (df["label_description"] != df["label"]).sum()
    print(f"✓ Loaded enriched taxonomy: {enriched_count}/{len(df)} labels enriched")

    return df
