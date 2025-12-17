import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple


class TfidfRetriever:
    """
    Retriever bazat pe TF-IDF și similaritate cosinus.
    
    Fit-ul se face pe vocabularul combinat (companii + etichete) pentru
    a asigura un spațiu vectorial comun.
    """
    
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_features: int = 200000
    ):
        """
        Inițializează retriever-ul cu parametrii TF-IDF.
        
        Args:
            ngram_range: Range-ul de n-grame (ex: (1,2) pentru unigrame și bigrame)
            min_df: Frecvența minimă a termenilor (ignoră termeni rari)
            max_features: Numărul maxim de features
        """
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            lowercase=True,  # Deși textele sunt deja lowercase, asigurăm consistency
            strip_accents='unicode',
            token_pattern=r'\b\w+\b'  # Token = secvență de caractere alfanumerice
        )
        
        self.company_vectors = None
        self.label_vectors = None
        self.is_fitted = False
    
    def fit(self, company_texts: pd.Series, label_texts: pd.Series) -> 'TfidfRetriever':
        """
        Antrenează vectorizer-ul pe vocabularul combinat.
        
        Creează un vocabular comun din toate textele (companii + etichete)
        pentru a asigura că ambele sunt în același spațiu vectorial.
        
        Args:
            company_texts: Serie cu textele companiilor
            label_texts: Serie cu textele etichetelor
            
        Returns:
            Self (pentru chaining)
        """
        print("Fitting TF-IDF vectorizer...")
        
        # Combină toate textele pentru vocabular comun
        all_texts = pd.concat([company_texts, label_texts], ignore_index=True)
        
        # Fit pe toate textele
        self.vectorizer.fit(all_texts)
        
        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"✓ Fitted TF-IDF vectorizer: {vocab_size} features")
        
        # Transform textele companiilor și etichetelor
        print("Transforming company texts...")
        self.company_vectors = self.vectorizer.transform(company_texts)
        
        print("Transforming label texts...")
        self.label_vectors = self.vectorizer.transform(label_texts)
        
        print(f"  - Company vectors shape: {self.company_vectors.shape}")
        print(f"  - Label vectors shape: {self.label_vectors.shape}")
        
        self.is_fitted = True
        
        return self
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Calculează matricea de similaritate cosinus între companii și etichete.
        
        Returns:
            Matrice numpy shape (n_companies, n_labels) cu scoruri de similaritate
            
        Raises:
            ValueError: Dacă retriever-ul nu a fost antrenat
        """
        if not self.is_fitted:
            raise ValueError("Retriever must be fitted before computing similarity.")
        
        print("Computing cosine similarity matrix...")
        
        # Calculează similaritatea cosinus între companii (rows) și etichete (columns)
        similarity_matrix = cosine_similarity(
            self.company_vectors,
            self.label_vectors
        )
        
        print(f"Computed similarity matrix: {similarity_matrix.shape}")
        print(f"  - Min score: {similarity_matrix.min():.4f}")
        print(f"  - Max score: {similarity_matrix.max():.4f}")
        print(f"  - Mean score: {similarity_matrix.mean():.4f}")
        print(f"  - Median score: {np.median(similarity_matrix):.4f}")
        
        return similarity_matrix
    
    def get_top_k_labels(
        self,
        similarity_matrix: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pentru fiecare companie, extrage top-k etichete și scorurile lor.
        
        Args:
            similarity_matrix: Matrice de similaritate (n_companies, n_labels)
            k: Numărul de etichete de returnat pentru fiecare companie
            
        Returns:
            Tuple de:
            - top_indices: Matrice (n_companies, k) cu indicii top-k etichete
            - top_scores: Matrice (n_companies, k) cu scorurile top-k
        """
        n_companies, n_labels = similarity_matrix.shape
        k = min(k, n_labels)  # Nu poate fi mai mare decât numărul de etichete
        
        # Găsește indicii top-k pentru fiecare rând (companie)
        # argsort sortează crescător, deci luăm ultimele k valori
        top_indices = np.argsort(similarity_matrix, axis=1)[:, -k:]
        
        # Inversează ordinea pentru a avea scorurile descrescător
        top_indices = np.flip(top_indices, axis=1)
        
        # Extrage scorurile corespunzătoare
        top_scores = np.take_along_axis(
            similarity_matrix,
            top_indices,
            axis=1
        )
        
        return top_indices, top_scores


def build_retrieval_results(
    company_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Construiește un DataFrame cu rezultatele retrieval-ului.
    
    Pentru fiecare companie, include top-k etichete și scorurile lor.
    
    Args:
        company_df: DataFrame cu companiile (trebuie să aibă 'company_text')
        taxonomy_df: DataFrame cu taxonomia (trebuie să aibă 'label')
        similarity_matrix: Matrice de similaritate (n_companies, n_labels)
        top_k: Numărul de etichete top de extras
        
    Returns:
        DataFrame cu coloanele originale + top_k_labels și top_k_scores
    """
    # Extrage top-k pentru fiecare companie
    top_indices = np.argsort(similarity_matrix, axis=1)[:, -top_k:]
    top_indices = np.flip(top_indices, axis=1)
    
    top_scores = np.take_along_axis(
        similarity_matrix,
        top_indices,
        axis=1
    )
    
    # Convertește indicii în label-uri
    labels_array = taxonomy_df['label'].values
    
    top_labels_list = []
    top_scores_list = []
    
    for i in range(len(company_df)):
        # Extrage label-urile și scorurile pentru compania i
        labels = [labels_array[idx] for idx in top_indices[i]]
        scores = top_scores[i].tolist()
        
        top_labels_list.append(labels)
        top_scores_list.append(scores)
    
    # Adaugă la dataframe
    result_df = company_df.copy()
    result_df['top_k_labels'] = top_labels_list
    result_df['top_k_scores'] = top_scores_list
    
    return result_df