import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def fit_preprocess(df):
    """
    Entraîne (fit) et applique (transform) les encodeurs et le scaler 
    sur le jeu de données d'entraînement.
    Retourne le DataFrame transformé, le dictionnaire d'encodeurs et le scaler.
    """
    df_processed = df.copy()
    
    # 1. Encodage des variables catégorielles
    encoders = {}
    categorical_cols = ['historique_paiement', 'type_contrat']
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        encoders[col] = le
        
    # 2. Normalisation des variables continues
    scaler = StandardScaler()
    continuous_cols = ['revenus', 'dettes']
    df_processed[continuous_cols] = scaler.fit_transform(df_processed[continuous_cols])
    
    return df_processed, encoders, scaler

def transform_preprocess(df, encoders, scaler):
    """
    Applique (transform) les encodeurs et le scaler déjà entraînés
    sur de nouvelles données (comme le jeu de test ou les entrées de l'utilisateur).
    """
    df_processed = df.copy()
    
    # 1. Application des encodeurs
    categorical_cols = ['historique_paiement', 'type_contrat']
    for col in categorical_cols:
        le = encoders[col]
        df_processed[col] = le.transform(df_processed[col])
        
    # 2. Application du scaler
    continuous_cols = ['revenus', 'dettes']
    df_processed[continuous_cols] = scaler.transform(df_processed[continuous_cols])
    
    return df_processed
