import os
# Activer le stockage de fichiers locaux pour MLflow (requis dans les versions récentes)
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Importation de TensorFlow pour le réseau de neurones
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Importation de MLflow pour le suivi des expériences
# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn
# pyrefly: ignore [missing-import]
import mlflow.tensorflow

# Importations de nos modules personnalisés
from preprocessing import fit_preprocess, transform_preprocess

# Trouver le chemin de credit_data.csv de façon robuste
if os.path.exists('credit_data.csv'):
    data_path = 'credit_data.csv'
elif os.path.exists('../credit_data.csv'):
    data_path = '../credit_data.csv'
else:
    raise FileNotFoundError("Fichier 'credit_data.csv' introuvable. Veuillez d'abord lancer le script de téléchargement.")

def create_nn_model(input_dim):
    """Crée un modèle de Réseau de Neurones Artificiels avec Keras."""
    model = Sequential([
        # pyrefly: ignore [unexpected-keyword]
        Dense(16, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # 1. Chargement des données
    print("Chargement des données...")
    df = pd.read_csv(data_path)
    
    # 2. Séparation Features (X) et Target (y)
    X = df.drop('solvable', axis=1)
    y = df['solvable']
    
    # Split Train/Test (80% / 20%)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Prétraitement des données
    # On entraîne le préprocesseur sur le train et on l'applique sur le test
    X_train, encoders, scaler = fit_preprocess(X_train_raw)
    X_test = transform_preprocess(X_test_raw, encoders, scaler)
    
    # Configuration de MLflow
    # On utilise un chemin relatif pour éviter que MLflow ne confonde la lettre de lecteur 'C:' avec un protocole
    mlflow.set_tracking_uri("../mlruns")
    mlflow.set_experiment("Credit_Solvency_Scoring")



    
    best_model = None
    best_f1 = -1
    best_model_name = ""
    
    # Importation retardée des modules d'évaluation et de sauvegarde
    # que nous allons écrire à l'étape suivante.
    from evaluate import evaluate_model
    from save_model import save_model_assets
    
    # --- MODÈLE 1 : Régression Logistique ---
    print("\nEntraînement de la Régression Logistique...")
    with mlflow.start_run(run_name="Logistic_Regression"):
        log_model = LogisticRegression(max_iter=1000, random_state=42)
        log_model.fit(X_train, y_train)
        
        # Prédiction et évaluation
        y_pred = log_model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, "Regression Logistique")
        
        # Logger les paramètres et métriques dans MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(log_model, "model")
        
        if metrics['f1_solvable'] > best_f1:
            best_f1 = metrics['f1_solvable']
            best_model = log_model
            best_model_name = "Regression Logistique"

    # --- MODÈLE 2 : Arbre de Décision ---
    print("\nEntraînement de l'Arbre de Décision...")
    with mlflow.start_run(run_name="Decision_Tree"):
        tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
        tree_model.fit(X_train, y_train)
        
        # Prédiction et évaluation
        y_pred = tree_model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, "Arbre de Decision")
        
        # Logger dans MLflow
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("max_depth", 5)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(tree_model, "model")
        
        if metrics['f1_solvable'] > best_f1:
            best_f1 = metrics['f1_solvable']
            best_model = tree_model
            best_model_name = "Arbre de Decision"

    # --- MODÈLE 3 : Réseau de Neurones (TensorFlow / Keras) ---
    print("\nEntraînement du Réseau de Neurones...")
    with mlflow.start_run(run_name="Neural_Network"):
        nn_model = create_nn_model(X_train.shape[1])
        
        # Entraînement
        history = nn_model.fit(
            X_train, y_train, 
            epochs=50, 
            batch_size=32, 
            validation_split=0.1, 
            verbose=0
        )
        
        # Prédiction (sigmoid donne des probabilités, on seuille à 0.5)
        y_pred_prob = nn_model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        metrics = evaluate_model(y_test, y_pred, "Reseau de Neurones")
        
        # Logger dans MLflow
        mlflow.log_param("model_type", "Keras_Sequential")
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 32)
        mlflow.log_metrics(metrics)
        mlflow.tensorflow.log_model(nn_model, "model")
        
        if metrics['f1_solvable'] > best_f1:
            best_f1 = metrics['f1_solvable']
            best_model = nn_model
            best_model_name = "Reseau de Neurones"

    # 4. Sauvegarde de tous les modèles pour permettre la comparaison dans l'interface
    print(f"\nLe meilleur modèle est : {best_model_name} (F1-Score: {best_f1:.4f})")
    print("Sauvegarde de tous les modèles...")
    save_model_assets(log_model, "Regression Logistique", "regression_logistique", encoders, scaler)
    save_model_assets(tree_model, "Arbre de Decision", "arbre_de_decision", encoders, scaler)
    save_model_assets(nn_model, "Reseau de Neurones", "reseau_de_neurones", encoders, scaler)
    
    # Détermination du nom de fichier du meilleur modèle
    best_model_filename = "arbre_de_decision"
    if best_model_name == "Regression Logistique":
        best_model_filename = "regression_logistique"
    elif best_model_name == "Reseau de Neurones":
        best_model_filename = "reseau_de_neurones"
        
    # Enregistrement des métadonnées du meilleur modèle
    import json
    metadata = {
        'best_model_name': best_model_name,
        'best_model_filename': best_model_filename
    }
    
    actual_dir = '.'
    if not os.path.exists('model_metadata.json') and os.path.exists('../model'):
        actual_dir = '../model'
    elif os.path.exists('model'):
        actual_dir = 'model'
        
    metadata_path = os.path.join(actual_dir, 'model_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Métadonnées du meilleur modèle sauvegardées dans : {metadata_path}")

if __name__ == "__main__":
    main()
