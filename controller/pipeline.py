import os
import json
import joblib
import pandas as pd
import numpy as np

class CreditPredictionPipeline:
    def __init__(self, model_choice="Modèle Automatique", model_dir='model'):
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir) and os.path.exists(os.path.join('..', model_dir)):
            self.model_dir = os.path.join('..', model_dir)
            
        # 1. Chargement du scaler et des encodeurs (communs)
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.encoders = joblib.load(os.path.join(self.model_dir, 'encoders.pkl'))
        
        # 2. Chargement des métadonnées du meilleur modèle
        metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        # 3. Détermination du fichier et du type de modèle à charger
        if model_choice == "Régression Logistique":
            self.model_name = "Regression Logistique"
            self.model_type = "sklearn"
            model_filename = "regression_logistique.pkl"
        elif model_choice == "Arbre de Décision":
            self.model_name = "Arbre de Decision"
            self.model_type = "sklearn"
            model_filename = "arbre_de_decision.pkl"
        else:  # "Modèle Automatique" (Recommande)
            self.model_name = self.metadata['best_model_name']
            model_filename = f"{self.metadata['best_model_filename']}.pkl"
            
        # 4. Chargement du modèle sélectionné
        model_path = os.path.join(self.model_dir, model_filename)
        self.model = joblib.load(model_path)
            
    def preprocess_input(self, raw_input_df):
        """
        Prépare les données en entrée en appliquant les encodeurs et le scaler.
        """
        df_processed = raw_input_df.copy()
        
        for col in ['historique_paiement', 'type_contrat']:
            le = self.encoders[col]
            df_processed[col] = le.transform(df_processed[col])
            
        continuous_cols = ['revenus', 'dettes']
        df_processed[continuous_cols] = self.scaler.transform(df_processed[continuous_cols])
        
        return df_processed

    def predict(self, raw_input_df):
        """
        Réalise la prédiction et retourne le diagnostic (1/0) et la probabilité.
        """
        processed_df = self.preprocess_input(raw_input_df)
        
        prediction = int(self.model.predict(processed_df)[0])
        if hasattr(self.model, 'predict_proba'):
            prob = float(self.model.predict_proba(processed_df)[0][1])
        else:
            prob = None
                
        return prediction, prob
