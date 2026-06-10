import os
import sys
import pandas as pd

# Permettre l'importation de pipeline.py même si on lance depuis un autre dossier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import CreditPredictionPipeline

# Pool de pipelines pour mettre en cache les modèles déjà chargés
pipelines = {}

def get_pipeline(model_choice):
    global pipelines
    if model_choice not in pipelines:
        pipelines[model_choice] = CreditPredictionPipeline(model_choice=model_choice, model_dir='model')
    return pipelines[model_choice]

def predict_credit_score(revenus, dettes, historique, type_contrat, model_choice="Modèle Automatique"):
    """
    Fonction appelée par l'interface Streamlit.
    Prend les valeurs saisies et le modèle choisi, réalise la prédiction et retourne les résultats.
    """
    # 1. Formater en DataFrame d'une ligne avec les colonnes correspondantes
    raw_data = pd.DataFrame([{
        'revenus': revenus,
        'dettes': dettes,
        'historique_paiement': historique,
        'type_contrat': type_contrat
    }])
    
    # 2. Obtenir le pipeline correspondant au modèle choisi
    pipe = get_pipeline(model_choice)
    prediction, prob = pipe.predict(raw_data)
    
    # Retourner la prédiction, la probabilité et le nom du modèle réellement utilisé
    return prediction, prob, pipe.model_name
