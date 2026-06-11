import os
import sys
import streamlit as st

# Résolution robuste des chemins pour importer le contrôleur et les composants
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'controller'))

import pipeline
import predict
import importlib
importlib.reload(pipeline)
importlib.reload(predict)
from predict import predict_credit_score
from components import (
    render_header, 
    render_input_form, 
    render_prediction_result, 
    render_model_evaluation_tabs
)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Credit Solvency Predictor",
    page_icon="💰",
    layout="centered"
)

# 1. Rendu de l'en-tête (titre et design)
render_header()

# 2. Rendu du formulaire et récupération des données saisies
revenus, dettes, historique, type_contrat = render_input_form()

# Choix du modèle pour la démonstration d'examen
st.subheader("🤖 Choix du Modèle IA")
model_choice = st.selectbox(
    "Quel algorithme de Machine Learning souhaitez-vous interroger ?",
    options=[
        "Modèle Automatique (Meilleur)",
        "Régression Logistique",
        "Arbre de Décision"
    ],
    help="Sélectionnez le modèle qui effectuera la prédiction."
)

# Ajouter un espace visuel
st.markdown("<br>", unsafe_allow_html=True)

# 3. Bouton d'action pour lancer la prédiction
if st.button("🔮 Évaluer la solvabilité", use_container_width=True):
    with st.spinner("Analyse du profil en cours..."):
        try:
            # Appel du contrôleur pour réaliser la prédiction avec le modèle choisi
            prediction, probability, model_name = predict_credit_score(
                revenus, dettes, historique, type_contrat, model_choice
            )
            
            # Affichage du résultat de solvabilité et de la jauge
            render_prediction_result(prediction, probability, model_name)
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.info("Vérifiez que le modèle a bien été entraîné au préalable.")

# Ligne de séparation
st.markdown("---")

# 4. Section comparative des matrices de confusion
render_model_evaluation_tabs()
