import os
# pyrefly: ignore [missing-import]
import streamlit as st

def render_header():
    """Affiche une bannière premium pour l'application."""
    st.markdown("""
    <div style="background-color:#1E3A8A;padding:20px;border-radius:10px;margin-bottom:25px;text-align:center;">
        <h1 style="color:white;margin:0;font-family:'Outfit',sans-serif;font-weight:600;">💰 Credit Solvency Predictor</h1>
        <p style="color:#93C5FD;margin:5px 0 0 0;font-size:1.1rem;">Système intelligent d'aide à la décision pour l'octroi de crédits</p>
    </div>
    """, unsafe_allow_html=True)

def render_input_form():
    """Affiche le formulaire de saisie dans une mise en page soignée."""
    st.subheader("📋 Informations du Client")
    
    col1, col2 = st.columns(2)
    with col1:
        revenus = st.number_input(
            "Revenus mensuels (€)", 
            min_value=0, 
            value=3000, 
            step=100, 
            help="Saisissez le revenu net mensuel moyen du demandeur."
        )
        historique = st.selectbox(
            "Historique de paiement", 
            options=["Bon", "Moyen", "Mauvais"],
            help="Qualité des remboursements sur les crédits passés."
        )
        
    with col2:
        dettes = st.number_input(
            "Montant total des dettes (€)", 
            min_value=0, 
            value=1500, 
            step=100, 
            help="Encours total des dettes du demandeur."
        )
        type_contrat = st.selectbox(
            "Type de contrat de travail", 
            options=["CDI", "CDD", "Freelance", "Sans emploi"],
            help="Statut professionnel actuel du demandeur."
        )
        
    return revenus, dettes, historique, type_contrat

def render_prediction_result(prediction, probability, model_name):
    """Affiche le résultat de la prédiction de façon premium."""
    st.subheader("🎯 Résultat de la Prédiction")
    
    # Encadré selon le résultat de solvabilité
    if prediction == 1:
        color = "#10B981"  # Vert émeraude
        bg_color = "#ECFDF5"
        border_color = "#A7F3D0"
        title = "Client Solvable ✅"
        desc = "Le profil du client présente un risque de crédit faible. L'octroi du prêt est recommandé."
    else:
        color = "#EF4444"  # Rouge rubis
        bg_color = "#FEF2F2"
        border_color = "#FCA5A5"
        title = "Client Non Solvable ❌"
        desc = "Le profil du client présente un risque de défaut important. L'octroi du prêt est déconseillé."
        
    st.markdown(f"""
    <div style="background-color:{bg_color};border:1px solid {border_color};padding:20px;border-radius:10px;margin-bottom:20px;">
        <h3 style="color:{color};margin-top:0;font-weight:600;">{title}</h3>
        <p style="color:#374151;margin-bottom:0;">{desc}</p>
        <p style="color:#6B7280;font-size:0.85rem;margin-top:10px;font-style:italic;">Décision prise par le modèle : <b>{model_name}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    if probability is not None:
        # Affichage d'une jauge de confiance sous forme de barre de progression
        pct_conf = int(probability * 100) if prediction == 1 else int((1 - probability) * 100)
        st.markdown(f"**Indice de confiance de la décision :** `{pct_conf}%`")
        st.progress(pct_conf / 100)

def render_model_evaluation_tabs():
    """Affiche un comparatif des matrices de confusion des modèles entraînés."""
    st.subheader("📊 Comparaison et Évaluation des Modèles")
    st.markdown("Ces graphiques (générés durant la phase d'entraînement) montrent les erreurs et réussites de chaque modèle (matrice de confusion) :")
    
    tab1, tab2, tab3 = st.tabs([
        "Régression Logistique", 
        "Arbre de Décision", 
        "Réseau de Neurones (Keras)"
    ])
    
    with tab1:
        img_path = get_image_path("cm_regression_logistique.png")
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.warning("Matrice de confusion introuvable pour la Régression Logistique.")
            
    with tab2:
        img_path = get_image_path("cm_arbre_de_decision.png")
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.warning("Matrice de confusion introuvable pour l'Arbre de Décision.")
            
    with tab3:
        img_path = get_image_path("cm_reseau_de_neurones.png")
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.warning("Matrice de confusion introuvable pour le Réseau de Neurones.")

def get_image_path(filename):
    """Recherche de manière robuste le chemin de l'image de la matrice de confusion."""
    options = [
        filename,
        os.path.join('view', filename),
        os.path.join('..', 'view', filename),
        os.path.join('Projet1', 'view', filename),
        os.path.join('..', 'Projet1', 'view', filename)
    ]
    for opt in options:
        if os.path.exists(opt):
            return opt
    return None
