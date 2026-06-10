# 💰 Credit Solvency Predictor

Ce projet implémente un système intelligent d'aide à la décision pour prédire la solvabilité de clients demandant un crédit bancaire (1 = Solvable, 0 = Non solvable), en se basant sur le jeu de données réel **German Credit de l'UCI**.

Le projet est conçu de façon modulaire en respectant l'architecture **MVC (Modèle-Vue-Contrôleur)** et intègre le suivi d'expériences avec **MLflow** et un modèle d'apprentissage profond avec **TensorFlow/Keras**.

---

## 📁 Architecture du Projet (MVC)

L'organisation des fichiers respecte une séparation stricte des responsabilités :

*   **Model (Modèle)** : Gère le téléchargement des données, leur prétraitement (normalisation, encodage), l'entraînement de plusieurs modèles et leur évaluation.
*   **Controller (Contrôleur)** : Charge le modèle configuré en production, applique la pipeline de prétraitement sur les saisies brutes en temps réel et génère les prédictions.
*   **View (Vue)** : Interface utilisateur interactive développée avec **Streamlit** pour tester les profils en direct et visualiser les performances des modèles.

---

## 🛠️ Structure des Fichiers

```text
Projet1/
│
├── README.md                # Guide du projet (ce fichier)
├── credit_data.csv          # Dataset nettoyé et équilibré (1000 observations)
├── download_uci_data.py     # Script de téléchargement et d'adaptation du dataset UCI
├── requirements.txt         # Dépendances Python nécessaires
├── notebook_credit.ipynb    # Notebook Jupyter complet et documenté
│
├── model/                   # --- COUCHE MODÈLE (M) ---
│   ├── preprocessing.py     # Pipeline d'encodage (LabelEncoder) & normalisation (StandardScaler)
│   ├── evaluate.py          # Calcul des scores et tracé des matrices de confusion
│   ├── save_model.py        # Module de sauvegarde dynamique (PKL pour Sklearn, Keras pour NN)
│   ├── train.py             # Script principal d'entraînement et tracking MLflow
│   ├── best_credit_model.pkl# Fichier du meilleur modèle sauvegardé en production
│   ├── encoders.pkl         # Encodeurs entraînés sauvegardés
│   ├── scaler.pkl           # Scaler entraîné sauvegardé
│   └── model_metadata.json  # Configuration et nom du modèle actif en production
│
├── controller/              # --- COUCHE CONTRÔLEUR (C) ---
│   ├── pipeline.py          # Logique d'inférence (prétraitement des entrées brutes)
│   └── predict.py           # Point d'entrée des prédictions (Singleton)
│
└── view/                    # --- COUCHE VUE (V) ---
    ├── app.py               # Application web Streamlit principale
    ├── components.py        # Éléments graphiques (bannière, formulaires, onglets)
    ├── cm_regression_logistique.png
    ├── cm_arbre_de_decision.png
    └── cm_reseau_de_neurones.png
```

---

## 🚀 Installation et Démarrage

### 1. Installation des dépendances
Assurez-vous d'avoir Python 3.10 ou supérieur installé. À la racine du projet, lancez :
```bash
pip install -r requirements.txt
```

### 2. Récupération et préparation des données
Téléchargez les données réelles de l'UCI et générez le dataset adapté :
```bash
python download_uci_data.py
```

### 3. Entraînement et évaluation des modèles
Lancez l'entraînement des modèles (Régression Logistique, Arbre de Décision et Réseau de Neurones). Le script va évaluer les performances, logger les résultats dans **MLflow** et enregistrer le meilleur modèle en production :
```bash
cd model
python train.py
cd ..
```

### 4. Lancement de l'interface Streamlit
Pour ouvrir l'application web interactive dans votre navigateur, lancez depuis le dossier racine :
```bash
python -m streamlit run view/app.py
```

---

## 📊 Performances des Modèles

Le script d'entraînement choisit automatiquement le modèle ayant le meilleur **F1-Score** global. Sur notre jeu de données logique UCI, les résultats obtenus sont :

| Modèle | Précision (Accuracy) | F1-Score (Solvable) | Technologie | Statut |
| :--- | :---: | :---: | :---: | :---: |
| **Arbre de Décision** | **94 %** | **0.9434** | Scikit-Learn | **Meilleur Modèle (Actif)** |
| **Réseau de Neurones (MLP)** | **93 %** | **0.9300** | TensorFlow / Keras | Alternative |
| **Régression Logistique** | **80 %** | **0.8100** | Scikit-Learn | Ligne de base |

---

## 💡 Concepts clés démontrés dans ce projet
*   **Classification Supervisée** : Modélisation binaire (Solvable vs Non solvable).
*   **Prétraitement sans Fuite (No Data Leakage)** : Application séparée du `.fit()` (sur Train uniquement) et du `.transform()` (sur Test & Production).
*   **Tracking MLOps** : Enregistrement complet des hyperparamètres et métriques sous **MLflow**.
*   **Architecture logicielle MVC** : Structuration de niveau industriel garantissant la lisibilité et la testabilité du code.
