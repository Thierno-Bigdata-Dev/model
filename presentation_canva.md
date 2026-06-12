# 🎨 Présentation Canva - Système de Scoring de Crédit (Modèles Classiques)

Ce guide diapositive par diapositive est conçu pour vous aider à créer rapidement une présentation professionnelle et premium sur **Canva**. Vous y trouverez les éléments visuels, les textes clés à intégrer sur les diapositives et le script oral à lire ou à adapter pour votre passage devant le jury.

---

## 💎 Recommandations de Style Canva (À configurer en premier)

*   **Mots-clés de recherche de modèles dans Canva** : 
    *   *« Pitch Deck Finance »*, *« Tech Presentation »*, *« Corporate Dark Blue »*, *« Analytics Dashboard »*.
*   **Palette de couleurs recommandée (Premium & Confiance)** :
    *   🔵 **Bleu Marine Sourd (60% - Fond des diapositives de titre / conteneurs principaux)** : `#0F172A` ou `#1E3A8A` (inspire la sécurité bancaire et le sérieux).
    *   ⚪ **Blanc & Gris très clair (30% - Textes et clarté)** : `#F8FAFC` et `#E2E8F0`.
    *   🟢 **Vert Émeraude (10% - Accents et points clés)** : `#10B981` (symbole de solvabilité et de validation).
*   **Typographie** :
    *   *Titres principaux* : **Outfit** ou **Montserrat** (Modernes, géométriques et professionnels).
    *   *Corps de texte* : **Inter** (Simple, épuré et très lisible).

---

## 📂 Structure des Diapositives (Slide-by-Slide)

### 📊 Diapositive 1 : Titre & Introduction
*   **Mise en page** : Fond bleu marine sombre épuré. Titre principal centré en blanc, sous-titre en vert émeraude. Une icône de graphique de crédit ou un bouclier discret.
*   **Contenu visuel** :
    *   **Titre Principal** : Système Intelligent d'Évaluation du Risque de Crédit
    *   **Sous-titre** : Aide à la décision d'octroi de prêt par Machine Learning (MVC & MLflow)
    *   **Informations de bas de page** : Présenté par [Votre Nom] — Juin 2026
*   **🗣️ Script Oral (Ce qu'il faut dire)** :
    > « Bonjour à tous. Je vous présente aujourd'hui mon projet de fin d'études portant sur le développement d'un système intelligent de scoring de crédit. L'objectif de ce projet est de concevoir un outil robuste et explicable pour aider les institutions financières à évaluer la solvabilité des emprunteurs. »

---

### 🎯 Diapositive 2 : La Problématique Métier
*   **Mise en page** : Deux blocs ou colonnes contrastés (ex: un bloc rouge pour le défi/risque, un bloc vert pour l'opportunité/solution).
*   **Contenu visuel** :
    *   **Titre** : Le Défi du Risque Bancaire
    *   **À Gauche (Le Défi)** : 
        *   Augmentation des défauts de paiement.
        *   Temps de traitement manuel des dossiers trop long.
        *   Besoin d'objectivité et de décisions standardisées.
    *   **À Droite (L'Objectif ML)** : 
        *   Classifier instantanément un profil en :
            *   **Solvable (1)** $\rightarrow$ Prêt accordé.
            *   **Non Solvable (0)** $\rightarrow$ Risque de défaut trop élevé.
*   **🗣️ Script Oral (Ce qu'il faut dire)** :
    > « Pour une banque, prêter de l'argent représente un équilibre permanent entre rentabilité et risque de perte. Les méthodes traditionnelles d'analyse manuelle sont lentes et peuvent manquer d'homogénéité. Notre approche utilise le Machine Learning pour automatiser ce classement de solvabilité sous forme d'une classification binaire fiable et instantanée. »

---

### 📁 Diapositive 3 : Les Données (German Credit Dataset)
*   **Mise en page** : 4 cartes alignées horizontalement ou en grille 2x2, chacune avec une icône moderne.
*   **Contenu visuel** :
    *   **Titre** : Origine des Données & Variables Clés
    *   **Sous-titre** : Base de données réelle *German Credit de l'UCI* (adaptée en français).
    *   **Les 4 variables décisionnelles (Features)** :
        1.  💶 **Revenus** : Revenus nets mensuels du demandeur (numérique).
        2.  📉 **Dettes** : Encours total des dettes financières (numérique).
        3.  ⏱️ **Historique** : Qualité des remboursements passés (Bon, Moyen, Mauvais).
        4.  💼 **Contrat** : Type de contrat de travail (CDI, CDD, Freelance, Sans emploi).
*   **🗣️ Script Oral (Ce qu'il faut dire)** :
    > « Pour entraîner et valider nos modèles, nous nous sommes appuyés sur un jeu de données de référence : le German Credit dataset de l'UCI. Nous avons sélectionné les quatre facteurs les plus représentatifs de la solvabilité financière : les revenus, l'encours de dettes actuel, l'historique de remboursement et le statut contractuel du demandeur. »

---

### ⚙️ Diapositive 4 : Pipeline de Prétraitement des Données
*   **Mise en page** : Une frise chronologique linéaire avec des flèches directionnelles de gauche à droite.
*   **Contenu visuel** :
    *   **Titre** : Rigueur du Pipeline de Données
    *   **Étapes** :
        *   **Split Stratifié (80/20)** : Conservation exacte de la proportion de clients insolvables dans les jeux de Train et de Test.
        *   **Encodage Catégoriel** : Passage de variables textuelles à des indices numériques via `LabelEncoder`.
        *   **Normalisation** : Application de `StandardScaler` pour mettre les revenus et les dettes sur la même échelle.
    *   **💡 Point clé** : *Zéro Data Leakage* (Fuite de données) $\rightarrow$ Ajustement (`fit`) sur le Train uniquement, simple transformation (`transform`) sur le Test.
*   **🗣️ Script Oral (Ce qu'il faut dire)** :
    > « La préparation des données est une étape cruciale. Les algorithmes de Machine Learning exigent des données numériques et équilibrées en échelle. Nous avons converti les textes et normalisé les montants continus. Afin de garantir la rigueur scientifique de l'étude et éviter le "Data Leakage", tous nos calculs de normalisation ont été entraînés exclusivement sur le jeu d'entraînement avant d'être appliqués aux tests. »

---

### 🤖 Diapositive 5 : Expérimentation et Suivi (MLflow)
*   **Mise en page** : Visuels illustrant MLflow (ou des courbes de performance) avec un bloc descriptif à gauche.
*   **Contenu visuel** :
    *   **Titre** : Expérimentation Structurée avec MLflow
    *   **Concepts** :
        *   **Suivi Reproductible** : Enregistrement automatique des hyperparamètres et des métriques de chaque exécution.
        *   **Modèles Comparés** :
            *   *Régression Logistique* (Modèle probabiliste linéaire de base).
            *   *Arbre de Décision* (Modèle géométrique de règles logiques).
*   **🗣️ Script Oral (Ce qu'il faut dire)** :
    > « Dans une démarche professionnelle, il est essentiel d'assurer la traçabilité de nos expérimentations. Nous avons utilisé la plateforme MLOps MLflow pour suivre précisément et comparer nos runs d'entraînement. Nous nous sommes concentrés sur deux familles d'algorithmes classiques et hautement interprétables : la régression logistique et l'arbre de décision. »

---

### 🏆 Diapositive 6 : Résultats des Modèles
*   **Mise en page** : Un tableau comparatif centralisé très lisible, avec la ligne de l'Arbre de Décision surlignée en vert émeraude.
*   **Contenu visuel** :
    *   **Titre** : Comparaison des Performances Réelles
    *   **Tableau** :

| Modèle | Précision (Accuracy) | F1-Score (Solvable) | Rôle / Statut |
| :--- | :---: | :---: | :---: |
| 🌲 **Arbre de Décision** | **94.00 %** | **0.9434** | **MEILLEUR MODÈLE (Production)** |
| 📈 Régression Logistique | 80.00 % | 0.8100 | Ligne de base (Baseline) |

*   **Avantages clés de l'Arbre de Décision** :
    *   Performance globale nettement supérieure (F1-Score de 94%).
    *   **Haute Explicabilité** : Permet à un banquier ou un auditeur de comprendre précisément pourquoi une demande de crédit est acceptée ou refusée.
*   **🗣️ Script Oral (Ce qu'il faut dire)** :
    > « Sur notre jeu de test, l'Arbre de Décision surpasse nettement la Régression Logistique avec un taux de précision de 94% et un F1-score de 0,943. En plus de ses performances élevées, l'Arbre de Décision présente un intérêt majeur en environnement bancaire : sa grande explicabilité. Il permet de tracer précisément le chemin de décision qui a mené à l'acceptation ou au refus d'un dossier. »

---

### 💻 Diapositive 7 : Architecture Logicielle (Pattern MVC)
*   **Mise en page** : 3 colonnes verticales représentant les couches du patron d'architecture MVC.
*   **Contenu visuel** :
    *   **Titre** : Architecture Applicative (MVC)
    *   **Les Trois Couches** :
        *   📦 **Modèle (Model)** : Contient les données d'apprentissage, le script d'entraînement (`train.py`) et les fichiers de modèles sérialisés (`.pkl`).
        *   ⚙️ **Contrôleur (Controller)** : Gère le pipeline de prétraitement en temps réel (`pipeline.py`) et l'orchestration des prédictions (`predict.py`).
        *   🖥️ **Vue (View)** : L'interface graphique utilisateur conçue en Python avec **Streamlit** (`app.py`, `components.py`).
*   **🗣️ Script Oral (Ce qu'il faut dire)** :
    > « Pour le déploiement applicatif, nous avons choisi une architecture propre et standardisée : le patron MVC (Modèle-Vue-Contrôleur). Cette séparation étanche garantit la maintenabilité du code : l'intelligence artificielle est isolée dans le Modèle, la logique métier réside dans le Contrôleur, et l'interface utilisateur dynamique Streamlit forme la Vue. »

---

### 🚀 Diapositive 8 : Démonstration Applicative & Conclusion
*   **Mise en page** : Capture d'écran ou aperçu visuel élégant de l'interface Streamlit à gauche, conclusion à droite.
*   **Contenu visuel** :
    *   **Titre** : Déploiement & Synthèse
    *   **Bilan** :
        *   Application web Streamlit ergonomique et réactive en temps réel.
        *   Indice de confiance dynamique fourni pour chaque prédiction.
        *   Intégration d'un module d'évaluation visuel (matrices de confusion).
        *   Code de qualité professionnelle, versionné et documenté.
*   **🗣️ Script Oral (Ce qu'il faut dire)** :
    > « Pour conclure, nous avons mis en place une application web fonctionnelle qui permet d'utiliser le modèle en conditions réelles. L'utilisateur saisit simplement les informations du client et obtient immédiatement une recommandation d'octroi de prêt accompagnée d'un indice de confiance. Ce projet démontre qu'il est possible de lier rigueur scientifique en Machine Learning et développement logiciel de qualité. Je vous remercie pour votre attention et je suis prêt à répondre à vos questions. »
