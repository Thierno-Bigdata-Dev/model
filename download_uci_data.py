import pandas as pd
import numpy as np

print("Téléchargement du German Credit Dataset depuis l'UCI...")

# URL officielle du jeu de données UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
df = pd.read_csv(url, sep='\s+', header=None)

# 1. Mappage de l'historique de paiement
# A30, A31, A32 -> Bon / Moyen (comportements sains)
# A33, A34 -> Mauvais (retards / comptes critiques)
history_map = {
    'A30': 'Bon', 'A31': 'Bon',
    'A32': 'Moyen',
    'A33': 'Mauvais', 'A34': 'Mauvais'
}
historique_paiement = df[2].map(history_map)

# 2. Mappage du type de contrat / emploi
job_map = {
    'A171': 'Sans emploi',
    'A172': 'CDD',
    'A173': 'CDI',
    'A174': 'Freelance'
}
type_contrat = df[16].map(job_map)

# 3. Mappage des dettes
dettes = df[4]

# 4. Estimation des revenus
np.random.seed(42)
age = df[12]
bruit_revenu = np.random.randint(-300, 300, size=len(df))

revenus = np.select(
    [
        type_contrat == 'CDI',
        type_contrat == 'Freelance',
        type_contrat == 'CDD',
        type_contrat == 'Sans emploi'
    ],
    [
        3200 + age * 15 + bruit_revenu,
        3800 + age * 10 + bruit_revenu,
        1800 + age * 5 + bruit_revenu,
        500 + bruit_revenu
    ],
    default=2000
)
revenus = np.clip(revenus, 500, None)

# === CORRECTION LOGIQUE DE LA CIBLE SOLVABLE ===
# On génère une cible logique basée sur les vraies caractéristiques physiques :
# + revenus, - dettes, + bon historique, + CDI
score = (
    0.6 * (revenus / 10000) 
    - 0.7 * (dettes / 10000)
    + np.where(historique_paiement == 'Bon', 0.35, np.where(historique_paiement == 'Moyen', 0.1, -0.4))
    + np.where(type_contrat == 'CDI', 0.25, np.where(type_contrat == 'Freelance', 0.1, np.where(type_contrat == 'CDD', 0.05, -0.45)))
)
# Ajout d'un léger bruit pour garder un problème réaliste
bruit = np.random.normal(0, 0.1, size=len(df))
score_final = score + bruit

# Classe cible (1 = Solvable si score_final > 0.1, sinon 0)
solvable = (score_final > 0.1).astype(int)

# Création du DataFrame final
credit_data = pd.DataFrame({
    'revenus': revenus,
    'dettes': dettes,
    'historique_paiement': historique_paiement,
    'type_contrat': type_contrat,
    'solvable': solvable
})

# Sauvegarde en format CSV
credit_data.to_csv('credit_data.csv', index=False)
print("Fichier 'credit_data.csv' recréé avec succès avec une logique cohérente !")
print(credit_data['solvable'].value_counts(normalize=True))
