from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(y_true, y_pred, model_name, output_dir='view'):
    """
    Affiche le rapport de classification et sauvegarde l'image de la matrice de confusion.
    Retourne les métriques clés pour l'enregistrement MLflow.
    """
    print(f"\n================ Évaluation du modèle : {model_name} ================")
    report = classification_report(y_true, y_pred)
    print(report)
    
    # 1. Calcul de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # 2. Dessin et sauvegarde de la heatmap
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non Solvable', 'Solvable'],
                yticklabels=['Non Solvable', 'Solvable'])
    plt.title(f"Matrice de confusion - {model_name}")
    plt.ylabel('Vrai')
    plt.xlabel('Prédit')
    plt.tight_layout()
    
    # S'assurer que le dossier (par défaut 'view') existe pour sauvegarder le graphique
    # On gère le cas où on l'appelle depuis la racine ou le dossier model/
    actual_output_dir = output_dir
    if not os.path.exists(actual_output_dir) and os.path.exists(os.path.join('..', output_dir)):
        actual_output_dir = os.path.join('..', output_dir)
        
    os.makedirs(actual_output_dir, exist_ok=True)
    filename = f"cm_{model_name.lower().replace(' ', '_')}.png"
    fig_path = os.path.join(actual_output_dir, filename)
    plt.savefig(fig_path)
    plt.close()
    
    print(f"Graphique de la matrice de confusion sauvegardé dans : {fig_path}")
    
    # 3. Extraction des métriques pour MLflow
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    metrics = {
        'accuracy': report_dict['accuracy'],
        'precision_solvable': report_dict['1']['precision'],
        'recall_solvable': report_dict['1']['recall'],
        'f1_solvable': report_dict['1']['f1-score'],
        'precision_non_solvable': report_dict['0']['precision'],
        'recall_non_solvable': report_dict['0']['recall'],
        'f1_non_solvable': report_dict['0']['f1-score'],
    }
    return metrics
