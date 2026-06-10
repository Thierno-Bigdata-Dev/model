# pyrefly: ignore [missing-import]
import joblib
import json
import os

def save_model_assets(model, model_name, filename_prefix, encoders, scaler, output_dir='.'):
    """
    Sauvegarde un modèle spécifique (Keras ou Scikit-Learn) sous un nom de fichier préfixé,
    et sauvegarde également les encodeurs et le scaler.
    """
    actual_dir = output_dir
    if not os.path.exists(actual_dir) and os.path.exists(os.path.join('..', output_dir)):
        actual_dir = os.path.join('..', output_dir)
        
    # 1. Sauvegarde du modèle sous son nom préfixé
    if hasattr(model, 'save') and not hasattr(model, 'predict_proba'):
        model_type = 'keras'
        model_path = os.path.join(actual_dir, f"{filename_prefix}.keras")
        model.save(model_path)
        print(f"Modèle {model_name} (Keras) sauvegardé dans : {model_path}")
    else:
        model_type = 'sklearn'
        model_path = os.path.join(actual_dir, f"{filename_prefix}.pkl")
        joblib.dump(model, model_path)
        print(f"Modèle {model_name} (Scikit-Learn) sauvegardé dans : {model_path}")
        
    # 2. Sauvegarde des encodeurs et du scaler (communs à tous les modèles)
    encoders_path = os.path.join(actual_dir, 'encoders.pkl')
    scaler_path = os.path.join(actual_dir, 'scaler.pkl')
    
    joblib.dump(encoders, encoders_path)
    joblib.dump(scaler, scaler_path)
