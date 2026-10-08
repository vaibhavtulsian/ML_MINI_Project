import os
import joblib
import pandas as pd
from typing import Dict, Any

class TriageInference:
    def __init__(self, config_path: str = "configs/config.yaml"):
        import yaml
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.saved_weights_dir = self.config["models"]["saved_weights_dir"]
        self.targets = self.config["targets"]
        
    def score_patient(self, patient_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Takes a dictionary of patient data (must match features in config)
        and returns a dictionary of risk probabilities for each target.
        """
        results = {}
        for target in self.targets:
            model_path = os.path.join(self.saved_weights_dir, f"{target}_best.joblib")
            if not os.path.exists(model_path):
                results[target] = "Model not found. Run training first."
                continue
                
            artifact = joblib.load(model_path)
            model = artifact["model"]
            scaler = artifact["scaler"]
            features = artifact["features"]
            
            # Convert to DataFrame
            df = pd.DataFrame([patient_data], columns=features)
            
            # Validate missing inputs
            if df.isnull().values.any():
                raise ValueError("Patient data contains missing fields for required features.")
                
            # Scale
            X_scaled = scaler.transform(df)
            
            # Predict
            prob = model.predict_proba(X_scaled)[0, 1]
            results[target] = float(prob)
            
        return results

if __name__ == "__main__":
    # Example direct inference from CLI
    sample_patient = {
        "Age": 68,
        "Gender": 1,
        "Has_Pneumonia": 1,
        "Has_Diabetes": 1,
        "Has_COPD": 0,
        "Has_Asthma": 0,
        "Is_Immunosuppressed": 0,
        "Has_Hypertension": 1,
        "Has_Cardiovascular": 1,
        "Is_Smoker": 1,
        "Is_Obese": 1
    }
    
    inference = TriageInference()
    try:
        scores = inference.score_patient(sample_patient)
        print("\nPatient Triage Scores:")
        for t, s in scores.items():
            if isinstance(s, float):
                print(f"  - {t}: {s*100:.1f}%")
            else:
                print(f"  - {t}: {s}")
    except Exception as e:
        print(f"Inference error: {e}")
