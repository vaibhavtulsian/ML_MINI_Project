import os
import joblib
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.evaluate import Evaluator

# Suppress some convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class Trainer:
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.evaluator = Evaluator()
        
        self.random_state = self.preprocessor.random_state
        self.n_jobs = self.preprocessor.n_jobs
        self.saved_weights_dir = self.data_loader.config["models"]["saved_weights_dir"]
        os.makedirs(self.saved_weights_dir, exist_ok=True)
        
    def _get_models(self):
        # FR-03: Initialize the 5 models
        return {
            "LR": LogisticRegression(
                penalty='l2', 
                random_state=self.random_state, 
                max_iter=1000
            ),
            "RF": RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced', 
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            "SVM": SVC(
                kernel='rbf', 
                probability=True, 
                random_state=self.random_state
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(64, 32), 
                max_iter=500, 
                random_state=self.random_state
            ),
            "LGBM": LGBMClassifier(
                random_state=self.random_state, 
                n_jobs=self.n_jobs, 
                verbose=-1
            )
        }
        
    def run_training_pipeline(self):
        targets = self.data_loader.targets
        
        for target in targets:
            print(f"\n{'#'*50}")
            print(f"Starting Training Pipeline for Target: {target}")
            print(f"{'#'*50}")
            
            # Load and prepare data
            X, y = self.data_loader.get_X_y(target)
            X_train_res, X_test, y_train_res, y_test, scaler = self.preprocessor.prepare_data(X, y)
            
            models = self._get_models()
            best_model_name = None
            best_auprc = -1
            best_model = None
            
            for name, model in models.items():
                print(f"-> Training {name}...")
                model.fit(X_train_res, y_train_res)
                
                # Predict
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Evaluate
                metrics = self.evaluator.evaluate_model(name, target, y_test, y_pred, y_prob)
                
                if metrics["auprc"] > best_auprc:
                    best_auprc = metrics["auprc"]
                    best_model_name = name
                    best_model = model
            
            print(f"\n*** Best Model for {target} was {best_model_name} with AUPRC: {best_auprc:.4f} ***")
            
            # FR-05: Serialization
            artifact = {
                "model": best_model,
                "model_name": best_model_name,
                "scaler": scaler,
                "features": self.data_loader.features
            }
            save_path = os.path.join(self.saved_weights_dir, f"{target}_best.joblib")
            joblib.dump(artifact, save_path)
            print(f"Saved best model artifact to {save_path}")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run_training_pipeline()
