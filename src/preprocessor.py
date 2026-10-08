import yaml
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN

class Preprocessor:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.random_state = self.config["training"]["random_seed"]
        self.n_jobs = self.config["training"]["n_jobs"]
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """
        Splits data, normalizes, and applies SMOTEENN on the training set only.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # FR-02: Z-score normalization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # FR-02: Resampling using SMOTEENN on training partition only
        smote_enn = SMOTEENN(random_state=self.random_state, n_jobs=self.n_jobs)
        X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_scaled, y_train)
        
        return X_train_resampled, X_test_scaled, y_train_resampled, np.array(y_test), scaler
