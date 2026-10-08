import pandas as pd
import yaml
import os
from typing import Tuple

class DataLoader:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.raw_path = self.config["data"]["raw_path"]
        self.features = self.config["features"]
        self.targets = self.config["targets"]
        
    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Data file not found at {self.raw_path}")
            
        df = pd.read_csv(self.raw_path)
        self._validate_schema(df)
        return df
        
    def _validate_schema(self, df: pd.DataFrame):
        expected_columns = self.features + self.targets
        
        # FR-01: Verify presence of all mandatory columns
        for col in expected_columns:
            if col not in df.columns:
                raise ValueError(f"Missing mandatory column: {col}")
                
        # FR-01: Enforce binary constraints and Age range
        for feature in self.features:
            if feature == "Age":
                if not df[feature].between(0, 120).all():
                    raise ValueError("Age contains values outside [0, 120]")
            else:
                if not set(df[feature].dropna().unique()).issubset({0, 1}):
                    raise ValueError(f"Feature {feature} must be binary {0, 1}")
                    
        for target in self.targets:
            if not set(df[target].dropna().unique()).issubset({0, 1}):
                raise ValueError(f"Target {target} must be binary {0, 1}")
                
        # Drop rows with nulls as part of sanitization
        if df[expected_columns].isnull().any().any():
            print("Warning: Missing values found. Dropping rows with null values.")
            df.dropna(subset=expected_columns, inplace=True)
            
    def get_X_y(self, target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.load_data()
        X = df[self.features]
        y = df[target_name]
        return X, y
