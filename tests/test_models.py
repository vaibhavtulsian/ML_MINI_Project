import pytest
import os
import joblib
from src.inference import TriageInference
from src.data_loader import DataLoader

def test_inference_engine():
    # Make sure we have a model trained
    dl = DataLoader()
    target = dl.targets[0]
    model_path = f"models/saved_weights/{target}_best.joblib"
    
    if os.path.exists(model_path):
        inference = TriageInference()
        sample = {f: 0 for f in dl.features}
        sample["Age"] = 55
        
        scores = inference.score_patient(sample)
        assert len(scores) == 3
        
        for t in dl.targets:
            if t in scores and isinstance(scores[t], float):
                assert 0.0 <= scores[t] <= 1.0
