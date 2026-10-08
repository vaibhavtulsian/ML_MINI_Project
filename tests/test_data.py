import pytest
import pandas as pd
from src.data_loader import DataLoader

def test_data_schema():
    dl = DataLoader()
    df = dl.load_data()
    
    expected_columns = dl.features + dl.targets
    for col in expected_columns:
        assert col in df.columns
        
def test_data_ranges():
    dl = DataLoader()
    df = dl.load_data()
    
    assert df["Age"].between(0, 120).all()
    
    for feature in dl.features:
        if feature != "Age":
            assert set(df[feature].dropna().unique()).issubset({0, 1})
            
def test_no_nulls_after_load():
    dl = DataLoader()
    df = dl.load_data()
    expected_columns = dl.features + dl.targets
    assert not df[expected_columns].isnull().any().any()
