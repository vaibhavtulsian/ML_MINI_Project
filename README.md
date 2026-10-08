# COVID-19 Clinical Triage Machine Learning System

This project is a machine learning pipeline designed to predict the clinical severity and triage outcomes for incoming COVID-19 patients. By analyzing a patient's medical history (age, gender, and pre-existing comorbidities), the system predicts the risk probabilities for three critical targets:
1. **ICU Admission** (`Admitted_ICU`)
2. **Need for Intubation/Ventilation** (`Is_Intubated`)
3. **Mortality Risk** (`Is_Deceased`)

## Features
* **Automated Data Processing:** Handles missing data, applies Z-score normalization, and uses `SMOTEENN` to automatically balance rare medical outcomes in the training set.
* **Multi-Model Evaluation:** Trains and evaluates 5 different classification models:
  * Logistic Regression (LR)
  * Random Forest (RF)
  * Support Vector Machine (SVM)
  * Multi-Layer Perceptron / Neural Network (MLP)
  * LightGBM (LGBM)
* **Automated Model Selection:** Automatically identifies the best model for each specific target based on the **AUPRC** (Area Under the Precision-Recall Curve) metric, and saves the "frozen" brain of the model as a `.joblib` file.
* **Performance Visualizations:** Automatically generates Confusion Matrices and ROC/PR Curves for every model to prove clinical reliability.
* **Ready-to-use Inference:** Includes an inference script that can take a new patient's profile and instantly output their personalized risk percentages.

## Project Structure
```text
covid-triage-ml/
├── configs/             # Configuration files (features, targets, hyperparameters)
├── data/                # Raw patient data (InputData.csv)
├── models/              # Saved model artifacts (.joblib files)
├── outputs/             # Generated visual reports (Confusion Matrices, Curves)
├── src/                 # Core Python scripts
│   ├── data_loader.py   # Schema validation and data loading
│   ├── preprocessor.py  # Train/test splitting, scaling, and SMOTEENN
│   ├── train.py         # Main training and evaluation pipeline
│   ├── evaluate.py      # Metric calculations and plotting
│   └── inference.py     # End-user script to predict risks for new patients
└── tests/               # Unit testing suite
```

## Setup & Installation

1. **Navigate to the project directory:**
   ```bash
   cd covid-triage-ml
   ```

2. **Activate the virtual environment:**
   This project uses a virtual environment to manage dependencies.
   ```bash
   source venv/bin/activate
   ```
   *(If you are setting this up from scratch, you can run `pip install -r requirements.txt` after activating).*

## Usage Guide

### 1. Training the Models
To train the models from scratch on your dataset, run:
```bash
python -m src.train
```
*This will evaluate all 5 models against all 3 targets, output the classification metrics to the console, save the visual plots to `outputs/figures/`, and serialize the winning models into `models/saved_weights/`.*

### 2. Running Inference (Predicting for a New Patient)
To use the trained models to triage a new patient, run:
```bash
python -m src.inference
```
*To test different patient profiles, simply open `src/inference.py` in your text editor and modify the values inside the `sample_patient` dictionary before running the command.*

### 3. Running Tests
To ensure the pipeline is functioning correctly, you can run the test suite:
```bash
pytest tests/
```
