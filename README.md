# SVM Heart Disease Predictor

This project trains and evaluates Support Vector Machine (SVM) models to predict heart disease using 10 categorical health parameters.

## Data
- CSV: `Dataset/heart_disease.csv`
- Features (categorical):
  - `Exercise Habits`, `Smoking`, `Family Heart Disease`, `Diabetes`, `High Blood Pressure`,
    `Low HDL Cholesterol`, `High LDL Cholesterol`, `Alcohol Consumption`, `Stress Level`, `Sugar Consumption`
- Target: `Heart Disease Status` (encoded via LabelEncoder)

## What the script does
- Label-encodes the target (No=0, Yes=1 depending on dataset ordering; printed mapping)
- One-hot encodes the 10 categorical features (handle_unknown=ignore)
- Stratified train/test split (80/20)
- Trains and compares:
  1. Linear SVM (baseline)
  2. Default RBF SVM
  3. Tuned RBF SVM (GridSearchCV over C and gamma)
- Evaluates on test set with accuracy, classification report, confusion matrix heatmap
- Saves tuned model and encoder
- Provides an interactive live prediction using the saved artifacts

## Requirements
Install dependencies (recommended in a virtual environment):

```
pip install -r requirements.txt
```

## Run
Simply run the script:

```
python Heart_Diseasepy.py
```

Outputs will be saved to the `outputs/` folder:
- `confusion_matrix.png`
- `model_accuracies.png`
- `classification_report.txt`

Artifacts will be saved to `models/`:
- `svm_tuned_rbf.joblib`
- `onehot_encoder.joblib`

## Live Prediction
After training, the script will ask if you want to perform a live prediction. If yes, it will prompt for the 10 input parameters, transform them with the saved encoder, and print a human-readable result.

## Notes
- Missing categorical values are filled with `"Unknown"` and treated as a separate category.
- The OneHotEncoder uses `handle_unknown='ignore'` so inputs outside training categories won't crash; they will be ignored.
- Plots are saved using a non-interactive backend. Set `SHOW_PLOTS = True` in the script to display figures interactively (if supported).
