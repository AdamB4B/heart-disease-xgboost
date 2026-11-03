import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix
)
import warnings

import joblib

# Pokus o import XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("CHYBA: Knižnica 'xgboost' nie je nainštalovaná.")

# Potlačenie warningov
warnings.filterwarnings('ignore')

# =============================================================================
# --- 1. KONFIGURÁCIA ---
# =============================================================================

seed = 42
TEST_SIZE = 0.2

# --- DEFINÍCIA GRIDU ---
# Všetky hodnoty musia byť v zozname [], aj keď je tam len jedno číslo.
PARAM_GRID = {
    'classifier__n_estimators': [3000],
    'classifier__learning_rate': [0.001],
    'classifier__max_depth': [3, 4],
    'classifier__min_child_weight': [3, 4, 5],
    'classifier__gamma': [0, 1, 2, 3],
    'classifier__subsample': [0.8, 0.7, 0.9],
    'classifier__colsample_bytree': [0.8, 0.7, 0.9],
    'classifier__reg_alpha': [0.1, 1],
    'classifier__reg_lambda': [1, 2]
}

# =============================================================================
# --- 2. NAČÍTANIE A PRÍPRAVA DÁT ---
# =============================================================================
try:
    data = pd.read_csv('heart-disease_data.csv')
    print("Dataset úspešne načítaný.")
except FileNotFoundError:
    print("POZOR: CSV nenájdené, generujem náhodné dáta...")
    from sklearn.datasets import make_classification

    X_dummy, y_dummy = make_classification(n_samples=500, n_features=13, random_state=seed, flip_y=0.1)
    data = pd.DataFrame(X_dummy, columns=[f'feat_{i}' for i in range(13)])
    data['cp'] = np.random.choice([0, 1, 2, 3], size=500)
    data['thal'] = np.random.choice([0, 1, 2, 3], size=500)
    data['age'] = np.random.randint(20, 80, size=500)
    data['target'] = y_dummy

spojite_rozsahy = {
    'age': (0, 120), 'trestbps': (60, 250), 'chol': (80, 700),
    'thalach': (50, 230), 'oldpeak': (0, 5)
}
kategoricke_hodnoty = {
    'sex': [0, 1], 'cp': [0, 1, 2, 3], 'fbs': [0, 1], 'restecg': [0, 1, 2],
    'exang': [0, 1], 'slope': [0, 1, 2], 'ca': [0, 1, 2, 3], 'thal': [0, 1, 2, 3]
}


def remove_invalid_values(X_in):
    X_out = X_in.copy()
    for col, (min_v, max_v) in spojite_rozsahy.items():
        if col in X_out.columns:
            mask = (X_out[col] < min_v) | (X_out[col] > max_v)
            X_out.loc[mask, col] = np.nan
    for col, allowed in kategoricke_hodnoty.items():
        if col in X_out.columns:
            mask = ~X_out[col].isin(allowed)
            X_out.loc[mask, col] = np.nan
    return X_out


data_clean = remove_invalid_values(data)
X_full = data_clean.drop(columns=["target"])
y_full = data_clean["target"]

X_train_cv, X_test_final, y_train_cv, y_test_final = train_test_split(
    X_full, y_full, test_size=TEST_SIZE, random_state=seed, stratify=y_full
)

print(f"Train Set: {X_train_cv.shape} | Test Set: {X_test_final.shape}")
print("-" * 60)

# =============================================================================
# --- 3. PIPELINE A GRID SEARCH ---
# =============================================================================

xgb_pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=3)),
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(
        random_state=seed,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1
    ))
])

print("\nSpúšťam GridSearchCV (Optimalizácia podľa MCC)...")

# scoring='matthews_corrcoef' zabezpečí, že GridSearch vyberie model s najlepším MCC
grid_search = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=PARAM_GRID,
    scoring='matthews_corrcoef',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_cv, y_train_cv)

# =============================================================================
# --- 4. VÝPIS KANDIDÁTOV (TOP 10 PODĽA MCC) ---
# =============================================================================

print("\n" + "=" * 80)
print("TOP 10 VÝSLEDKOV GRID SEARCH (Zoradené podľa MCC):")
print("=" * 80)

results_df = pd.DataFrame(grid_search.cv_results_)
cols_to_keep = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
results_df = results_df[cols_to_keep]

# Zoradenie podľa ranku (1 = najlepší)
results_df = results_df.sort_values(by='rank_test_score')

# --- ZMENA TU: Vyberieme iba prvých 10 riadkov ---
top_10_results = results_df.head(10)

for index, row in top_10_results.iterrows():
    rank = row['rank_test_score']
    score = row['mean_test_score']  # Toto je teraz hodnota MCC
    std = row['std_test_score']
    params = row['params']

    clean_params = {k.replace('classifier__', ''): v for k, v in params.items()}

    print(f"#{rank} | MCC: {score:.4f} (+/- {std:.4f}) | Params: {clean_params}")

print("=" * 80 + "\n")

# =============================================================================
# --- 5. FINALIZÁCIA ---
# =============================================================================

print(f"Najlepšie MCC z CV: {grid_search.best_score_:.4f}")
print("Vyhodnocujem najlepší model na NEZÁVISLOM TESTE...")

best_model = grid_search.best_estimator_
y_pred_final = best_model.predict(X_test_final)

print("\nMetriky na nezávislom teste:")
print(f"MCC:       {matthews_corrcoef(y_test_final, y_pred_final):.4f}  <-- Kľúčová metrika")
print(f"F1-Score:  {f1_score(y_test_final, y_pred_final):.4f}")
print(f"Accuracy:  {accuracy_score(y_test_final, y_pred_final):.4f}")
print(f"Recall:    {recall_score(y_test_final, y_pred_final):.4f}")

# --- 6. Uloženie finálnej pipeline ---
joblib.dump(best_model, 'final_pipeline_XGBoost.joblib')
print("\nFinálna pipeline uložená ako 'final_pipeline_XGBoost.joblib'.")