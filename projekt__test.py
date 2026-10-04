import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer
import warnings

# Potlačenie warningov
warnings.filterwarnings('ignore')

# --- IMPORT NOVÝCH MODELOV ---
try:
    from xgboost import XGBClassifier

    print("XGBoost úspešne načítaný.")
except ImportError:
    XGBClassifier = None
    print("XGBoost nie je nainštalovaný. Model bude preskočený.")

try:
    from catboost import CatBoostClassifier

    print("CatBoost úspešne načítaný.")
except ImportError:
    CatBoostClassifier = None
    print("CatBoost nie je nainštalovaný. Model bude preskočený.")

# =============================================================================
# --- 1. NAČÍTANIE A PRÍPRAVA DÁT ---
# =============================================================================
try:
    data = pd.read_csv('heart-disease_data.csv')
except FileNotFoundError:
    print("CHYBA: Súbor 'heart-disease_data.csv' sa nenašiel.")
    exit()

spojite_rozsahy = {
    'age': (0, 120), 'trestbps': (60, 250), 'chol': (80, 700),
    'thalach': (50, 230), 'oldpeak': (0, 5)
}
kategoricke_hodnoty = {
    'sex': [0, 1], 'cp': [0, 1, 2, 3], 'fbs': [0, 1], 'restecg': [0, 1, 2],
    'exang': [0, 1], 'slope': [0, 1, 2], 'ca': [0, 1, 2, 3], 'thal': [0, 1, 2, 3]
}

cols_to_encode = ['cp', 'restecg', 'slope', 'thal']
data = pd.get_dummies(data, columns=cols_to_encode, drop_first=True)

X = data.drop(columns=["target"])
y = data["target"]


def remove_invalid_values(X_df, continuous_ranges, categorical_values):
    X_clean = X_df.copy()
    for col, (min_val, max_val) in continuous_ranges.items():
        if col in X_df.columns:
            mask_invalid = (X_df[col] < min_val) | (X_df[col] > max_val)
            X_clean.loc[mask_invalid, col] = np.nan
    for col, allowed in categorical_values.items():
        if col in X_df.columns:
            mask_invalid = ~X_df[col].isin(allowed)
            X_clean.loc[mask_invalid, col] = np.nan
    return X_clean


X_clean = remove_invalid_values(X, spojite_rozsahy, kategoricke_hodnoty)
print(f"Dataset pripravený (100% dát). Shape: {X_clean.shape}")

# =============================================================================
# --- 2. KONFIGURÁCIA (ROZŠÍRENÉ PARAMETRE) ---
# =============================================================================

models_config = {
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'classifier__C': [0.05, 0.1, 0.5, 1, 5, 10],
            'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'classifier__gamma': ['scale', 'auto', 0.1, 1],
            'classifier__class_weight': [None, 'balanced']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [100, 300],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__class_weight': [None, 'balanced']
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 4, 5],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', None]
        }
    }
}

if XGBClassifier:
    models_config['XGBoost'] = {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 5, 7],
            'classifier__gamma': [0, 0.1, 0.2, 1],
            'classifier__min_child_weight': [1, 3, 5],
            'classifier__subsample': [0.7, 0.8, 1.0],
            'classifier__colsample_bytree': [0.7, 0.8, 1.0],
            'classifier__reg_alpha': [0, 0.1, 1],
            'classifier__reg_lambda': [1, 2]
        }
    }

if CatBoostClassifier:
    models_config['CatBoost'] = {
        'model': CatBoostClassifier(verbose=0, random_state=42),
        'params': {
            'classifier__iterations': [100, 300],
            'classifier__depth': [4, 6],
            'classifier__learning_rate': [0.01, 0.03, 0.1],
            'classifier__l2_leaf_reg': [1, 3, 5, 9],
            'classifier__random_strength': [1, 5],
            'classifier__bagging_temperature': [0, 1]
        }
    }


def get_pipeline(classifier):
    return Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('var_thresh', VarianceThreshold(threshold=0)),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif, k='all')),
        ('classifier', classifier)
    ])


# =============================================================================
# --- 3. SPUSTENIE STANDARD GRID SEARCH (BEZ NESTED SLUČKY) ---
# =============================================================================

best_models_info = {}

print("\n" + "=" * 60)
print("SPÚŠŤAM ŠTANDARDNÝ GRID SEARCH (5-FOLD CV)")
print(f"Cieľ: Nájsť jedny najlepšie parametre pre každý model.")
print("=" * 60)

for name, config in models_config.items():
    print(f"\n>>> Optimalizujem: {name} ...")

    pipeline = get_pipeline(config['model'])

    # Tu je zmena: Žiadna vonkajšia slučka.
    # GridSearchCV urobí 5-fold CV priamo na celých dátach.
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=config['params'],
        cv=5,  # 5-fold Cross Validation
        scoring='matthews_corrcoef',  # Optimalizujeme na MCC
        n_jobs=-1
    )

    try:
        grid.fit(X_clean, y)

        print(f"   Najlepšie MCC: {grid.best_score_:.4f}")
        print(f"   Najlepšie parametre: {grid.best_params_}")

        best_models_info[name] = {
            'best_score': grid.best_score_,
            'best_params': grid.best_params_
        }

    except Exception as e:
        print(f"CHYBA pri modeli {name}: {e}")

# =============================================================================
# --- 4. ZHRNUTIE VÝSLEDKOV ---
# =============================================================================

print("\n" + "=" * 60)
print("FINÁLNE POROVNANIE (Najlepšie nájdené konfigurácie)")
print("=" * 60)

# Zoradíme modely podľa MCC
sorted_models = sorted(best_models_info.items(), key=lambda x: x[1]['best_score'], reverse=True)

for name, info in sorted_models:
    print(f"{name:<20} | MCC: {info['best_score']:.4f}")

winner_name = sorted_models[0][0]
print("-" * 60)
print(f"CELKOVÝ VÍŤAZ: {winner_name}")
print(f"Parametre víťaza: {best_models_info[winner_name]['best_params']}")
print("=" * 60)