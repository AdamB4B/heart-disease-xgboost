import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
import warnings
from xgboost import XGBClassifier

# Potlačenie warningov pre krajší výpis
warnings.filterwarnings('ignore')

# =============================================================================
# --- 1. KONFIGURÁCIA ---
# =============================================================================

seed = 42
VYBRANY_MODEL = "RandomForest"

# Sem skopíruj "best_params", ktoré ti vypísal predchádzajúci GridSearch
# Príklad: {'clf__max_depth': 10, 'clf__n_estimators': 100, 'imputer__strategy': 'mean'}
# NASTAVENIA = {
#     'classifier__C': 0.5,
#     'classifier__kernel': 'linear',
#     'classifier__gamma': 'scale',
#     'class_weight':'balanced',
#     'classifier__tol': 1e-4
# }
# NASTAVENIA = {
#     'classifier__n_estimators': 200,
#     'classifier__max_depth': 4,
#     'classifier__min_samples_split': 4,
#     'classifier__min_samples_leaf': 3,
#     'classifier__max_features': 'sqrt',
#     'min_impurity_decrease': 0.002,
#     'max_samples': 0.9,
#     'max_leaf_nodes': None,
#     'classifier__class_weight': 'balanced',
#     'criterion': 'gini',
#     'ccp_alpha': 0.0,
#     'bootstrap': True
# }

VYBRANY_MODEL = "XGBoost"
NASTAVENIA = {
    'classifier__n_estimators': 300,
    'classifier__learning_rate': 0.01,
    'classifier__max_depth': 3,
    'classifier__min_child_weight':3,
    'classifier__gamma':0.1,
    'classifier__subsample': 0.8,  # Prevencia preučenia (použije len 80% dát pre strom)
    'classifier__colsample_bytree': 0.8,  # Prevencia preučenia (použije len 80% stĺpcov)
    'classifier__reg_alpha':0.001,
    'classifier__reg_lambda':2
}


# =============================================================================
# --- 2. NAČÍTANIE A PRÍPRAVA DÁT (BEZ SPLITU) ---
# =============================================================================
try:
    data = pd.read_csv('heart-disease_data.csv')
except FileNotFoundError:
    # Fallback pre demonštráciu, ak nemáš súbor (v tvojom PC to zbehne s CSV)
    print("POZOR: CSV nenájdené, generujem náhodné dáta pre ukážku kódu...")
    from sklearn.datasets import make_classification

    X_dummy, y_dummy = make_classification(n_samples=300, n_features=13, random_state=seed)
    data = pd.DataFrame(X_dummy, columns=[f'feat_{i}' for i in range(13)])
    data['target'] = y_dummy

# Definícia pravidiel pre čistenie
spojite_rozsahy = {
    'age': (0, 120), 'trestbps': (60, 250), 'chol': (80, 700),
    'thalach': (50, 230), 'oldpeak': (0, 5)
}
kategoricke_hodnoty = {
    'sex': [0, 1], 'cp': [0, 1, 2, 3], 'fbs': [0, 1], 'restecg': [0, 1, 2],
    'exang': [0, 1], 'slope': [0, 1, 2], 'ca': [0, 1, 2, 3], 'thal': [0, 1, 2, 3]
}

# One-Hot Encoding (ak stĺpce existujú)
cols_to_encode = ['cp', 'restecg', 'slope', 'thal']
existing_cols = [c for c in cols_to_encode if c in data.columns]
data = pd.get_dummies(data, columns=existing_cols, drop_first=True)

X = data.drop(columns=["target"])
y = data["target"]


# Funkcia čistenia
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


# Vyčistíme celý dataset naraz
X_clean = remove_invalid_values(X)

print(f"Dataset pripravený na Cross-Validáciu: {X_clean.shape}")


# =============================================================================
# --- 3. PIPELINE ---
# =============================================================================

def build_custom_pipeline(model_type, params):
    steps = []
    # 1. Imputer
    steps.append(('imputer', KNNImputer(n_neighbors=5)))
    # 2. Scaler
    steps.append(('scaler', StandardScaler()))
    # 3. Model
    if model_type == "SVM":
        clf = SVC(random_state=seed, probability=True)
    elif model_type == "RandomForest":
        clf = RandomForestClassifier(random_state=seed)
    elif model_type == "GradientBoosting":
        clf = GradientBoostingClassifier(random_state=seed)
    # >>> PRIDANÁ PODPORA PRE XGBOOST <<<
    elif model_type == "XGBoost":
        # Nastavíme eval_metric='logloss' pre potlačenie varovaní,
        # class_weight parameter nie je priamo v XGBClassifier, ale
        # vyvažuje sa cez scale_pos_weight ak je to potrebné.
        clf = XGBClassifier(random_state=seed, eval_metric='logloss')
    # >>> KONIEC PRIDANEJ PODPORY <<<
    else:
        raise ValueError(f"Neznámy model: {model_type}")

    steps.append(('classifier', clf))
    pipeline = Pipeline(steps)

    valid_params = {k: v for k, v in params.items() if '__' in k}
    pipeline.set_params(**valid_params)
    return pipeline


model = build_custom_pipeline(VYBRANY_MODEL, NASTAVENIA)

# =============================================================================
# --- 4. MANUÁLNA KRÍŽOVÁ VALIDÁCIA (5-FOLD) ---
# =============================================================================

print("\n" + "=" * 60)
print(f"SPÚŠŤAM 5-FOLD CROSS VALIDATION")
print("=" * 60)

# StratifiedKFold zabezpečí, že pomer zdravých/chorých je v každom folde rovnaký
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

acc_scores = []
mcc_scores = []

fold_no = 1
for train_index, test_index in skf.split(X_clean, y):
    # Rozdelenie na Train a Test pre tento konkrétny "fold"
    X_train_fold = X_clean.iloc[train_index]
    X_test_fold = X_clean.iloc[test_index]
    y_train_fold = y.iloc[train_index]
    y_test_fold = y.iloc[test_index]

    # Trénovanie
    model.fit(X_train_fold, y_train_fold)

    # Predikcia
    y_pred_fold = model.predict(X_test_fold)

    # Metriky
    acc = accuracy_score(y_test_fold, y_pred_fold)
    mcc = matthews_corrcoef(y_test_fold, y_pred_fold)

    acc_scores.append(acc)
    mcc_scores.append(mcc)

    print(f"Fold {fold_no}: Accuracy = {acc:.2%} | MCC = {mcc:.4f}")
    fold_no += 1

# =============================================================================
# --- 5. FINÁLNE VYHODNOTENIE ---
# =============================================================================

print("-" * 60)
print(f"PRIEMERNÉ VÝSLEDKY ({VYBRANY_MODEL}):")
print(f"Accuracy: {np.mean(acc_scores):.2%}  (+/- {np.std(acc_scores):.2%})")
print(f"MCC:      {np.mean(mcc_scores):.4f}  (+/- {np.std(mcc_scores):.4f})")
print("-" * 60)

if np.std(mcc_scores) > 0.10:
    print(">>> VAROVANIE: Model je nestabilný (veľká odchýlka medzi foldami).")
    print("    Skús znížiť max_depth alebo zvýšiť n_estimators.")
else:
    print(">>> OK: Model je stabilný.")

# =============================================================================
# --- 6. GRAF KRIVKY UČENIA ---
# =============================================================================
# Toto nám ukáže, či by pomohlo viac dát

train_sizes, train_scores, test_scores = learning_curve(
    model, X_clean, y, cv=5, scoring='matthews_corrcoef', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Tréningové skóre")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-Val skóre")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.title(f"Krivka učenia (Learning Curve) - {VYBRANY_MODEL}")
plt.xlabel("Počet trénovacích vzoriek")
plt.ylabel("MCC Skóre")
plt.legend(loc="best")
plt.grid(True)
plt.savefig('learning_curve_cv.png')  # Uložíme graf
plt.show()