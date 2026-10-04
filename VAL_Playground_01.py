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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
import warnings

# Pokus o import XGBoost (ak nie je nainštalovaný, skript nespadne hneď, ale až pri použití)
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("POZOR: Knižnica 'xgboost' nie je nainštalovaná. Model XGBoost nebude fungovať.")

# Potlačenie warningov pre krajší výpis
warnings.filterwarnings('ignore')

# =============================================================================
# --- 1. KONFIGURÁCIA ---
# =============================================================================

seed = 42

# --- VÝBER MODELU (Odkomentuj ten, ktorý chceš použiť) ---

# --- A) MODEL: ADABOOST ---
# VYBRANY_MODEL = "AdaBoost"
# NASTAVENIA = {
#     'classifier__n_estimators': 100,
#     'classifier__learning_rate': 1.0,  # Menší learning rate zvyčajne vyžaduje viac n_estimators
#     'classifier__algorithm': 'SAMME',  # 'SAMME' alebo 'SAMME.R'
# }

# --- B) MODEL: XGBOOST ---
VYBRANY_MODEL = "XGBoost"
NASTAVENIA = {
    'classifier__n_estimators': 300,
    'classifier__learning_rate': 0.001,
    'classifier__max_depth': 3,
    'classifier__min_child_weight':3,
    'classifier__gama':0.1  ,
    'classifier__subsample': 0.8,  # Prevencia preučenia (použije len 80% dát pre strom)
    'classifier__colsample_bytree': 0.7,  # Prevencia preučenia (použije len 80% stĺpcov)
    'classifier__reg_alpha':0.001,
    'classifier__reg_lamba':2
}

# --- C) MODEL: RANDOM FOREST (Pôvodný) ---
# VYBRANY_MODEL = "RandomForest"
# NASTAVENIA = {
#     'classifier__n_estimators': 200,
#     'classifier__max_depth': 6,
#     'classifier__min_samples_split': 2,
#     'classifier__min_samples_leaf': 2,
#     'classifier__max_features': 'sqrt',
#     'classifier__class_weight': 'balanced'
# }

# --- D) MODEL: SVM (Pôvodný) ---
# VYBRANY_MODEL = "SVM"
# NASTAVENIA = {
#     'classifier__C': 1.0,
#     'classifier__kernel': 'rbf',
#     'classifier__gamma': 'scale',
#     'classifier__class_weight': 'balanced',
#     'classifier__probability': True
# }


# =============================================================================
# --- 2. NAČÍTANIE A PRÍPRAVA DÁT ---
# =============================================================================
try:
    data = pd.read_csv('heart-disease_data.csv')
except FileNotFoundError:
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

# One-Hot Encoding
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


# Vyčistíme celý dataset
X_clean = remove_invalid_values(X)
print(f"Dataset pripravený na Cross-Validáciu: {X_clean.shape}")


# =============================================================================
# --- 3. PIPELINE ---
# =============================================================================

def build_custom_pipeline(model_type, params):
    steps = []
    # 1. Imputer (Doplnenie chýbajúcich hodnôt)
    steps.append(('imputer', KNNImputer(n_neighbors=5)))

    # 2. Scaler (Škálovanie)
    # Poznámka: Pre XGBoost a AdaBoost nie je scaler nutný, ale v Pipeline nevadí a zachováva konzistenciu
    steps.append(('scaler', StandardScaler()))

    # 3. Model
    if model_type == "SVM":
        clf = SVC(random_state=seed, probability=True)

    elif model_type == "RandomForest":
        clf = RandomForestClassifier(random_state=seed)

    elif model_type == "GradientBoosting":
        clf = GradientBoostingClassifier(random_state=seed)

    elif model_type == "AdaBoost":
        # AdaBoost zvyčajne používa DecisionTreeClassifier(max_depth=1) ako základ
        clf = AdaBoostClassifier(random_state=seed, algorithm='SAMME')

    elif model_type == "XGBoost":
        if XGBClassifier is None:
            raise ImportError("XGBoost nie je nainštalovaný. Spusti: pip install xgboost")
        # eval_metric='logloss' odstráni varovania v novších verziách
        clf = XGBClassifier(random_state=seed, eval_metric='logloss')

    else:
        raise ValueError(f"Neznámy model: {model_type}")

    steps.append(('classifier', clf))
    pipeline = Pipeline(steps)

    # Filtrovanie parametrov, ktoré patria pipeline (majú '__')
    valid_params = {k: v for k, v in params.items() if '__' in k}
    pipeline.set_params(**valid_params)
    return pipeline


# Zostavenie pipeline
model = build_custom_pipeline(VYBRANY_MODEL, NASTAVENIA)

# =============================================================================
# --- 4. MANUÁLNA KRÍŽOVÁ VALIDÁCIA (5-FOLD) ---
# =============================================================================

print("\n" + "=" * 60)
print(f"SPÚŠŤAM 5-FOLD CV PRE: {VYBRANY_MODEL}")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

acc_scores = []
mcc_scores = []

fold_no = 1
for train_index, test_index in skf.split(X_clean, y):
    X_train_fold = X_clean.iloc[train_index]
    X_test_fold = X_clean.iloc[test_index]
    y_train_fold = y.iloc[train_index]
    y_test_fold = y.iloc[test_index]

    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_test_fold)

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

# =============================================================================
# --- 6. GRAF KRIVKY UČENIA ---
# =============================================================================

print("Generujem Learning Curve...")
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
plt.savefig(f'learning_curve_{VYBRANY_MODEL}.png')
plt.show()
print("Hotovo.")