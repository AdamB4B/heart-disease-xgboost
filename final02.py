import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix,
    roc_curve
)
from sklearn.tree import DecisionTreeClassifier
import warnings

# Pokus o import XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("POZOR: Knižnica 'xgboost' nie je nainštalovaná. Model XGBoost nebude fungovať.")

# Potlačenie warningov
warnings.filterwarnings('ignore')

# =============================================================================
# --- 1. KONFIGURÁCIA ---
# =============================================================================

seed = 42
TEST_SIZE = 0.2  # 20% dát pôjde do nezávislého testovacieho setu

# --- VÝBER MODELU (Odkomentuj ten, ktorý chceš použiť) ---

# --- A) MODEL: ADABOOST ---
# VYBRANY_MODEL = "AdaBoost"
# NASTAVENIA = {
#     'classifier__n_estimators': 100,
#     'classifier__learning_rate': 1.0,
#     'classifier__algorithm': 'SAMME.R',
# }

# --- B) MODEL: XGBOOST ---
VYBRANY_MODEL = "XGBoost"
NASTAVENIA = {
    'classifier__n_estimators': 3000,
    'classifier__learning_rate': 0.001,
    'classifier__max_depth': 4,
    'classifier__min_child_weight': 5,
    'classifier__gamma': 0,
    'classifier__subsample': 0.7,
    'classifier__colsample_bytree': 0.8,
    'classifier__reg_alpha': 0.1,
    'classifier__reg_lambda': 1
}


# NASTAVENIA = {
#     'classifier__n_estimators': 300,
#     'classifier__learning_rate': 0.01,
#     'classifier__max_depth': 6,
#     'classifier__min_child_weight': 3,
#     'classifier__gamma': 1,
#     'classifier__subsample': 0.8,
#     'classifier__colsample_bytree': 0.7,
#     'classifier__reg_alpha': 0,
#     'classifier__reg_lambda': 1
# }



# --- C) MODEL: RANDOM FOREST ---
# VYBRANY_MODEL = "RandomForest"
# NASTAVENIA = {
#     'classifier__n_estimators': 200,
#     'classifier__max_depth': 4,
#     'classifier__min_samples_split': 4,
#     'classifier__min_samples_leaf': 3,
#     'classifier__max_features': 'log2',
#     'classifier__min_impurity_decrease': 0.002,
#     'classifier__class_weight': 'balanced',
#     'classifier__criterion': 'gini',
#     'classifier__ccp_alpha': 0.0,
#     'classifier__bootstrap': True,
#     'classifier__random_state': seed
# }


# --- D) MODEL: SVM ---
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

    # Generujeme vzorky
    X_dummy, y_dummy = make_classification(n_samples=400, n_features=13, random_state=seed, flip_y=0.1)
    data = pd.DataFrame(X_dummy, columns=[f'feat_{i}' for i in range(13)])
    # Pridáme fiktívne stĺpce s názvami, aby kód prešiel
    data['cp'] = np.random.choice([0, 1, 2, 3], size=400)
    data['thal'] = np.random.choice([0, 1, 2, 3], size=400)
    data['age'] = np.random.randint(20, 80, size=400)
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


def remove_invalid_values(X_in):
    """
    Funkcia nastaví hodnoty mimo rozsahu na NaN.
    Dôležité: Musí sa volať PRED One-Hot Encodingom.
    """
    X_out = X_in.copy()
    # 1. Čistenie spojitých premenných
    for col, (min_v, max_v) in spojite_rozsahy.items():
        if col in X_out.columns:
            mask = (X_out[col] < min_v) | (X_out[col] > max_v)
            X_out.loc[mask, col] = np.nan

    # 2. Čistenie kategorických premenných
    for col, allowed in kategoricke_hodnoty.items():
        if col in X_out.columns:
            mask = ~X_out[col].isin(allowed)
            X_out.loc[mask, col] = np.nan
    return X_out


# --- ZMENA LOGIKY ---
# 1. Najprv vyčistíme dáta, kým majú pôvodné názvy stĺpcov
data_clean = remove_invalid_values(data)

# 2. Rozdelenie na X a y
X_full = data_clean.drop(columns=["target"])
y_full = data_clean["target"]

# 3. Poznámka k Encodingu:
# Tu sme odstránili pd.get_dummies().
# Prečo? Chceme, aby KNN Imputer videl napr. 'cp' ako jeden stĺpec s hodnotami 0,1,2,3 a NaN.
# Ak by sme spravili dummies, NaN by zmizli (boli by 0 všade) a Imputer by nemal čo opravovať.
# Modely ako RandomForest/XGBoost si poradia aj s číselnou reprezentáciou kategórií.

# Rozdelenie na TRAIN (pre CV) a NEZÁVISLÝ TEST
X_train_cv, X_test_final, y_train_cv, y_test_final = train_test_split(
    X_full, y_full, test_size=TEST_SIZE, random_state=seed, stratify=y_full
)

print(f"Dataset pripravený.")
print(f"  Tréning + CV Set: {X_train_cv.shape}")
print(f"  Nezávislý Test Set: {X_test_final.shape}")
print(f"  Počet chýbajúcich hodnôt (NaN) v Train sete: {X_train_cv.isna().sum().sum()}")


# =============================================================================
# --- 3. PIPELINE a POMOCNÉ FUNKCIE ---
# =============================================================================

def build_custom_pipeline(model_type, params):
    steps = []
    # 1. Imputer (KNN) - teraz reálne imputuje aj kategórie (lebo sú NaN)
    steps.append(('imputer', KNNImputer(n_neighbors=3)))

    # 2. Scaler
    steps.append(('scaler', StandardScaler()))

    # 3. Model
    if model_type == "SVM":
        clf = SVC(random_state=seed, probability=True)

    elif model_type == "RandomForest":
        clf = RandomForestClassifier(random_state=seed)

    elif model_type == "GradientBoosting":
        clf = GradientBoostingClassifier(random_state=seed)

    elif model_type == "AdaBoost":
        base_estimator = DecisionTreeClassifier(max_depth=1, random_state=seed)
        clf = AdaBoostClassifier(estimator=base_estimator, random_state=seed)

    elif model_type == "XGBoost":
        if XGBClassifier is None:
            raise ImportError("XGBoost nie je nainštalovaný.")
        clf = XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric='logloss')

    else:
        raise ValueError(f"Neznámy model: {model_type}")

    steps.append(('classifier', clf))
    pipeline = Pipeline(steps)

    # Filtrovanie parametrov
    valid_params = {k: v for k, v in params.items() if '__' in k}

    # Bezpečnejšie nastavenie parametrov
    try:
        pipeline.set_params(**valid_params)
    except ValueError as e:
        print(f"Varovanie: Parameter nesedí pre {model_type}: {e}")

    return pipeline


def evaluate_model(y_true, y_pred, y_proba=None, print_result=True, context=""):
    """Vypočíta a voliteľne vytlačí sadu metrík."""
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['F1-Score'] = f1_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred)
    metrics['Recall'] = recall_score(y_true, y_pred)
    if y_proba is not None:
        try:
            metrics['AUC-ROC'] = roc_auc_score(y_true, y_proba[:, 1])
        except ValueError:
            metrics['AUC-ROC'] = np.nan

    if print_result:
        print(f"\n--- {context} METRIKY ---")
        print(f"Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"MCC:       {metrics['MCC']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall:    {metrics['Recall']:.4f}")
        if 'AUC-ROC' in metrics:
            print(f"AUC-ROC:   {metrics['AUC-ROC']:.4f}")

    return metrics


# Zostavenie pipeline
model = build_custom_pipeline(VYBRANY_MODEL, NASTAVENIA)

# =============================================================================
# --- 4. MANUÁLNA KRÍŽOVÁ VALIDÁCIA (5-FOLD) ---
# =============================================================================

print(f"\nSPÚŠŤAM 5-FOLD CV PRE: {VYBRANY_MODEL} (Na Train/CV Sete)")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# Zoznamy na uloženie všetkých metrík
all_metrics = {key: [] for key in ['Accuracy', 'F1-Score', 'MCC', 'Precision', 'Recall', 'AUC-ROC']}

fold_no = 1
for train_index, test_index in skf.split(X_train_cv, y_train_cv):
    X_train_fold = X_train_cv.iloc[train_index]
    X_test_fold = X_train_cv.iloc[test_index]
    y_train_fold = y_train_cv.iloc[train_index]
    y_test_fold = y_train_cv.iloc[test_index]

    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_test_fold)

    try:
        y_proba_fold = model.predict_proba(X_test_fold)
    except AttributeError:
        y_proba_fold = None

    # Získanie metrík bez výpisu (výpis spravíme ručne o riadok nižšie)
    metrics = evaluate_model(y_test_fold, y_pred_fold, y_proba_fold, print_result=False)

    print(
        f"Fold {fold_no}: Acc={metrics['Accuracy']:.2%} | F1={metrics['F1-Score']:.4f} | MCC={metrics['MCC']:.4f} | AUC={metrics.get('AUC-ROC', np.nan):.4f}")

    for key, value in metrics.items():
        all_metrics[key].append(value)

    fold_no += 1

# =============================================================================
# --- 5. FINÁLNE VYHODNOTENIE KRÍŽOVEJ VALIDÁCIE ---
# =============================================================================

print(f"\nSÚHRNNÉ VÝSLEDKY KRÍŽOVEJ VALIDÁCIE (CV) PRE {VYBRANY_MODEL}:")

for metric_name in ['Accuracy', 'F1-Score', 'MCC', 'Precision', 'Recall', 'AUC-ROC']:
    scores = [s for s in all_metrics[metric_name] if not np.isnan(s)]
    if scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric_name:9}: {mean_score:.4f}  (+/- {std_score:.4f})")
    else:
        print(f"{metric_name:9}: N/A")

print("-" * 60)

# =============================================================================
# --- 6. VYHODNOTENIE NA NEZÁVISLOM TESTOVACOM SPLITE ---
# =============================================================================

print(f"VYHODNOTENIE NA NEZÁVISLOM TESTE PRE {VYBRANY_MODEL}")

# Natrénujeme model na celom Train/CV sete
model_final = build_custom_pipeline(VYBRANY_MODEL, NASTAVENIA)
model_final.fit(X_train_cv, y_train_cv)

# Predikcia na Nezávislom Teste
y_pred_final = model_final.predict(X_test_final)

try:
    y_proba_final = model_final.predict_proba(X_test_final)
except AttributeError:
    y_proba_final = None

# Vyhodnotenie
evaluate_model(y_test_final, y_pred_final, y_proba_final, context="NEZÁVISLÝ TEST")

# Konfúzna matica
print("\nKonfúzna Matica (Nezávislý Test):")
print(confusion_matrix(y_test_final, y_pred_final))

# =============================================================================
# --- 7. GRAF KRIVKY UČENIA (VOLITEĽNÉ) ---
# =============================================================================
# Kód pre krivky učenia je zakomentovaný, aby sa zrýchlil beh, ak ho nepotrebuješ
print("\nHotovo.")