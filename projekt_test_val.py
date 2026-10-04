import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
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
# --- 1. NAČÍTANIE DÁT A KONFIGURÁCIA ---
# =============================================================================
try:
    data = pd.read_csv('heart-disease_data.csv')
except FileNotFoundError:
    print("CHYBA: Súbor 'heart-disease_data.csv' sa nenašiel.")
    exit()

# Definícia pravidiel
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

# --- 2. PRÍPRAVA DÁT (FULL DATASET) ---
X = data.drop(columns=["target"])
y = data["target"]


# Funkcia na čistenie
def remove_invalid_values(X_df, continuous_ranges, categorical_values):
    X_clean = X_df.copy()
    report_rows = []

    for col, (min_val, max_val) in continuous_ranges.items():
        if col in X_df.columns:
            mask_invalid = (X_df[col] < min_val) | (X_df[col] > max_val)
            n_changes = mask_invalid.sum()
            X_clean.loc[mask_invalid, col] = np.nan
            if n_changes > 0: report_rows.append([col, "continuous", n_changes])

    for col, allowed in categorical_values.items():
        if col in X_df.columns:
            mask_invalid = ~X_df[col].isin(allowed)
            n_changes = mask_invalid.sum()
            X_clean.loc[mask_invalid, col] = np.nan
            if n_changes > 0: report_rows.append([col, "categorical", n_changes])

    if report_rows:
        report_df = pd.DataFrame(report_rows, columns=["feature", "type", "num_replaced"])
    else:
        report_df = pd.DataFrame(columns=["feature", "type", "num_replaced"])

    return X_clean, report_df


# ČISTENIE CELÉHO DATASETU
X_clean, cleaning_report = remove_invalid_values(X, spojite_rozsahy, kategoricke_hodnoty)

print(f"Dataset pripravený (100% dát). Shape: {X_clean.shape}")

# =============================================================================
# --- 3. KONFIGURÁCIA MODELOV (ROZŠÍRENÉ PARAMETRE) ---
# =============================================================================

models_config = {
    # 'SVM': {
    #     'model': SVC(probability=True, random_state=42),
    #     'params': {
    #         'classifier__C': [0.001,0.005,0.01,0.05, 0.1, 0.5, 1, 5, 10],
    #         'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    #         'classifier__gamma': ['scale', 'auto', 0.1, 1],
    #         'classifier__degree': [2, 3],
    #         'classifier__coef0': [0.0, 1.0, 5.0],
    #         'classifier__class_weight': [None, 'balanced']
    #     }
    # },
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
#     'GradientBoosting': {
#         'model': GradientBoostingClassifier(random_state=42),
#         'params': {
#             'classifier__n_estimators': [100, 200, 300],
#             'classifier__learning_rate': [0.01, 0.05, 0.1],
#             'classifier__max_depth': [3, 4, 5],
#             'classifier__subsample': [0.8, 0.9, 1.0],
#             'classifier__min_samples_leaf': [1, 2, 4],
#             'classifier__max_features': ['sqrt', None]
#         }
#     }
 }
#
# # --- PRIDANIE XGBOOST (ROZŠÍRENÉ) ---
# if XGBClassifier:
#     models_config['XGBoost'] = {
#         'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
#         'params': {
#             'classifier__n_estimators': [200,300,500],
#             'classifier__learning_rate': [0.01],
#             'classifier__max_depth': [2,3,4],
#             'classifier__min_child_weight': [1,3,5],
#             'classifier__gamma': [0.0,0.1,0.5,1],
#             'classifier__subsample': [0.7,0.8,1],
#             'classifier__colsample_bytree': [0.7,0.9],
#             'classifier__reg_alpha': [0,0.001,0.1],
#             'classifier__reg_lambda': [1,2,5]
#         }
#     }

# --- PRIDANIE CATBOOST (ROZŠÍRENÉ) ---
# if CatBoostClassifier:
#     models_config['CatBoost'] = {
#         'model': CatBoostClassifier(verbose=0, random_state=42),
#         'params': {
#             'classifier__iterations': [100, 300],
#             'classifier__depth': [4, 6],
#             'classifier__learning_rate': [0.01, 0.03, 0.1],
#             'classifier__l2_leaf_reg': [1, 3, 5, 9],
#             'classifier__random_strength': [1, 5],
#             'classifier__bagging_temperature': [0, 1]
#         }
#     }
#
#
def get_pipeline(classifier):
    return Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('var_thresh', VarianceThreshold(threshold=0)),  # Odstráni konštantné stĺpce
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif, k='all')),
        ('classifier', classifier)
    ])


def print_fold_results(model_name, results):
    df = pd.DataFrame(results)
    print(f"\n[Súhrn {model_name}] Priemer MCC: {df['mcc'].mean():.4f} | Priemer ACC: {df['acc'].mean():.4f}")
    return df


# =============================================================================
# --- 4. NESTED CROSS-VALIDATION (MCC OPTIMIZED + DETAILNÝ VÝPIS) ---
# =============================================================================

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

nested_cv_results = {}
best_overall_mcc = -1
best_model_name = ""

print("\n" + "=" * 60)
print("SPÚŠŤAM FULL NESTED CROSS-VALIDATION (Optimalizácia na MCC)")
print(f"Testujem modely: {list(models_config.keys())}")
print("=" * 60)

for name, config in models_config.items():
    print(f"\n>>> MODEL: {name}")
    print("-" * 60)

    pipeline = get_pipeline(config['model'])
    fold_results = {'fold': [], 'acc': [], 'mcc': [], 'best_params': []}

    # Outer Loop
    for i, (train_idx, val_idx) in enumerate(outer_cv.split(X_clean, y)):

        X_out_train, X_out_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
        y_out_train, y_out_val = y.iloc[train_idx], y.iloc[val_idx]

        # Inner Loop (GridSearchCV)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=config['params'],
            cv=inner_cv,
            scoring='matthews_corrcoef',  # Optimalizácia podľa MCC
            n_jobs=-1
        )

        try:
            grid.fit(X_out_train, y_out_train)
        except Exception as e:
            print(f"CHYBA pri trénovaní {name} vo folde {i + 1}: {e}")
            continue

        # Validácia
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_out_val)

        acc = accuracy_score(y_out_val, y_pred)
        mcc = matthews_corrcoef(y_out_val, y_pred)

        # --- VÝPIS PRE DANÝ FOLD ---
        print(f"Fold {i + 1}/5:")
        print(f"   MCC: {mcc:.4f} | ACC: {acc:.4f}")
        print(f"   Best Params: {grid.best_params_}")
        print("-" * 30)

        fold_results['fold'].append(i + 1)
        fold_results['acc'].append(acc)
        fold_results['mcc'].append(mcc)
        fold_results['best_params'].append(grid.best_params_)

    # Uloženie a výpis výsledkov
    if fold_results['acc']:
        df_res = print_fold_results(name, fold_results)

        mean_mcc = df_res['mcc'].mean()

        # Uložíme najčastejšiu konfiguráciu
        best_params_mode = df_res['best_params'].mode()
        if not best_params_mode.empty:
            # Ak sú parametre slovníky, musíme ich previesť na hashovateľný typ pre mode()
            # Alebo jednoducho zoberieme prvý najlepší zoznam
            # Pre jednoduchosť: zoberieme ten s najlepším MCC ak je zhoda, inak prvý
            # Tu berieme prvý mode
            final_best_params = best_params_mode[0]
        else:
            final_best_params = df_res['best_params'][0]

        nested_cv_results[name] = {'score': mean_mcc, 'best_params': final_best_params}

        if mean_mcc > best_overall_mcc:
            best_overall_mcc = mean_mcc
            best_model_name = name

print("\n" + "*" * 60)
if best_model_name:
    print(f"CELKOVÝ VÍŤAZ (podľa MCC): {best_model_name}")
    print(f"Dosiahnuté priemerné MCC: {best_overall_mcc:.4f}")
else:
    print("Žiadny model sa nepodarilo úspešne natrénovať.")
    exit()
print("*" * 60)

# =============================================================================
# --- 5. FINÁLNY MODEL A KRIVKA UČENIA (MCC) ---
# =============================================================================

print(f"\nGenerujem krivku učenia pre víťaza ({best_model_name}) na celom datasete...")

winner_config = models_config[best_model_name]
final_pipeline = get_pipeline(winner_config['model'])

# Fixné parametre pre krivku (použijeme tie najlepšie zistené)
# GridSearchCV potrebuje parametre ako zoznamy
fixed_params = {k: [v] for k, v in nested_cv_results[best_model_name]['best_params'].items()}

final_grid_for_curve = GridSearchCV(
    estimator=final_pipeline,
    param_grid=fixed_params,
    cv=inner_cv,
    scoring='matthews_corrcoef',
    n_jobs=-1
)

# Vykreslenie krivky
train_sizes, train_scores, test_scores = learning_curve(
    final_grid_for_curve, X_clean, y, cv=5, scoring='matthews_corrcoef', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(10, 6))
plt.title(f"Krivka Učenia (MCC) - {best_model_name} (Full Validation)")
plt.xlabel("Počet trénovacích vzoriek")
plt.ylabel("MCC Skóre")
plt.grid()

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_mean, 'o-', color="r", label="Train MCC")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-Validation MCC")

plt.legend(loc="best")
plt.savefig('learning_curve.png') # Saving the plot as an image file.
print("Graf krivky učenia bol uložený ako 'learning_curve.png'.")
print("\nHotovo. Výsledky sú optimalizované pre MCC.")

# ---