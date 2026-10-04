import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
# PRIDANÉ: VarianceThreshold na odstránenie konštantných stĺpcov
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

# =============================================================================
# --- 1. NAČÍTANIE DÁT A KONFIGURÁCIA ---
# =============================================================================
try:
    data = pd.read_csv('heart-disease_data.csv')
    print("Dáta úspešne načítané.")
except FileNotFoundError:
    print("CHYBA: Súbor 'heart-disease_data.csv' sa nenašiel.")
    data = pd.DataFrame()

spojite_rozsahy = {
    'age': (0, 120),
    'trestbps': (60, 250),
    'chol': (80, 700),
    'thalach': (50, 230),
    'oldpeak': (0, 5)
}

kategoricke_hodnoty = {
    'sex': [0, 1],
    'cp': [0, 1, 2, 3],
    'fbs': [0, 1],
    'restecg': [0, 1, 2],
    'exang': [0, 1],
    'slope': [0, 1, 2],
    'ca': [0, 1, 2, 3],
    'thal': [0, 1, 2, 3]
}

# One-Hot Encoding
cols_to_encode = ['cp', 'restecg', 'slope', 'thal']
cols_to_encode = [c for c in cols_to_encode if c in data.columns]
data = pd.get_dummies(data, columns=cols_to_encode, drop_first=True)
print(f"Počet stĺpcov po One-Hot Encodingu: {data.shape[1]}")

# =============================================================================
# --- 2. ROZDELENIE DÁT (SPLIT) ---
# =============================================================================
X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Reset indexov
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# =============================================================================
# --- 3. ČISTENIE DÁT (REMOVE INVALID) ---
# =============================================================================
def remove_invalid_values(
        X: pd.DataFrame,
        continuous_ranges: dict,
        categorical_values: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_clean = X.copy()
    report_rows = []

    for col, (min_val, max_val) in continuous_ranges.items():
        if col in X.columns:
            mask_invalid = (X[col] < min_val) | (X[col] > max_val)
            n_changes = mask_invalid.sum()
            if n_changes > 0:
                X_clean.loc[mask_invalid, col] = np.nan
            report_rows.append([col, "continuous", n_changes])

    for col, allowed in categorical_values.items():
        if col in X.columns:
            mask_invalid = ~X[col].isin(allowed)
            n_changes = mask_invalid.sum()
            if n_changes > 0:
                X_clean.loc[mask_invalid, col] = np.nan
            report_rows.append([col, "categorical", n_changes])

    report_df = pd.DataFrame(report_rows, columns=["feature", "type", "num_replaced"])
    return X_clean, report_df


X_train_clean, cleaning_report = remove_invalid_values(
    X=X_train,
    continuous_ranges=spojite_rozsahy,
    categorical_values=kategoricke_hodnoty
)

print("\n--- Report čistenia (Trénovacia sada) ---")
print(cleaning_report[cleaning_report['num_replaced'] > 0])

# =============================================================================
# --- 4. KONFIGURÁCIA MODELOV A PIPELINE ---
# =============================================================================

models_config = {
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        }
    }
}


def get_pipeline(classifier):
    """
    Pipeline s opravou varovaní:
    1. Imputer: Doplní NaN.
    2. VarianceThreshold: Odstráni stĺpce, ktoré majú nulovú varianciu (samé 0 alebo samé 1).
    3. Scaler: Normalizácia.
    4. Selector: Výber rysov.
    5. Classifier: Model.
    """
    return Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('var_thresh', VarianceThreshold(threshold=0)),  # TOTO JE PRIDANÉ (Odstráni konštanty)
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif, k='all')),
        ('classifier', classifier)
    ])


def print_fold_results(model_name, results):
    df = pd.DataFrame(results)
    print(f"\nModel: {model_name} | Priemer ACC: {df['acc'].mean():.4f} | Priemer MCC: {df['mcc'].mean():.4f}")
    return df


# =============================================================================
# --- 5. NESTED CROSS-VALIDATION ---
# =============================================================================
# Pre ticho od sklearn (voliteľné, ak by sa stále objavovali nejaké warnings)
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

nested_cv_results = {}
best_overall_score = -1
best_model_name = ""

print("\n" + "=" * 50)
print("SPÚŠŤAM NESTED CROSS-VALIDATION")
print("=" * 50)

for name, config in models_config.items():
    print(f"--> Trénujem: {name} ...")

    pipeline = get_pipeline(config['model'])
    fold_results = {'fold': [], 'acc': [], 'mcc': [], 'best_params': []}

    for i, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_clean, y_train)):
        X_out_train = X_train_clean.iloc[train_idx]
        X_out_val = X_train_clean.iloc[val_idx]
        y_out_train = y_train.iloc[train_idx]
        y_out_val = y_train.iloc[val_idx]

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=config['params'],
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )

        grid.fit(X_out_train, y_out_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_out_val)

        acc = accuracy_score(y_out_val, y_pred)
        mcc = matthews_corrcoef(y_out_val, y_pred)

        fold_results['fold'].append(i)
        fold_results['acc'].append(acc)
        fold_results['mcc'].append(mcc)
        fold_results['best_params'].append(grid.best_params_)

    df_res = print_fold_results(name, fold_results)
    mean_acc = df_res['acc'].mean()
    nested_cv_results[name] = mean_acc

    if mean_acc > best_overall_score:
        best_overall_score = mean_acc
        best_model_name = name

print("\n" + "*" * 50)
print(f"VÍŤAZ NESTED CV: {best_model_name} (ACC: {best_overall_score:.4f})")
print("*" * 50)

# =============================================================================
# --- 6. FINÁLNY TRÉNING A TESTOVANIE ---
# =============================================================================

print(f"\nPrebieha finálny tréning modelu {best_model_name} na celom X_train...")

X_test_clean, _ = remove_invalid_values(
    X=X_test,
    continuous_ranges=spojite_rozsahy,
    categorical_values=kategoricke_hodnoty
)

winner_config = models_config[best_model_name]
final_pipeline = get_pipeline(winner_config['model'])

final_grid = GridSearchCV(
    estimator=final_pipeline,
    param_grid=winner_config['params'],
    cv=inner_cv,
    scoring='accuracy',
    n_jobs=-1
)

final_grid.fit(X_train_clean, y_train)

final_model = final_grid.best_estimator_
y_test_pred = final_model.predict(X_test_clean)

final_acc = accuracy_score(y_test, y_test_pred)
final_mcc = matthews_corrcoef(y_test, y_test_pred)

print("\n" + "=" * 40)
print(f"VÝSLEDKY NA NEZÁVISLEJ TESTOVACEJ SADE")
print("=" * 40)
print(f"Víťazný model:      {best_model_name}")
print(f"Najlepšie parametre: {final_grid.best_params_}")
print("-" * 40)
print(f"Accuracy:           {final_acc:.4f}")
print(f"MCC:                {final_mcc:.4f}")
print("=" * 40)