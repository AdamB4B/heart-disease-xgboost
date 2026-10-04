import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from xgboost import XGBClassifier

# =============================================================================
# --- 1. KONFIGURÁCIA (TU SI MANUÁLNE NASTAV VÍŤAZNÝ MODEL) ---
# =============================================================================

# Sem napíš typ modelu: "SVM", "RandomForest" alebo "GradientBoosting"
# VYBRANY_MODEL = "SVM"
#
# # Sem skopíruj "best_params", ktoré ti vypísal predchádzajúci GridSearch
# # Príklad: {'clf__max_depth': 10, 'clf__n_estimators': 100, 'imputer__strategy': 'mean'}
# NASTAVENIA = {
#     'classifier__C': 1,
#     'classifier__kernel': 'linear',
#     'classifier__gamma': 'scale'
# }
# VYBRANY_MODEL = "RandomForest"
seed = 42

# Tvoje nastavenia
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
# VYBRANY_MODEL = "XGBoost"
# NASTAVENIA = {
#     'classifier__n_estimators': 300,
#     'classifier__learning_rate': 0.01,
#     'classifier__max_depth': 3,
#     'classifier__min_child_weight':3,
#     'classifier__gamma':0.1,
#     'classifier__subsample': 0.8,  # Prevencia preučenia (použije len 80% dát pre strom)
#     'classifier__colsample_bytree': 0.8,  # Prevencia preučenia (použije len 80% stĺpcov)
#     'classifier__reg_alpha':0.001,
#     'classifier__reg_lambda':2
# }

# vyskusat alfu ...

# =============================================================================
# --- 2. NAČÍTANIE A PRÍPRAVA DÁT ---
# =============================================================================
data = pd.read_csv('heart-disease_data.csv')

# Pravidlá pre čistenie
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
data = pd.get_dummies(data, columns=cols_to_encode, drop_first=True)

X = data.drop(columns=["target"])
y = data["target"]

# --- SPLIT 80/20 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=seed
)


# Funkcia na čistenie (vkladanie NaN tam, kde sú nezmysly)
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


# Aplikujeme čistenie na Train aj Test
X_train_clean = remove_invalid_values(X_train)
X_test_clean = remove_invalid_values(X_test)

print(f"Dáta pripravené. Train: {X_train_clean.shape}, Test: {X_test_clean.shape}")


# =============================================================================
# --- 3. ZOSTAVENIE MODELU PODĽA KONFIGURÁCIE ---
# =============================================================================

def build_custom_pipeline(model_type, params):
    steps = []

    # 1. Imputer
    if 'imputer__n_neighbors' in params:
        steps.append(('imputer', KNNImputer(n_neighbors=params['imputer__n_neighbors'])))
    else:
        strat = params.get('imputer__strategy', 'mean')
        steps.append(('imputer', SimpleImputer(strategy=strat)))

    # 2. Scaler (dáme vždy, pre stromy nevadí, pre SVM nutný)
    steps.append(('scaler', StandardScaler()))

    # 3. Classifier
    if model_type == "SVM":
        clf = SVC(random_state=seed, probability=True)
    elif model_type == "RandomForest":
        clf = RandomForestClassifier(random_state=seed)
    elif model_type == "GradientBoosting":
        clf = GradientBoostingClassifier(random_state=seed)
    elif model_type == "XGBoost":
        # eval_metric='logloss' pomôže potlačiť niektoré varovania
        clf = XGBClassifier(random_state=seed, eval_metric='logloss')
    else:
        raise ValueError(f"Neznámy model: {model_type}")
    steps.append(('classifier', clf))
    pipeline = Pipeline(steps)

    # Nastavenie parametrov (očistenie od prefixov ak treba, ale Pipeline to zvláda s 'clf__')
    # Filtrujeme len tie parametre, ktoré sedia do pipeline
    valid_params = {k: v for k, v in params.items() if '__' in k}
    pipeline.set_params(**valid_params)

    return pipeline


# Vytvorenie modelu
try:
    model = build_custom_pipeline(VYBRANY_MODEL, NASTAVENIA)
    print(f"\nModel {VYBRANY_MODEL} úspešne zostavený.")
except Exception as e:
    print(f"\nCHYBA pri stavaní modelu: {e}")
    print("Skontroluj názvy parametrov v sekcii KONFIGURÁCIA!")
    exit()

# =============================================================================
# --- 4. TRÉNOVANIE A TESTOVANIE (S KONTROLOU PREÚČANIA VRÁTANE MCC) ---
# =============================================================================

print("Trénujem model na 80% dát...")
model.fit(X_train_clean, y_train)

# 1. Predikcia na TRÉNOVACÍCH dátach
y_train_pred = model.predict(X_train_clean)
acc_train = accuracy_score(y_train, y_train_pred)
mcc_train = matthews_corrcoef(y_train, y_train_pred)

# 2. Predikcia na TESTOVACÍCH dátach
y_pred = model.predict(X_test_clean)
acc_test = accuracy_score(y_test, y_pred)
mcc_test = matthews_corrcoef(y_test, y_pred)

print("\n" + "="*50)
print("ANALÝZA PREÚČANIA (ACCURACY + MCC)")
print("="*50)

# Tabuľkový výpis
print(f"{'METRIKA':<12} | {'TRAIN (Učené)':<15} | {'TEST (Nové)':<15} | {'ROZDIEL':<15}")
print("-" * 65)
print(f"{'Accuracy':<12} | {acc_train*100:.2f} %{'':<8} | {acc_test*100:.2f} %{'':<8} | {(acc_train-acc_test)*100:+.2f} %")
print(f"{'MCC':<12} | {mcc_train:.4f}{'':<9} | {mcc_test:.4f}{'':<9} | {mcc_train-mcc_test:+.4f}")
print("-" * 65)

# Vyhodnotenie na základe Accuracy (pre jednoduchosť)
rozdiel_acc = (acc_train - acc_test) * 100

if rozdiel_acc > 10:
    print(">>> POZOR: Model sa výrazne PREÚČA (Overfitting)!")
    print("    MCC na Train je omnoho vyššie ako na Test.")
elif rozdiel_acc < -2:
    print(">>> Zaujímavé: Model má lepšie výsledky na testovacích dátach.")
else:
    print(">>> STAV OK: Hodnoty ACC aj MCC sú vyrovnané (Model je robustný).")

print("=" * 50)

# =============================================================================
# --- 5. VÝSLEDKY ---
# =============================================================================

acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("\n" + "=" * 30)
print(f"VÝSLEDKY VALIDÁCIE ({VYBRANY_MODEL})")
print("=" * 30)
print(f"Accuracy (Presnosť): {acc:.4f}")
print(f"MCC (Matthews Corel.): {mcc:.4f}")
print("-" * 30)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Vizualizácia konfúznej matice
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predikcia modelu')
plt.ylabel('Skutočná hodnota')
plt.title(f'Confusion Matrix - {VYBRANY_MODEL}')
plt.show()