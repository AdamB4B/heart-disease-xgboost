# model_training_evaluation.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Načítanie finálnych dát ---
try:
    # Načítame finálne sady vytvorené v predchádzajúcom skripte
    X_train = pd.read_csv('X_train_final.csv')
    X_test = pd.read_csv('X_test_final.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()
    y_test = pd.read_csv('y_test.csv').squeeze()
except FileNotFoundError:
    print("❌ Chyba: Finálne súbory neboli nájdené. Uistite sa, že súbory existujú.")
    exit()

print(f"Trénovacia sada: {X_train.shape}, Testovacia sada: {X_test.shape}")
print("-" * 50)

# --- 2. Škálovanie (Standardizácia) ---
# Trénujeme scaler na X_train a transformujeme obe sady.
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# --- 3. Funkcia pre trénovanie a vyhodnotenie ---

def evaluate_model(model, X_train_s, y_train_s, X_test_s, y_test_s, model_name):
    """Trénuje model, predikuje na testovacej sade a vypočíta metriky."""

    # Trénovanie modelu
    model.fit(X_train_s, y_train_s)

    # Predikcia
    y_pred = model.predict(X_test_s)

    # Matica zámen (Confusion Matrix)
    cm = confusion_matrix(y_test_s, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Matthews Correlation Coefficient (hlavná metrika projektu)
    mcc = matthews_corrcoef(y_test_s, y_pred)

    # Presnosť
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"### Výsledky pre model: {model_name} ###")
    print(f"Matica zámen (TN, FP / FN, TP):\n {cm}")
    print(f"True Positives (TP): {TP}, True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}, False Negatives (FN): {FN}")
    print(f"Presnosť (Accuracy): {accuracy:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print("-" * 50)

    return mcc, model


# --- 4. Trénovanie a vyhodnotenie vybraných modelov ---

best_mcc = -1
best_model_name = ""
best_model = None

# A) Logistická regresia (Vyžaduje škálované dáta)
lr = LogisticRegression(random_state=42, solver='liblinear')
mcc_lr, model_lr = evaluate_model(lr, X_train_scaled, y_train, X_test_scaled, y_test, "Logistická regresia")
if mcc_lr > best_mcc:
    best_mcc = mcc_lr
    best_model_name = "Logistická regresia"
    best_model = model_lr

# B) Random Forest (Nevyžaduje škálované dáta)
rf = RandomForestClassifier(random_state=42, n_estimators=100)
mcc_rf, model_rf = evaluate_model(rf, X_train, y_train, X_test, y_test, "Random Forest")
if mcc_rf > best_mcc:
    best_mcc = mcc_rf
    best_model_name = "Random Forest"
    best_model = model_rf

# C) Support Vector Machine (SVC) (Vyžaduje škálované dáta)
svc = SVC(random_state=42, kernel='linear')
mcc_svc, model_svc = evaluate_model(svc, X_train_scaled, y_train, X_test_scaled, y_test, "SVC (Linear)")
if mcc_svc > best_mcc:
    best_mcc = mcc_svc
    best_model_name = "SVC (Linear)"
    best_model = model_svc

# --- 5. Hyperparameter Tuning pre najlepší model ---

print(f"🔥 Optimálizácia najlepšieho modelu: {best_model_name}")

# Použijeme GridSearchCV na najlepší model s 5-násobnou krížovou validáciou
# a optimalizujeme priamo na metriku MCC.

if best_model_name == "Logistická regresia":
    model_to_tune = LogisticRegression(solver='liblinear', random_state=42)
    param_grid = {'C': [0.01, 0.1, 1, 10]}
    X_data = X_train_scaled
    X_test_data = X_test_scaled

elif best_model_name == "Random Forest":
    model_to_tune = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
    X_data = X_train
    X_test_data = X_test

elif "SVC" in best_model_name:
    model_to_tune = SVC(random_state=42)
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    X_data = X_train_scaled
    X_test_data = X_test_scaled

else:
    # Ak by nebol nájdený žiadny model
    print("Pre hyperparameter tuning nebol vybraný žiadny model.")
    exit()

grid_search = GridSearchCV(
    model_to_tune,
    param_grid,
    cv=5,
    scoring='matthews_corrcoef',
    verbose=1,
    n_jobs=-1  # Použije všetky jadrá
)

grid_search.fit(X_data, y_train)

# Vyhodnotenie optimalizovaného modelu na testovacej sade
best_tuned_model = grid_search.best_estimator_
mcc_tuned, _ = evaluate_model(
    best_tuned_model,
    X_data,
    y_train,
    X_test_data,
    y_test,
    f"TUNED {best_model_name} (Najlepšie params: {grid_search.best_params_})"
)

if mcc_tuned > best_mcc:
    best_mcc = mcc_tuned
    best_model_name = f"TUNED {best_model_name}"
    best_model = best_tuned_model

# --- 6. Finálny Výstup ---

print(f"\n==========================================")
print(f"🚀 Celkovo NAJLEPŠÍ VÝSLEDOK: {best_model_name} s MCC: {best_mcc:.4f}")
print(f"==========================================")

# Uloženie najlepšieho modelu
joblib.dump(best_model, 'final_cardio_model.joblib')
print(f"Finálny model uložený ako 'final_cardio_model.joblib'.")