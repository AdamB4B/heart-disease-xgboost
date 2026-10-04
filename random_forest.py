import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
import joblib

# --- 1. Načítanie finálnych dát ---
try:
    X_train = pd.read_csv('X_train_final.csv')
    X_test = pd.read_csv('X_test_final.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()
    y_test = pd.read_csv('y_test.csv').squeeze()
except FileNotFoundError:
    print("❌ Chyba: Finálne súbory neboli nájdené.")
    exit()

print(f"Trénovacia sada: {X_train.shape}, Testovacia sada: {X_test.shape}")
print("-" * 50)


# --- 2. Funkcia na trénovanie a vyhodnotenie ---
def evaluate_model(model, X_train_s, y_train_s, X_test_s, y_test_s):
    """Trénuje model, predikuje a vypočíta MCC a confusion matrix."""

    model.fit(X_train_s, y_train_s)
    y_pred = model.predict(X_test_s)

    cm = confusion_matrix(y_test_s, y_pred)
    TN, FP, FN, TP = cm.ravel()
    mcc = matthews_corrcoef(y_test_s, y_pred)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"Matica zámen (TN, FP / FN, TP):\n{cm}")
    print(f"Presnosť (Accuracy): {accuracy:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print("-" * 50)

    return mcc, model


# --- 3. Trénovanie základného Random Forest ---
rf = RandomForestClassifier(random_state=42, n_estimators=200) 
print("Trénovanie základného Random Forest...")
mcc_rf, model_rf = evaluate_model(rf, X_train, y_train, X_test, y_test)


# --- 4. Hyperparameter Tuning ---
print("🔥 Spúšťam GridSearchCV pre Random Forest...")

param_grid = {'n_estimators': [50,100,200,300], 'max_depth': [1,3,5,7]}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='matthews_corrcoef',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print(f"Najlepšie parametre: {grid_search.best_params_}")

# # Vyhodnotenie tunovaného modelu
# print("Vyhodnotenie tunovaného Random Forest...")
# mcc_tuned, _ = evaluate_model(best_rf, X_train, y_train, X_test, y_test)

# --- 5. Uloženie finálneho modelu ---
joblib.dump(best_rf, 'final_rf_cardio_model.joblib')
print("✅ Finálny Random Forest model uložený ako 'final_rf_cardio_model.joblib'.")
