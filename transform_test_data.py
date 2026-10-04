import pandas as pd
import numpy as np
import joblib

def remove_invalid_values(
    X: pd.DataFrame,
    continuous_ranges: dict[str, tuple[float, float]],
    categorical_values: dict[str, list[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nahradí nesmyslné nebo nepovolené hodnoty v datasetu NaN hodnotami.

    Parametry
    ---------
    X : pd.DataFrame
        Vstupní dataset s příznaky.
    continuous_ranges : Dict[str, Tuple[float, float]]
        Slovník: název sloupce -> (min, max) pro spojité proměnné.
    categorical_values : Dict[str, List[int]]
        Slovník: název sloupce -> seznam povolených kategorií.

    Návratové hodnoty
    -----------------
    Tuple[pd.DataFrame, pd.DataFrame]
        - nový DataFrame s nahrazenými nesmyslnými hodnotami,
        - DataFrame s počty nahrazených hodnot pro jednotlivé sloupce.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X musí být pandas DataFrame.")

    # Kopie pro úpravy
    X_clean = X.copy()
    report_rows = []

    # ---- Spojité proměnné ----
    for col, (min_val, max_val) in continuous_ranges.items():
        if col in X.columns:
            mask_invalid = (X[col] < min_val) | (X[col] > max_val)
            n_changes = mask_invalid.sum()
            X_clean.loc[mask_invalid, col] = np.nan
            report_rows.append([col, "continuous", n_changes])

    # ---- Kategoriální proměnné ----
    for col, allowed in categorical_values.items():
        if col in X.columns:
            mask_invalid = ~X[col].isin(allowed)
            n_changes = mask_invalid.sum()
            X_clean.loc[mask_invalid, col] = np.nan
            report_rows.append([col, "categorical", n_changes])

    report_df = pd.DataFrame(report_rows, columns=["feature", "type", "num_replaced"])
    return X_clean, report_df

# Trénovacie dáta
X_train = pd.read_csv('X_train_imputed.csv')
y_train = pd.read_csv('y_train.csv').squeeze()

# Načítanie surových testovacích dát a targetu
X_test_raw = pd.read_csv('X_test_raw.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

# Načítanie logických pravidiel a imputačných hodnôt
imputation_values = joblib.load('imputation_values_median.joblib')
spojite_rozsahy = joblib.load('spojite_rozsahy.joblib')
kategoricke_hodnoty = joblib.load('kategoricke_hodnoty.joblib')

# Aplikácia pravidiel čistenia
X_test_clean, _ = remove_invalid_values(
    X=X_test_raw,
    continuous_ranges=spojite_rozsahy,
    categorical_values=kategoricke_hodnoty
)

# Imputácia pomocou naučených hodnôt (medián/móda z X_train)
X_test_imputed = X_test_clean.copy()
kategoricke_sloupce = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Kontrola, že všechny kategorické sloupce existují
for col in kategoricke_sloupce:
    if col not in X_test_imputed.columns:
        raise KeyError(f"Sloupec '{col}' chýba v X_test_imputed! Skontrolujte pipeline.")


for col, val in imputation_values.items():
    X_test_imputed[col].fillna(val, inplace=True)

for col in kategoricke_sloupce:
    if X_test_imputed[col].isna().sum() > 0:
        print(f"⚠️ Upozornenie: Sloupec '{col}' má stále NaN hodnoty po imputácii!")
        print("   -> kompenzujem doplnením najčastejšej hodnoty z X_train")
        most_freq = imputation_values[col]
        X_test_imputed[col].fillna(most_freq, inplace=True)

    X_test_imputed[col] = X_test_imputed[col].astype(int)

# Zabezpečenie celočíselného typu
for col in kategoricke_sloupce:
    X_test_imputed[col] = X_test_imputed[col].astype(int)

print("✅ Testovacie dáta X_test_imputed pripravené.")


# --- 3. One-Hot Encoding (Kľúčový krok!) ---

# Sloupce, ktoré potrebujeme konvertovať
OHE_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']

# Zabezpečíme, že trénovacie a testovacie dáta budú mať rovnaké stĺpce!
# pd.get_dummies je pre OHE, columns= definuje, ktoré stĺpce konvertujeme
X_train_final = pd.get_dummies(X_train, columns=OHE_cols, drop_first=True, dtype=int)
X_test_final = pd.get_dummies(X_test_imputed, columns=OHE_cols, drop_first=True, dtype=int)

# Dôležitá kontrola pre modely:
print("\nPočet stĺpcov po OHE:")
print(f"X_train_final stĺpcov: {X_train_final.shape[1]}")
print(f"X_test_final stĺpcov: {X_test_final.shape[1]}")

# Kontrola konzistencie stĺpcov
missing_in_test = set(X_train_final.columns) - set(X_test_final.columns)
missing_in_train = set(X_test_final.columns) - set(X_train_final.columns)

if missing_in_test or missing_in_train:
    print("❌ CHYBA: Sady majú rozdielne stĺpce po OHE. To musíme opraviť!")
    # Automatická oprava chýbajúcich stĺpcov:
    for col in missing_in_test:
        X_test_final[col] = 0
    for col in missing_in_train:
        X_train_final[col] = 0
    # Obnovenie poradia stĺpcov
    X_test_final = X_test_final[X_train_final.columns]
    print("✅ Stĺpce boli synchronizované (pridané nuly za chýbajúce kategórie).")

# Finálna kontrola: sloupce musia byť úplne identické
if list(X_train_final.columns) != list(X_test_final.columns):
    raise ValueError("❌ Train a test nemajú rovnaké sloupce ani po synchronizácii!")
else:
    print("✅ Sloupce 100% matchujú – bezpečné pre modelovanie.")

# --- 4. Uloženie finálnych dát ---
X_train_final.to_csv('X_train_final.csv', index=False)
X_test_final.to_csv('X_test_final.csv', index=False)

print("\n✅ Finálne trénovacie a testovacie sady sú pripravené na modelovanie!")