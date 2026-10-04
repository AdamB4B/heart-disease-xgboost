import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from typing import Sequence
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from typing import List, Tuple, Dict, Optional
from scipy.stats import shapiro, mannwhitneyu, ttest_ind, chi2_contingency


data = pd.read_csv('heart-disease_data.csv')

array_data = data.to_numpy()

print(data.head())


pravidla = {
    'age': 'Věk pacienta – reálné hodnoty od 0 do přibližně 120 let',
    'sex': 'Pohlaví pacienta – 0 = žena, 1 = muž',
    'cp': 'Typ bolesti na hrudi – 0 = typická angina, 1 = atypická angina, 2 = neanginózní bolest, 3 = bez bolesti',
    'trestbps': 'Klidový krevní tlak – typicky 80 až 200 mmHg, reálné hodnoty',
    'chol': 'Sérový cholesterol – obvykle 100 až 600 mg/dl, reálné hodnoty',
    'fbs': 'Hladina cukru nalačno – 0 = ≤120 mg/dl, 1 = >120 mg/dl',
    'restecg': 'Výsledek klidového EKG – 0 = normální, 1 = abnormalita ST-T, 2 = hypertrofie levé komory',
    'thalach': 'Maximální dosažená tepová frekvence – reálné hodnoty přibližně 60 až 220 tepů/min',
    'exang': 'Angina vyvolaná zátěží – 0 = ne, 1 = ano',
    'oldpeak': 'Pokles ST úseku oproti klidu – reálné hodnoty (netuším rozsah)',
    'slope': 'Sklon ST úseku při zátěži – 0, 1, 2',
    'ca': 'Počet hlavních cév obarvených fluoroskopií – hodnoty 0 až 3',
    'thal': 'Typ thalasémie – 0 = žádná, 1 = normální, 2 = fixní defekt, 3 = reverzibilní defekt',
    'target': 'Cílová proměnná (onemocnění srdce) – 0 = bez onemocnění, 1 = s onemocněním'
}
# Vytvoření DataFrame pro přehlednou tabulku
tabulka_pravidel = pd.DataFrame(list(pravidla.items()), columns=['Příznak', 'Možné hodnoty / význam'])

# Zobrazení přehledu
print("\n Přehled logických pravidel pro všechny příznaky:\n")
print(tabulka_pravidel.to_string(index=False))

from sklearn.model_selection import train_test_split


# Rozdělení na příznaky (X) a cílovou proměnnou (y)
X = data.drop(columns=["target"])
y = data["target"]

# Stratifikované rozdělení 80/20 (vhodné kvůli zachování poměru tříd)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Oprava indexování
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)


# kontrola tvarů a poměrů tříd
print("Shapes:")
print("X_train:", X_train.shape, "| y_train:", y_train.shape)
print("X_test: ", X_test.shape,  "| y_test: ", y_test.shape)

print("\nPoměr tříd (celá data):")
print(y.value_counts(normalize=True).round(3))
print("\nPoměr tříd (train):")
print(y_train.value_counts(normalize=True).round(3))
print("\nPoměr tříd (test):")
print(y_test.value_counts(normalize=True).round(3))

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


## SAMOTNÉ ODTRAŇOVÁNÍ
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

X_train_clean, cleaning_report = remove_invalid_values(
    X=X_train,
    continuous_ranges=spojite_rozsahy,
    categorical_values=kategoricke_hodnoty
)

# print(cleaning_report)
#
# ## Oprava Nan pomocí KNN - doplnuju Nan z nejbližších kamošu z jejich prumeru mně chybějící hodnoty
#
# # Normalizace číselných hodnot
# scaler = StandardScaler()
# X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_clean), columns=X_train_clean.columns)
#
# # KNN nastavení a provedení
# imputer = KNNImputer(n_neighbors=5) # určuji z kolika nejbližších kámošů dělám průměr
# X_train_imputed_scaled = pd.DataFrame(imputer.fit_transform(X_train_scaled), columns=X_train_clean.columns)
#
# # 3Vrácení původního měřítka
# X_train_knn = pd.DataFrame(scaler.inverse_transform(X_train_imputed_scaled), columns=X_train_clean.columns)
#
# # Zaokrouhlení kategoriálních proměnných
# categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
# for col in categorical_cols:
#     X_train_knn[col] = X_train_knn[col].round().astype(int)
#
# # Výsledek
# print(X_train_knn.head())
#
# ##Funkce na imputaci pomocí medianu/modu
#
# def impute_with_simple_imputer(
#     data: pd.DataFrame,
#     continuous_columns: Sequence[str],
#     categorical_columns: Sequence[str],
# ) -> pd.DataFrame:
#     """
#     Doplní chybějící hodnoty (NaN) v zadaném DataFrame pomocí SimpleImputeru.
#
#     U spojitých (numerických) sloupců se chybějící hodnoty nahradí mediánem.
#     U kategoriálních sloupců se chybějící hodnoty nahradí nejčastější hodnotou (módem).
#
#     Funkce nevytváří žádné vedlejší efekty – vstupní DataFrame nijak neupravuje
#     a vrací nový DataFrame s doplněnými hodnotami.
#
#     Parametry
#     ---------
#     data : pd.DataFrame
#         Vstupní tabulka dat, která může obsahovat NaN hodnoty.
#     continuous_columns : Sequence[str]
#         Názvy sloupců se spojitými (numerickými) daty, kde se použije medián.
#     categorical_columns : Sequence[str]
#         Názvy kategoriálních sloupců, kde se použije nejčastější hodnota.
#
#     Návratová hodnota
#     -----------------
#     pd.DataFrame
#         Nový DataFrame obsahující všechny vybrané sloupce s doplněnými NaN hodnotami.
#
#     Výjimky
#     -------
#     TypeError
#         Pokud parametr `data` není typu pandas.DataFrame.
#     ValueError
#         Pokud některý z uvedených sloupců v DataFrame neexistuje.
#     """
#     if not isinstance(data, pd.DataFrame):
#         raise TypeError("Parameter 'data' musí být pandas DataFrame.")
#
#     # Ověření, že požadované sloupce opravdu existují
#     missing_cols = [
#         col for col in continuous_columns + categorical_columns
#         if col not in data.columns
#     ]
#     if missing_cols:
#         raise ValueError(f"V DataFrame chybí tyto sloupce: {missing_cols}")
#
#     # Imputery
#     numeric_imputer = SimpleImputer(strategy="median")
#     categorical_imputer = SimpleImputer(strategy="most_frequent")
#
#     # ColumnTransformer přiřadí správný imputer ke správným sloupcům
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numeric_imputer, continuous_columns),
#             ("cat", categorical_imputer, categorical_columns),
#         ],
#         remainder="drop",
#     )
#
#     # Transformace dat (fit + transform)
#     transformed = preprocessor.fit_transform(data)
#
#     # Složení výsledného DataFrame
#     result = pd.DataFrame(
#         transformed,
#         columns=list(continuous_columns) + list(categorical_columns)
#     )
#
#     return result
#
# ## Overení funkce
#
# spojite = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
# kategorie = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
#
# X_train_filled = impute_with_simple_imputer(
#     data=X_train_clean,
#     continuous_columns=spojite,
#     categorical_columns=kategorie,
# )
#
# ## Oprava Nan pomocí mediánu u spojitých a modusu u kategoriálních
#
# # Kopie dat
# X_train_filled = X_train_clean.copy()
#
# # Spojité a kategoriální proměnné
# spojite_sloupce = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
# kategoricke_sloupce = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
#
# # Doplňování spojitých proměnných mediánem
# for col in spojite_sloupce:
#     if col in X_train_filled.columns:
#         median = X_train_filled[col].median()
#         n_changed = 0
#
#         # všechny řádky
#         for i in range(len(X_train_filled)):
#             hodnota = X_train_filled.at[i, col]
#
#             # Pokud je hodnota NaN → nahradím mediánem
#             if pd.isna(hodnota):
#                 X_train_filled.at[i, col] = median
#                 n_changed += 1
#
#         print(f"{col}: doplněno {n_changed} hodnot mediánem = {round(median,2)}")
#
# # Doplňování kategoriálních proměnných modusem
# for col in kategoricke_sloupce:
#     if col in X_train_filled.columns:
#         mode_values = X_train_filled[col].mode()
#
#         # Ověření, jestli existuje aspoň jedna nejčastější hodnota
#         if len(mode_values) > 0:
#             mode_val = mode_values[0]
#         else:
#             mode_val = 0  # záložní hodnota
#
#         n_changed = 0
#
#         for i in range(len(X_train_filled)):
#             hodnota = X_train_filled.at[i, col]
#
#             if pd.isna(hodnota):
#                 X_train_filled.at[i, col] = mode_val
#                 n_changed += 1
#
#         print(f"{col}: doplněno {n_changed} hodnot módou = {mode_val}")
#
# # Kontrola, zda už nemáme žádné NaN
# print("\nPočet zbývajících NaN po doplnění:")
# print(X_train_filled.isna().sum())
#
#
# def evaluate_feature_significance(
#     X: pd.DataFrame,
#     y: pd.Series,
#     continuous_features: List[str],
#     binary_features: List[str],
#     categorical_features: List[str]
# ) -> Tuple[pd.DataFrame, List[str]]:
#     """
#     Provede vhodné statistické testy pro všechny sloupce v X a vrátí jejich p-hodnoty.
#
#     Parametry
#     ---------
#     X : pd.DataFrame
#         Trénovací vstupní data (příznaky).
#     y : pd.Series
#         Target proměnná (0/1).
#     continuous_features : List[str]
#         Názvy spojitých proměnných.
#     binary_features : List[str]
#         Názvy binárních proměnných.
#     categorical_features : List[str]
#         Názvy kategoriálních proměnných.
#
#     Návratová hodnota
#     -----------------
#     Tuple[pd.DataFrame, List[str]]
#         - DataFrame s výsledky (feature, použitý test, p-hodnota)
#         - seznam statisticky významných příznaků (p < 0.05)
#
#     Výjimky
#     --------
#     ValueError
#         Pokud X a y nemají stejný počet řádků nebo obsahují jiná než binární data v targetu.
#     """
#
#     # ---- Validace vstupů ----
#     if not isinstance(X, pd.DataFrame):
#         raise TypeError("X musí být pandas DataFrame.")
#
#     if not isinstance(y, pd.Series):
#         raise TypeError("y musí být pandas Series.")
#
#     if len(X) != len(y):
#         raise ValueError("X a y musí mít stejný počet řádků.")
#
#     if sorted(y.unique()) not in [[0, 1], [0], [1]]:
#         raise ValueError("Target y musí být binární (0/1).")
#
#     results: List[Dict[str, object]] = []
#     significant_features: List[str] = []
#
#     # ---- Pro každý sloupec proveď test ----
#     for col in X.columns:
#         x = X[col]
#         group_0 = x[y == 0]
#         group_1 = x[y == 1]
#
#         # Spojité proměnné
#         if col in continuous_features:
#             _, p0 = shapiro(group_0)
#             _, p1 = shapiro(group_1)
#
#             if p0 < 0.05 or p1 < 0.05:
#                 _, p_val = mannwhitneyu(group_0, group_1)
#                 test_used = "Mann-Whitney U"
#             else:
#                 _, p_val = ttest_ind(group_0, group_1, equal_var=False)
#                 test_used = "t-test"
#
#         # Binární + kategoriální
#         elif col in binary_features + categorical_features:
#             table = pd.crosstab(x, y)
#             _, p_val, _, _ = chi2_contingency(table)
#             test_used = "Chi² test"
#
#         # Nezařazené proměnné
#         else:
#             p_val = np.nan
#             test_used = "Unknown"
#
#         # Uložení výsledku
#         results.append({
#             "feature": col,
#             "test": test_used,
#             "p_value": p_val
#         })
#
#         if isinstance(p_val, float) and p_val < 0.05:
#             significant_features.append(col)
#
#     results_df = pd.DataFrame(results).sort_values("p_value")
#     return results_df, significant_features
#
# # ## Vyhodnocení významnosti pžíznaků pro X_train_knn (y_train je pro všechny train data stejné)
# # datasets = {
# #     "Původní": X_train,
# #     "KNN imputace": X_train_knn,
# #     "Medián imputace": X_train_filled
# # }
#
# results_df, significant_features_train_knn = evaluate_feature_significance(
#     X=X_train_knn,
#     y=y_train,
#     continuous_features=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
#     binary_features=['sex', 'fbs', 'exang'],
#     categorical_features=['cp', 'restecg', 'slope', 'ca', 'thal']
# )
#
# unsignificant_features = set(X_train.columns) - set(significant_features_train_knn)
#
# print("Významné příznaky (p < 0.05):")
# print(significant_features_train_knn)
# print("Vyřazené příznaky:")
# print(unsignificant_features)
# print("Výsledky:")
# print(results_df)
#
#
# # ## Vyhodnocení významnosti pžíznaků pro X_train_filled (y_train je pro všechny train data stejné)
# # datasets = {
# #     "Původní": X_train,
# #     "KNN imputace": X_train_knn,
# #     "Medián imputace": X_train_filled
# # }
#
# results_df, significant_features_train_filled = evaluate_feature_significance(
#     X=X_train_filled,
#     y=y_train,
#     continuous_features=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
#     binary_features=['sex', 'fbs', 'exang'],
#     categorical_features=['cp', 'restecg', 'slope', 'ca', 'thal']
# )
#
# unsignificant_features = set(X_train.columns) - set(significant_features_train_filled)
#
# print("Významné příznaky (p < 0.05):")
# print(significant_features_train_filled)
# print("Vyřazené příznaky:")
# print(unsignificant_features)
# print("Výsledky:")
# print(results_df)
#
# ## FUKCE provádějící standardizaci a PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
#
# def run_pca(X, treshold=0.95):
#     """
#     Provede škálování + PCA na datech.
#
#     Parametry:
#     X : vstupní data (numpy array nebo DataFrame)
#     treshold : float, kolik variance má PCA zachovat (0–1)
#
#     Vrací:
#     X_pca : transformovaná (redukovaná) data
#     pca_model : vytrénovaný PCA objekt
#     scaler : použitý scaler (pro transformaci test data)
#     """
#
#     # 1) Standardizace dat
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # 2) PCA s daným tresholdem
#     pca_model = PCA(n_components=treshold)
#     X_pca = pca_model.fit_transform(X_scaled)
#
#     return X_pca, pca_model, scaler
#
#
# ## PCA pro knn
# X_knn_reduced, pca_knn, scaler_knn = run_pca(X_train_knn, treshold=0.95)
# print(f"Počet komponent KNN: {X_knn_reduced.shape[1]}")
#
#
# ## PCA pro medián
# X_median_reduced, pca_median, scaler_median = run_pca(X_train_filled, treshold=0.95)
# print(f"Počet komponent Medián: {X_median_reduced.shape[1]}")
#
#
# # výběr jen významných featur
# X_knn_sig = X_train_knn[significant_features_train_knn]
# X_median_sig = X_train_filled[significant_features_train_filled]
#
# treshold = 0.95
#
# # PCA pro KNN na významných featurách
# X_knn_reduced_sig, pca_knn_sig, scaler_knn_sig = run_pca(X_knn_sig, treshold=treshold)
# print(f"Počet vstupujících feature {len(significant_features_train_knn)}")
# print(f"Počet komponent KNN po PCA (významné featury): {X_knn_reduced_sig.shape[1]}")
#
# # PCA pro medián na významných featurách
# X_median_reduced_sig, pca_median_sig, scaler_median_sig = run_pca(X_median_sig, treshold=treshold)
# print(f"Počet vstupujících feature {len(significant_features_train_filled)}")
# print(f"Počet komponent Medián po PCA (významné featury): {X_median_reduced_sig.shape[1]}")


