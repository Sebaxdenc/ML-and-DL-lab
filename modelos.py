import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------
# BLOQUE 0 - Inicio
# ---------------------------------------------------------------------
print("=== INICIO DEL PROCESO ===")
print("Vamos a limpiar, procesar y entrenar modelos sobre el dataset de laptops.\n")

# ---------------------------------------------------------------------
# BLOQUE 1 - Carga de dataset
# ---------------------------------------------------------------------
print("1) Cargando dataset...")
df = pd.read_csv("laptop_price.csv", encoding="latin-1")
print(f"   Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.\n")

# ---------------------------------------------------------------------
# BLOQUE 2 - Preprocesamiento manual (parseo de columnas)
# ---------------------------------------------------------------------
print("2) Preprocesamiento inicial...")

# Limpiamos columnas numéricas
df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype(float)
df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype(float)
df['Price_euros'] = df['Price_euros'].astype(float)

# Parseamos columna de almacenamiento
def parse_storage(mem):
    parts = re.split(r'\+|,', str(mem))
    total = 0.0
    for p in parts:
        m = re.search(r'(\d+(\.\d+)?)\s*(TB|GB)', p, flags=re.I)
        if m:
            num = float(m.group(1))
            if m.group(3).upper() == 'TB':
                num *= 1000
            total += num
    return total if total > 0 else np.nan

df['TotalStorageGB'] = df['Memory'].apply(parse_storage)

# Parseamos resolución
def parse_resolution(sr):
    m = re.search(r'(\d{3,4})\s*[x×]\s*(\d{3,4})', str(sr))
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)

res = df['ScreenResolution'].apply(parse_resolution)
df['res_x'] = res.apply(lambda x: x[0])
df['res_y'] = res.apply(lambda x: x[1])
df['ppi'] = np.sqrt(df['res_x'].fillna(0)**2 + df['res_y'].fillna(0)**2) / df['Inches']

# Extraemos si tiene GPU dedicada
def gpu_brand(g):
    g = str(g).lower()
    if 'nvidia' in g or 'amd' in g:
        return 1
    elif 'intel' in g:
        return 0
    else:
        return np.nan

df['Has_dedicated_gpu'] = df['Gpu'].apply(gpu_brand)
print("   Preprocesamiento inicial completado.\n")

# ---------------------------------------------------------------------
# BLOQUE 3 - Selección de features y pipeline de limpieza
# ---------------------------------------------------------------------
print("3) Selección de features y creación de pipeline...")

features = ['Inches', 'Ram', 'Weight', 'TotalStorageGB', 'ppi', 'Has_dedicated_gpu']
target = 'Price_euros'

X = df[features].copy()
y = df[target].copy()

# Definimos pipeline de preprocesamiento
preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Aplicamos pipeline a los datos (para train/val/test se ajusta solo en train)
X_processed = preprocessor.fit_transform(X)

print(f"   Dataset procesado -> X tiene shape: {X_processed.shape}, y tiene {y.shape[0]} valores.\n")

# ---------------------------------------------------------------------
# BLOQUE 4 - División de datos
# ---------------------------------------------------------------------
print("4) Dividiendo dataset en train, val y test...")
X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
print(f"   Train: {X_train.shape[0]} filas | Val: {X_val.shape[0]} filas | Test: {X_test.shape[0]} filas\n")

# ---------------------------------------------------------------------
# BLOQUE 5 - Modelo 1: kNN
# ---------------------------------------------------------------------
print("5) Entrenando kNN...")
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

knn_results = {}
for split_name, X_split, y_split in [("Train", X_train, y_train),
                                    ("Val", X_val, y_val),
                                    ("Test", X_test, y_test)]:
    y_pred = knn.predict(X_split)
    knn_results[split_name] = {
        "RMSE": np.sqrt(mean_squared_error(y_split, y_pred)),
        "R2": r2_score(y_split, y_pred)
    }

# ---------------------------------------------------------------------
# BLOQUE 6 - Modelo 2: Random Forest
# ---------------------------------------------------------------------
print("6) Entrenando Random Forest...")
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

rf_results = {}
for split_name, X_split, y_split in [("Train", X_train, y_train),
                                    ("Val", X_val, y_val),
                                    ("Test", X_test, y_test)]:
    y_pred = rf.predict(X_split)
    rf_results[split_name] = {
        "RMSE": np.sqrt(mean_squared_error(y_split, y_pred)),
        "R2": r2_score(y_split, y_pred)
    }

# ---------------------------------------------------------------------
# BLOQUE 7 - Modelo 3: DNN
# ---------------------------------------------------------------------
print("7) Entrenando Red Neuronal Densa...")
dnn = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

dnn.compile(optimizer="adam", loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

history = dnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=32,
    verbose=0,
    callbacks=[early_stop]
)

dnn_results = {}
for split_name, X_split, y_split in [("Train", X_train, y_train),
                                    ("Val", X_val, y_val),
                                    ("Test", X_test, y_test)]:
    y_pred = dnn.predict(X_split, verbose=0).flatten()
    dnn_results[split_name] = {
        "RMSE": np.sqrt(mean_squared_error(y_split, y_pred)),
        "R2": r2_score(y_split, y_pred)
    }

# ---------------------------------------------------------------------
# BLOQUE 8 - Comparación de resultados
# ---------------------------------------------------------------------
print("\n=== RESULTADOS COMPARATIVOS ===")
rows = []
for model_name, res_dict in [("kNN", knn_results), ("RandomForest", rf_results), ("DNN", dnn_results)]:
    for split in ["Train", "Val", "Test"]:
        rows.append({
            "Modelo": model_name,
            "Split": split,
            "RMSE": res_dict[split]["RMSE"],
            "R2": res_dict[split]["R2"]
        })

df_results = pd.DataFrame(rows)
pivot = df_results.pivot(index="Modelo", columns="Split", values=["RMSE", "R2"])
print(pivot)

# ---------------------------------------------------------------------
# BLOQUE 9 - Interpretación de resultados
# ---------------------------------------------------------------------
print("\n=== INTERPRETACIÓN ===")
best_model = pivot["R2"]["Val"].idxmax()
print(f"El modelo con mejor desempeño en validación fue: {best_model}")

for model_name, res_dict in [("kNN", knn_results), ("RandomForest", rf_results), ("DNN", dnn_results)]:
    r2_train = res_dict["Train"]["R2"]
    r2_val = res_dict["Val"]["R2"]
    if r2_train - r2_val > 0.15:
        print(f"{model_name} parece tener overfitting (R2 train muy alto comparado con val).")
    elif r2_val < 0.3:
        print(f"{model_name} podría tener underfitting (R2 bajo en validación).")
    else:
        print(f"{model_name} generaliza bien.")

print(f"\nPara producción elegiríamos {best_model} porque logró el mejor equilibrio entre bias y varianza en val/test.\n")
print("=== FIN DEL PROCESO ===")
