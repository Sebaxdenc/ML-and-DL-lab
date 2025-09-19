# =====================================
# ANALISIS EXPLORATORIO DE DATOS (EDA)
# =====================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# ---------------------------
# PASO 1: CARGA Y EXPLORACIÓN
# ---------------------------
print("📂 Cargando el dataset...")
df = pd.read_csv("laptop_price.csv", encoding="latin-1")

print("✅ Dataset cargado correctamente.\n")
print(f"Tenemos {df.shape[0]} filas (laptops) y {df.shape[1]} columnas (características).")
print("\n👀 Veamos las primeras filas para conocer cómo se ve:\n")
print(df.head(), "\n")
print("🔎 Revisando si hay valores nulos:\n")
print(df.isnull().sum(), "\n")
print("💡 Así identificamos si tenemos que hacer limpieza antes de analizar.\n")

# ---------------------------
# PASO 2: LIMPIEZA Y FEATURES
# ---------------------------
print("🔧 Limpiando datos y creando nuevas columnas para que sea más fácil analizarlos...")

# --- RAM ---
print("\n➡️ Convirtiendo la RAM de texto ('8GB') a número (8)...")
df['Ram'] = df['Ram'].str.replace('GB','', regex=False).astype(float)

# --- Peso ---
print("➡️ Convirtiendo el peso de texto ('2.3kg') a número (2.3)...")
df['Weight'] = df['Weight'].str.replace('kg','', regex=False).astype(float)

# --- Precio ---
print("➡️ Asegurando que el precio sea numérico...")
df['Price_euros'] = df['Price_euros'].astype(float)

# --- Almacenamiento total ---
print("➡️ Calculando el almacenamiento total (GB) aunque tenga SSD + HDD juntos...")
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

# --- Resolución y PPI ---
print("➡️ Extrayendo resolución de la pantalla (ej: '1920x1080') y calculando PPI...")
def parse_resolution(sr):
    m = re.search(r'(\d{3,4})\s*[x×]\s*(\d{3,4})', str(sr))
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)

res = df['ScreenResolution'].apply(parse_resolution)
df['res_x'] = res.apply(lambda x: x[0])
df['res_y'] = res.apply(lambda x: x[1])
df['ppi'] = np.sqrt(df['res_x'].fillna(0)**2 + df['res_y'].fillna(0)**2) / df['Inches']

# --- GPU dedicada ---
print("➡️ Clasificando GPUs: 1 = dedicada (Nvidia/AMD), 0 = integrada (Intel)...")
def gpu_brand(g):
    g = str(g).lower()
    if 'nvidia' in g or 'amd' in g:
        return 1
    elif 'intel' in g:
        return 0
    else:
        return np.nan

df['Has_dedicated_gpu'] = df['Gpu'].apply(gpu_brand)

# --- Log del precio ---
print("➡️ Creando columna log(Precio) para que la distribución de precios no sea tan sesgada.")
df['log_price'] = np.log1p(df['Price_euros'])

print("\n✅ Datos listos para analizar. 🎯\n")

# ---------------------------
# PASO 3: ESTADÍSTICAS DESCRIPTIVAS
# ---------------------------
print("📊 Veamos estadísticas básicas (promedio, min, max)...\n")
print(df[['Price_euros','Inches','Ram','Weight','TotalStorageGB','ppi']].describe())
print("\n💡 Aquí ya podemos ver cosas interesantes, como el precio promedio y el rango de valores.")

# ---------------------------
# PASO 4: DISTRIBUCIONES
# ---------------------------
print("\n🎨 Graficando distribuciones para entender cómo se reparten los datos...")

plt.figure(figsize=(8,5))
sns.histplot(df['Price_euros'], kde=True)
plt.title("Distribución de precios")
plt.show()
print("💡 Los precios están muy cargados hacia la izquierda: hay muchas laptops baratas y pocas muy caras.\n")

plt.figure(figsize=(8,5))
sns.boxplot(x=df['Price_euros'])
plt.title("Boxplot de precios")
plt.show()
print("💡 El boxplot confirma que hay outliers (laptops muy caras) que pueden afectar el análisis.\n")

plt.figure(figsize=(6,4))
sns.countplot(x=df['Ram'])
plt.title("Distribución de RAM")
plt.show()
print("💡 La mayoría tiene 4GB, 8GB o 16GB de RAM. Eso nos dice qué es común en el mercado.\n")

# ---------------------------
# PASO 5: CORRELACIONES Y RELACIONES
# ---------------------------
print("🔗 Calculando correlaciones entre variables numéricas...\n")
num_features = ['Price_euros','log_price','Inches','Ram','Weight','TotalStorageGB','ppi','Has_dedicated_gpu']
pearson = df[num_features].corr(method='pearson')

print("Correlación con el precio:\n")
print(pearson['Price_euros'].sort_values(ascending=False))
print("\n💡 Vemos que RAM, PPI y almacenamiento tienen buena correlación positiva con el precio.\n")

plt.figure(figsize=(10,8))
sns.heatmap(pearson, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Matriz de correlación")
plt.show()

# Top features más correlacionadas
top_corr = pearson['Price_euros'].drop('Price_euros').abs().sort_values(ascending=False)
top_features = top_corr.head(4).index.tolist()

for feat in top_features:
    print(f"📈 Graficando {feat} vs Precio...")
    plt.figure(figsize=(7,4))
    sns.regplot(x=feat, y='Price_euros', data=df, scatter_kws={'s':10})
    plt.title(f"{feat} vs Precio")
    plt.show()
    print(f"💡 Cuando {feat} aumenta, el precio generalmente también sube.\n")

# ---------------------------
# PASO 6: VERIFICACIONES
# ---------------------------
print("🔎 Revisemos si los cálculos de PPI y GPUs quedaron bien:")
cols_to_check = ['Company','ScreenResolution','Inches','res_x','res_y','ppi','Gpu','Has_dedicated_gpu','Price_euros']
print(df[cols_to_check].head(10))

print("\n📊 Estadísticas de PPI:")
print(df['ppi'].describe())

print("\n🎮 Conteo de GPUs dedicadas vs integradas:")
print(df['Has_dedicated_gpu'].value_counts())

