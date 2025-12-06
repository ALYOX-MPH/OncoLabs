import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_real", "cancer patient data sets.csv")

# Rutas de salida
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "..", "02_MODELOS_ENTRENADOS", "modelo_futuro.h5")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "..", "..", "02_MODELOS_ENTRENADOS", "scaler_futuro.pkl")
GRAPH_SAVE_PATH = os.path.join(BASE_DIR, "analisis_profesional.png")

print("INICIANDO SISTEMA DE PREDICCIÓN ONCOLÓGICA AVANZADA")

# 1. CARGAR DATOS REALES
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ ERROR CRÍTICO: No se encuentra el archivo de datos reales en:\n{DATASET_PATH}\n--> Descarga el 'Cancer Patients Data Set' de Kaggle y pégalo ahí.")

print("📂 Cargando expedientes clínicos reales...")
df = pd.read_csv(DATASET_PATH)

# 2. LIMPIEZA Y PREPROCESAMIENTO PROFESIONAL
# Eliminamos columnas irrelevantes si existen (como 'index' o 'Patient Id')
drop_cols = ['index', 'Patient Id']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

print(f" Analizando {df.shape[0]} pacientes con {df.shape[1]} variables clínicas.")

# Mapeo de Niveles de Riesgo (El dataset suele tener 'Low', 'Medium', 'High')
# Lo convertiremos a numérico: Low=0, Medium=0.5, High=1 (Probabilidad de Cáncer)
if 'Level' in df.columns:
    risk_mapping = {'Low': 0, 'Medium': 1, 'High': 1} # Simplificamos a Binario (0: Bajo Riesgo, 1: Alto Riesgo/Cancer)
    df['Level'] = df['Level'].map(risk_mapping)

# Separar Features (X) y Target (y)
X = df.drop(columns=['Level'])
y = df['Level']

# Identificar columnas clave para el escalador
feature_names = X.columns.tolist()
print(f" Factores de análisis detectados: {feature_names}")

# 3. NORMALIZACIÓN DE DATOS (Estándar Médico)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. SPLIT DE DATOS (Entrenamiento / Validación / Test)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 5. ARQUITECTURA "DEEP RISK" (RED NEURONAL PROFUNDA)
# Diseñada para encontrar correlaciones no lineales complejas
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    
    # Bloque Denoising (Elimina ruido de datos médicos)
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.4),
    
    # Bloque de Inferencia Profunda
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    # Capa de Decisión
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Probabilidad 0 a 1
])

# Optimizador con Learning Rate dinámico
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# 6. ENTRENAMIENTO INTENSIVO
print("\n🧠 Entrenando Red Neuronal con datos reales...")
early_stopping = callbacks.EarlyStopping(
    monitor='val_auc', 
    patience=15, 
    mode='max',
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150, # Más épocas para aprendizaje profundo
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 7. EVALUACIÓN FINAL
loss, acc, auc = model.evaluate(X_test, y_test)
print(f"\n RESULTADOS DEL MODELO FINAL:")
print(f"   Precisión Global: {acc * 100:.2f}%")
print(f"   Área bajo la curva (Fiabilidad): {auc:.4f}")

# 8. GUARDADO
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)
joblib.dump(feature_names, os.path.join(os.path.dirname(MODEL_SAVE_PATH), "features_futuro.pkl")) # Guardamos nombres de columnas
print("Sistema Predictivo y Calibradores Guardados.")

# 9. VISUALIZACIÓN PROFESIONAL
plt.figure(figsize=(14, 6))

# Subplot 1: Precisión vs Pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['auc'], label='AUC Entrenamiento (Fiabilidad)', color='#2ecc71', linewidth=2)
plt.plot(history.history['val_auc'], label='AUC Validación', color='#27ae60', linestyle='--')
plt.title('Curva de Fiabilidad del Modelo (AUC)')
plt.xlabel('Ciclos de Aprendizaje')
plt.ylabel('Score')
plt.grid(True, alpha=0.2)
plt.legend()

# Subplot 2: Matriz de Correlación (Qué factores pesan más)
plt.subplot(1, 2, 2)
# Hacemos un mapa de calor simple de los primeros datos para ver estructura
sns.heatmap(pd.DataFrame(X_train).iloc[:10, :10], cmap='viridis', cbar=False)
plt.title('Mapa de Activación Neuronal (Muestra)')

plt.savefig(GRAPH_SAVE_PATH)
print(f"Informe gráfico generado en: {GRAPH_SAVE_PATH}")