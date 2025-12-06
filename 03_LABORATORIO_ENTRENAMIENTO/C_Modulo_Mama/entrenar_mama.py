import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # Para guardar el calibrador de datos

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "..", "02_MODELOS_ENTRENADOS", "modelo_mama.h5")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "..", "..", "02_MODELOS_ENTRENADOS", "scaler_mama.pkl")

print(" Cargando Dataset de Biopsias de Wisconsin...")
# Usamos solo 5 características clave para que la interfaz sea fácil de usar
# (Radio, Textura, Perímetro, Área, Suavidad)
data = load_breast_cancer()
# Índices de las 5 columnas principales (mean radius, mean texture, mean perimeter, mean area, mean smoothness)
selected_features = [0, 1, 2, 3, 4] 
X = data.data[:, selected_features] 
y = data.target # 0 = Maligno, 1 = Benigno (OJO: en este dataset 0 es maligno)

# Invertir etiquetas para que 1 sea Cancer (Maligno) y 0 sea Sano (Benigno)
# Esto es para mantener coherencia con nuestros otros módulos
y = np.where(y == 0, 1, 0) 

print(f"Datos cargados: {X.shape[0]} pacientes.")

# 1. Dividir entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Normalizar datos (CRÍTICO para redes neuronales numéricas)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Crear la Red Neuronal (ANN)
model = models.Sequential([
    layers.Input(shape=(5,)), # Esperamos 5 datos de entrada
    layers.Dense(16, activation='relu'), # Capa oculta 1
    layers.Dense(32, activation='relu'), # Capa oculta 2
    layers.Dense(16, activation='relu'), # Capa oculta 3
    layers.Dense(1, activation='sigmoid') # Salida (Probabilidad 0-1)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Entrenar
print("Entrenando red neuronal con datos clínicos...")
model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=1)

# 5. Evaluar
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nPrecisión del modelo: {accuracy * 100:.2f}%")

# 6. Guardar
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH) # Guardamos el escalador

print(f" Modelo guardado: {MODEL_SAVE_PATH}")
print(f" Escalador guardado: {SCALER_SAVE_PATH}")