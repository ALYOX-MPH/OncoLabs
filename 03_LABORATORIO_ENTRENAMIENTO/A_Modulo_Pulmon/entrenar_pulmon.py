import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_pulmon")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "..", "02_MODELOS_ENTRENADOS", "modelo_pulmon.h5")
CHECKPOINT_WEIGHTS = os.path.join(BASE_DIR, "checkpoint_pulmon_best.weights.h5")

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 16
EPOCHS_FASE_1 = 15 
EPOCHS_FASE_2 = 50  
LOSS_FN = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

print(" INICIANDO SISTEMA DE ENTRENAMIENTO PARA CÁNCER DE PULMÓN")
print(f" Directorio de trabajo: {BASE_DIR}")

# --- 2. GENERACIÓN DE DATOS  ---
def verificar_estructura_datos():
    if not os.path.exists(DATASET_DIR):
        print(" ERROR CRÍTICO: No se encuentra la carpeta 'dataset_pulmon'.")
        print("   Por favor, coloca tus imágenes reales organizadas en carpetas: 'normal', 'benigno', 'maligno'.")
        return False
    
    # Verificacion si hay imágenes reales 
    total_imgs = sum([len(files) for r, d, files in os.walk(DATASET_DIR)])
    if total_imgs < 10:
        print("ADVERTENCIA: Hay muy pocas imágenes. El modelo no aprenderá correctamente.")
        print("   Asegúrate de borrar las imágenes de 'ruido' generadas anteriormente si existen.")
    return True

if not verificar_estructura_datos():
    raise SystemExit("Deteniendo entrenamiento: faltan datos reales en 'dataset_pulmon'.")

# --- 3. PREPARACIÓN Y AUMENTO DE DATOS (DATA AUGMENTATION) ---
print("📸 Preparando generadores de imágenes...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,      
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.1,
    brightness_range=(0.85, 1.15),
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# --- 4. CÁLCULO DE PESOS PARA BALANCEAR CLASES ---
print("⚖️  Calculando pesos para equilibrar clases...")
counter = Counter(train_generator.classes)
total_samples = len(train_generator.classes)
num_classes = len(counter)
class_weights = {cls: (total_samples / (num_classes * count)) for cls, count in counter.items()}

# Mostrar resumen claro de qué datos se están usando
indices_inv = {v: k for k, v in train_generator.class_indices.items()}
print("Imágenes originales detectadas por clase:")
for i, count in counter.items():
    print(f"   Clase {indices_inv[i]}: {count} archivos")

for i, w in class_weights.items():
    print(f"   Clase {indices_inv[i]}: Peso {w:.2f}")

# --- 5. ARQUITECTURA DEL MODELO (VGG16 + GAP) ---
print("🏗️  Construyendo arquitectura VGG16 modificada...")

# Cargar base VGG16 sin la parte superior (include_top=False)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Congelar TODA la base convolucional inicialmente
base_model.trainable = False

# Crear modelo nuevo
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.6),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dense(num_classes, activation='softmax')
])

# --- 6. FASE 1: ENTRENAMIENTO DE CABECERA ---
print("\nFASE 1: Entrenando solo las nuevas capas densas...")
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss=LOSS_FN,
              metrics=['accuracy'])

history_1 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS_FASE_1,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    class_weight=class_weights,
    verbose=1
)

# --- 7. FASE 2: FINE TUNING  ---
print("\n❄️  FASE 2: Descongelando últimos bloques para ajuste fino...")

# Descongelar la base VGG16
base_model.trainable = True

# Congelar todo EXCEPTO el último bloque (Block 5)
set_trainable = False
for layer in base_model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Re-compilar con Learning Rate MUY BAJO (esencial para no destruir lo aprendido)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), # 0.00001
              loss=LOSS_FN,
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1)
checkpoint_cb = ModelCheckpoint(
    CHECKPOINT_WEIGHTS,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

history_2 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS_FASE_2,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr, checkpoint_cb],
    class_weight=class_weights,
    verbose=1
)

if os.path.exists(CHECKPOINT_WEIGHTS):
    model.load_weights(CHECKPOINT_WEIGHTS)
    print(f" Pesos restaurados desde: {CHECKPOINT_WEIGHTS}")

# --- 8. GUARDADO DEL MODELO ---
print("\n Guardando modelo final...")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f" Modelo guardado exitosamente en:\n   {os.path.abspath(MODEL_SAVE_PATH)}")

# --- 9. GRAFICADO DE RESULTADOS ---
def concat_history(hist1, hist2):
    result = {}
    for k in hist1.history.keys():
        result[k] = hist1.history[k] + hist2.history[k]
    return result

full_history = concat_history(history_1, history_2)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(full_history['accuracy'], label='Precisión Entrenamiento')
plt.plot(full_history['val_accuracy'], label='Precisión Validación')
plt.title('Precisión - Cáncer de Pulmón')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Error (Loss)
plt.subplot(1, 2, 2)
plt.plot(full_history['loss'], label='Error Entrenamiento')
plt.plot(full_history['val_loss'], label='Error Validación')
plt.title('Error - Cáncer de Pulmón')
plt.xlabel('Época')
plt.ylabel('Error')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "grafica_entrenamiento_pulmon.png"))
plt.show()

print("¡Entrenamiento finalizado!")