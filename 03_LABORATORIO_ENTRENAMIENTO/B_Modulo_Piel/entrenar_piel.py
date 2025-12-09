import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_piel")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "..", "02_MODELOS_ENTRENADOS", "modelo_piel.h5")
GRAPH_SAVE_PATH = os.path.join(BASE_DIR, "reporte_dermatoscopico.png")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20 

print(f"INICIANDO TRANSFER LEARNING CON MOBILENET V2")
print(f" Dataset: {DATASET_DIR}")

# --- 1. PREPARACIÓN DE DATOS ---

train_datagen = ImageDataGenerator(
    rescale=1./255,   
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# --- 2. ARQUITECTURA: TRANSFER LEARNING ---
# Descargamos el cerebro pre-entrenado de Google (MobileNetV2)

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# CONGELAMOS el cerebro base para no dañar lo que ya sabe
base_model.trainable = False 

# Creamos nuestro modelo montado encima
inputs = tf.keras.Input(shape=(224, 224, 3))

# Capa de adaptación interna: MobileNet espera valores entre -1 y 1
x = layers.Lambda(lambda x: (x * 2.0) - 1.0)(inputs) 

x = base_model(x, training=False) 
x = layers.GlobalAveragePooling2D()(x) 
x = layers.Dropout(0.2)(x) 
outputs = layers.Dense(1, activation='sigmoid')(x) 

model = models.Model(inputs, outputs)

# Compilamos
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. ENTRENAMIENTO ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)
]

print("\n ENTRENANDO CAPA SUPERIOR...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

# --- 4. REPORTE ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Precisión Entrenamiento')
plt.plot(val_acc, label='Precisión Validación')
plt.title('Precisión (Transfer Learning)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Error Entrenamiento')
plt.plot(val_loss, label='Error Validación')
plt.title('Error (Debe bajar suavemente)')
plt.legend()

plt.savefig(GRAPH_SAVE_PATH)
print(f" Gráficas generadas: {GRAPH_SAVE_PATH}")
print(f"Modelo guardado: {MODEL_SAVE_PATH}")