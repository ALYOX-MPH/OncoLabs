import os
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_pulmon")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "..", "02_MODELOS_ENTRENADOS", "modelo_pulmon.h5")

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 5 

print(" INICIANDO LABORATORIO DE IA - MÓDULO PULMÓN")
print(f" Directorio de trabajo: {BASE_DIR}")

# --- 1. GENERADOR DE DATOS DE PRUEBA 
def verificar_o_crear_datos():
    clases = ['normal', 'cancer']
    if not os.path.exists(DATASET_DIR):
        print(" No se encontró dataset real. Generando imágenes sintéticas de prueba...")
        for clase in clases:
            path_clase = os.path.join(DATASET_DIR, clase)
            os.makedirs(path_clase, exist_ok=True)
            # Crear 20 imágenes de ruido aleatorio por clase
            for i in range(20):
                img_array = np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(path_clase, f"img_{i}.jpg"))
        print("Datos sintéticos generados exitosamente.")
    else:
        print("Dataset encontrado.")

verificar_o_crear_datos()

# --- 2. PREPARACIÓN DE IMÁGENES (PREPROCESSING) ---

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

print("  Cargando imágenes...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary', 
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# --- 3. ARQUITECTURA DE LA IA (CNN) ---
print(" Construyendo la Red Neuronal Convolucional (CNN)...")
model = models.Sequential([
    # Capa 1: Convolución (Detecta características básicas como bordes)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)), # Reduce el tamaño para procesar menos datos
    
    # Capa 2: Convolución (Detecta formas más complejas)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Capa 3: Convolución (Detecta patrones específicos)
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Aplanar: Convierte la imagen 2D en una lista larga de números
    layers.Flatten(),
    
    
    # Capa Densa: Cerebro final que toma la decisión
    layers.Dense(512, activation='relu'),
    
    # Capa de Salida: 1 sola neurona (0 = Normal, 1 = Cáncer)
    layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- 4. ENTRENAMIENTO ---
print("\n ENTRENANDO EL MODELO (Esto puede tardar unos segundos)...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1
)

# --- 5. GUARDADO ---
print("\n Guardando el modelo entrenado...")
# Asegurarnos de que el directorio de destino exista
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

model.save(MODEL_SAVE_PATH)
print(f" ¡ÉXITO! Modelo guardado en:\n {os.path.abspath(MODEL_SAVE_PATH)}")
print(" Ahora ve a la Aplicación Final y prueba el botón de 'Cáncer de Pulmón'.")