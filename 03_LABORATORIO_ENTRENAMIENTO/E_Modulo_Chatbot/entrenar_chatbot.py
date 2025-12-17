import random
import json
import pickle
import numpy as np
import os

import nltk
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('wordnet')
nltk.download('omw-1.4')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "02_MODELOS_ENTRENADOS")

lemmatizer = WordNetLemmatizer()

print(" Cargando intenciones...")
# Verificación de archivo intents.json
if not os.path.exists(INTENTS_PATH):
    print(f"Error: No se encuentra intents.json en {INTENTS_PATH}")
    exit()

with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# 1. PROCESAMIENTO DE TEXTO (TOKENIZACIÓN)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Separar en palabras
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Asociar palabras con su etiqueta
        documents.append((word_list, intent['tag']))
        # Agregar etiqueta a la lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematizar (Convertir palabras a su raíz: "jugando" -> "jugar")
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

print(f"Palabras únicas aprendidas: {len(words)}")
print(f" Clases detectadas: {len(classes)}")

# Guardar estructuras de datos (necesarias para que la App entienda al usuario)
os.makedirs(MODEL_DIR, exist_ok=True)
pickle.dump(words, open(os.path.join(MODEL_DIR, 'words.pkl'), 'wb'))
pickle.dump(classes, open(os.path.join(MODEL_DIR, 'classes.pkl'), 'wb'))

# 2. PREPARACIÓN DE DATOS DE ENTRENAMIENTO (BAG OF WORDS)
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Si la palabra existe en el patrón, poner 1, si no 0
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Mezclar datos y convertir a array
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# 3. CREAR RED NEURONAL (CEREBRO)
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) 

# Compilar
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# 4. ENTRENAR
print("Entrenando OncoBot...")
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Guardar
model.save(os.path.join(MODEL_DIR, 'chatbot_model.h5'))
print("OncoBot entrenado y guardado con éxito.")