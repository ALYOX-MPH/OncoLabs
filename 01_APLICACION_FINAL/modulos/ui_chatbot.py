import customtkinter as ctk
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import os
import threading
from tkinter import filedialog
from PIL import Image, ImageTk
import pytesseract
import cv2
import re

# --- CONFIGURACIÓN TESSERACT  ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ChatbotWindow(ctk.CTkToplevel):
    def __init__(self, parent, model_dir):
        super().__init__(parent)
        self.title("Módulo F: OncoBot - Asesor Médico Avanzado")
        self.geometry("650x800")
        self.resizable(False, False)
        self.configure(fg_color="#E2E1E1") 
        
        self.model_dir = model_dir
        self.lemmatizer = WordNetLemmatizer()
        
        self.intents = None
        self.words = None
        self.classes = None
        self.model = None
        
        self.create_ui()
        threading.Thread(target=self.load_brain, daemon=True).start()

    def create_ui(self):
        # Area de Chat (Scroll)
        self.chat_frame = ctk.CTkScrollableFrame(self, fg_color="#D1CFCF", label_text_color="#0E0E0E")
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Mensaje bienvenida profesional
        self.add_message("Bot", "¡Hola! Soy OncoBot Pro. Soy tu asistente médico IA. Puedo responder dudas sobre salud o analizar tus hemogramas completos. Sube una foto de tu analítica para un reporte detallado.")

        # Area de entrada
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.pack(fill="x", padx=10, pady=10)

        # Botón Adjuntar Imagen (Clip)
        self.btn_attach = ctk.CTkButton(self.input_frame, text="📷 Analizar Hemograma", width=160, height=40, 
                                        fg_color="#F39C12", font=ctk.CTkFont(size=14, weight="bold"),
                                        text_color="#FFFFFF", hover_color="#D35400",
                                        command=self.upload_image)
        self.btn_attach.pack(side="left", padx=(0, 5))

        # Caja de Texto
        self.entry_msg = ctk.CTkEntry(self.input_frame, placeholder_text="Escribe tu consulta...", height=40,
                                      fg_color="#FFFFFF", text_color="#0E0E0E", placeholder_text_color="#747474")
        self.entry_msg.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.entry_msg.bind("<Return>", lambda event: self.send_message())

        # Botón Enviar
        self.btn_send = ctk.CTkButton(self.input_frame, text="➤", width=50, height=40, 
                                      fg_color="#67C090", font=ctk.CTkFont(size=20, weight="bold"),
                                      text_color="#FFFFFF", hover_color="#4E9F75",
                                      command=self.send_message)
        self.btn_send.pack(side="right")

    def load_brain(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            intents_path = os.path.join(base_dir, "..", "..", "03_LABORATORIO_ENTRENAMIENTO", "E_Modulo_Chatbot", "intents.json")
            
            self.intents = json.loads(open(intents_path, encoding='utf-8').read())
            self.words = pickle.load(open(os.path.join(self.model_dir, 'words.pkl'), 'rb'))
            self.classes = pickle.load(open(os.path.join(self.model_dir, 'classes.pkl'), 'rb'))
            self.model = tf.keras.models.load_model(os.path.join(self.model_dir, 'chatbot_model.h5'))
            print(" OncoBot cargado correctamente.")
        except Exception as e:
            print(f"Error cargando bot: {e}")
            self.after(0, lambda: self.add_message("Sistema", "Error: Cerebro no encontrado. Entrena el bot primero."))

    # --- LÓGICA DE VISIÓN MÉDICA (OCR PRO MEJORADO) ---
    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes Clínicas", "*.jpg;*.png;*.jpeg;*.pdf")])
        if not path: return

        filename = os.path.basename(path)
        self.add_message("Tú", f" [Documento clínico enviado: {filename}]")
        self.add_message("Bot", "Recibido. Iniciando análisis espectral del documento... Un momento, por favor.")

        threading.Thread(target=self.process_ocr, args=(path,), daemon=True).start()

    def process_ocr(self, path):
        try:
            # 1. PREPROCESAMIENTO ROBUSTO
            img = cv2.imread(path)
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Aumentar contraste (CLAHE - Mejor que threshold simple para fotos con sombras)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrasted = clahe.apply(gray)
            
            # Binarización suave
            _, thresh = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR con configuración laxa (PSM 6 asume bloque de texto, PSM 4 columna)
            # Probamos PSM 3 (auto) que suele ser el más general
            custom_config = r'--oem 3 --psm 3' 
            text = pytesseract.image_to_string(thresh, config=custom_config)
            
            # Debug: Ver qué está leyendo en la consola
            print("\n--- TEXTO LEÍDO POR OCR ---")
            print(text)
            print("---------------------------\n")

            # Análisis Médico Profundo
            reporte = self.analyze_medical_text(text)
            
            self.after(0, lambda: self.add_message("Bot", reporte))

        except Exception as e:
            err_msg = f"Error técnico de lectura: {e}. Intenta recortar la imagen solo a la tabla de resultados."
            self.after(0, lambda: self.add_message("Bot", err_msg))

    def analyze_medical_text(self, text):
        # Normalización agresiva
        text = text.lower()
        text = text.replace(',', '.') 
        text = text.replace(':', ' ')
        
        # Diccionario Médico EXPANDIDO (Más sinónimos)
        parametros = {
            'HGB': {'keys': ['hemoglobina', 'hgb', 'hb', 'hemoglobin'], 'min': 12.0, 'max': 16.5, 'unit': 'g/dL', 'val': None},
            'HTO': {'keys': ['hematocrito', 'hto', 'hct', 'hmt'], 'min': 36.0, 'max': 50.0, 'unit': '%', 'val': None},
            'VCM': {'keys': ['vcm', 'mcv', 'volumen corpuscular'], 'min': 80.0, 'max': 100.0, 'unit': 'fL', 'val': None},
            'WBC': {'keys': ['leucocitos', 'wbc', 'globulos blancos', 'leucocito'], 'min': 4.5, 'max': 11.0, 'unit': 'x10³/µL', 'val': None},
            'NEU': {'keys': ['neutrofilos', 'neu', 'segmentados'], 'min': 40.0, 'max': 75.0, 'unit': '%', 'val': None},
            'LIN': {'keys': ['linfocitos', 'lym', 'linfo'], 'min': 20.0, 'max': 45.0, 'unit': '%', 'val': None},
            'PLT': {'keys': ['plaquetas', 'plt', 'trombocitos', 'platelets'], 'min': 150, 'max': 450, 'unit': 'x10³/µL', 'val': None},
            'GLU': {'keys': ['glucosa', 'glucose', 'glicemia', 'glu'], 'min': 70, 'max': 105, 'unit': 'mg/dL', 'val': None}
        }

        found_count = 0
        reporte_valores = "📋 **DETALLE DE VALORES DETECTADOS:**\n"
        
        for p_code, p_data in parametros.items():
            for key in p_data['keys']:
                if key in text:

                    pattern = rf"{key}[^0-9\n]{{0,20}}?(\d+(\.\d+)?)"
                    match = re.search(pattern, text)
                    
                    if match:
                        try:
                            val_str = match.group(1)
                            val = float(val_str)
                            
                            
                            if p_code == "HGB" and val > 25: continue
                            if p_code == "PLT" and val < 10: val = val * 100 
                            if p_code == "WBC" and val > 100: val = val / 1000 

                            p_data['val'] = val
                            
                            estado = "✅ Normal"
                            if val < p_data['min']: estado = "🔻 BAJO"
                            elif val > p_data['max']: estado = "🔺 ALTO"
                            
                            reporte_valores += f"• **{p_code} ({key.title()})**: {val} {p_data['unit']} -> {estado}\n"
                            found_count += 1
                            break # Ya encontramos este, siguiente parámetro
                        except: pass

        if found_count < 2:
            return "⚠️ La imagen es difícil de leer. Detecté texto pero no pude aislar los valores numéricos con seguridad. Intenta tomar la foto más cerca y con buena luz."

        # MOTOR DE DIAGNÓSTICO
        diagnostico = "\n👨‍⚕️ **INTERPRETACIÓN CLÍNICA PRELIMINAR:**\n"
        alertas = []

        hgb = parametros['HGB']['val']
        vcm = parametros['VCM']['val']
        wbc = parametros['WBC']['val']
        plt = parametros['PLT']['val']

        # Anemias
        if hgb and hgb < 12.0:
            if vcm and vcm < 80: alertas.append("🩸 **Posible Anemia Microcítica (Falta de Hierro):** Común en dietas pobres en hierro o pérdidas de sangre.")
            elif vcm and vcm > 100: alertas.append("🩸 **Posible Anemia Macrocítica:** Sugiere déficit de Vitamina B12.")
            else: alertas.append("🩸 **Anemia Normocítica:** Hemoglobina baja. Requiere revisión médica.")

        # Infecciones
        if wbc and wbc > 11.5: alertas.append("🦠 **Leucocitosis:** Defensas altas. Sugiere una infección activa o inflamación.")
        if wbc and wbc < 4.0: alertas.append("🛡️ **Leucopenia:** Defensas bajas. Sistema inmune debilitado.")

        # Coagulación
        if plt and plt < 140: alertas.append("⚠️ **Plaquetas Bajas:** Riesgo de sangrado. Evite golpes fuertes.")

        if not alertas:
            diagnostico += "✅ Todo parece estar en orden. Tus valores principales están dentro del rango saludable."
        else:
            for a in alertas: diagnostico += f"{a}\n\n"
            diagnostico += "⚠️ *Consulta a un médico para confirmar.*"

        return reporte_valores + diagnostico

    # --- LÓGICA DE CHAT (NLP) ---
    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s: bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        p = self.bow(sentence, self.words)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, intents_list, intents_json):
        if not intents_list: return "No entendí eso. Intenta preguntar de otra forma."
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag: return random.choice(i['responses'])
        return "..."

    def send_message(self):
        msg = self.entry_msg.get()
        if not msg: return
        self.entry_msg.delete(0, 'end')
        self.add_message("Tú", msg)

        if self.model:
            ints = self.predict_class(msg)
            res = self.get_response(ints, self.intents)
            self.add_message("Bot", res)

    def add_message(self, sender, text):
        frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        frame.pack(fill="x", pady=5)
        
        # Colores para tema claro
        if sender == "Bot":
            color_label = "#3498DB" # Azul Bot
            align = "w"
            bg_msg = "#FFFFFF" # Burbuja blanca
            text_color = "#0E0E0E" # Texto negro
        else:
            color_label = "#67C090" # Verde Tú
            align = "e"
            bg_msg = "#67C090" # Burbuja verde
            text_color = "#FFFFFF" # Texto blanco para leer sobre verde
        
        ctk.CTkLabel(frame, text=sender, font=ctk.CTkFont(size=10, weight="bold"), text_color=color_label).pack(anchor=align)
        ctk.CTkLabel(frame, text=text, fg_color=bg_msg, text_color=text_color, corner_radius=10, padx=15, pady=10, wraplength=550, justify="left", font=ctk.CTkFont(size=12)).pack(anchor=align)
        
        self.chat_frame._parent_canvas.yview_moveto(1.0)