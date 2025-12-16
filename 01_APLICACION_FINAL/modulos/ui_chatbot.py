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

class ChatbotWindow(ctk.CTkToplevel):
    def __init__(self, parent, model_dir):
        super().__init__(parent)
        self.title("Módulo F: OncoBot - Asesoría IA")
        self.geometry("500x700")
        self.resizable(False, False)
        
        self.model_dir = model_dir
        self.lemmatizer = WordNetLemmatizer()
        
        # Cargar cerebro (en hilo para no congelar)
        self.intents = None
        self.words = None
        self.classes = None
        self.model = None
        
        # UI
        self.create_ui()
        threading.Thread(target=self.load_brain, daemon=True).start()

    def create_ui(self):
        # Area de Chat (Scroll)
        self.chat_frame = ctk.CTkScrollableFrame(self, fg_color="#1a1a1a")
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Mensaje bienvenida
        self.add_message("Bot", "¡Hola! Soy OncoBot. Pregúntame sobre costos, prevención o cómo usar la app.")

        # Area de entrada
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.pack(fill="x", padx=10, pady=10)

        self.entry_msg = ctk.CTkEntry(self.input_frame, placeholder_text="Escribe tu duda aquí...", height=40)
        self.entry_msg.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.entry_msg.bind("<Return>", lambda event: self.send_message())

        self.btn_send = ctk.CTkButton(self.input_frame, text="ENVIAR", width=100, height=40, 
                                      fg_color="#3498DB", command=self.send_message)
        self.btn_send.pack(side="right")

    def load_brain(self):
        try:
            # Rutas
            base_dir = os.path.dirname(os.path.abspath(__file__))
            intents_path = os.path.join(base_dir, "..", "..", "03_LABORATORIO_ENTRENAMIENTO", "E_Modulo_Chatbot", "intents.json")
            
            # Cargar archivos
            self.intents = json.loads(open(intents_path, encoding='utf-8').read())
            self.words = pickle.load(open(os.path.join(self.model_dir, 'words.pkl'), 'rb'))
            self.classes = pickle.load(open(os.path.join(self.model_dir, 'classes.pkl'), 'rb'))
            self.model = tf.keras.models.load_model(os.path.join(self.model_dir, 'chatbot_model.h5'))
            print("🧠 OncoBot cargado correctamente.")
        except Exception as e:
            print(f"Error cargando bot: {e}")
            self.after(0, lambda: self.add_message("Sistema", "Error: No se encontró el cerebro del bot. Entrénalo primero."))

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
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
        if not intents_list:
            return "Lo siento, no entendí eso. Intenta preguntar sobre costos o salud."
        
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

    def send_message(self):
        msg = self.entry_msg.get()
        if not msg: return
        
        self.entry_msg.delete(0, 'end')
        self.add_message("Tú", msg)

        if self.model:
            # Procesar respuesta
            ints = self.predict_class(msg)
            res = self.get_response(ints, self.intents)
            self.add_message("Bot", res)

    def add_message(self, sender, text):
        frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        frame.pack(fill="x", pady=5)
        
        color = "#3498DB" if sender == "Bot" else "#95a5a6"
        align = "w" if sender == "Bot" else "e"
        bg_msg = "#2B2B2B" if sender == "Bot" else "#2c3e50"
        
        ctk.CTkLabel(frame, text=sender, font=ctk.CTkFont(size=10, weight="bold"), text_color=color).pack(anchor=align)
        
        msg_lbl = ctk.CTkLabel(frame, text=text, fg_color=bg_msg, corner_radius=10, padx=10, pady=5, wraplength=350, justify="left")
        msg_lbl.pack(anchor=align)
        
        # Auto scroll al fondo
        self.chat_frame._parent_canvas.yview_moveto(1.0)