import customtkinter as ctk
from tkinter import messagebox
import numpy as np
import threading
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

tf = None 
def cargar_tensorflow_lazy():
    global tf
    if tf is None:
        import tensorflow as _tf
        tf = _tf
    return tf

class FuturePredictionWindow(ctk.CTkToplevel):
    def __init__(self, parent, model_path, scaler_path):
        super().__init__(parent)
        self.title("Módulo D: Predicción de Riesgo Oncológico (Datos Reales)")
        self.geometry("1200x800")
        self.configure(fg_color="#E2E1E1") 
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        base = os.path.dirname(model_path)
        self.features_path = os.path.join(base, "features_futuro.pkl")
        
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.inputs = {}

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.create_form_panel()
        self.create_result_panel()

        threading.Thread(target=self.init_system, daemon=True).start()

    def create_form_panel(self):
        self.scroll = ctk.CTkScrollableFrame(self, label_text="EVALUACIÓN CLÍNICA COMPLETA", corner_radius=0, fg_color="#D1CFCF", label_text_color="#0E0E0E")
        self.scroll.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.lbl_loading_form = ctk.CTkLabel(self.scroll, text="Cargando parámetros del modelo...", text_color="#F39C12")
        self.lbl_loading_form.pack(pady=20)

        self.btn_calc = ctk.CTkButton(self.scroll, text="EJECUTAR ANÁLISIS PREDICTIVO", height=50, 
                                      fg_color="#67C090", text_color="#FFFFFF", font=ctk.CTkFont(size=15, weight="bold"),
                                      state="disabled", command=self.predict, hover_color="#4E9F75")
        self.btn_calc.pack(pady=30, padx=20, fill="x", side="bottom")

    def create_result_panel(self):
        self.panel_right = ctk.CTkFrame(self, fg_color="#F0F2F5") 
        self.panel_right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.panel_right, text="PROYECCIÓN DE RIESGO", font=ctk.CTkFont(size=20, weight="bold"), text_color="#0E0E0E").pack(pady=30)
        
        self.lbl_risk = ctk.CTkLabel(self.panel_right, text="---", font=ctk.CTkFont(size=60, weight="bold"), text_color="#0E0E0E")
        self.lbl_risk.pack(pady=20)

        self.chart_frame = ctk.CTkFrame(self.panel_right, fg_color="transparent")
        self.chart_frame.pack(fill="both", expand=True, padx=20, pady=20)

    def init_system(self):
        try:
            tf_module = cargar_tensorflow_lazy()
            self.model = tf_module.keras.models.load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            
            if os.path.exists(self.features_path):
                self.feature_names = joblib.load(self.features_path)
            else:
                self.feature_names = ['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'Occupational Hazards', 'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Snoring']

            self.after(0, self.generate_form_fields)

        except Exception as e:
            print(f"Error init: {e}")
            self.after(0, lambda: messagebox.showerror("Error Crítico", f"No se pudo cargar el modelo real.\n{str(e)}"))

    def generate_form_fields(self):
        self.lbl_loading_form.destroy()
        
        traducciones = {
            'Age': 'Edad (Años)',
            'Gender': 'Género',
            'Air Pollution': 'Contaminación del Aire',
            'Alcohol use': 'Consumo de Alcohol',
            'Dust Allergy': 'Alergia al Polvo',
            'OccuPational Hazards': 'Riesgos Laborales',
            'Occupational Hazards': 'Riesgos Laborales', 
            'Genetic Risk': 'Riesgo Genético',
            'chronic Lung Disease': 'Enf. Pulmonar Crónica',
            'Balanced Diet': 'Dieta Balanceada',
            'Obesity': 'Obesidad',
            'Smoking': 'Fumador Activo',
            'Passive Smoker': 'Fumador Pasivo',
            'Chest Pain': 'Dolor de Pecho',
            'Coughing of Blood': 'Tos con Sangre',
            'Fatigue': 'Fatiga / Cansancio',
            'Weight Loss': 'Pérdida de Peso',
            'Shortness of Breath': 'Dificultad para Respirar',
            'Wheezing': 'Sibilancias (Silbidos)',
            'Swallowing Difficulty': 'Dificultad para Tragar',
            'Clubbing of Finger Nails': 'Dedos en Palillo de Tambor',
            'Frequent Cold': 'Resfriados Frecuentes',
            'Dry Cough': 'Tos Seca',
            'Snoring': 'Ronquidos'
        }

        for feature in self.feature_names:
            frame = ctk.CTkFrame(self.scroll, fg_color="transparent")
            frame.pack(fill="x", pady=2, padx=5)
            
            raw_name = feature.strip()
            label_text = traducciones.get(raw_name, raw_name) 
            
            ctk.CTkLabel(frame, text=label_text, anchor="w", font=ctk.CTkFont(weight="bold"), text_color="#0E0E0E").pack(side="left")
            
            if 'age' in feature.lower():
                var = ctk.StringVar(value="30")
                ctk.CTkEntry(frame, textvariable=var, width=60, fg_color="#FFFFFF", text_color="#0E0E0E").pack(side="right")
            elif 'gender' in feature.lower():
                var = ctk.IntVar(value=1)
                ctk.CTkSwitch(frame, text="M / F", variable=var, onvalue=1, offvalue=2, progress_color="#67C090", text_color="#0E0E0E").pack(side="right")
            else:
                var = ctk.IntVar(value=1)
                ctk.CTkSlider(frame, from_=1, to=8, variable=var, width=120, height=15, progress_color="#67C090", button_color="#67C090", button_hover_color="#4E9F75").pack(side="right", padx=5)
                lbl_val = ctk.CTkLabel(frame, textvariable=var, width=20, text_color="#747474")
                lbl_val.pack(side="right")
            
            self.inputs[feature] = var

        self.btn_calc.configure(state="normal")

    def predict(self):
        try:
            input_data = []
            for feature in self.feature_names:
                val = float(self.inputs[feature].get())
                input_data.append(val)
            
            data_array = np.array([input_data])
            data_scaled = self.scaler.transform(data_array)
            
            riesgo = self.model.predict(data_scaled)[0][0] * 100
            
            color = "#67C090"
            status = "BAJO RIESGO"
            if riesgo > 40:
                color = "#F39C12"
                status = "RIESGO MODERADO"
            if riesgo > 75:
                color = "#e74c3c"
                status = "ALTO RIESGO"

            self.lbl_risk.configure(text=f"{riesgo:.1f}%", text_color=color)
            self.draw_gauge(riesgo, color, status)

        except Exception as e:
            messagebox.showerror("Error", f"Error en cálculo: {e}")

    def draw_gauge(self, value, color, msg):
        for w in self.chart_frame.winfo_children(): w.destroy()
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        fig.patch.set_facecolor('#F0F2F5')
        ax.set_facecolor('#F0F2F5')
        ax.axis('equal')
        
        val_plot = max(0, min(100, value))
        
        ax.pie([val_plot, 100-val_plot], startangle=90, colors=[color, '#D1CFCF'], wedgeprops={'width': 0.3}, counterclock=False)
        ax.text(0, 0, f"{msg}\n\nPredicción\nIA", ha='center', va='center', color='#0E0E0E', fontsize=12, fontweight='bold')
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)