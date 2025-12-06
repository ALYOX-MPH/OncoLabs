import customtkinter as ctk
from tkinter import messagebox
import numpy as np
import threading
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Lazy loading TF
tf = None 

def cargar_tensorflow_lazy():
    global tf
    if tf is None:
        import tensorflow as _tf
        tf = _tf
    return tf

class BreastDiagnosticWindow(ctk.CTkToplevel):
    def __init__(self, parent, model_path, scaler_path):
        super().__init__(parent)
        self.title("Módulo C: Análisis Clínico de Mama")
        self.geometry("1100x700")
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None

        # Variables para los Inputs (Sliders)
        self.var_radius = ctk.DoubleVar(value=14.0)
        self.var_texture = ctk.DoubleVar(value=19.0)
        self.var_perimeter = ctk.DoubleVar(value=90.0)
        self.var_area = ctk.DoubleVar(value=650.0)
        self.var_smoothness = ctk.DoubleVar(value=0.09)

        # Layout
        self.grid_columnconfigure(0, weight=1) # Panel Controles
        self.grid_columnconfigure(1, weight=2) # Panel Visualización
        self.grid_rowconfigure(0, weight=1)

        self.create_left_panel()
        self.create_right_panel()

        # Iniciar carga silenciosa
        threading.Thread(target=self.init_system, daemon=True).start()

    def create_left_panel(self):
        panel = ctk.CTkFrame(self, corner_radius=0)
        panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(panel, text="DATOS BIOPSIA", font=ctk.CTkFont(size=20, weight="bold"), text_color="#E96E9C").pack(pady=20)

        # Generar Sliders
        self.create_slider(panel, "Radio Medio (mm)", self.var_radius, 6.0, 30.0)
        self.create_slider(panel, "Textura Media", self.var_texture, 9.0, 40.0)
        self.create_slider(panel, "Perímetro (mm)", self.var_perimeter, 40.0, 190.0)
        self.create_slider(panel, "Área (mm²)", self.var_area, 140.0, 2500.0)
        self.create_slider(panel, "Suavidad (0-0.2)", self.var_smoothness, 0.05, 0.20)

        self.btn_predict = ctk.CTkButton(panel, text="ANALIZAR DATOS", command=self.predict, 
                                         height=50, fg_color="#8e44ad", state="disabled", font=ctk.CTkFont(weight="bold"))
        self.btn_predict.pack(fill="x", padx=20, pady=30)

        self.lbl_status = ctk.CTkLabel(panel, text="Cargando calibradores...", text_color="gray")
        self.lbl_status.pack(side="bottom", pady=10)

    def create_slider(self, parent, title, variable, min_val, max_val):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=10)
        
        # Etiqueta y Valor numérico al lado
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x")
        ctk.CTkLabel(header, text=title, font=ctk.CTkFont(weight="bold")).pack(side="left")
        val_lbl = ctk.CTkLabel(header, textvariable=variable) # Se actualiza solo
        val_lbl.pack(side="right")

        slider = ctk.CTkSlider(frame, from_=min_val, to=max_val, variable=variable, 
                               number_of_steps=100, progress_color="#E96E9C")
        slider.pack(fill="x", pady=(5,0))

    def create_right_panel(self):
        panel = ctk.CTkFrame(self, fg_color="#1a1a1a")
        panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(panel, text="RIESGO CALCULADO", font=ctk.CTkFont(size=16), text_color="gray").pack(pady=(20, 5))
        
        self.lbl_result = ctk.CTkLabel(panel, text="Esperando...", font=ctk.CTkFont(size=36, weight="bold"))
        self.lbl_result.pack(pady=10)

        # Area Gráfica
        self.chart_frame = ctk.CTkFrame(panel, fg_color="transparent")
        self.chart_frame.pack(fill="both", expand=True, padx=20, pady=20)

    def init_system(self):
        try:
            # 1. Cargar TensorFlow
            tf_module = cargar_tensorflow_lazy()
            # 2. Cargar Modelo
            self.model = tf_module.keras.models.load_model(self.model_path)
            # 3. Cargar Escalador (Importante)
            self.scaler = joblib.load(self.scaler_path)

            self.after(0, lambda: self.lbl_status.configure(text="Sistema Calibrado", text_color="#2ecc71"))
            self.after(0, lambda: self.btn_predict.configure(state="normal"))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.configure(text=f" Error: {str(e)}", text_color="red"))

    def predict(self):
        # Obtener valores
        features = np.array([[
            self.var_radius.get(),
            self.var_texture.get(),
            self.var_perimeter.get(),
            self.var_area.get(),
            self.var_smoothness.get()
        ]])

        # Escalar datos (La IA aprendió con datos escalados, debemos hacer lo mismo)
        features_scaled = self.scaler.transform(features)

        # Predecir
        prob = self.model.predict(features_scaled)[0][0]
        
        is_malignant = prob > 0.5
        percentage = prob * 100
        
        text = "ALTO RIESGO (Maligno)" if is_malignant else "BAJO RIESGO (Benigno)"
        color = "#e74c3c" if is_malignant else "#2ecc71"

        self.lbl_result.configure(text=f"{percentage:.1f}%", text_color=color)
        self.draw_chart(percentage)

    def draw_chart(self, risk_percent):
        for w in self.chart_frame.winfo_children(): w.destroy()

        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')

        # Gráfico de indicador tipo "Gauge" simplificado (Barra horizontal)
        categories = ['Benigno', 'Riesgo', 'Maligno']
        
        # Dibujamos una barra de progreso visual
        ax.barh(['Riesgo'], [100], color='#333333', height=0.5) # Fondo
        ax.barh(['Riesgo'], [risk_percent], color='#e74c3c' if risk_percent > 50 else '#2ecc71', height=0.5) # Valor

        ax.set_xlim(0, 100)
        ax.set_title("Probabilidad de Malignidad", color="white")
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)