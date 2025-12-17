import customtkinter as ctk
from tkinter import messagebox
import numpy as np
import threading
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        self.configure(fg_color="#E2E1E1")
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None

        self.var_radius = ctk.DoubleVar(value=14.0)
        self.var_texture = ctk.DoubleVar(value=19.0)
        self.var_perimeter = ctk.DoubleVar(value=90.0)
        self.var_area = ctk.DoubleVar(value=650.0)
        self.var_smoothness = ctk.DoubleVar(value=0.09)

        self.grid_columnconfigure(0, weight=1) 
        self.grid_columnconfigure(1, weight=2) 
        self.grid_rowconfigure(0, weight=1)

        self.create_left_panel()
        self.create_right_panel()

        threading.Thread(target=self.init_system, daemon=True).start()

    def create_left_panel(self):
        panel = ctk.CTkFrame(self, corner_radius=0, fg_color="#D1CFCF")
        panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(panel, text="DATOS BIOPSIA", font=ctk.CTkFont(size=20, weight="bold"), text_color="#0E0E0E").pack(pady=20)

        self.create_slider(panel, "Radio Medio (mm)", self.var_radius, 6.0, 30.0)
        self.create_slider(panel, "Textura Media", self.var_texture, 9.0, 40.0)
        self.create_slider(panel, "Perímetro (mm)", self.var_perimeter, 40.0, 190.0)
        self.create_slider(panel, "Área (mm²)", self.var_area, 140.0, 2500.0)
        self.create_slider(panel, "Suavidad (0-0.2)", self.var_smoothness, 0.05, 0.20)

        self.btn_predict = ctk.CTkButton(panel, text="ANALIZAR DATOS", command=self.predict, 
                                         height=50, fg_color="#67C090", text_color="#FFFFFF", state="disabled", font=ctk.CTkFont(weight="bold"), hover_color="#4E9F75")
        self.btn_predict.pack(fill="x", padx=20, pady=30)

        self.lbl_status = ctk.CTkLabel(panel, text="Cargando calibradores...", text_color="#747474")
        self.lbl_status.pack(side="bottom", pady=10)

    def create_slider(self, parent, title, variable, min_val, max_val):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=10)
        
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x")
        ctk.CTkLabel(header, text=title, font=ctk.CTkFont(weight="bold"), text_color="#0E0E0E").pack(side="left")
        val_lbl = ctk.CTkLabel(header, textvariable=variable, text_color="#0E0E0E") 
        val_lbl.pack(side="right")

        slider = ctk.CTkSlider(frame, from_=min_val, to=max_val, variable=variable, 
                               number_of_steps=100, progress_color="#67C090", button_color="#67C090", button_hover_color="#4E9F75")
        slider.pack(fill="x", pady=(5,0))

    def create_right_panel(self):
        panel = ctk.CTkFrame(self, fg_color="#F0F2F5")
        panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(panel, text="RIESGO CALCULADO", font=ctk.CTkFont(size=16), text_color="#747474").pack(pady=(20, 5))
        
        self.lbl_result = ctk.CTkLabel(panel, text="Esperando...", font=ctk.CTkFont(size=36, weight="bold"), text_color="#0E0E0E")
        self.lbl_result.pack(pady=10)

        self.chart_frame = ctk.CTkFrame(panel, fg_color="transparent")
        self.chart_frame.pack(fill="both", expand=True, padx=20, pady=20)

    def init_system(self):
        try:
            tf_module = cargar_tensorflow_lazy()
            self.model = tf_module.keras.models.load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)

            self.after(0, lambda: self.lbl_status.configure(text="Sistema Calibrado", text_color="#67C090"))
            self.after(0, lambda: self.btn_predict.configure(state="normal"))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.configure(text=f" Error: {str(e)}", text_color="red"))

    def predict(self):
        features = np.array([[
            self.var_radius.get(),
            self.var_texture.get(),
            self.var_perimeter.get(),
            self.var_area.get(),
            self.var_smoothness.get()
        ]])

        features_scaled = self.scaler.transform(features)

        prob = self.model.predict(features_scaled)[0][0]
        
        is_malignant = prob > 0.5
        percentage = prob * 100
        
        text = "ALTO RIESGO (Maligno)" if is_malignant else "BAJO RIESGO (Benigno)"
        color = "#e74c3c" if is_malignant else "#67C090"

        self.lbl_result.configure(text=f"{percentage:.1f}%", text_color=color)
        self.draw_chart(percentage)

    def draw_chart(self, risk_percent):
        for w in self.chart_frame.winfo_children(): w.destroy()

        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        fig.patch.set_facecolor('#F0F2F5')
        ax.set_facecolor('#F0F2F5')

        categories = ['Benigno', 'Riesgo', 'Maligno']
        
        ax.barh(['Riesgo'], [100], color='#D1CFCF', height=0.5) 
        ax.barh(['Riesgo'], [risk_percent], color='#e74c3c' if risk_percent > 50 else '#67C090', height=0.5) 

        ax.set_xlim(0, 100)
        ax.set_title("Probabilidad de Malignidad", color="#0E0E0E")
        ax.tick_params(colors='#0E0E0E')
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)