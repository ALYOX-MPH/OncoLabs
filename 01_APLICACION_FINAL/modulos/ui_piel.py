import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Lazy loading TF
tf = None 

def cargar_tensorflow_lazy():
    global tf
    if tf is None:
        import tensorflow as _tf
        tf = _tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError: pass
    return tf

class SkinDiagnosticWindow(ctk.CTkToplevel):
    def __init__(self, parent, model_path):
        super().__init__(parent)
        self.title("Módulo B: Análisis Dermatoscópico")
        self.geometry("1000x700")
        self.model_path = model_path
        self.model = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # --- PANEL IZQUIERDO ---
        self.panel_left = ctk.CTkFrame(self, corner_radius=0)
        self.panel_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(self.panel_left, text="DERMATOSCOPIA", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)
        
        self.btn_load = ctk.CTkButton(self.panel_left, text="📂 Cargar Imagen Piel", command=self.load_image, state="disabled", height=40, fg_color="#F39C12", hover_color="#D35400")
        self.btn_load.pack(fill="x", padx=20, pady=20)

        self.lbl_status = ctk.CTkLabel(self.panel_left, text="Iniciando motor...", text_color="gray")
        self.lbl_status.pack(pady=10)

        # Espacio para gráfica
        self.chart_frame = ctk.CTkFrame(self.panel_left, fg_color="transparent")
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- PANEL DERECHO ---
        self.panel_right = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.panel_right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.lbl_img = ctk.CTkLabel(self.panel_right, text="[VISTA PREVIA]", text_color="gray")
        self.lbl_img.pack(expand=True)

        self.lbl_result = ctk.CTkLabel(self.panel_right, text="", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_result.pack(pady=20)

        threading.Thread(target=self.init_model, daemon=True).start()

    def init_model(self):
        try:
            tf_module = cargar_tensorflow_lazy()
            self.model = tf_module.keras.models.load_model(self.model_path)
            self.after(0, lambda: self.lbl_status.configure(text=" Sistema Listo", text_color="#2ecc71"))
            self.after(0, lambda: self.btn_load.configure(state="normal"))
        except Exception:
            self.after(0, lambda: self.lbl_status.configure(text=" Error Modelo", text_color="red"))

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg;*.png;*.jpeg")])
        if not file_path: return

        # Visualizar
        img_show = Image.open(file_path)
        # Resize manteniendo aspecto
        ratio = img_show.size[0] / img_show.size[1]
        new_h = 400
        new_w = int(new_h * ratio)
        
        img_ctk = ctk.CTkImage(light_image=img_show, dark_image=img_show, size=(new_w, new_h))
        self.lbl_img.configure(image=img_ctk, text="")
        self.lbl_img.image = img_ctk

        # Predecir
        self.lbl_result.configure(text="Analizando...", text_color="#3498db")
        threading.Thread(target=self.predict, args=(file_path,), daemon=True).start()

    def predict(self, file_path):
        img = Image.open(file_path).convert('RGB').resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = self.model.predict(img_array)[0][0]
        
        # Interpretación (Asumiendo 0: Maligno, 1: Benigno o viceversa según dataset)
        # Ajustaremos: < 0.5 Maligno, > 0.5 Benigno
        malignancy = (1 - pred) * 100
        benignity = pred * 100
        
        is_malignant = pred < 0.5
        text = " ALTA PROBABILIDAD MELANOMA" if is_malignant else "LESIÓN BENIGNA"
        color = "#e74c3c" if is_malignant else "#2ecc71"

        self.after(0, lambda: self.update_ui(text, color, benignity, malignancy))

    def update_ui(self, text, color, benignity, malignancy):
        self.lbl_result.configure(text=text, text_color=color)
        self.draw_chart(benignity, malignancy)

    def draw_chart(self, val_benign, val_malign):
        # Limpiar gráfico anterior
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Crear figura Matplotlib
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
        fig.patch.set_facecolor('#2B2B2B') # Fondo oscuro match UI
        ax.set_facecolor('#2B2B2B')

        categories = ['Benigno', 'Maligno']
        values = [val_benign, val_malign]
        colors = ['#2ecc71', '#e74c3c']

        bars = ax.bar(categories, values, color=colors)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Probabilidad (%)', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        # Añadir etiquetas encima de barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', color='white')

        # Insertar en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)