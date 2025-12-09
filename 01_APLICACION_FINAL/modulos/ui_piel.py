import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

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
        self.title("Módulo B: Análisis Dermatoscópico de Alta Precisión")
        self.geometry("1100x750")
        self.model_path = model_path
        self.model = None
        
        # Grid Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # --- PANEL IZQUIERDO (CONTROLES) ---
        self.panel_left = ctk.CTkFrame(self, corner_radius=0)
        self.panel_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(self.panel_left, text="DERMATOSCOPIA AI", font=ctk.CTkFont(size=22, weight="bold"), text_color="#F39C12").pack(pady=20)
        
        self.lbl_instruct = ctk.CTkLabel(self.panel_left, text="Suba una imagen clara del lunar\no lesión cutánea.", text_color="gray")
        self.lbl_instruct.pack(pady=5)

        self.btn_load = ctk.CTkButton(self.panel_left, text="📂 CARGAR IMAGEN", command=self.load_image, state="disabled", height=50, fg_color="#D35400", font=ctk.CTkFont(weight="bold"))
        self.btn_load.pack(fill="x", padx=20, pady=30)

        self.lbl_status = ctk.CTkLabel(self.panel_left, text="Inicializando Red Neuronal...", text_color="orange")
        self.lbl_status.pack(pady=10)

        # Espacio para gráfica de barras
        self.chart_frame = ctk.CTkFrame(self.panel_left, fg_color="transparent")
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- PANEL DERECHO (VISUALIZACIÓN) ---
        self.panel_right = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.panel_right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Marco de imagen
        self.img_frame = ctk.CTkFrame(self.panel_right, fg_color="#000000")
        self.img_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.lbl_img = ctk.CTkLabel(self.img_frame, text="\n[ VISTA PREVIA ]", text_color="gray")
        self.lbl_img.pack(expand=True)

        # Resultado
        self.res_frame = ctk.CTkFrame(self.panel_right, fg_color="#2B2B2B", height=100)
        self.res_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        self.lbl_result = ctk.CTkLabel(self.res_frame, text="Esperando análisis...", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_result.pack(pady=20)

        # Hilo de carga
        threading.Thread(target=self.init_model, daemon=True).start()

    def init_model(self):
        try:
            tf_module = cargar_tensorflow_lazy()
            # AQUÍ ESTÁ EL CAMBIO: Agregamos safe_mode=False
            self.model = tf_module.keras.models.load_model(self.model_path, safe_mode=False)
            
            self.after(0, lambda: self.lbl_status.configure(text="Motor IA Calibrado", text_color="#2ecc71"))
            self.after(0, lambda: self.btn_load.configure(state="normal"))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.configure(text=" Error de Carga", text_color="red"))
            print(f"Error detallado: {e}")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes Dermatoscópicas", "*.jpg;*.png;*.jpeg")])
        if not file_path: return

        # Mostrar imagen
        try:
            img_show = Image.open(file_path)
            # Mantener aspecto
            ratio = img_show.size[0] / img_show.size[1]
            new_h = 450
            new_w = int(new_h * ratio)
            
            img_ctk = ctk.CTkImage(light_image=img_show, dark_image=img_show, size=(new_w, new_h))
            self.lbl_img.configure(image=img_ctk, text="")
            self.lbl_img.image = img_ctk # Keep ref

            # Procesar
            self.lbl_result.configure(text="🔬 Analizando texturas...", text_color="#3498db")
            self.btn_load.configure(state="disabled")
            
            # Lanzar predicción en hilo para no congelar UI
            threading.Thread(target=self.predict, args=(file_path,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")

    def predict(self, file_path):
        try:
            # Preprocesamiento EXACTO al del entrenamiento
            # Nota: Entrenamos con (224, 224)
            img = Image.open(file_path).convert('RGB').resize((224, 224))
            img_array = np.array(img)
            img_array = img_array / 255.0 # Normalizar
            img_array = np.expand_dims(img_array, axis=0) # Batch dim

            # Predicción
            prediction = self.model.predict(img_array)
            prob_maligno = prediction[0][0] # Valor cercano a 1 es Maligno, 0 es Benigno
            
            prob_benigno = 1.0 - prob_maligno
            
            # Umbral de decisión
            is_cancer = prob_maligno > 0.5
            
            # Enviar a UI
            self.after(0, lambda: self.update_ui(is_cancer, prob_benigno, prob_maligno))
            
        except Exception as e:
            print(f"Error predicción: {e}")
            self.after(0, lambda: self.lbl_result.configure(text="Error en análisis"))
        finally:
             self.after(0, lambda: self.btn_load.configure(state="normal"))

    def update_ui(self, is_cancer, prob_benigno, prob_maligno):
        confianza = prob_maligno * 100 if is_cancer else prob_benigno * 100
        
        if is_cancer:
            text = " DETECCIÓN: POSIBLE MELANOMA"
            color = "#e74c3c" # Rojo
        else:
            text = " DIAGNÓSTICO: LESIÓN BENIGNA"
            color = "#2ecc71" # Verde

        self.lbl_result.configure(text=f"{text}\nConfianza: {confianza:.2f}%", text_color=color)
        self.draw_chart(prob_benigno * 100, prob_maligno * 100)

    def draw_chart(self, val_benign, val_malign):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        fig.patch.set_facecolor('#2B2B2B')
        ax.set_facecolor('#2B2B2B')

        labels = ['Benigno', 'Maligno']
        sizes = [val_benign, val_malign]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0, 0.1) if val_malign > val_benign else (0.1, 0) # Destacar el mayor

        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'color':"white", 'weight':'bold'})
        ax.axis('equal') 

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)