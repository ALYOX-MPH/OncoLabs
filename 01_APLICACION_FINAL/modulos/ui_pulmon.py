import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import threading
import os

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
            except RuntimeError as e:
                print(e)
    return tf

class LungDiagnosticWindow(ctk.CTkToplevel):
    def __init__(self, parent, model_path):
        super().__init__(parent)
        self.title("Módulo A: Análisis Pulmonar")
        self.geometry("900x700")
        self.configure(fg_color="#E2E1E1")
        self.model_path = model_path
        self.model = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.panel_left = ctk.CTkFrame(self, corner_radius=0, fg_color="#D1CFCF")
        self.panel_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(self.panel_left, text="CONTROLES", font=ctk.CTkFont(size=20, weight="bold"), text_color="#0E0E0E").pack(pady=20)

        self.status_frame = ctk.CTkFrame(self.panel_left, fg_color="#E2E1E1")
        self.status_frame.pack(fill="x", padx=10, pady=10)
        self.lbl_status = ctk.CTkLabel(self.status_frame, text="⚡ Cargando IA...", text_color="#747474")
        self.lbl_status.pack(pady=10)

        self.btn_load = ctk.CTkButton(self.panel_left, text="📂 Subir Radiografía", command=self.load_image, state="disabled", height=50, fg_color="#67C090", text_color="#FFFFFF", hover_color="#4E9F75")
        self.btn_load.pack(fill="x", padx=20, pady=20)

        self.progress = ctk.CTkProgressBar(self.panel_left, progress_color="#67C090")
        self.progress.set(0)
        self.progress.pack(fill="x", padx=20, pady=10)

        self.panel_right = ctk.CTkFrame(self, fg_color="transparent")
        self.panel_right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.img_frame = ctk.CTkFrame(self.panel_right, fg_color="#000000")
        self.img_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        self.lbl_img = ctk.CTkLabel(self.img_frame, text="\n\n[ VISTA PREVIA IMAGEN ]\n\n", text_color="gray")
        self.lbl_img.pack(expand=True)

        self.res_frame = ctk.CTkFrame(self.panel_right, height=150, fg_color="#D1CFCF")
        self.res_frame.pack(fill="x")

        self.lbl_result_title = ctk.CTkLabel(self.res_frame, text="DIAGNÓSTICO:", font=ctk.CTkFont(size=14), text_color="#747474")
        self.lbl_result_title.pack(pady=(10,0))
        
        self.lbl_result = ctk.CTkLabel(self.res_frame, text="Esperando datos...", font=ctk.CTkFont(size=24, weight="bold"), text_color="#0E0E0E")
        self.lbl_result.pack(pady=10)

        threading.Thread(target=self.init_model, daemon=True).start()

    def init_model(self):
        try:
            tf_module = cargar_tensorflow_lazy()
            self.model = tf_module.keras.models.load_model(self.model_path)
            
            self.after(0, lambda: self.lbl_status.configure(text="✅ IA LISTA", text_color="#67C090"))
            self.after(0, lambda: self.btn_load.configure(state="normal"))
            self.after(0, lambda: self.progress.set(1))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.configure(text="❌ Error de Carga", text_color="red"))

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg;*.png;*.jpeg")])
        if not file_path: return

        img_show = Image.open(file_path)
        ratio = img_show.size[0] / img_show.size[1]
        new_h = 400
        new_w = int(new_h * ratio)
        
        img_ctk = ctk.CTkImage(light_image=img_show, dark_image=img_show, size=(new_w, new_h))
        self.lbl_img.configure(image=img_ctk, text="")
        self.lbl_img.image = img_ctk

        self.lbl_result.configure(text="Analizando...", text_color="#3498DB")
        self.progress.set(0.5)
        threading.Thread(target=self.predict, args=(file_path,), daemon=True).start()

    def predict(self, file_path):
        img = Image.open(file_path).convert('RGB').resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        # Ajuste para modelo de 3 clases (Normal, Benigno, Maligno)
        pred_class = np.argmax(prediction[0])
        conf = np.max(prediction[0]) * 100

        clases = ["Normal", "Benigno", "Maligno"]
        colores = ["#67C090", "#F39C12", "#e74c3c"] 

        text = f"RESULTADO: {clases[pred_class]}"
        color = colores[pred_class]

        self.after(0, lambda: self.update_result(f"{text}\nConfianza: {conf:.2f}%", color, conf))

    def update_result(self, text, color, conf):
        self.lbl_result.configure(text=text, text_color=color)
        self.progress.set(1)