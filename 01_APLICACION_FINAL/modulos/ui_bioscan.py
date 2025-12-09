import customtkinter as ctk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from math import hypot

class BioScanWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Módulo E: BioScan - Detector de Fatiga y Salud")
        self.geometry("1200x800")
        
        # --- CONFIGURACIÓN DE IA (MediaPipe) ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

        # Variables de estado
        self.is_running = True
        self.cap = None
        self.fatigue_counter = 0
        self.blink_counter = 0
        self.status_text = "ESCANEO ACTIVO"
        self.status_color = "#2ecc71"

        # --- DISEÑO (Grid Layout) ---
        self.grid_columnconfigure(0, weight=3) # Video Grande
        self.grid_columnconfigure(1, weight=1) # Panel Datos
        self.grid_rowconfigure(0, weight=1)

        self.create_video_panel()
        self.create_data_panel()

        # Iniciar cámara en hilo separado (Para no congelar la App)
        self.cap = cv2.VideoCapture(0) # 0 es la webcam por defecto
        threading.Thread(target=self.video_loop, daemon=True).start()

        # Al cerrar la ventana, apagar cámara
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_video_panel(self):
        self.video_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.lbl_video = ctk.CTkLabel(self.video_frame, text="Iniciando Sensores Ópticos...", text_color="white")
        self.lbl_video.pack(expand=True, fill="both")

    def create_data_panel(self):
        self.data_panel = ctk.CTkFrame(self, corner_radius=0, fg_color="#1a1a1a")
        self.data_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(self.data_panel, text="BIOSCAN TIME-REAL", font=ctk.CTkFont(size=20, weight="bold"), text_color="#3498DB").pack(pady=30)

        # Indicador de Fatiga
        self.card_fatiga = self.create_metric_card("ESTADO DE ALERTA", "Analizando...", "#2ecc71")
        
        # Indicador de Parpadeos
        self.card_parpadeo = self.create_metric_card("PARPADEOS / MIN", "0", "#F39C12")

        # Mensaje Económico (Para la Hackathon)
        self.create_economic_insight()

        self.btn_stop = ctk.CTkButton(self.data_panel, text="DETENER ESCANEO", fg_color="#e74c3c", command=self.on_close)
        self.btn_stop.pack(side="bottom", pady=30, padx=20, fill="x")

    def create_metric_card(self, title, initial_value, color):
        frame = ctk.CTkFrame(self.data_panel, fg_color="#2B2B2B")
        frame.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="gray").pack(anchor="w", padx=10, pady=(10,0))
        lbl_val = ctk.CTkLabel(frame, text=initial_value, font=ctk.CTkFont(size=24, weight="bold"), text_color=color)
        lbl_val.pack(anchor="w", padx=10, pady=(0,10))
        return lbl_val

    def create_economic_insight(self):
        frame = ctk.CTkFrame(self.data_panel, fg_color="#2B2B2B", border_color="#3498DB", border_width=1)
        frame.pack(fill="x", padx=15, pady=30)
        ctk.CTkLabel(frame, text="ANÁLISIS DE PRODUCTIVIDAD", font=ctk.CTkFont(size=14, weight="bold"), text_color="#3498DB").pack(pady=10)
        self.lbl_eco = ctk.CTkLabel(frame, text="El sujeto muestra niveles\nóptimos de energía.\nRiesgo laboral nulo.", 
                                    text_color="white", font=ctk.CTkFont(size=12))
        self.lbl_eco.pack(pady=(0, 10))

    def video_loop(self):
        last_blink_time = time.time()
        
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            # Espejo y Color (OpenCV usa BGR, Tkinter usa RGB)
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar IA
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 1. DIBUJAR LA MALLA FACIAL (Efecto WOW)
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )

                    # 2. CÁLCULO MATEMÁTICO DE OJOS
                    landmarks = face_landmarks.landmark
                    
                    # Puntos clave del ojo izquierdo (Arriba: 159, Abajo: 145)
                    left_up = landmarks[159]
                    left_down = landmarks[145]
                    
                    # Calcular distancia (apertura del ojo)
                    # Multiplicamos por 1000 para tener números manejables
                    eye_open_dist = abs(left_up.y - left_down.y) * 1000
                    
                    # Umbral: Si la distancia es menor a 12, el ojo está cerrado
                    if eye_open_dist < 12: 
                        self.fatigue_counter += 1
                    else:
                        self.fatigue_counter = 0

                    # Detectar parpadeo (cierre rápido)
                    if eye_open_dist < 12 and (time.time() - last_blink_time) > 0.3:
                        self.blink_counter += 1
                        last_blink_time = time.time()

                    # Enviar datos a la UI (usamos self.after para no chocar hilos)
                    self.after(0, lambda: self.update_metrics(self.fatigue_counter))

            # Convertir imagen para mostrar en Tkinter
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
            
            if self.is_running:
                self.after(0, lambda: self.lbl_video.configure(image=imgtk, text=""))

    def update_metrics(self, fatiga):
        # Si lleva más de 30 frames (aprox 1.5 seg) con ojos cerrados -> ALERTA
        if fatiga > 30: 
            self.card_fatiga.configure(text="⚠️ FATIGA CRÍTICA", text_color="#e74c3c") # Rojo
            self.lbl_eco.configure(text="ALERTA ECONÓMICA:\nRiesgo de accidente laboral.\nSe recomienda pausa activa\ninmediata.", text_color="#e74c3c")
        else:
            self.card_fatiga.configure(text="✅ VIGILIA ACTIVA", text_color="#2ecc71") # Verde
            self.lbl_eco.configure(text="PRODUCTIVIDAD:\nEl sujeto está apto para\nlabores de alta precisión.", text_color="white")

        self.card_parpadeo.configure(text=f"{self.blink_counter}")

    def on_close(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()
