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
        self.configure(fg_color="#E2E1E1")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

        self.is_running = True
        self.cap = None
        self.fatigue_counter = 0
        self.blink_counter = 0
        self.status_text = "ESCANEO ACTIVO"
        self.status_color = "#67C090"

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.create_video_panel()
        self.create_data_panel()

        self.cap = cv2.VideoCapture(0)
        threading.Thread(target=self.video_loop, daemon=True).start()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_video_panel(self):
        self.video_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.lbl_video = ctk.CTkLabel(self.video_frame, text="Iniciando Sensores Ópticos...", text_color="white")
        self.lbl_video.pack(expand=True, fill="both")

    def create_data_panel(self):
        self.data_panel = ctk.CTkFrame(self, corner_radius=0, fg_color="#D1CFCF")
        self.data_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(self.data_panel, text="BIOSCAN TIME-REAL", font=ctk.CTkFont(size=20, weight="bold"), text_color="#0E0E0E").pack(pady=30)

        self.card_fatiga = self.create_metric_card("ESTADO DE ALERTA", "Analizando...", "#67C090")
        
        self.card_parpadeo = self.create_metric_card("PARPADEOS / MIN", "0", "#F39C12")

        self.create_economic_insight()

        self.btn_stop = ctk.CTkButton(self.data_panel, text="DETENER ESCANEO", fg_color="#e74c3c", command=self.on_close)
        self.btn_stop.pack(side="bottom", pady=30, padx=20, fill="x")

    def create_metric_card(self, title, initial_value, color):
        frame = ctk.CTkFrame(self.data_panel, fg_color="#F0F2F5")
        frame.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="#747474").pack(anchor="w", padx=10, pady=(10,0))
        lbl_val = ctk.CTkLabel(frame, text=initial_value, font=ctk.CTkFont(size=24, weight="bold"), text_color=color)
        lbl_val.pack(anchor="w", padx=10, pady=(0,10))
        return lbl_val

    def create_economic_insight(self):
        frame = ctk.CTkFrame(self.data_panel, fg_color="#F0F2F5", border_color="#3498DB", border_width=1)
        frame.pack(fill="x", padx=15, pady=30)
        ctk.CTkLabel(frame, text="ANÁLISIS DE PRODUCTIVIDAD", font=ctk.CTkFont(size=14, weight="bold"), text_color="#3498DB").pack(pady=10)
        self.lbl_eco = ctk.CTkLabel(frame, text="El sujeto muestra niveles\nóptimos de energía.\nRiesgo laboral nulo.", 
                                    text_color="#0E0E0E", font=ctk.CTkFont(size=12))
        self.lbl_eco.pack(pady=(0, 10))

    def video_loop(self):
        last_blink_time = time.time()
        
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )

                    landmarks = face_landmarks.landmark
                    
                    left_up = landmarks[159]
                    left_down = landmarks[145]
                    
                    eye_open_dist = abs(left_up.y - left_down.y) * 1000
                    
                    if eye_open_dist < 12: 
                        self.fatigue_counter += 1
                    else:
                        self.fatigue_counter = 0

                    if eye_open_dist < 12 and (time.time() - last_blink_time) > 0.3:
                        self.blink_counter += 1
                        last_blink_time = time.time()

                    self.after(0, lambda: self.update_metrics(self.fatigue_counter))

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
            
            if self.is_running:
                self.after(0, lambda: self.lbl_video.configure(image=imgtk, text=""))

    def update_metrics(self, fatiga):
        if fatiga > 30: 
            self.card_fatiga.configure(text=" CRÍTICA", text_color="#e74c3c")
            self.lbl_eco.configure(text="ALERTA ECONÓMICA:\nRiesgo de accidente laboral.\nSe recomienda pausa activa\ninmediata.", text_color="#e74c3c")
        else:
            self.card_fatiga.configure(text="VIGILIA ACTIVA", text_color="#67C090")
            self.lbl_eco.configure(text="PRODUCTIVIDAD:\nEl sujeto está apto para\nlabores de alta precisión.", text_color="#0E0E0E")

        self.card_parpadeo.configure(text=f"{self.blink_counter}")

    def on_close(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()