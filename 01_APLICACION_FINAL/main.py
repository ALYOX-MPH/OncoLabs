import customtkinter as ctk
from tkinter import messagebox
import os
from PIL import Image

# --- IMPORTACIÓN DE MÓDULOS ---
ui_pulmon = None
ui_piel = None
ui_mama = None
ui_futuro = None
ui_bioscan = None
ui_wallet = None
ui_chatbot = None

# Intentamos importar uno por uno para saber cuál falla
try:
    from modulos import ui_pulmon
    from modulos import ui_piel
    from modulos import ui_mama
    from modulos import ui_futuro
    from modulos import ui_chatbot
except ImportError as e:
    print(f"Error cargando módulos médicos: {e}")

try:
    from modulos import ui_bioscan
except ImportError as e:
    print(f" Error cargando BioScan (Falta opencv o mediapipe): {e}")

# --- PALETA DE COLORES ---
COLOR_BG_MAIN = "#E2E1E1"      
COLOR_BG_CARD = "#D1CFCF"     
COLOR_ACCENT_PINK = "#67C090"  
COLOR_ACCENT_ORANGE = "#F39C12"
COLOR_ACCENT_BLUE = "#3498DB"  
COLOR_TEXT_WHITE = "#0E0E0E"
COLOR_TEXT_GRAY = "#747474"

ctk.set_appearance_mode("Dark")

class OncoAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración Ventana
        self.title("OncoLabs AI - Plataforma Integral")
        self.geometry("1280x800")
        self.configure(fg_color=COLOR_BG_MAIN)

        # Directorios
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.join(self.base_dir, "assets")
        self.models_dir = os.path.join(self.base_dir, "..", "02_MODELOS_ENTRENADOS")

        # Layout Principal
        self.grid_columnconfigure(0, weight=0, minsize=250) 
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar
        self.create_sidebar()

        # 2. Contenido Principal
        self.main_content = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=40, pady=40)
        
        self.create_dashboard_content()

    def load_image(self, filename, size):
        path = os.path.join(self.assets_dir, filename)
        if os.path.exists(path):
            return ctk.CTkImage(light_image=Image.open(path), dark_image=Image.open(path), size=size)
        else:
            # Retorna imagen vacía si falla para no romper la app
            return ctk.CTkImage(Image.new("RGBA", size, (0,0,0,0)), size=size)

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, fg_color=COLOR_BG_CARD, corner_radius=0, width=250)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1) # Espaciador al final

        # Logo
        logo_img = self.load_image("logo_oncolabs.png", (220, 80)) 
        lbl_logo = ctk.CTkLabel(self.sidebar, text="", image=logo_img)
        lbl_logo.grid(row=0, column=0, padx=20, pady=(40, 40))

        # Menú Lateral
        self.create_menu_btn("Inicio", 1, True, None)
        
        # --- AQUÍ ESTÁ EL BOTÓN DE BIOSCAN EN EL SIDEBAR ---
        self.create_menu_btn("BioScan Facial", 3, False, self.abrir_bioscan)
        
        self.create_menu_btn("OncoBot", 4, False, self.abrir_chatbot)
        
        self.create_menu_btn("Sobre Nosotros", 5, False, self.show_about)


    def create_menu_btn(self, text, row, is_active, command_func):
        fg_color = COLOR_ACCENT_PINK if is_active else "transparent"
        text_color = COLOR_TEXT_WHITE if is_active else COLOR_TEXT_GRAY
        
        btn = ctk.CTkButton(self.sidebar, text=text, anchor="w",
                            fg_color=fg_color, text_color=text_color,
                            hover_color=COLOR_BG_MAIN,
                            font=ctk.CTkFont(size=16, weight="bold" if is_active else "normal"),
                            height=50, corner_radius=8,
                            command=command_func) # Ahora aceptamos comandos
        btn.grid(row=row, column=0, padx=20, pady=10, sticky="ew")

    def create_dashboard_content(self):
        # Títulos
        title_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 30))
        
        ctk.CTkLabel(title_frame, text="Bienvenido a ", font=ctk.CTkFont(size=32, weight="bold"), text_color=COLOR_TEXT_WHITE).pack(side="left")
        ctk.CTkLabel(title_frame, text="OncoLabs", font=ctk.CTkFont(size=32, weight="bold"), text_color=COLOR_ACCENT_PINK).pack(side="left")
        
        # Grid de Tarjetas
        cards_grid = ctk.CTkFrame(self.main_content, fg_color="transparent")
        cards_grid.pack(fill="both", expand=True)
        cards_grid.grid_columnconfigure(0, weight=1)
        cards_grid.grid_columnconfigure(1, weight=1)

        # Tarjetas Fila 1
        self.create_module_card(cards_grid, 0, 0, "Cáncer de Pulmón", "Análisis de Rayos X y TC.", "img_pulmon.jpg", COLOR_ACCENT_PINK, "pulmon", self.abrir_pulmon)
        self.create_module_card(cards_grid, 0, 1, "Cáncer de Piel", "Dermatoscopia avanzada.", "img_piel.jpg", COLOR_ACCENT_PINK, "piel", self.abrir_piel)

        # Tarjetas Fila 2
        self.create_module_card(cards_grid, 1, 0, "Cáncer de Mama", "Análisis de mamografías.", "img_mama.jpg", COLOR_ACCENT_PINK, "mama", self.abrir_mama)
        self.create_module_card(cards_grid, 1, 1, "Predicción 5 Años", "Algoritmos predictivos.", "img_futuro.jpg", COLOR_ACCENT_PINK, "futuro", self.abrir_futuro)

 

    def create_module_card(self, parent, row, col, title, desc, img_name, btn_color, btn_text, command):
        card = ctk.CTkFrame(parent, fg_color=COLOR_BG_CARD, corner_radius=20)
        card.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=2)

        img_cover = self.load_image(img_name, (180, 180))
        lbl_img = ctk.CTkLabel(card, text="", image=img_cover)
        lbl_img.grid(row=0, column=0, rowspan=3, padx=20, pady=20)

        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=22, weight="bold"), text_color=COLOR_TEXT_WHITE).grid(row=0, column=1, padx=(0,20), pady=(25,5), sticky="w")
        ctk.CTkLabel(card, text=desc, font=ctk.CTkFont(size=14), text_color=COLOR_TEXT_GRAY, wraplength=300, justify="left").grid(row=1, column=1, padx=(0,20), pady=(0,20), sticky="nw")

        btn = ctk.CTkButton(card, text=btn_text.upper(), fg_color=btn_color, hover_color=COLOR_BG_MAIN, font=ctk.CTkFont(weight="bold"), height=40,
                            command=command if command else lambda: messagebox.showinfo("Info", "En desarrollo"))
        btn.grid(row=2, column=1, padx=(0,20), pady=(0,25), sticky="ew")

    # --- LÓGICA DE APERTURA DE MÓDULOS ---
    def abrir_pulmon(self):
        model_path = os.path.join(self.models_dir, "modelo_pulmon.h5")
        if not os.path.exists(model_path):
            messagebox.showwarning("Alerta", "Modelo Pulmón no encontrado.")
            return
        if ui_pulmon: ui_pulmon.LungDiagnosticWindow(self, model_path)
        else: messagebox.showerror("Error", "Módulo Pulmón no cargado.")

    def abrir_piel(self):
        model_path = os.path.join(self.models_dir, "modelo_piel.h5")
        if not os.path.exists(model_path):
            messagebox.showwarning("Alerta", "Modelo Piel no encontrado.")
            return
        if ui_piel: ui_piel.SkinDiagnosticWindow(self, model_path)
        else: messagebox.showerror("Error", "Módulo Piel no cargado.")

    def abrir_mama(self):
        model_path = os.path.join(self.models_dir, "modelo_mama.h5")
        scaler_path = os.path.join(self.models_dir, "scaler_mama.pkl")
        if not os.path.exists(model_path):
            messagebox.showwarning("Alerta", "Modelo Mama no encontrado.")
            return
        if ui_mama: ui_mama.BreastDiagnosticWindow(self, model_path, scaler_path)
        else: messagebox.showerror("Error", "Módulo Mama no cargado.")

    def abrir_futuro(self):
         model_path = os.path.join(self.models_dir, "modelo_futuro.h5")
         scaler_path = os.path.join(self.models_dir, "scaler_futuro.pkl")
         if not os.path.exists(model_path):
             messagebox.showwarning("Alerta", "Modelo Futuro no encontrado.")
             return
         if ui_futuro: ui_futuro.FuturePredictionWindow(self, model_path, scaler_path)
         else: messagebox.showerror("Error", "Módulo Predicción no cargado.")

    def abrir_bioscan(self):
         if ui_bioscan:
            ui_bioscan.BioScanWindow(self)
         else:
             # Si entras aquí es porque falló el import de cv2 o mediapipe
             messagebox.showerror("Error Crítico", "No se pudo cargar BioScan.\n\nAsegúrate de instalar:\npip install opencv-python mediapipe")

    def show_about(self):
        messagebox.showinfo("Sobre OncoLabs AI",
                            "OncoLabs AI - Plataforma Integral de Diagnóstico Asistido por IA\n\n"
                            "Desarrollado por el equipo de CincoBits.\n"
                            "Integrantes:\n\n"
                            "Alvaro Miguel\n"
                            "Johaly Concepcion\n"
                            "Jeremy Hernandez\n"
                            "Manuel Rosario\n"
                            "Edgar Rosario\n"
                            "\n\n"
                            "© 2024 OncoLabs. Todos los derechos reservados.")
        
    def abrir_chatbot(self):
        if ui_chatbot:
            ui_chatbot.ChatbotWindow(self, self.models_dir)
        else:
            messagebox.showerror("Error", "Módulo Chatbot no cargado (Instala nltk).")  

if __name__ == "__main__":
    app = OncoAIApp()
    app.mainloop()