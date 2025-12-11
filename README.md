🧬 OncoLabs AI

Plataforma de Inteligencia Artificial para la Recuperación Económica y Salud Preventiva

Desarrollado por: Team 5 Bits

Contexto: Hackathon "Recuperación Económica Post-Pandemia"

💡 Propuesta de Valor: Economía + Salud

"Sin salud, no hay economía."

OncoLabs no es solo una aplicación médica; es una herramienta de infraestructura económica. Las enfermedades detectadas tardíamente (especialmente el cáncer) cuestan a la economía global más de $200 mil millones anuales en pérdida de productividad y gastos paliativos.

Nuestra solución democratiza el triaje preventivo, llevando un laboratorio de IA al bolsillo de cada ciudadano. Esto permite:

Reducir la carga hospitalaria del Estado.

Evitar la quiebra financiera de familias por diagnósticos tardíos.

Mantener la fuerza laboral activa y productiva (BioScan).

🚀 Módulos de Inteligencia Artificial

El sistema consta de 5 módulos integrados, impulsados por Deep Learning y Visión Artificial:

🫁 A. Módulo Pulmonar (CNN)

Tecnología: Red Neuronal Convolucional (Custom CNN).

Dataset: IQ-OTH/NCCD Lung Cancer Dataset.

Función: Análisis de Tomografías Computarizadas (TC) y Rayos X para la detección temprana de nódulos y carcinomas.

🔬 B. Módulo Dermatológico (Transfer Learning)

Tecnología: MobileNetV2 (Transfer Learning) optimizada para dispositivos móviles.

Dataset: HAM10000 (Human Against Machine).

Precisión: >85% en validación.

Función: Clasificación de lesiones cutáneas en Benignas vs. Malignas (Melanoma) analizando asimetría, bordes y texturas.

🎗️ C. Módulo Clínico de Mama (ANN)

Tecnología: Red Neuronal Artificial (Dense Neural Network).

Dataset: Wisconsin Diagnostic Breast Cancer (WDBC).

Función: Análisis de datos de biopsias (radio, textura, perímetro) para determinar malignidad con alta precisión estadística.

🔮 D. Motor Predictivo a 5 Años (Deep Learning)

Tecnología: Deep Neural Network (DNN) con regularización Dropout.

Dataset: Cancer Patients Data Set (Kaggle).

Función: Cruza datos de estilo de vida (fumador, dieta), biometría y genética para calcular la probabilidad porcentual de desarrollar patologías a futuro.

👁️ E. BioScan Facial (Computer Vision)

Tecnología: MediaPipe Face Mesh + OpenCV.

Función: Escáner en tiempo real de 468 puntos faciales.

Impacto Económico: Detecta fatiga laboral y niveles de alerta en empleados, previniendo accidentes y midiendo la productividad en tiempo real.

✅ Certificaciones y Seguridad

En 5 Bits, la ética es innegociable.

Validación Técnica: El sistema ha sido revisado y avalado técnicamente por la Dra. Dilcia Peña Peña, especialista en el área, quien certifica que la herramienta:

Es no invasiva.

No infiere daños ni altera parámetros de salud del usuario.

Funciona correctamente como herramienta de soporte y segunda opinión.

Estado Regulatorio: Actualmente en proceso de presentación de protocolo de investigación ante instituciones de salud pública de la República Dominicana para validación clínica retrospectiva.

💻 Instalación y Uso

Sigue estos pasos para ejecutar el entorno de desarrollo en tu máquina local.

Prerrequisitos

Python 3.9 - 3.11 (Recomendado 3.10)

Windows / Mac / Linux

1. Clonar el Repositorio

git clone [https://github.com/ALYOX-MPH/OncoLabs.git](https://github.com/ALYOX-MPH/OncoLabs.git)
cd OncoLabs


2. Crear Entorno Virtual

python -m venv .venv
# Activar en Windows:
.\.venv\Scripts\activate
# Activar en Mac/Linux:
source .venv/bin/activate


3. Instalar Dependencias (CRÍTICO)

Para evitar conflictos entre TensorFlow y MediaPipe, usa este comando exacto:

pip install -r requirements.txt


O manualmente:

pip install tensorflow==2.15.0 mediapipe==0.10.9 opencv-python protobuf==3.20.3 ml-dtypes==0.2.0 customtkinter pillow pandas scikit-learn seaborn


4. Ejecutar la Aplicación

python 01_APLICACION_FINAL/main.py


📂 Estructura del Proyecto

PROYECTO_ONCO_AI/
├── 01_APLICACION_FINAL/       # Código Fuente de la Interfaz (Frontend)
│   ├── assets/                # Imágenes y recursos gráficos
│   ├── modulos/               # Lógica de cada ventana (BioScan, Piel, etc.)
│   └── main.py                # Punto de entrada (Dashboard)
│
├── 02_MODELOS_ENTRENADOS/     # Cerebros de la IA (.h5 y .pkl)
│
├── 03_LABORATORIO/            # Scripts de Entrenamiento (Backend IA)
│   ├── A_Modulo_Pulmon/
│   ├── B_Modulo_Piel/
│   ├── C_Modulo_Mama/
│   └── D_Modulo_Prediccion/
│
└── requirements.txt           # Lista de librerías


👥 Equipo "5 Bits"

Innovando para un futuro saludable y económicamente sostenible.

Disclaimer: OncoLabs es una herramienta de apoyo al diagnóstico y triaje. No sustituye el criterio de un médico profesional. En caso de alerta, acuda siempre a un especialista.

Verciones y requerimientos:
customtkinter==5.2.2
tensorflow==2.15.0
mediapipe==0.10.9
opencv-python
protobuf==3.20.3
ml-dtypes==0.2.0
pandas
numpy
scikit-learn
matplotlib
seaborn
pillow
joblib
