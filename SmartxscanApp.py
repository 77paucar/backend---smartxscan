import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# =============================================
# CONFIGURACIÓN PRINCIPAL (AJUSTA ESTOS VALORES)
# =============================================
MODELO1 = "modelo1_radiografias_vs_otros.h5"
MODELO2 = "modelo2_clasificador_tb.h5"
TAMANO_IMAGEN = (224, 224)

# Umbrales mejorados para calidad de imagen
PENETRACION_OPTIMA_MIN = 80    # Valores menores = demasiado claros (antes 40)
PENETRACION_OPTIMA_MAX = 160   # Valores mayores = demasiado oscuros (antes 185)
CONTRASTE_MINIMO = 40          # Diferencia mínima entre zonas claras/oscuras

# =============================================
# CLASE PRINCIPAL DEL DETECTOR (MODIFICADA)
# =============================================
class DetectorTBC:
    def __init__(self):
        try:
            self.modelo1 = load_model(MODELO1)
            self.modelo2 = load_model(MODELO2)
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar modelos:\n{str(e)}")
            raise

    def _analizar_penetracion(self, img_gris):
        """Analiza la calidad de la imagen con criterios mejorados"""
        # Calculamos percentiles clave
        p5 = np.percentile(img_gris, 5)    # Áreas más oscuras
        p50 = np.percentile(img_gris, 50)  # Mediana (intensidad típica)
        p95 = np.percentile(img_gris, 95)   # Áreas más claras
        contraste = p95 - p5                # Diferencia entre claros/oscuros

        # Diagnóstico de penetración
        if p50 < PENETRACION_OPTIMA_MIN:
            clasificacion = "INSUFICIENTE"
            problema = "IMAGEN DEMASIADO CLARA\n• Posible causa: Baja exposición/kVp"
            color = "#FF6D00"  # Naranja
        elif p50 > PENETRACION_OPTIMA_MAX:
            clasificacion = "EXCESIVA"
            problema = "IMAGEN DEMASIADO OSCURA\n• Posible causa: Alta exposición/kVp"
            color = "#D50000"  # Rojo
        else:
            clasificacion = "ÓPTIMA"
            problema = "Rango óptimo de penetración"
            color = "#00C853"  # Verde

        # Verificación adicional de contraste
        if contraste < CONTRASTE_MINIMO and clasificacion == "ÓPTIMA":
            clasificacion = "CONTRASTE BAJO"
            problema += "\n• CONTRASTE INSUFICIENTE\n• Posible causa: kVp inadecuado o dispersión"
            color = "#FFAB00"  # Amarillo/naranja

        return {
            "clasificacion": clasificacion,
            "hallazgos": (
                f"• Intensidad media: {p50:.1f} (óptimo: {PENETRACION_OPTIMA_MIN}-{PENETRACION_OPTIMA_MAX})\n"
                f"• Contraste: {contraste:.1f} (mínimo recomendado: {CONTRASTE_MINIMO})\n"
                f"• Rango dinámico: {p5:.1f} (oscuros) a {p95:.1f} (claros)\n"
                f"• DIAGNÓSTICO TÉCNICO: {problema}\n"
                f"• CONSECUENCIA: {'Pérdida de detalles diagnósticos' if clasificacion != 'ÓPTIMA' else 'Visualización adecuada'}"
            ),
            "color": color
        }

    def predecir(self, ruta_imagen):
        try:
            # Paso 1: Verificar si es radiografía
            img = Image.open(ruta_imagen).convert('RGB').resize(TAMANO_IMAGEN)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            es_radiografia = self.modelo1.predict(img_array, verbose=0)[0]
            if np.argmax(es_radiografia) == 0:
                return {
                    "diagnostico": "NoRadiografia",
                    "confianza": float(np.max(es_radiografia)),
                    "detalle": "No es una radiografía"
                }
            
            # Paso 2: Clasificación médica
            pred = self.modelo2.predict(img_array, verbose=0)[0]
            prob_tb, prob_normal, prob_baja = pred
            
            # Paso 3: Análisis técnico (modificado)
            img_gris = cv2.imread(ruta_imagen, 0)
            if img_gris is None:
                raise ValueError("No se pudo leer la imagen en escala de grises")
                
            calidad = self._analizar_penetracion(img_gris)
            
            # Diagnóstico final (ahora más estricto con calidad)
            if (prob_baja > 0.5 or calidad["clasificacion"] != "ÓPTIMA"):
                diagnostico = "BajaCalidad"
                mostrar_prob = {"Tuberculosis": float(prob_tb)}
            elif prob_tb > 0.5:
                diagnostico = "Tuberculosis"
                mostrar_prob = {"Tuberculosis": float(prob_tb)}
            else:
                diagnostico = "Normal"
                mostrar_prob = {"Normal": float(prob_normal)}
            
            return {
                "diagnostico": diagnostico,
                "confianza": float(np.max(pred)),
                "probabilidades": mostrar_prob,
                "prob_tuberculosis": float(prob_tb),
                "calidad": calidad,
                "es_radiografia": True
            }
            
        except Exception as e:
            return {"error": str(e)}

# =============================================
# INTERFAZ GRÁFICA (NO SE MODIFICÓ)
# =============================================
class Aplicacion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SmartXScan - Detección de Tuberculosis Pulmonar")
        self.geometry("1000x850")
        self.configure(bg="#f5f5f5")
        self.detector = DetectorTBC()
        self._configurar_interfaz()
    
    def _configurar_interfaz(self):
        # Panel superior
        frame_superior = tk.Frame(self, bg="#ffffff", padx=15, pady=15)
        frame_superior.pack(fill=tk.X)
        
        btn_cargar = tk.Button(
            frame_superior,
            text=" CARGAR RADIOGRAFÍA TORÁCICA",
            command=self._cargar_imagen,
            font=("Arial", 12, "bold"),
            bg="#1976D2",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.GROOVE
        )
        btn_cargar.pack(side=tk.LEFT)
        
        self.panel_imagen = tk.Label(frame_superior, bg="#e0e0e0", bd=2, relief=tk.SUNKEN)
        self.panel_imagen.pack(side=tk.RIGHT, padx=10)
        
        # Panel de resultados
        frame_resultados = tk.Frame(self, bg="#ffffff", padx=20, pady=20)
        frame_resultados.pack(fill=tk.BOTH, expand=True)
        
        # Diagnóstico principal
        self.lbl_diagnostico = tk.Label(
            frame_resultados,
            text="DIAGNÓSTICO PRELIMINAR: --",
            font=("Arial", 14, "bold"),
            bg="#ffffff",
            fg="#0D47A1"
        )
        self.lbl_diagnostico.pack(anchor="w", pady=(0, 5))
        
        self.lbl_confianza = tk.Label(
            frame_resultados,
            text="Nivel de confianza: --",
            font=("Arial", 12),
            bg="#ffffff",
            fg="#424242"
        )
        self.lbl_confianza.pack(anchor="w")
        
        # Contenedor para probabilidades
        self.frame_probs = tk.LabelFrame(
            frame_resultados,
            text="INFORMACIÓN ADICIONAL",
            font=("Arial", 11, "bold"),
            bg="#ffffff",
            fg="#616161",
            padx=10,
            pady=10
        )
        self.frame_probs.pack(fill=tk.X, pady=10)
        
        # Informe de calidad (mejorado)
        frame_calidad = tk.LabelFrame(
            frame_resultados,
            text="INFORME TÉCNICO - ESTÁNDARES ICRP",
            font=("Arial", 11, "bold"),
            bg="#ffffff",
            fg="#616161",
            padx=10,
            pady=10
        )
        frame_calidad.pack(fill=tk.X)
        
        self.lbl_clasif_pen = tk.Label(
            frame_calidad,
            text="PENETRACION EN LA IMAGEN: --",
            font=("Arial", 11, "bold"),
            bg="#ffffff"
        )
        self.lbl_clasif_pen.pack(anchor="w", padx=10, pady=(0, 5))
        
        self.lbl_hallazgos = tk.Label(
            frame_calidad,
            text="HALLAZGOS TÉCNICOS:\n--",
            font=("Courier New", 10),
            bg="#ffffff",
            justify=tk.LEFT,
            wraplength=700
        )
        self.lbl_hallazgos.pack(anchor="w", padx=10, pady=(0, 10))
        
        # Recomendación clínica
        self.lbl_recomendacion = tk.Label(
            frame_calidad,
            text="RECOMENDACIÓN CLÍNICA:\n--",
            font=("Arial", 11, "bold"),
            bg="#FFF8E1",
            fg="#D32F2F",
            justify=tk.LEFT,
            wraplength=700,
            bd=2,
            relief=tk.GROOVE,
            padx=10,
            pady=10
        )
        self.lbl_recomendacion.pack(fill=tk.X, padx=5, pady=5)
    
    def _cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar radiografía torácica",
            filetypes=[("Imágenes médicas", "*.png *.jpg *.jpeg *.dcm")]
        )
        
        if not ruta:
            return
            
        try:
            # Mostrar imagen
            img = Image.open(ruta)
            img.thumbnail((450, 450))
            img_tk = ImageTk.PhotoImage(img)
            self.panel_imagen.config(image=img_tk)
            self.panel_imagen.image = img_tk
            
            # Generar informe
            resultado = self.detector.predecir(ruta)
            if "error" in resultado:
                raise Exception(resultado["error"])
            
            self._actualizar_interfaz(resultado)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en análisis:\n{str(e)}")
    
    def _actualizar_interfaz(self, resultado):
        # Limpiar frame de probabilidades
        for widget in self.frame_probs.winfo_children():
            widget.destroy()
        
        # Configurar diagnóstico principal
        if resultado["diagnostico"] == "Tuberculosis":
            color_dx = "#D32F2F"  # Rojo
            texto_confianza = f"Confianza TB: {resultado['prob_tuberculosis']:.2%}"
        elif resultado["diagnostico"] == "BajaCalidad":
            color_dx = "#FFA000"  # Ámbar
            texto_confianza = f"Prob. TB: {resultado['prob_tuberculosis']:.2%}"
        elif resultado["diagnostico"] == "Normal":
            color_dx = "#388E3C"  # Verde
            texto_confianza = "Imagen normal"
        else:
            color_dx = "#0D47A1"  # Azul
            texto_confianza = "No aplica"
            
        self.lbl_diagnostico.config(
            text=f"DIAGNÓSTICO: {resultado['diagnostico']}",
            fg=color_dx
        )
        self.lbl_confianza.config(text=texto_confianza)
        
        # Mostrar información técnica mejorada
        if resultado.get("es_radiografia", False):
            calidad = resultado["calidad"]
            self.lbl_clasif_pen.config(
                text=f"PENETRACION EN LA IMAGEN: {calidad['clasificacion']}",
                fg=calidad["color"]
            )
            self.lbl_hallazgos.config(text=f"HALLAZGOS TÉCNICOS:\n{calidad['hallazgos']}")
            
            # Recomendaciones específicas
            if resultado["diagnostico"] == "Tuberculosis":
                recomendacion = (
                    "RECOMENDACIÓN CLÍNICA:\n\n"
                    "HALLazGO SOSPECHOSO DE TUBERCULOSIS\n\n"
                    "1. Confirmar con cultivo de esputo y prueba molecular (Xpert MTB/RIF)\n"
                    "2. Iniciar terapia antituberculosa inmediata (4 fármacos)\n"
                    "3. Aislamiento respiratorio y estudio de contactos\n"
                    "URGENCIA: Derivación inmediata a NEUMOLOGÍA"
                )
            elif resultado["diagnostico"] == "BajaCalidad":
                recomendacion = (
                    "RECOMENDACIÓN CLÍNICA:\n\n"
                    "PENETRACION EN LA IMAGEN INSUFICIENTE\n\n"
                    "1. Repetir estudio con técnica adecuada:\n"
                    f"   • {'Aumentar exposición (mAs)' if calidad['clasificacion']=='INSUFICIENTE' else 'Reducir exposición' if calidad['clasificacion']=='EXCESIVA' else 'Ajustar kVp'}\n"
                    "2. Considerar tomografía computarizada si persiste duda diagnóstica\n"
                    "PRECAUCIÓN: No descartar TB solo por calidad de imagen"
                )
            else:
                recomendacion = (
                    "RECOMENDACIÓN CLÍNICA:\n\n"
                    "✓ IMAGEN NORMAL SIN HALLazGOS SOSPECHOSOS\n\n"
                    "1. Continuar seguimiento clínico habitual\n"
                    "2. Control anual en pacientes de riesgo\n"
                    "3. Reevaluar si aparecen síntomas respiratorios\n\n"
                )
            
            self.lbl_recomendacion.config(text=recomendacion)
        else:
            self.lbl_clasif_pen.config(text="CALIDAD: No aplica")
            self.lbl_hallazgos.config(text="HALLAZGOS:\nNo es una radiografía torácica válida")
            self.lbl_recomendacion.config(
                text="RECOMENDACIÓN:\n\n"
                     "1. Verificar que el archivo sea una radiografía PA de tórax\n"
                     "2. Usar formato DICOM preferentemente\n"
                     "3. Evitar imágenes comprimidas o con artefactos"
            )

# =============================================
# INICIO DE LA APLICACIÓN
# =============================================
if __name__ == "__main__":
    if not os.path.exists(MODELO1) or not os.path.exists(MODELO2):
        messagebox.showerror("Error", "No se encontraron los archivos de modelo necesarios")
    else:
        app = Aplicacion()
        app.mainloop()