import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import os

MODELO1_PATH = "modelo1_radiografias_vs_otros.h5"
MODELO2_PATH = "modelo2_clasificador_tb.h5"
TAMANO_IMAGEN = (224, 224)

# Umbrales mejorados para calidad de imagen
PENETRACION_OPTIMA_MIN = 80
PENETRACION_OPTIMA_MAX = 160
CONTRASTE_MINIMO = 40


class DetectorTBC:
    def __init__(self):
        try:
            if not os.path.exists(MODELO1_PATH) or not os.path.exists(MODELO2_PATH):
                raise FileNotFoundError("Uno o ambos modelos no fueron encontrados.")
            self.modelo1 = load_model(MODELO1_PATH)
            self.modelo2 = load_model(MODELO2_PATH)
        except Exception as e:
            print(f"[ERROR] No se pudo cargar los modelos: {str(e)}")
            raise

    def _analizar_penetracion(self, img_gris):
        p5 = np.percentile(img_gris, 5)
        p50 = np.percentile(img_gris, 50)
        p95 = np.percentile(img_gris, 95)
        contraste = p95 - p5

        if p50 < PENETRACION_OPTIMA_MIN:
            clasificacion = "INSUFICIENTE"
            problema = "IMAGEN DEMASIADO CLARA"
        elif p50 > PENETRACION_OPTIMA_MAX:
            clasificacion = "EXCESIVA"
            problema = "IMAGEN DEMASIADO OSCURA"
        else:
            clasificacion = "ÓPTIMA"
            problema = "Rango óptimo de penetración"

        if contraste < CONTRASTE_MINIMO and clasificacion == "ÓPTIMA":
            clasificacion = "CONTRASTE BAJO"
            problema += " - Contraste insuficiente"

        return {
            "clasificacion": clasificacion,
            "problema": problema,
            "penetracion": float(p50),
            "contraste": float(contraste),
        }

    def predecir(self, image_data):
        try:
            # Abrir desde bytes
            img = Image.open(image_data).convert('RGB').resize(TAMANO_IMAGEN)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Modelo 1: ¿Es radiografía?
            es_radiografia = self.modelo1.predict(img_array, verbose=0)[0]
            if np.argmax(es_radiografia) == 0:
                return {
                    "diagnostico": "NoRadiografia",
                    "confianza": float(np.max(es_radiografia)),
                    "detalle": "No es una radiografía"
                }

            # Modelo 2: Clasificación médica
            pred = self.modelo2.predict(img_array, verbose=0)[0]
            prob_tb, prob_normal, prob_baja = pred

            # Análisis técnico
            img = Image.open(image_data).convert('L')  # Escala de grises
            img_gris_np = np.array(img)
            calidad = self._analizar_penetracion(img_gris_np)

            # Diagnóstico final
            if prob_baja > 0.5 or calidad["clasificacion"] != "ÓPTIMA":
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
                "calidad": calidad,
                "es_radiografia": True
            }

        except Exception as e:
            return {"error": str(e)}