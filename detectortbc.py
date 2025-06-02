import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import os
import time  # Para medir el tiempo

MODELO1_PATH = "modelo1_radiografias_vs_otros.h5"
MODELO2_PATH = "modelo2_clasificador_tb.h5"
TAMANO_IMAGEN = (224, 224)

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
        rango_dinamico = (p5, p95)

        if p50 < PENETRACION_OPTIMA_MIN:
            clasificacion = "INSUFICIENTE"
            problema = "IMAGEN DEMASIADO CLARA"
            recomendacion = "Repetir estudio con técnica adecuada: Reducir exposición"
        elif p50 > PENETRACION_OPTIMA_MAX:
            clasificacion = "EXCESIVA"
            problema = "IMAGEN DEMASIADO OSCURA"
            recomendacion = "Repetir estudio con técnica adecuada: Reducir exposición"
        else:
            clasificacion = "ÓPTIMA"
            problema = "Rango óptimo de penetración"
            recomendacion = "No requiere repetición"

        if contraste < CONTRASTE_MINIMO and clasificacion == "ÓPTIMA":
            clasificacion = "CONTRASTE BAJO"
            problema += " - Contraste insuficiente"
            recomendacion = "Repetir estudio con técnica adecuada: Mejorar contraste"

        return {
            "clasificacion": clasificacion,
            "problema": problema,
            "penetracion": float(p50),
            "contraste": float(contraste),
            "intensidad_media": float(np.mean(img_gris)),
            "rango_dinamico": {"oscuros": float(p5), "claros": float(p95)},
            "recomendacion": recomendacion
        }

    def predecir(self, image_data):
        try:
            # Inicia la medición de tiempo
            tiempo_inicio = time.time()

            # Preparar imagen RGB
            img = Image.open(image_data).convert('RGB').resize(TAMANO_IMAGEN)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Modelo 1: Verificar si es radiografía
            es_radiografia = self.modelo1.predict(img_array, verbose=0)[0]
            if np.argmax(es_radiografia) == 0:
                tiempo_fin = time.time()
                tiempo_total = tiempo_fin - tiempo_inicio
                return {
                    "diagnostico": "NoRadiografia",
                    "confianza": float(np.max(es_radiografia)),
                    "detalle": "No es una radiografía",
                    "tiempo_de_espera": round(tiempo_total, 1)  # en segundos
                }

            # Modelo 2: Clasificación médica
            pred = self.modelo2.predict(img_array, verbose=0)[0]
            prob_tb, prob_normal, prob_baja = pred

            # Imagen en escala de grises
            img = Image.open(image_data).convert('L')
            img_gris_np = np.array(img)

            # Análisis técnico de calidad
            calidad = self._analizar_penetracion(img_gris_np)

            # Diagnóstico técnico
            diagnostico_tecnico = "IMAGEN DEMASIADO CLARA" if calidad["clasificacion"] == "INSUFICIENTE" else \
                                   "IMAGEN DEMASIADO OSCURA" if calidad["clasificacion"] == "EXCESIVA" else \
                                   "IMAGEN CON CONTRASTE BAJO" if calidad["clasificacion"] == "CONTRASTE BAJO" else \
                                   "IMAGEN ÓPTIMA"

            # Diagnóstico final y recomendación
            if prob_baja > 0.5 or calidad["clasificacion"] != "ÓPTIMA":
                diagnostico_final = "BajaCalidad"
                probabilidad = prob_tb
                recomendacion_clinica = "PRECAUCIÓN: No descartar TB solo por calidad de imagen"
            elif prob_tb > 0.5:
                diagnostico_final = "Tuberculosis"
                probabilidad = prob_tb
                recomendacion_clinica = "Continuar con protocolo diagnóstico para TB"
            else:
                diagnostico_final = "Normal"
                probabilidad = prob_normal
                recomendacion_clinica = "No hay hallazgos patológicos relevantes"

            # Finaliza la medición de tiempo
            tiempo_fin = time.time()
            tiempo_total = tiempo_fin - tiempo_inicio

            # Formato final tipo reporte
            reporte = {
                "diagnostico": f"{diagnostico_final}",
                "probabilidad_tb": round(float(prob_tb) * 100, 2),
                "confianza": round(float(np.max(pred)) * 100, 2),
                "hallazgos_tecnicos": {
                    "intensidad_media": calidad["intensidad_media"],
                    "contraste": calidad["contraste"],
                    "rango_dinamico": calidad["rango_dinamico"],
                    "diagnostico_tecnico": diagnostico_tecnico,
                    "posible_causa": calidad["problema"],
                    "consecuencia": "Pérdida de detalles diagnósticos" if calidad["clasificacion"] != "ÓPTIMA" else "Óptima calidad diagnóstica"
                },
                "recomendacion_clinica": recomendacion_clinica,
                "es_radiografia": True,
                "tiempo_de_espera": round(tiempo_total, 1)  # en segundos
            }

            return reporte

        except Exception as e:
            return {"error": str(e)}    