from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from app.detectortbc import DetectorTBC

app = FastAPI()
detector = DetectorTBC()

@app.post("/upload-image/")
async def analizar_imagen(file: UploadFile = File(...)):
    allowed_types = {"image/png", "image/jpeg", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Formato de archivo no permitido.")

    try:
        # Leer contenido del archivo en memoria
        contents = await file.read()
        image_data = BytesIO(contents)

        # Analizar imagen
        resultado = detector.predecir(image_data)
        return JSONResponse(content=resultado)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

@app.get("/")
def read_root():
    return {"Hello": "World"}