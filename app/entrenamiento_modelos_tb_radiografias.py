import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Ruta base
RUTA_BASE = os.environ.get("DATOS_PATH", "./datos/TB_Chest_Radiography_Database")

if not os.path.isdir(RUTA_BASE):
    raise FileNotFoundError(f"No se encontró la carpeta de datos en: {RUTA_BASE}")


# Función para crear modelo CNN
def crear_modelo(num_clases):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_clases, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Función mejorada para manejar subcarpetas
def copiar_archivos_recursivamente(origen, destino):
    import shutil
    for item in os.listdir(origen):
        item_path = os.path.join(origen, item)
        if os.path.isdir(item_path):  # Si es subcarpeta, procesarla recursivamente
            copiar_archivos_recursivamente(item_path, destino)
        elif item.lower().endswith(('.png', '.jpg', '.jpeg')):  # Solo imágenes
            try:
                shutil.copy(item_path, destino)
            except PermissionError:
                print(f"Advertencia: No se pudo copiar {item_path} (permisos)")

def entrenar_modelo1_radiografias_vs_otras():
    print("Entrenando modelo 1: Radiografías vs NoRadiografías")

    subdirs = {
        'Radiografias': ['Tuberculosis', 'Normal', 'BajaCalidad'],
        'NoRadiografias': ['NoRadiografia']
    }

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    def generar_directorio_virtual(subdir_dict):
        import tempfile
        temp_root = tempfile.mkdtemp()
        for clase, carpetas in subdir_dict.items():
            destino = os.path.join(temp_root, clase)
            os.makedirs(destino, exist_ok=True)
            for carpeta in carpetas:
                origen = os.path.join(RUTA_BASE, carpeta)
                if not os.path.exists(origen):
                    print(f"¡Advertencia! No existe: {origen}")
                    continue
                copiar_archivos_recursivamente(origen, destino)
        return temp_root

    temp_dir = generar_directorio_virtual(subdirs)

    train_generator = datagen.flow_from_directory(
        temp_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    val_generator = datagen.flow_from_directory(
        temp_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = crear_modelo(num_clases=2)
    model.fit(train_generator, validation_data=val_generator, epochs=10,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
    model.save("modelo1_radiografias_vs_otros.h5")
    print("Modelo 1 guardado.")

def entrenar_modelo2_radiografias_tb():
    print("Entrenando modelo 2: Tuberculosis, Normal, BajaCalidad")

    subdirs = {
        'Tuberculosis': ['Tuberculosis'],
        'Normal': ['Normal'],
        'BajaCalidad': ['BajaCalidad']
    }

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    def generar_directorio_virtual(subdir_dict):
        import tempfile
        temp_root = tempfile.mkdtemp()
        for clase, carpetas in subdir_dict.items():
            destino = os.path.join(temp_root, clase)
            os.makedirs(destino, exist_ok=True)
            for carpeta in carpetas:
                origen = os.path.join(RUTA_BASE, carpeta)
                if not os.path.exists(origen):
                    print(f"¡Advertencia! No existe: {origen}")
                    continue
                copiar_archivos_recursivamente(origen, destino)
        return temp_root

    temp_dir = generar_directorio_virtual(subdirs)

    train_generator = datagen.flow_from_directory(
        temp_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    val_generator = datagen.flow_from_directory(
        temp_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = crear_modelo(num_clases=3)
    model.fit(train_generator, validation_data=val_generator, epochs=10,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
    model.save("modelo2_clasificador_tb.h5")
    print("Modelo 2 guardado.")


if __name__ == "__main__":
    entrenar_modelo1_radiografias_vs_otras()
    entrenar_modelo2_radiografias_tb()