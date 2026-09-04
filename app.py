import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Predictor de prendas",
    page_icon="👕",
    layout="wide",
)


CLASS_NAMES = [
    "Camiseta/top",
    "Pantalón",
    "Jersey",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla",
    "Bolso",
    "Botín",
]


MODEL_PATH = "prendas/prendas.keras"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"No se encontró el archivo '{MODEL_PATH}' "
            "en la misma carpeta que la aplicación."
        )
        return None

    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convierte una imagen a escala de grises 28x28 y la normaliza."""
    gray = image.convert("L")
    gray = gray.resize((28, 28), Image.Resampling.LANCZOS)

    array = np.asarray(gray, dtype=np.float32)
    array = array / 255.0

    return array


def predict_from_image(image: Image.Image):
    model = load_model()

    if model is None:
        return None

    image_array = preprocess_image(image)

    input_tensor = image_array.reshape(1, 28, 28).astype(np.float32)

    prediction = model.predict(input_tensor, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])

    return predicted_index, confidence


st.title("Predictor de prendas con TensorFlow")

st.write(
    "Sube una imagen o dibuja una prenda sobre el canvas para que el modelo "
    "la clasifique. La red fue entrenada con Softmax y entradas 28x28 "
    "en escala de grises."
)


tab_dibujo, tab_archivo = st.tabs(["Dibujar", "Subir imagen"])


# ============================================================
# DIBUJAR
# ============================================================

with tab_dibujo:
    st.subheader("Dibujo en canvas")

    canvas = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True,
    )

    if st.button("Predecir desde el dibujo"):
        try:
            image_data = canvas.image_data

            if image_data is None:
                st.warning("Dibuja algo antes de predecir.")
            else:
                rgb = image_data[:, :, :3]

                gray = np.mean(rgb, axis=2).astype(np.uint8)

                pil_image = Image.fromarray(gray)

                result = predict_from_image(pil_image)

                if result is not None:
                    pred_index, confidence = result

                    st.success(
                        f"Predicción: {CLASS_NAMES[pred_index]} "
                        f"({confidence * 100:.1f}%)"
                    )

                    st.bar_chart(
                        np.asarray([confidence])
                    )
                else:
                    st.warning("No se pudo cargar el modelo.")

        except RuntimeError:
            st.warning(
                "El canvas todavía no tiene una imagen disponible. "
                "Dibuja una prenda y vuelve a pulsar el botón."
            )


# ============================================================
# SUBIR IMAGEN
# ============================================================

with tab_archivo:
    st.subheader("Subir imagen")

    uploaded_file = st.file_uploader(
        "Selecciona una imagen de la prenda",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Imagen cargada",
            use_column_width=True,
        )

        if st.button("Predecir imagen subida"):
            result = predict_from_image(image)

            if result is not None:
                pred_index, confidence = result

                st.success(
                    f"Predicción: {CLASS_NAMES[pred_index]} "
                    f"({confidence * 100:.1f}%)"
                )

                st.bar_chart(
                    np.asarray([confidence])
                )
            else:
                st.warning("No se pudo cargar el modelo.")


# ============================================================
# INSTRUCCIONES
# ============================================================

st.markdown("---")

st.subheader("Instrucciones")

st.markdown(
    """
    - Usa imágenes con fondo negro y la prenda en tonos claros o blancos,
      como en el conjunto de entrenamiento.
    - Para dibujar, usa un ancho de lápiz medio y traza la prenda centrada
      en el canvas.
    - Para imágenes cargadas, procura que sean parecidas a las páginas
      del dataset original o a prendas con silueta y proporciones similares.
    - El modelo espera una imagen 28x28 en escala de grises normalizada
      entre 0 y 1.
    - Las clases disponibles son: Camiseta/top, Pantalón, Jersey, Vestido,
      Abrigo, Sandalia, Camisa, Zapatilla, Bolso y Botín.
    - La red neuronal se entrenó con salida Softmax, por lo que la predicción
      se interpreta como la clase con mayor probabilidad.
    """
)
