import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

# --- CAMBIO 1: Apuntar al nuevo modelo TFLite ---
MODEL_FILE = 'modelo_optimizado.tflite'
# Las clases deben coincidir con el orden del entrenamiento.
# Basado en tu script, el orden es: 'Grande', 'Camioneta', 'family sedan'
# ¡Asegúrate de que este orden sea el correcto!
# Lo ajustaré a los nombres que usas en el HTML para que se vea bien.
CLASS_NAMES = ['Vehículo de carga', 'Camioneta', 'Sedan'] 

# --- CAMBIO 2: Cargar el modelo con el Intérprete de TFLite ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Modelo '{MODEL_FILE}' cargado correctamente con TFLite.")
except Exception as e:
    print(f"❌ Error al cargar el modelo TFLite: {e}")
    interpreter = None

def preprocess_image(image_bytes):
    """Preprocesa la imagen para que sea compatible con el modelo TFLite."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS) # Usar un método de remuestreo de alta calidad
    img_array = np.array(img) / 255.0
    # TFLite espera el tipo de dato float32
    img_array = img_array.astype(np.float32)
    # Añadir la dimensión del batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    """Renderiza la página principal."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recibe una imagen, la procesa y devuelve la predicción del modelo."""
    if not interpreter:
        return jsonify({'error': 'El modelo no se ha cargado correctamente en el servidor.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ningún archivo en la solicitud.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo.'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)

        # --- CAMBIO 3: Realizar la predicción con la API de TFLite ---
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()  # Ejecutar la inferencia
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_class_index = int(np.argmax(prediction))
        
        # Verificación de seguridad
        if predicted_class_index >= len(CLASS_NAMES):
            return jsonify({'error': 'El modelo devolvió un resultado inesperado.'}), 500

        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(prediction))

        return jsonify({
            'class': predicted_class_name,
            'confidence': f"{confidence:.2%}"
        })

    except Exception as e:
        print(f"❌ Error durante la predicción: {e}")
        # Proporcionar un mensaje de error más útil para el frontend
        return jsonify({'error': 'No se pudo procesar la imagen. Intente con otra.'}), 500

if __name__ == '__main__':
    # Esto es para pruebas locales, Render usará Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=True)
