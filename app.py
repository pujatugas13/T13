import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Muat model yang telah dilatih
MODEL_PATH = 'traffic_mobilenet.h5'
model = load_model(MODEL_PATH)

# Sesuaikan dengan urutan kelas dataset kamu
CLASS_NAMES = ['High Traffic Density', 'Low Traffic Density', 'Medium Traffic Density']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='Tidak ada file yang diunggah')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='File belum dipilih')

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocessing Gambar
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            preds = model.predict(img_array)
            result_idx = np.argmax(preds[0])
            confidence = round(100 * float(np.max(preds[0])), 2)
            label = CLASS_NAMES[result_idx]

            return render_template('index.html', label=label, confidence=confidence, img_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)