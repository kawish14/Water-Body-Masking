from flask import Flask, request, send_file, jsonify, make_response
from flask_cors import CORS
import numpy as np
import os
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import datetime
import zipfile
from pyproj import CRS
from xml.etree.ElementTree import Element, SubElement, ElementTree

# Import the model from model.py
from model import ViT

app = Flask(__name__)
CORS(app) 

#OUTPUT_DIR = "saved_masks"
#os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/predict", methods=["POST"])

def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']

    img = load_img(BytesIO(file.read()), target_size=(128, 128))
    img = np.expand_dims(img, axis=0)

    mask = ViT.predict(img)
    if float(request.form['threshold']) == 0.0:
        mask = np.squeeze(mask)
    else:
        threshold = float(request.form.get('threshold', 0.5))
        mask = (mask > threshold).astype(np.uint8)
        mask = np.squeeze(mask)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')  # 'L' = grayscale
    

    original_image = Image.open(file)
    original_size = original_image.size  # (width, height)
    print(original_size)
    mask_img_r = mask_img.resize(original_size, resample=Image.NEAREST)

    #mask_io = BytesIO()
    mask_io_r = BytesIO()
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #save_path = os.path.join(OUTPUT_DIR, f"mask_{timestamp}.png")
    #mask_img.save(save_path)

    #mask_img.save(mask_io, format="PNG")
    mask_img_r.save(mask_io_r, format="JPEG")

    #mask_io.seek(0)
    mask_io_r.seek(0)

    response = make_response(send_file(mask_io_r, mimetype='image/png'))
    response.headers['Content-Length'] = str(mask_io_r.getbuffer().nbytes)
    return response
    #return send_file(zip_io, mimetype='application/zip', as_attachment=True, download_name='output.zip')
    #return jsonify({"message": "Mask saved", "file": save_path}), 200


if __name__ == "__main__":
    app.run(debug=True)