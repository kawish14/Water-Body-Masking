from flask import Flask, request, send_file, jsonify, make_response, render_template
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
from rasterio import Affine
from rasterio.features import shapes
import geopandas as gpd
import json


# Import the model from model.py
from model import ViT

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disables oneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)
CORS(app) 

#OUTPUT_DIR = "saved_masks"
#os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_jgw(jgw_bytes):
    lines = jgw_bytes.decode('utf-8').splitlines()
    x_res = float(lines[0])  # pixel width
    y_res = float(lines[3])  # pixel height (typically negative)
    x_origin = float(lines[4])  # top left X
    y_origin = float(lines[5])  # top left Y
    return x_res, y_res, x_origin, y_origin

def extract_crs_from_xml(xml_bytes):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_bytes)
    srs_element = root.find("SRS")
    if srs_element is not None:
        return CRS.from_wkt(srs_element.text.strip())
    else:
        raise ValueError("No <SRS> element found in XML.")
    
@app.route("/")
def index():
    return render_template("front.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    jgw = request.files['jgw']
    xml = request.files['xml']

    img = load_img(BytesIO(file.read()), target_size=(128, 128))
    img = np.expand_dims(img, axis=0)

    mask = ViT.predict(img)
    if float(request.form['threshold']) == 0.0:
        mask = np.squeeze(mask)
    else:
        threshold = float(request.form.get('threshold', 0.5))
        mask = (mask > threshold).astype(np.uint8)
        mask = np.squeeze(mask)
    mask_scaled = (mask * 255).astype(np.uint8)
    reclassified_mask = np.zeros_like(mask_scaled)
    reclassified_mask[(mask_scaled >= 0) & (mask_scaled <= 50)] = 0
    reclassified_mask[(mask_scaled > 50) & (mask_scaled <= 170)] = 1
    reclassified_mask[(mask_scaled > 170) & (mask_scaled <= 255)] = 2

    visible_mask = np.zeros_like(reclassified_mask)
    visible_mask[reclassified_mask == 1] = 127
    visible_mask[reclassified_mask == 2] = 255
    mask_img = Image.fromarray(visible_mask, mode='L')
    #mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')  # 'L' = grayscale
    
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

    jgw_bytes = jgw.read()
    xml_bytes = xml.read()

    x_res, y_res, x_origin, y_origin = parse_jgw(jgw_bytes)
    print(x_res, y_res, x_origin, y_origin)
    transform = Affine(x_res, 0, x_origin, 0, y_res, y_origin)

    # Convert mask image to NumPy array
    resized_mask = Image.fromarray(reclassified_mask, mode='L').resize(original_size, resample=Image.NEAREST)
    resized_mask_array = np.array(resized_mask)

    results = (
    {'properties': {'value': int(v)}, 'geometry': s}
    for s, v in shapes(resized_mask_array, transform=transform)
    if int(v) != 0  # Skip background
    )

    # Build GeoDataFrame
    crs = extract_crs_from_xml(xml_bytes)
    gdf = gpd.GeoDataFrame.from_features(results, crs=crs)
    if not crs.is_projected:
        lon, lat = (gdf.bounds.minx + gdf.bounds.maxx) / 2, (gdf.bounds.miny + gdf.bounds.maxy) / 2
        zone = int((lon + 180) / 6) + 1
        utm_crs = CRS.from_dict({'proj': 'utm', 'zone': zone, 'ellps': 'WGS84'})
    
        # Reproject to the calculated UTM CRS
        gdf = gdf.to_crs(utm_crs)
        #raise ValueError("CRS must be projected to calculate area correctly.")
    
    gdf['area_sqkm'] = gdf['geometry'].area / 1e6  # 1,000,000 m² in 1 km²

    polygon_io = BytesIO()
    gdf.to_file(polygon_io, driver='GeoJSON')
    polygon_io.seek(0)

    wgs84_io = BytesIO()
    gdf_wgs84 = gdf.to_crs(epsg=4326)              # reproject
    gdf_wgs84.to_file(wgs84_io, driver="GeoJSON") # fresh buffer
    wgs84_io.seek(0)

    zip_io = BytesIO()
    with zipfile.ZipFile(zip_io, 'w') as zf:
        #zf.writestr('image/image.jpg', original_image_bytes)
        zf.writestr('mask.jpg', mask_io_r.getvalue())
        zf.writestr('mask.jgw', jgw_bytes)
        zf.writestr('mask.jpg.aux.xml', xml_bytes)
        zf.writestr('mask_polygons.geojson', polygon_io.getvalue())
        zf.writestr("geojsonWGS1984.geojson", wgs84_io.getvalue())

    zip_io.seek(0)
    return send_file(zip_io, mimetype='application/zip', as_attachment=True, download_name='output.zip')

    #response = make_response(send_file(mask_io_r, mimetype='image/jpg'))
    #response.headers['Content-Length'] = str(mask_io_r.getbuffer().nbytes)
    #return response
    #return send_file(zip_io, mimetype='application/zip', as_attachment=True, download_name='output.zip')
    #return jsonify({"message": "Mask saved", "file": save_path}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)