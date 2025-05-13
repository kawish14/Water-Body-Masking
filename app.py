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

def create_jgw(extent, width, height):
    xres = (extent['xmax'] - extent['xmin']) / float(width)
    yres = (extent['ymax'] - extent['ymin']) / float(height)
    return f"""{xres}
0.0
0.0
{-yres}
{extent['xmin'] + xres / 2}
{extent['ymax'] - yres / 2}
"""


def create_aux_xml(xmin, ymin, mask_array):
    # Convert PIL Image to NumPy array
    mask_array = np.array(mask_array)
    
    # Flatten mask and compute statistics
    mask_flat = mask_array.flatten()
    hist_counts, bin_edges = np.histogram(mask_flat, bins=256, range=(0, 255))
    hist_counts_str = '|'.join(map(str, hist_counts))
    hist_min = bin_edges[0]
    hist_max = bin_edges[-1]

    stats_mean = float(np.mean(mask_flat))
    stats_std = float(np.std(mask_flat))
    stats_min = float(np.min(mask_flat))
    stats_max = float(np.max(mask_flat))
    stats_cov = float(np.cov(mask_flat)) if mask_flat.size > 1 else 0.0

    # Set the SRS to WGS 84 / UTM Zone 42N (EPSG:32642)
    crs = CRS.from_epsg(32642)
    srs_wkt = crs.to_wkt()

    return f"""<PAMDataset>
<SRS>{srs_wkt}</SRS>
<Metadata domain=\"IMAGE_STRUCTURE\">
  <MDI key=\"COMPRESSION\">JPEG</MDI>
  <MDI key=\"INTERLEAVE\">PIXEL</MDI>
  <MDI key=\"SOURCE_COLOR_SPACE\">YCbCr</MDI>
</Metadata>
<Metadata>
  <MDI key=\"DataType\">Generic</MDI>
</Metadata>
<PAMRasterBand band=\"1\">
  <NoDataValue>2.56000000000000E+002</NoDataValue>
  <Histograms>
    <HistItem>
      <HistMin>-0.5</HistMin>
      <HistMax>255.5</HistMax>
      <BucketCount>256</BucketCount>
      <IncludeOutOfRange>1</IncludeOutOfRange>
      <Approximate>0</Approximate>
      <HistCounts>{hist_counts_str}</HistCounts>
    </HistItem>
  </Histograms>
  <Metadata domain=\"IMAGE_STRUCTURE\">
    <MDI key=\"COMPRESSION\">JPEG</MDI>
  </Metadata>
  <Metadata>
    <MDI key=\"SourceBandIndex\">0</MDI>
    <MDI key=\"STATISTICS_COVARIANCES\">{stats_cov}</MDI>
    <MDI key=\"STATISTICS_EXCLUDEDVALUES\"/>
    <MDI key=\"STATISTICS_MAXIMUM\">{stats_max}</MDI>
    <MDI key=\"STATISTICS_MEAN\">{stats_mean}</MDI>
    <MDI key=\"STATISTICS_MINIMUM\">{stats_min}</MDI>
    <MDI key=\"STATISTICS_SKIPFACTORX\">1</MDI>
    <MDI key=\"STATISTICS_SKIPFACTORY\">1</MDI>
    <MDI key=\"STATISTICS_STDDEV\">{stats_std}</MDI>
  </Metadata>
</PAMRasterBand>
</PAMDataset>"""

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