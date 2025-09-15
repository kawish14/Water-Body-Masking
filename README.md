# Water Body Masking from Satellite Images

This Flask web application allows users to upload RGB satellite images and receive segmented waterbody masks using a ViT-U-Net deep learning model.

## Features

- Upload RGB satellite images for prediction
- Waterbody segmentation using a trained ViT U-Net model
- Download the Image and Mask to a local directory
- Automatic resizing and threshold-based binary masking

## Requirements
- Python 3.11
- flask
- flask-cors
- tensorflow
- pillow
- numpy
- pandas
- matplotlib
- scikit-learn
- rasterio
- geopandas
- pyproj

You can install them using:
```bash
pip install -r requirements.txt
```
Install above libraries after creating virtual environment.

### 1. Clone the Repository
```bash
https://github.com/kawish14/Water-Body-Masking.git
cd Water-Body-Masking
```
### 2. Download Model
Download the pre-trained model from the following link and place it in the Application folder (where `model.py` is located):
https://drive.google.com/file/d/1RrnTxElyNg68sBN2ETpVcGlRW7E-zti3/view?usp=sharing

### 3. Create and Activate a Virtual Environment
If you get a script execution error, run the following:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

```
Then create and activate the environment:
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Run the App
```bash
python app.py
```
#### or
```
python app.py run --host=0.0.0.0 --port=5000
```

The application will start on http://127.0.0.1:5000/
