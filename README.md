# Water Body Masking from Satellite Images

This Flask web application allows users to take RGB satellite image and receive a segmented waterbody mask using a ViT-UNET model.

## Features

- Upload RGB satellite images for prediction
- Waterbody segmentation using a trained ViT U-Net model
- Download the Image and Mask to a local directory

## Requirements
- Python 3.8+
- Flask
- TensorFlow
- OpenCV (`cv2`)
- NumPy
- GDAL
- Matplotlib
- io
- Pillow

### 1. Clone the Repository
```bash
https://github.com/kawish14/Water-Body-Masking.git
cd Water-Body-Masking
```
### 2. Create and Activate a Virtual Environment
#### on windows
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Run the App
```bash
python app.py
```

