# 🎉 Dataset Utilities for Object Detection and Image Processing 🚀

Welcome to the **Dataset Utilities** repository! This collection of Python scripts provides powerful tools for converting between popular object detection dataset formats (e.g., **VOC**, **COCO**, **YOLO**), visualizing datasets, and performing image transformations like creating GIFs. It is designed to simplify the management and manipulation of datasets, especially for computer vision and machine learning tasks.

## ✨ Key Features
- **Dataset Format Conversions**: Convert datasets between different formats like VOC, COCO, YOLO, and CSV.
- **Visualization Tools**: Visualize annotations and predictions in different formats (COCO, VOC, CSV).
- **Image Processing Utilities**: Convert images to GIFs and perform various image cropping and manipulation operations.
  
## 🛠️ Available Scripts

### 📊 Dataset Conversion

- **`VOC_to_COCO.py`**: Convert VOC dataset annotations to COCO format.
- **`coco2VOC.py`**: Convert COCO dataset annotations back to VOC format.
- **`yolo2coco.py`**: Convert YOLO dataset annotations to COCO format.
- **`csv2coco.py`**: Convert CSV files into COCO format annotations.
- **`coco2csv.py`**: Convert COCO annotations to CSV format.
- **`coco_split.py`**: Split a COCO dataset into train/test/validation splits.

### 🎥 Visualization and Image Transformation

- **`images2gif.py`**: Convert a set of images into an animated GIF.
- **`visualize_coco.py`**: Visualize COCO annotations on images.
- **`visualize_voc.py`**: Visualize VOC annotations on images.
- **`visualize_csv.py`**: Visualize CSV annotations on images.
- **`visualize_proposals.py`**: Visualize proposed bounding boxes on images.
  
### 🔧 Utility Functions

- **`calc_metrics.py`**: Calculate various evaluation metrics for object detection models.
- **`confusion_matrix.py`**: Generate a confusion matrix for model predictions.
- **`match_id_csv_json.py`**: Match image IDs between CSV and JSON files.
- **`rename_json.py`**: Rename JSON files based on certain rules.
- **`create_preds_multi_modal.py`**: Generate predictions using multi-modal data.

### 🖼️ Image Cropping and Preprocessing

- **`crop_img_ann.py`**: Crop images and their corresponding annotations.
- **`same_size_mammo.py`**: Ensure all mammography images are resized to the same dimensions.
- **`mammo_crop_roi.py`**: Crop Regions of Interest (ROI) from mammography images.
- **`rsna_annot2coco.py`**: Convert RSNA mammography annotations to COCO format.
- **`rsna_split.py`**: Split RSNA dataset into smaller subsets.

### 💡 Miscellaneous

- **`change_coco_res.py`**: Adjust the resolution of images and annotations in COCO format.
- **`make_plot.py`**: Create plots from evaluation metrics or dataset statistics.

---

## 📑 Installation and Setup

### 📦 Requirements
Make sure to install the required libraries listed below:
- `pycocotools`
- `Pillow`
- `opencv-python`
- `matplotlib`
- `numpy`

Install them with:
```bash
pip install -r requirements.txt
