import csv
import json
import os
import cv2
from tqdm import tqdm  # Import tqdm for progress tracking

# Function to extract height and width from images
def get_image_dimensions(image_folder):
    image_dimensions = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                height, width, _ = img.shape
                image_dimensions[filename] = {"height": height, "width": width}
    return image_dimensions

# Function to convert CSV data to COCO format JSON
def csv_to_coco(csv_file_path, image_folder, output_json_path):
    image_dimensions = get_image_dimensions(image_folder)
    coco_data = {
        "info": {
            "description": "COCO format data without annotations",
            "version": "1.0",
            "year": 2024,
            "contributor": "Your name or organization",
            "date_created": "2024-04-03"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Your license name",
                "url": "URL to license"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "mal",
                "supercategory": None
            }
        ]
    }

    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        num_rows = sum(1 for row in csv_reader)  # Count the number of rows for tqdm
        csv_file.seek(0)  # Reset file pointer
        for row in tqdm(csv_reader, total=num_rows, desc="Processing CSV"):
            file_name = os.path.basename(row['img_path'])  # Extract filename from path
            if file_name in image_dimensions:
                image_id = len(coco_data["images"])   # Assign ID starting from 1
                laterality = row['laterality']
                height = image_dimensions[file_name]["height"]
                width = image_dimensions[file_name]["width"]

                # Construct image entry for COCO format
                image_entry = {
                    "id": image_id,
                    "file_name": file_name,
                    "height": height,
                    "width": width,
                    "uhid": image_id,
                    "laterality": laterality
                }

                coco_data["images"].append(image_entry)
                annotation_entry = {
                    "id": image_id,  # Assuming annotation ID is the same as image ID
                    "image_id": image_id,
                    "category_id": 1,  # Category ID for "mal" category
                    "iscrowd": 0,
                    "area": height * width,  # Calculate area if needed
                    "bbox": [0, 0, width, height],  # Bounding box coordinates [x, y, width, height]
                    "segmentation": None
                }
                coco_data["annotations"].append(annotation_entry)
            else:
                print(f"Warning: File '{file_name}' not found in image folder. Skipping.")

    # Save COCO format data as JSON
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

# Example usage
csv_file_path = '/home/tajamul/scratch/DA/DATA/Dmaster_Data/c_view_data/irch_cview_gt.csv'
image_folder = '/home/tajamul/scratch/DA/Datasets/Dmaster_Data/c_view_data/common_cview'
output_json_path = '/home/tajamul/scratch/DA/Datasets/Dmaster_Data/c_view_data/c_view.json'

csv_to_coco(csv_file_path, image_folder, output_json_path)
