import cv2
import numpy as np
import os
import json
from tqdm import tqdm

def convert_np_int64_to_python_int(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError

def remove_black_rows_and_columns(img, annotations):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the non-zero (non-black) elements along rows and columns
    non_zero_rows = np.any(gray_img != 0, axis=1)
    non_zero_columns = np.any(gray_img != 0, axis=0)

    # Crop the image to the region containing non-black rows and columns
    cropped_img = img[non_zero_rows][:, non_zero_columns]

    # Update image dimensions in annotations
    height, width, _ = cropped_img.shape
    updated_annotations = []

    for ann in annotations:
        updated_ann = ann.copy()

        # Adjust bounding box coordinates based on non-zero rows and columns
        updated_ann['bbox'][0] = max(0, int(ann['bbox'][0]) - np.argmax(non_zero_columns))
        updated_ann['bbox'][1] = max(0, int(ann['bbox'][1]) - np.argmax(non_zero_rows))
        updated_ann['bbox'][2] = min(int(ann['bbox'][2]) - np.argmax(non_zero_columns) + np.argmax(non_zero_columns), width)
        updated_ann['bbox'][3] = min(int(ann['bbox'][3]) - np.argmax(non_zero_rows) + np.argmax(non_zero_rows), height)

        updated_annotations.append(updated_ann)

    return cropped_img, updated_annotations, height, width


def process_images(input_folder, output_folder, coco_annotations):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize lists to store information
    processed_images_info = {'images': [], 'annotations': [], 'categories': coco_annotations['categories']}

    # Iterate through each image in the input folder with tqdm
    for img_info in tqdm(coco_annotations['images'], desc="Processing Images"):
        filename = img_info['file_name']
        img_annotations = [ann for ann in coco_annotations['annotations'] if ann['image_id'] == img_info['id']]

        # Read the mammogram image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        # Remove black rows and columns
        processed_img, updated_annotations, height, width = remove_black_rows_and_columns(img, img_annotations)

        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(output_path, processed_img)

        # Update image dimensions in the COCO annotations
        img_info['height'] = height
        img_info['width'] = width

        # Append information for each image to the lists
        processed_images_info['images'].append(img_info)
        processed_images_info['annotations'].extend(updated_annotations)

    # Save all processed images and annotations to a single JSON file
    output_annotations_path = os.path.join(os.path.dirname(output_folder), 'annotations', 'processed_annotations.json')

    with open(output_annotations_path, 'w') as json_file:
        json.dump(processed_images_info, json_file, default=convert_np_int64_to_python_int)

    print(f"Processing complete. Results saved to {output_folder}.")



# Specify the path to your input folder containing mammogram images
input_folder_path = '/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release_working/data/rsna_4k_coco/test2017'

# Specify the output folder for the processed images
output_folder_path = '/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release_working/data/rsna_4k_crop/test2017'

# Specify the path to your COCO annotations JSON file
coco_annotations_path = '/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release_working/data/rsna_4k_coco/annotations/image_info_test-dev2017.json'

# Load COCO annotations
with open(coco_annotations_path, 'r') as file:
    coco_annotations = json.load(file)

# Call the function to process all images in the input folder
process_images(input_folder_path, output_folder_path, coco_annotations)
  