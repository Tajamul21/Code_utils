import os
import cv2
import json
from tqdm import tqdm

def crop_and_update_annotations(img_info, max_height, max_width, input_dir, output_dir, annotations):
    img_path = os.path.join(input_dir, img_info['file_name'])

    # Load the image
    img = cv2.imread(img_path)

    if img is None:
        tqdm.write(f"Error loading image: {img_path}")
        return None, None

    # Crop the image with the specified dimensions
    cropped_img = img[:max_height, :max_width]

    # Update image information in annotations
    img_info['height'], img_info['width'], _ = cropped_img.shape
    img_info['date_captured'] = '2024-01-01'  # You might want to update the date

    # Update annotations bounding box coordinates
    updated_annotations = []
    for ann in [a for a in annotations if a['image_id'] == img_info['id']]:
        ann['bbox'][0] = int(ann['bbox'][0])
        ann['bbox'][1] = int(ann['bbox'][1])
        ann['bbox'][2] = int(min(ann['bbox'][2], max_width))
        ann['bbox'][3] = int(min(ann['bbox'][3], max_height))
        updated_annotations.append(ann)

    # Save the cropped image
    output_path = os.path.join(output_dir, img_info['file_name'])
    cv2.imwrite(output_path, cropped_img)

    return output_path, updated_annotations

# Load annotations from the JSON file
json_file_path = '/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release-aug/data_actual/Inbreast_4k_coco/annotations/instances_train2017.json'
with open(json_file_path, 'r') as file:
    coco_annotations = json.load(file)

# Output directory for the cropped images
output_dir = '/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release-aug/data/Inbreast_4k_coco/train2017'
# Output directory for the updated annotations in COCO format
output_annotations_path = '/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release-aug/data/Inbreast_4k_coco/annotations/'

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_annotations_path, exist_ok=True)

# Initialize variables to store information about the image with the largest non-black region
max_height, max_width = 0, 0

# Iterate through each image in the dataset with tqdm
for img_info in tqdm(coco_annotations['images'], desc="Finding Max Non-Black Region"):
    img_path = os.path.join('/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release-aug/data_actual/Inbreast_4k_coco/train2017', img_info['file_name'])

    # Load the image
    img = cv2.imread(img_path)

    if img is None:
        tqdm.write(f"Error loading image: {img_path}")
        continue

    # Update max height and width if the current image has a larger non-black region
    max_height = max(max_height, img.shape[0])
    max_width = max(max_width, img.shape[1])

# Specify the desired resolution
target_height = 3328
target_width = 2560

# Iterate through each image in the dataset for cropping
updated_annotations = []
for img_info in tqdm(coco_annotations['images'], desc="Cropping Images"):
    output_path, annotations = crop_and_update_annotations(
        img_info, target_height, target_width,
        '/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release-aug/data_actual/Inbreast_4k_coco/train2017',
        output_dir, coco_annotations['annotations']
    )

    if output_path is not None:
        updated_annotations.extend(annotations)

# Save the updated annotations in COCO format
updated_coco_annotations = {
    "images": coco_annotations['images'],
    "annotations": updated_annotations,
    "categories": coco_annotations['categories']
}
updated_annotations_file_path = os.path.join(output_annotations_path, 'instances_train2017.json')
with open(updated_annotations_file_path, 'w') as json_file:
    json.dump(updated_coco_annotations, json_file)

print(f"Updated annotations saved to: {updated_annotations_file_path}")
