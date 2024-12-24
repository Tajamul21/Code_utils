import cv2
import json
import os
from tqdm import tqdm

def resize_images_and_annotations(image_folder, annotation_file, output_folder, target_width, target_height):
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'annotations'), exist_ok=True)

    # Resize images and update annotations
    for img_info in tqdm(annotations['images'], desc='Resizing images'):
        img_path = os.path.join(image_folder, img_info['file_name'])
        img = cv2.imread(img_path)
        
        # Resize image
        resized_img = cv2.resize(img, (target_width, target_height))

        # Update image info
        img_info['width'] = target_width
        img_info['height'] = target_height

        # Update annotations
        for ann in annotations['annotations']:
            if ann['image_id'] == img_info['id']:
                # Scale bounding box coordinates
                x, y, width, height = ann['bbox']
                x_ratio = target_width / img.shape[1]
                y_ratio = target_height / img.shape[0]
                ann['bbox'] = [int(x * x_ratio), int(y * y_ratio), int(width * x_ratio), int(height * y_ratio)]

        # Save resized image
        output_path = os.path.join(output_folder, img_info['file_name'])
        cv2.imwrite(output_path, resized_img)

    # Save updated annotations
    output_annotation_file = os.path.join(output_folder, 'annotations', 'final_annotations.json')
    with open(output_annotation_file, 'w') as outfile:
        json.dump(annotations, outfile)

    return output_annotation_file


# Example usage:
image_folder = '/home/tajamul/scratch/DA/DATA/Dmaster_Data/c_view_data/common_cview'
annotation_file = '/home/tajamul/scratch/DA/DATA/Dmaster_Data/c_view_data/c_view.json'
output_folder = '/home/tajamul/scratch/DA/DATA/Dmaster_Data/c_view_data/common_cview_same_size'
target_width = 4072  # Change this to the desired width
target_height = 3100  # Change this to the desired height

resized_annotation_file = resize_images_and_annotations(image_folder, annotation_file, output_folder, target_width, target_height)
