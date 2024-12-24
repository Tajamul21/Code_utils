import json
import csv

# Load COCO annotations from JSON file
with open('/home/tajamul/scratch/DA/Datasets/Coco_Data/INBreast/annotations/image_info_test-dev2017.json', 'r') as f:
    coco_data = json.load(f)

# Initialize CSV writer and specify the output file path
output_path = '/home/tajamul/scratch/DA/Datasets/Foundation_Data/INBreast/test_id.csv'
with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write header row with column names
    writer.writerow(['image_id', 'image_names', 'cancer_present'])

    # Process each image in COCO dataset
    for img in coco_data['images']:
        image_id = img['id']  # Get the image ID
        image_name = img['file_name']
        
        # Check if the image has annotations (bounding boxes)
        has_bbox = any(ann['image_id'] == img['id'] for ann in coco_data['annotations'])

        # Determine if cancer is present (1 if has_bbox, else 0)
        cancer_present = 1 if has_bbox else 0

        # Write image ID, image name, and cancer presence to CSV file
        writer.writerow([image_id, image_name, cancer_present])
