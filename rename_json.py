import json
import os

def rename_files_with_spaces(json_file, output_file):
    # Load the COCO JSON file
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    # Check if 'images_dir' key exists in the JSON data
    images_dir = coco_data.get('images_dir', '')

    # Iterate through images and rename filenames with spaces
    for img in coco_data['images']:
        if ' ' in img['file_name']:
            new_file_name = img['file_name'].replace(' ', '_')
            if images_dir:
                # Rename the file if 'images_dir' is specified
                os.rename(os.path.join(images_dir, img['file_name']),
                          os.path.join(images_dir, new_file_name))
            # Update the file_name in COCO data
            img['file_name'] = new_file_name

    # Save the modified COCO data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)


import json

# Load the JSON data from the annotation file
with open('/home/tajamul/scratch/DA/Datasets/Coco_Data/RSNA/annotations/instances_train2017.json', 'r') as json_file:
    coco_data = json.load(json_file)

# Loop through the "images" list and remove ".dcm" from filenames
for image in coco_data['images']:
    image['file_name'] = image['file_name'].replace('.dcm', '')

# Save the modified JSON data back to the annotation file
with open('/home/tajamul/scratch/DA/Datasets/Coco_Data/INBreast/annotations/test_name_corrected.json', 'w') as json_file:
    json.dump(coco_data, json_file, indent=4)




# # Example usage
# json_file_path = '/home/tajamul/scratch/DA/Datasets/Coco_Data/AIIMS/annotations/instances_val2017.json'
# output_file_path = '/home/tajamul/scratch/DA/Datasets/Coco_Data/AIIMS/annotations_name_corrected/instances_val2017.json'
# rename_files_with_spaces(json_file_path, output_file_path)
