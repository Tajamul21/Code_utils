import json

# Load the JSON data from file
with open('/home/tajamul/scratch/DA/Datasets/Coco_Data/RSNA/annotations/instances_val2017.json', 'r') as json_file:
    coco_data = json.load(json_file)

# Extract filenames of images with annotations from COCO JSON data
filenames_with_annotations = set()
for annotation in coco_data['annotations']:
    filenames_with_annotations.add(annotation['image_id'])

image_filenames = []
for image in coco_data['images']:
    if image['id'] in filenames_with_annotations:
        image_filenames.append(image['file_name'])
    else:
        image_filenames.append(image['file_name'])  # Include all images

# Write filenames to a text file
with open('/home/tajamul/scratch/DA/Datasets/Voc_Data/RSNA/ImageSets/Main/val.txt', 'w') as txt_file:
    for filename in image_filenames:
        txt_file.write(filename + '\n')
