# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import os
# import math
# import pdb
# import json
# import random


# def visualize_batch(image_folder, annotation_file, num_images = 16):
#     with open(annotation_file,"r") as file:
#         annot_data = json.load(file)
#     id2name = {pt['id']: pt['file_name'] for pt in annot_data['images']}
#     bboxes = annot_data['annotations']
#     random.shuffle(bboxes)
    
#     fig = plt.figure("random batch", figsize=(40, 40))
#     size = int(math.sqrt(num_images))
#     for i,bbox in enumerate(bboxes[:num_images]):
#         image_path = os.path.join(image_folder, id2name[bbox["image_id"]])
#         image = cv2.imread(image_path)
#         bbox_coord = np.array([(bbox["bbox"][0], bbox["bbox"][0]+bbox["bbox"][2]),(bbox["bbox"][1], bbox["bbox"][1]+bbox["bbox"][3])], dtype=int)
#         image = cv2.rectangle(image, bbox_coord[0], bbox_coord[1], (255,0,0), 2)

#         ax = plt.subplot(size, size, i + 1)
#         plt.imshow(image)
    
#     plt.tight_layout()
#     plt.savefig('random_batch.png')
#     plt.clf()
              



# if __name__=="__main__":
#     DATA_DIR = "/home/tajamul/scratch/RSNA_NEW/new/rsna_coco"
#     # DATA_DIR = "/home/kshitiz/scratch/MAMMO/DATASETS/INBREAST/Inbreast_4k_coco"
    
#     split = "train"
#     image_folder = os.path.join(DATA_DIR, "{}2017".format(split))
#     if(split == "test"):
#         annotation_file = os.path.join(DATA_DIR, "annotations", "image_info_test-dev2017.json")
#     else:
#         annotation_file = os.path.join(DATA_DIR, "annotations", "instances_{}2017.json".format(split))
    
#     visualize_batch(image_folder, annotation_file, num_images = 16)

import json
import os
import cv2
from tqdm import tqdm

def visualize_bboxes_and_save(json_file_path, image_folder_path, output_folder_path):
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)

    images_data = coco_data['images']
    annotations_data = coco_data['annotations']
    categories_data = coco_data['categories']

    os.makedirs(output_folder_path, exist_ok=True)

    for image_info in tqdm(images_data, desc="Processing images"):
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_path = os.path.join(image_folder_path, image_filename)

        # Find all annotations for this image
        image_annotations = [ann for ann in annotations_data if ann['image_id'] == image_id]

        if len(image_annotations) == 0:
            # Skip images without bounding box annotations
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for ann in image_annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']

            category = next((cat for cat in categories_data if cat['id'] == category_id), None)
            category_name = category['name']

            x, y, width, height = bbox
            x, y, width, height = int(x), int(y), int(width), int(height)

            # Draw the bounding box on the image
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 20)
            cv2.putText(image, "Mal", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 8)

        # Save the image with bounding boxes in the output folder
        output_image_path = os.path.join(output_folder_path, image_filename)
        cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def visualize_bboxes_and_save_all(json_file_path, image_folder_path, output_folder_path):
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)

    images_data = coco_data['images']
    annotations_data = coco_data['annotations']
    categories_data = coco_data['categories']

    os.makedirs(output_folder_path, exist_ok=True)

    for image_info in tqdm(images_data, desc="Processing images"):
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_path = os.path.join(image_folder_path, image_filename)

        # Find all annotations for this image
        image_annotations = [ann for ann in annotations_data if ann['image_id'] == image_id]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for ann in image_annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']

            category = next((cat for cat in categories_data if cat['id'] == category_id), None)
            category_name = category['name']

            x, y, width, height = bbox
            x, y, width, height = int(x), int(y), int(width), int(height)

            # Draw the bounding box on the image
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 5)
            cv2.putText(image, "mal", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 7)

        # Save the image with bounding boxes in the output folder
        output_image_path = os.path.join(output_folder_path, image_filename)
        cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


json_file_path = '/home/tajamul/scratch/DA/DATA/Dmaster_Data/IRCH/annotations/instances_train2017.json'
image_folder_path = '/home/tajamul/scratch/DA/DATA/Dmaster_Data/IRCH/train2017'
output_folder_path = '/home/tajamul/scratch/DA/DATA/Dmaster_Data/IRCH/train2017_with_gt'
visualize_bboxes_and_save(json_file_path, image_folder_path, output_folder_path)
