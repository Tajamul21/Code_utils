import os
import json
import cv2
import shutil
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
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 10)
            cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 4)

        # Save the image with bounding boxes in the output folder
        output_image_path = os.path.join(output_folder_path, image_filename)
        cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def move_images_by_metrics(output_folder_all, output_folder_tp, output_folder_fp, output_folder_tn, output_folder_fn, gt_annotations, pred_annotations):
    os.makedirs(output_folder_tp, exist_ok=True)
    os.makedirs(output_folder_fp, exist_ok=True)
    os.makedirs(output_folder_tn, exist_ok=True)
    os.makedirs(output_folder_fn, exist_ok=True)

    for image_id in range(1, len(gt_annotations) + 1):
        image_filename = f"{image_id:012d}.jpg"
        image_path_all = os.path.join(output_folder_all, image_filename)

        # Find corresponding annotations
        gt_ann = next((gt for gt in gt_annotations if gt['image_id'] == image_id), None)
        pred_ann = next((pred for pred in pred_annotations if pred['image_id'] == image_id), None)

        if gt_ann is None or pred_ann is None:
            continue

        tp_condition = (
            (gt_ann['bbox'][0] < pred_ann['bbox'][0] + pred_ann['bbox'][2] / 2 < gt_ann['bbox'][0] + gt_ann['bbox'][2]) and
            (gt_ann['bbox'][1] < pred_ann['bbox'][1] + pred_ann['bbox'][3] / 2 < gt_ann['bbox'][1] + gt_ann['bbox'][3])
        )

        if tp_condition:
            shutil.move(image_path_all, os.path.join(output_folder_tp, image_filename))
        else:
            shutil.move(image_path_all, os.path.join(output_folder_fp, image_filename))

            if (gt_ann['bbox'][0] > pred_ann['bbox'][0] + pred_ann['bbox'][2] or
                gt_ann['bbox'][0] + gt_ann['bbox'][2] < pred_ann['bbox'][0] or
                gt_ann['bbox'][1] > pred_ann['bbox'][1] + pred_ann['bbox'][3] or
                gt_ann['bbox'][1] + gt_ann['bbox'][3] < pred_ann['bbox'][1]):
                shutil.move(image_path_all, os.path.join(output_folder_tn, image_filename))
            else:
                shutil.move(image_path_all, os.path.join(output_folder_fn, image_filename))

# Example usage
json_file_path_gt = '/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release-froc/data/Inbreast_4k_coco/annotations/image_info_test-dev2017.json'
json_file_path_pred = '/home/tajamul/scratch/Domain_Adaptation/Domain_Adaptation_expts/MRT/Aiims2inbreast/tables/Final_MRT/Source_only/predictions_87/evaluation_result_labels.json'
image_folder_path = '/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release-froc/data/Inbreast_4k_coco/test2017'
output_folder_all = '/home/tajamul/scratch/Domain_Adaptation/Domain_Adaptation_expts/MRT/Aiims2inbreast/tables/Final_MRT/Source_only/predictions_87/pred_images/all'
output_folder_tp = '/home/tajamul/scratch/Domain_Adaptation/Domain_Adaptation_expts/MRT/Aiims2inbreast/tables/Final_MRT/Source_only/predictions_87/pred_images/tp'
output_folder_fp = '/home/tajamul/scratch/Domain_Adaptation/Domain_Adaptation_expts/MRT/Aiims2inbreast/tables/Final_MRT/Source_only/predictions_87/pred_images/fp'
output_folder_tn = '/home/tajamul/scratch/Domain_Adaptation/Domain_Adaptation_expts/MRT/Aiims2inbreast/tables/Final_MRT/Source_only/predictions_87/pred_images/tn'
output_folder_fn = '/home/tajamul/scratch/Domain_Adaptation/Domain_Adaptation_expts/MRT/Aiims2inbreast/tables/Final_MRT/Source_only/predictions_87/pred_images/fn'

# Load annotations
with open(json_file_path_gt, 'r') as f:
    gt_data = json.load(f)

with open(json_file_path_pred, 'r') as f:
    pred_data = json.load(f)

gt_annotations = gt_data['annotations']
pred_annotations = pred_data['annotations']

# Visualize bounding boxes on images
visualize_bboxes_and_save(json_file_path_gt, image_folder_path, output_folder_all)

# Move images to respective folders based on TP, FP, TN, FN
if os.path.exists(image_path_all):
    shutil.move(image_path_all, os.path.join(output_folder_tp, image_filename))
