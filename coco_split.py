import json
import os
import shutil
from pathlib import Path

import json
import random


def create_subset_with_annotations(input_file, output_file, total_images, annotated_images):
    # Load JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Filter images with bounding box annotations
    annotated_image_ids = set(ann['image_id'] for ann in data['annotations'])
    images_with_annotations = [img for img in data['images'] if img['id'] in annotated_image_ids]

    # Randomly select images with annotations
    selected_annotated_images = random.sample(images_with_annotations, min(annotated_images, len(images_with_annotations)))

    # Randomly select the remaining images without annotations, with replacement if necessary
    available_non_annotated_images = len(data['images']) - len(images_with_annotations)
    non_annotated_images_needed = total_images - len(selected_annotated_images)

    if non_annotated_images_needed <= available_non_annotated_images:
        non_annotated_images = [img for img in data['images'] if img['id'] not in annotated_image_ids]
        selected_non_annotated_images = random.sample(non_annotated_images, non_annotated_images_needed)
    else:
        selected_non_annotated_images = random.choices(data['images'], k=non_annotated_images_needed)

    # Combine the selected images
    selected_images = selected_annotated_images + selected_non_annotated_images

    # Create a new JSON file with the selected images
    selected_data = {
        'categories': data['categories'],
        'images': selected_images,
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in [img['id'] for img in selected_annotated_images]]
    }

    # Save the selected data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(selected_data, f)
        
        
        
def split_coco_data(json_file_path, output_dir, train_images=100, val_images=29, test_images=1000):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the Coco JSON file
    with open(json_file_path, 'r') as file:
        coco_data = json.load(file)

    # Extract image and annotation information
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])

    # Shuffle the image information randomly
    random.shuffle(images)

    # Create a mapping from original image IDs to new image IDs
    image_id_mapping = {image['id']: i for i, image in enumerate(images)}

    # Update image IDs and create a mapping
    for image in images:
        image['id'] = image_id_mapping[image['id']]

    # Update annotation image IDs using the mapping
    for annotation in annotations:
        annotation['image_id'] = image_id_mapping[annotation['image_id']]

    # Calculate the sizes for train, validation, and test sets
    total_samples = len(images)
    train_size = min(train_images, total_samples)
    val_size = min(val_images, total_samples - train_size)
    test_size = min(test_images, total_samples - train_size - val_size)

    # Split the image information
    train_set = images[:train_size]
    val_set = images[train_size:train_size + val_size]
    test_set = images[train_size + val_size:train_size + val_size + test_size]

    # Split the annotations accordingly
    train_annotations = [annotation for annotation in annotations if annotation['image_id'] in {img['id'] for img in train_set}]
    val_annotations = [annotation for annotation in annotations if annotation['image_id'] in {img['id'] for img in val_set}]
    test_annotations = [annotation for annotation in annotations if annotation['image_id'] in {img['id'] for img in test_set}]

    # Save the split datasets into separate Coco JSON files
    save_path_train = os.path.join(output_dir, 'train_set.json')
    with open(save_path_train, 'w') as file:
        json.dump({'categories': coco_data.get('categories', []), 'images': train_set, 'annotations': train_annotations}, file, indent=2)

    save_path_val = os.path.join(output_dir, 'val_set.json')
    with open(save_path_val, 'w') as file:
        json.dump({'categories': coco_data.get('categories', []), 'images': val_set, 'annotations': val_annotations}, file, indent=2)

    save_path_test = os.path.join(output_dir, 'test_set.json')
    with open(save_path_test, 'w') as file:
        json.dump({'categories': coco_data.get('categories', []), 'images': test_set, 'annotations': test_annotations}, file, indent=2)



def merge_coco_datasets(annotations_folder, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    merged_annotations = {
        "info": [],
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    for split in ["val", "train"]:
        if split == "test":
            annotations_path = Path(annotations_folder) / f"image_info_{split}-dev2017.json"
        else:
            annotations_path = Path(annotations_folder) / f"instances_{split}2017.json"

        with open(annotations_path, "r") as f:
            data = json.load(f)

        # Update image ids to be unique across all splits
        for img in data["images"]:
            img["id"] += len(merged_annotations["images"])

        # Update annotation ids to be unique across all splits
        for ann in data["annotations"]:
            ann["id"] += len(merged_annotations["annotations"])
            ann["image_id"] += len(merged_annotations["images"])

        merged_annotations["images"].extend(data["images"])
        merged_annotations["annotations"].extend(data["annotations"])

    # Save merged annotations
    merged_annotations_path = output_folder / "instances_train.json"
    with open(merged_annotations_path, "w") as f:
        json.dump(merged_annotations, f)

# Rest of the script remains unchanged


def copy_images_based_on_json(json_file_path, source_folder, target_folder):
    # Check if the JSON file exists
    if not os.path.isfile(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return

    # Load the Coco JSON file
    try:
        with open(json_file_path, 'r') as file:
            coco_data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}")
        return

    # Extract image information
    images = coco_data.get('images', [])

    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Copy images to the target folder based on image filenames
    for image in images:
        image_filename = image.get('file_name')
        if not image_filename:
            print(f"Error: File name not found in JSON for {image}")
            continue

        source_image_path = os.path.join(source_folder, image_filename)
        target_image_path = os.path.join(target_folder, image_filename)

        # Check if the source image exists
        if not os.path.isfile(source_image_path):
            print(f"Error: Source image not found for {image_filename}")
            continue

        # Copy the image
        shutil.copyfile(source_image_path, target_image_path)


if __name__ == "__main__":
    annotations_folder = "/home/tajamul/scratch/DA/Datasets/Coco_Data/INBreast/ann"
    image_folders = "/home/tajamul/scratch/DA/Datasets/Voc_Data/watercolor/JPEGImages"
    output_folder = "/home/tajamul/scratch/DA/Datasets/Coco_Data/Natural/kitti/val_split"
    json_file_path = "/home/tajamul/scratch/DA/Datasets/Coco_Data/Natural/kitti/annotations/instances_val2017.json"
    # output_json = "/home/tajamul/scratch/DA/Datasets/Coco_Data/INBreast/annotations/balanced_dissimilar.json"
    # merge_coco_datasets(annotations_folder, output_folder)
    split_coco_data(json_file_path, output_folder)
    # copy_images_based_on_json(json_file_path, image_folders, output_folder)
    # create_subset_with_annotations(json_file_path, output_json, total_images=40, annotated_images=32)
    
    
    
    
    




