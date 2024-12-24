
import json
import random

# Load annotations from train, val, and test annotation files
def load_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        return json.load(f)

# Update annotations for a given set of images
def update_annotations(dataset, selected_images):
    new_image_id = 0
    new_image_ids = {}
    for image_info in dataset['images']:
        if image_info['id'] in selected_images:
            new_image_ids[image_info['id']] = new_image_id
            image_info['id'] = new_image_id
            new_image_id += 1

    new_annotations = []
    for annotation in dataset['annotations']:
        if annotation['image_id'] in selected_images:
            annotation['image_id'] = new_image_ids[annotation['image_id']]
            new_annotations.append(annotation)

    dataset['images'] = [image_info for image_info in dataset['images'] if image_info['id'] in selected_images]
    dataset['annotations'] = new_annotations

# Define paths to train, val, and test annotation files
train_annotation_file = '/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release_working/data/rsna_4k_coco/annotations/instances_train2017.json'
val_annotation_file = '/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release_working/data/rsna_4k_coco/annotations/instances_val2017.json'
test_annotation_file = '/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release_working/data/rsna_4k_coco/annotations/image_info_test-dev2017.json'

# Load annotations
train_dataset = load_annotations(train_annotation_file)
val_dataset = load_annotations(val_annotation_file)
test_dataset = load_annotations(test_annotation_file)

# Calculate total number of images in each original set
total_train_images = len(train_dataset['images'])
total_val_images = len(val_dataset['images'])
total_test_images = len(test_dataset['images'])

# Define the desired number of images in the test set
desired_test_images = 50
available_mal_images_test = len([image_info for image_info in test_dataset['images'] if 'malignant' in image_info and image_info['malignant']])
desired_mal_images_test = min(available_mal_images_test, 10)
desired_ben_images_test = desired_test_images - desired_mal_images_test

# Define the desired number of images in the val set
desired_val_images = 200
available_mal_images_val = len([image_info for image_info in val_dataset['images'] if 'malignant' in image_info and image_info['malignant']])
desired_mal_images_val = min(available_mal_images_val, 40)
desired_ben_images_val = desired_val_images - desired_mal_images_val

# Calculate the number of remaining images for the train set
remaining_train_images = total_train_images - (desired_test_images + desired_val_images)

# Separate images with malignant and benign cases
malignant_images_test = [image_info['id'] for image_info in test_dataset['images'] if 'malignant' in image_info and image_info['malignant']]
benign_images_test = [image_info['id'] for image_info in test_dataset['images'] if 'malignant' in image_info and not image_info['malignant']]

# Randomly select images for test set
random.seed(42)  # For reproducibility
selected_test_mal_images = random.sample(malignant_images_test, desired_mal_images_test)
selected_test_ben_images = random.sample(benign_images_test, desired_ben_images_test)
selected_test_images = selected_test_mal_images + selected_test_ben_images

# Remove selected test images from the test dataset
selected_test_images_set = set(selected_test_images)
remaining_test_images = [image_info for image_info in test_dataset['images'] if image_info['id'] not in selected_test_images_set]

# Separate images with malignant and benign cases in the val set
malignant_images_val = [image_info['id'] for image_info in val_dataset['images'] if 'malignant' in image_info and image_info['malignant']]
benign_images_val = [image_info['id'] for image_info in val_dataset['images'] if 'malignant' in image_info and not image_info['malignant']]

# Randomly select images for val set
selected_val_mal_images = random.sample(malignant_images_val, desired_mal_images_val)
selected_val_ben_images = random.sample(benign_images_val, desired_ben_images_val)
selected_val_images = selected_val_mal_images + selected_val_ben_images

# Remove selected val images from the val dataset
selected_val_images_set = set(selected_val_images)
remaining_val_images = [image_info for image_info in val_dataset['images'] if image_info['id'] not in selected_val_images_set]

# Randomly select remaining images for train set
selected_train_images = random.sample([image_info['id'] for image_info in train_dataset['images']], remaining_train_images)

print("Number of images in train set:", len(selected_train_images))
print("Number of images in val set:", len(selected_val_images))
print("Number of images in test set:", len(selected_test_images))

# Update annotations for train, val, and test sets
update_annotations(train_dataset, selected_train_images)
update_annotations(val_dataset, selected_val_images)
update_annotations(test_dataset, selected_test_images)

# Save updated annotations to
with open('train_annotations_updated.json', 'w') as f:
    json.dump(train_dataset, f)

with open('val_annotations_updated.json', 'w') as f:
    json.dump(val_dataset, f)

with open('test_annotations_updated.json', 'w') as f:
    json.dump(test_dataset, f)
