import numpy as np
import cv2
import os
import json
from tqdm import tqdm

def coco2class(root, data_split):
    annot_file = None
    if(data_split =="test"):
        annot_file = os.path.join(root, "annotations", "image_info_test-dev2017.json")
    else:
        annot_file = os.path.join(root, "annotations", "instances_{}2017.json".format(data_split))

    annotations_dict = None
    with open(annot_file, "r") as file:
        annotations_dict = json.load(file)
    
    id2idx = { item['id']:i  for i,item in enumerate(annotations_dict["images"]) }

    for bbox in annotations_dict["annotations"]:
        dict_idx = id2idx[bbox["image_id"]]
        w,h = annotations_dict["images"][dict_idx]["width"], annotations_dict["images"][dict_idx]["height"] 
        bbox["bbox"] = [0,0,w,h]
        bbox["area"] = w*h

    trgt_path = os.path.join(root, "annotations_clf")
    os.makedirs(trgt_path, exist_ok=True)
    
    annot_file_new = os.path.join(trgt_path, annot_file.split("/")[-1])
    f = open(annot_file_new, "w")
    json.dump(annotations_dict, f, indent = 4)
    f.close()      
    



def change_resolution(save_dir, root, data_split, factor = 0.5):
    src_img_folder = os.path.join(root,"{}2017".format(data_split)) 
    trg_img_folder = os.path.join(save_dir,"{}2017".format(data_split)) 
    os.makedirs(trg_img_folder, exist_ok=True)
    annot_file = None
    if(data_split =="test"):
        annot_file = os.path.join(root, "annotations", "image_info_test-dev2017.json")
    else:
        annot_file = os.path.join(root, "annotations", "instances_{}2017.json".format(data_split))

    for i, image_name in enumerate(tqdm(os.listdir(src_img_folder))):
        image = cv2.imread(os.path.join(src_img_folder, image_name))
        new_img = cv2.resize(image, (0, 0), fx = factor, fy = factor)
        trgt_img_path = os.path.join(trg_img_folder, image_name)
        cv2.imwrite(trgt_img_path, new_img)
    
    annotations_dict = None
    with open(annot_file, "r") as file:
        annotations_dict = json.load(file)
    
    for img_pt in annotations_dict['images']:
        img_pt["height"]*=factor
        img_pt["width"]*=factor
    for bbox in annotations_dict["annotations"]:
        bbox["area"]*=(factor*factor)
        bbox["bbox"]= list(np.array(bbox["bbox"])*factor)
    

    annot_file_new = os.path.join(save_dir, "annotations", annot_file.split("/")[-1])
    f = open(annot_file_new, "w")
    json.dump(annotations_dict, f, indent = 4)
    f.close()      
    
import os
import cv2
import json
from tqdm import tqdm

def change_resolution_same_size(save_dir, root, data_split):
    src_img_folder = os.path.join(root, "{}2017".format(data_split))
    trg_img_folder = os.path.join(save_dir, "{}2017".format(data_split))
    os.makedirs(trg_img_folder, exist_ok=True)

    annot_file = None
    if data_split == "test":
        annot_file = os.path.join(root, "annotations", "image_info_test-dev2017.json")
    else:
        annot_file = os.path.join(root, "annotations", "instances_{}2017.json".format(data_split))

    with open(annot_file, "r") as file:
        annotations_dict = json.load(file)

    max_height =  4062
    max_width = 3328

    for img_pt in annotations_dict['images']:
        height = img_pt["height"]
        width = img_pt["width"]
        # max_height = max(max_height, height)
        # max_width = max(max_width, width)

    target_size = (max_width, max_height)

    for i, image_name in enumerate(tqdm(os.listdir(src_img_folder))):
        image = cv2.imread(os.path.join(src_img_folder, image_name))

        # Resize image to the target size and pad if necessary
        new_img = resize_and_pad(image, target_size)
        
        trgt_img_path = os.path.join(trg_img_folder, image_name)
        cv2.imwrite(trgt_img_path, new_img)

    for img_pt in annotations_dict['images']:
        img_pt["height"] = target_size[1]
        img_pt["width"] = target_size[0]

    for bbox in annotations_dict["annotations"]:
        bbox["area"] = target_size[0] * target_size[1]
        # Update bbox coordinates accordingly

    annot_file_new = os.path.join(save_dir, "annotations", annot_file.split("/")[-1])
    with open(annot_file_new, "w") as f:
        json.dump(annotations_dict, f, indent=4)

def resize_and_pad(image, target_size):
    height, width, _ = image.shape
    target_width, target_height = target_size

    # Calculate padding
    pad_height = max(0, target_height - height)
    pad_width = max(0, target_width - width)

    # Pad image with black color
    padded_img = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Resize padded image to the target size
    resized_img = cv2.resize(padded_img, target_size)

    return resized_img

# Example usage:
# save_dir = "path/to/save"
# root = "path/to/root"
# data_split = "train"  # or "test" or "val"
# change_resolution_same_size(save_dir, root, data_split)



if __name__ == "__main__":
    # ROOT = "/home/kshitiz/scratch/MAMMO/DATASETS/AIIMS/AIIMS_1k_coco"
    # save_dir = "/home/kshitiz/scratch/MAMMO/DATASETS/AIIMS/AIIMS_1k_coco"
    ROOT = "/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release_working/data_dump/Inbreast_4k_coco_cropped"
    save_dir = "/home/tajamul/scratch/Domain_Adaptation/MRT/MRT-release_working/data_dump/Inbreast_check"
    # ROOT = "/home/kshitiz/scratch/MAMMO/DATA_MAMMO/DDSM/DDSM_2k_coco"
    # save_dir = "/home/kshitiz/scratch/MAMMO/DATA_MAMMO/DDSM/DDSM_1k_coco"
    # ROOT = "/home/kshitiz/scratch/MAMMO/DATA7/DDSM/DDSM_2k_coco_new"
    # save_dir = "/home/kshitiz/scratch/MAMMO/DATA7/DDSM/DDSM_1k_coco_new"
    # ROOT = "/home/kshitiz/scratch/MAMMO/DATA3/RSNA_annot/RSNA_4k_coco"
    # save_dir = "/home/kshitiz/scratch/MAMMO/DATA3/RSNA_annot/RSNA_2k_coco"
    splits = ["train", "val", "test"]
    for i,folder in enumerate(splits):
        # coco2class(ROOT, folder)
        os.makedirs(save_dir+"/annotations", exist_ok=True)
        # change_resolution(save_dir, ROOT, folder, factor = 0.5)
        # target_size = (2560, 3328)  # Set the desired target size
        change_resolution_same_size(save_dir, ROOT, folder)

    