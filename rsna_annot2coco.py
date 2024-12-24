import os
import glob
import pdb
import numpy as np
import shutil
import cv2
import json
from bs4 import BeautifulSoup
import json
import xml.etree.ElementTree as ET

def convert2coco(ROOT, save_dir, ids, split):
    save_image_dir = os.path.join(save_dir, split+"2017")
    os.makedirs(save_image_dir, exist_ok=True)

    annotation_dict = dict()
    annotation_dict['categories'] = [{'id':0,'name':"mal",'supercategory': None}]
    annotation_dict['images'] = []
    annotation_dict['annotations'] = []

    bbox_id = 0
    file_id = 0
    for i,file_path in enumerate(ids):
        id = file_path.split("/")[-1]
        box_path = os.path.join(ROOT, id+".xml")
        img_path = os.path.join(ROOT, id+".png")
        shutil.copy(img_path, os.path.join(save_image_dir,id+".png"))
        
        img_context = {}
        height,width = cv2.imread(img_path).shape[:2]
        img_context['file_name'] = id+".png"
        img_context['height'] = height
        img_context['width'] = width
        img_context['id'] = file_id
        img_context['depth'] = 3
        annotation_dict['images'].append(img_context)

        # import pdb; pdb.set_trace()
        if (os.path.exists(os.path.join(ROOT, id+".xml"))):
            root = ET.parse(os.path.join(ROOT, id+".xml")).getroot()
            for j,boxes in enumerate(root.iter("object")):
                bbox_dict = {}
                bbox_dict['id'] = bbox_id
                bbox_dict['image_id'] = file_id
                bbox_dict['category_id'] = 0
                bbox_dict['iscrowd'] = 0 
                
                ymin = int(boxes.find("bndbox/ymin").text)
                xmin = int(boxes.find("bndbox/xmin").text)
                ymax = int(boxes.find("bndbox/ymax").text)
                xmax = int(boxes.find("bndbox/xmax").text)
                
                bbox_dict['area']  = (xmax-xmin)*(ymax-ymin)
                bbox_dict['bbox'] = [xmin, ymin, xmax-xmin, ymax-ymin]
                bbox_dict['segmentation'] = None
                annotation_dict['annotations'].append(bbox_dict)
                bbox_id+=1
        file_id+=1 
    annot_file = os.path.join(save_dir, "annotations", "instances_{}2017.json".format(split))
    f = open(annot_file, "w")
    json.dump(annotation_dict, f, indent = 4)
    f.close()



def split_convert(ROOT, save_dir):
    # import pdb; pdb.set_trace()
    ids = [item[:-4] for item in glob.glob(ROOT + "/*.png")]
    train_ids = [item for item in ids if np.random.random() > 0.13]  # 87% for training
    remaining_ids = np.setdiff1d(ids, train_ids)
    val_ids = [item for item in remaining_ids if np.random.random() > 0.9]  # 3% for validation
    test_ids = np.setdiff1d(remaining_ids, val_ids)  # 10% for testing
    print(len(train_ids), len(val_ids), len(test_ids), len(ids))
    convert2coco(ROOT, save_dir, train_ids, "train")
    convert2coco(ROOT, save_dir, val_ids, "val")
    convert2coco(ROOT, save_dir, test_ids, "test")


    


if __name__ == '__main__':
    ROOT = "/home/tajamul/scratch/RSNA/RSNA_NEW/new/rsna_xml"
    save_dir = "/home/tajamul/scratch/RSNA/MICCAI_2024/RSNA_224/RSNA_4k_coco/"
    np.random.seed(42)
    os.makedirs(save_dir+"/annotations", exist_ok=True)
    split_convert(ROOT, save_dir)