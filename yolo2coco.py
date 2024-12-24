import os
import glob
import pdb
import numpy as np
import shutil
import cv2
import json
from tqdm import tqdm

def yolo2coco(image_folder, label_folder, num_bboxes, annotation_dict, save_image_dir):
    file_id = len(annotation_dict['images'])
    images = glob.glob(image_folder+"/*.png")
    for file_number, img_path in enumerate(tqdm(images)):
        img_name = os.path.basename(img_path)
        file_name_without_ext = os.path.splitext(img_name)[0]
        yolo_annotation_path  = os.path.join(label_folder, file_name_without_ext+ "." + 'txt')
        img_context = {}
        height,width = cv2.imread(img_path).shape[:2]
        shutil.copy(img_path, os.path.join(save_image_dir,img_name))
        img_context['file_name'] = img_name
        img_context['height'] = height
        img_context['width'] = width
        img_context['id'] = file_id
        img_context['depth'] = 3
        annotation_dict['images'].append(img_context)
        
        if(os.path.exists(yolo_annotation_path)):
            with open(yolo_annotation_path,'r') as f2:
                lines2 = f2.readlines() 
        else:
            lines2 = []
        for i,line in enumerate(lines2):
            line = line.split(' ')
            bbox_dict = {}
            # import pdb
            # pdb.set_trace()
            if(line[0:][0]=="mal" or len(line[0:])==4):
                if(line[0:][0]=="mal"):
                    class_id, x1_pascal,y1_pascal, x2_pascal,y2_pascal= line[0:]
                else:
                    x1_pascal,y1_pascal, x2_pascal,y2_pascal= line[0:]
                class_id=int(0)
                x1_pascal,y1_pascal, x2_pascal,y2_pascal,class_id= float(x1_pascal),float(y1_pascal),float(x2_pascal),float(y2_pascal),int(class_id)
                bbox_dict['id'] = num_bboxes
                bbox_dict['image_id'] = file_id
                bbox_dict['category_id'] = class_id
                bbox_dict['iscrowd'] = 0 
                img = cv2.imread(img_path)
                x_coco = round(x1_pascal)
                y_coco = round(y1_pascal)
                w, h = abs(x2_pascal-x1_pascal),abs(y2_pascal-y1_pascal)
                bbox_dict['area']  = h * w
                bbox_dict['bbox'] = [x_coco,y_coco,w,h]
                bbox_dict['segmentation'] = None
                annotation_dict['annotations'].append(bbox_dict)
                num_bboxes+=1    
            else:
                class_id, x_yolo, y_yolo, width_yolo, height_yolo= line[0:]
                x_yolo,y_yolo,width_yolo,height_yolo,class_id= float(x_yolo),float(y_yolo),float(width_yolo),float(height_yolo),int(class_id)
                bbox_dict['id'] = num_bboxes
                bbox_dict['image_id'] = file_id
                bbox_dict['category_id'] = class_id
                bbox_dict['iscrowd'] = 0 
                h,w = abs(height_yolo*height),abs(width_yolo*width)
                bbox_dict['area']  = h * w
                x_coco = round(x_yolo*width -(w/2))
                y_coco = round(y_yolo*height -(h/2))
                if x_coco <0:
                    x_coco = 1
                if y_coco <0:
                    y_coco = 1
                bbox_dict['bbox'] = [x_coco,y_coco,w,h]
                bbox_dict['segmentation'] = None
                annotation_dict['annotations'].append(bbox_dict)
                num_bboxes+=1
        file_id+=1
    return annotation_dict, num_bboxes

def convert2coco(save_dir, data_dir, data_split):
    num_bboxes = 1
    save_image_dir = os.path.join(save_dir, data_split+"2017")
    os.makedirs(save_image_dir, exist_ok=True)

    annotation_dict = dict()
    annotation_dict['categories'] = [{'id':0,'name':"mal",'supercategory': None}]
    annotation_dict['images'] = []
    annotation_dict['annotations'] = []
    # import pdb
    # pdb.set_trace()
    split_folder = os.path.join(data_dir, data_split)
    # if (data_split == "test" or data_split == "train" or data_split == "val"):
    #     classes = ["mal", "ben"]
    #     for i,cl_name in enumerate(classes):
    #         image_folder = os.path.join(split_folder, cl_name, "images")
    #         label_folder = os.path.join(split_folder, cl_name, "labels")
    #         annotation_dict, num_bboxes = yolo2coco(image_folder, label_folder, num_bboxes, annotation_dict, save_image_dir)
    # else:
    image_folder = os.path.join(split_folder, "images")
    label_folder = os.path.join(split_folder, "labels")
    annotation_dict, num_bboxes = yolo2coco(image_folder, label_folder, num_bboxes, annotation_dict, save_image_dir)

    if(data_split == "test"):
        annot_file = os.path.join(save_dir, "annotations", "image_info_test-dev2017.json")
    else:
        annot_file = os.path.join(save_dir, "annotations", "instances_{}2017.json".format(data_split))
    f = open(annot_file, "w")
    json.dump(annotation_dict, f, indent = 4)
    f.close()


def convert2coco2(save_dir, data_dir, data_split):
    num_bboxes = 1
    save_image_dir = os.path.join(save_dir, data_split+"2017")
    os.makedirs(save_image_dir, exist_ok=True)

    annotation_dict = dict()
    annotation_dict['categories'] = [{'id':0,'name':"mal",'supercategory': None}]
    annotation_dict['images'] = []
    annotation_dict['annotations'] = []


    split_folder = os.path.join(data_dir, data_split)
    classes = ["mal", "ben"]
    file_id = 0
    for i,cl_name in enumerate(classes):
        image_folder = os.path.join(split_folder, cl_name, "images")
        label_folder = os.path.join(split_folder, cl_name, "gt")
        images = glob.glob(image_folder+"/*.png")
        for file_number, img_path in enumerate(images):
            img_name = os.path.basename(img_path)
            file_name_without_ext = os.path.splitext(img_name)[0]
            yolo_annotation_path  = os.path.join(label_folder, file_name_without_ext+ "." + 'txt')
            img_context = {}
            height,width = cv2.imread(img_path).shape[:2]
            shutil.copy(img_path, os.path.join(save_image_dir,img_name))
            img_context['file_name'] = img_name
            img_context['height'] = height
            img_context['width'] = width
            img_context['id'] = file_id
            img_context['depth'] = 3
            annotation_dict['images'].append(img_context)
            
            if(os.path.exists(yolo_annotation_path)):
                with open(yolo_annotation_path,'r') as f2:
                    lines2 = f2.readlines() 
            else:
                lines2 = []

            for i,line in enumerate(lines2):
                line = line.split(' ')
                bbox_dict = {}
                class_id, x1_pascal,y1_pascal, x2_pascal,y2_pascal= line[0:]
                if(class_id=="mal"):
                    class_id=int(0)
                    x1_pascal,y1_pascal, x2_pascal,y2_pascal,class_id= float(x1_pascal),float(y1_pascal),float(x2_pascal),float(y2_pascal),int(class_id)
                    bbox_dict['id'] = num_bboxes
                    bbox_dict['image_id'] = file_id
                    bbox_dict['category_id'] = class_id
                    bbox_dict['iscrowd'] = 0 
                    # img = cv2.imread(img_path)
                    # print(img.shape)
                    # print(x1_pascal,y1_pascal, x2_pascal,y2_pascal)
                    # print(x1_pascal,y1_pascal, x2_pascal-x1_pascal,y2_pascal-y1_pascal)
                    # start = (int(x1_pascal),int(y1_pascal))
                    # end = (int(x2_pascal),int(y2_pascal))
                    # img2 = cv2.rectangle(img, start, end, (255,0,0), 4)
                    # cv2.imwrite("temp.jpg",img2)
                    # exit(0)
                    x_coco = round(x1_pascal)
                    y_coco = round(y1_pascal)
                    w, h = abs(x2_pascal-x1_pascal),abs(y2_pascal-y1_pascal)
                    bbox_dict['area']  = h * w
                    bbox_dict['bbox'] = [x_coco,y_coco,w,h]
                    bbox_dict['segmentation'] = None
                    annotation_dict['annotations'].append(bbox_dict)
                    num_bboxes+=1
            file_id+=1
    annot_file = os.path.join(save_dir, "annotations", "instances_{}2017.json".format(data_split))
    f = open(annot_file, "w")
    json.dump(annotation_dict, f, indent = 4)
    f.close()


if __name__ == '__main__':
    ROOT = "/home/kshitiz/scratch/MAMMO/DATA7/DDSM/DDSM_2k_yolo_raw/splits"
    ROOT = "/home/tajamul/scratch/RSNA/RSNA_200 copy"
    # ROOT = "/home/kshitiz/scratch/MAMMO/DATA_MAMMO/AIIMS/AIIMS_4k"
    # ROOT = "/home/kshitiz/scratch/MAMMO/DATA_MAMMO/INBREAST/Inbreast_yolo/4k"
    # save_dir = "/home/kshitiz/scratch/MAMMO/DATA_MAMMO/DDSM/DDSM_coco_2k_test"
    # save_dir = "/home/kshitiz/scratch/MAMMO/DATA7/DDSM/DDSM_2k_coco_new"
    save_dir = "/home/tajamul/scratch/RSNA/RSNA_NEW_coco"
    # save_dir = "/home/kshitiz/scratch/MAMMO/DATA_MAMMO/AIIMS/AIIMS_4k_coco"
    splits = ["train", "test", "val"]
    # splits = ["test"]
    for i,folder in enumerate(splits):
        os.makedirs(save_dir+"/annotations", exist_ok=True)
        convert2coco(save_dir, ROOT, folder)
