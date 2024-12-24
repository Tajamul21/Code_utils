import os
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    image_path = root.find('path').text
    
    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    
    return image_path, boxes

def visualize_bbox(image_path, boxes):
    image = cv2.imread(image_path)
    for xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return image

# Path to folders containing XML files and images
xml_folder = '/home/tajamul/scratch/DA/Datasets/Voc_Data/AIIMS/Annotations'  # Replace with the path to the folder containing XML files
image_folder = '/home/tajamul/scratch/DA/Datasets/Voc_Data/AIIMS/JPEGImages'  # Replace with the path to the folder containing images
output_folder = '/home/tajamul/scratch/DA/Datasets/Voc_Data/AIIMS/Images_annotated'  # Replace with the path to the output folder

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through XML files and visualize annotations
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_folder, xml_file)
        image_path, boxes = parse_voc_annotation(xml_path)
        image_path = os.path.join(image_folder, os.path.basename(image_path))
        annotated_image = visualize_bbox(image_path, boxes)
        
        # Save the annotated image to the output folder
        output_image_path = os.path.join(output_folder, f'annotated_{os.path.basename(xml_path)[:-4]}.jpg')
        cv2.imwrite(output_image_path, annotated_image)
