from pycocotools.coco import COCO
from pascal_voc_writer import Writer
import argparse
import os

def coco2voc(ann_file, output_dir):
    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    cat_idx = {c['id']: c['name'] for c in cats}
    
    annotations_dir = os.path.join(output_dir, 'Annotations')
    os.makedirs(annotations_dir, exist_ok=True)
    
    for img_id in coco.imgs:
        catIds = coco.getCatIds()
        annIds = coco.getAnnIds(imgIds=[img_id], catIds=catIds)
        if len(annIds) > 0:
            img_info = coco.imgs[img_id]
            img_fname = os.path.basename(img_info['file_name'])  # Use only the basename
            image_fname_ls = img_fname.split('.')
            image_fname_ls[-1] = 'xml'
            label_fname = '.'.join(image_fname_ls)
            writer = Writer(img_info['file_name'], img_info['width'], img_info['height'])
            anns = coco.loadAnns(annIds)
            for a in anns:
                bbox = a['bbox']
                bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                bbox = [str(b) for b in bbox]
                catname = cat_idx[a['category_id']]
                writer.addObject(catname, bbox[0], bbox[1], bbox[2], bbox[3])
            writer.save(os.path.join(annotations_dir, label_fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert COCO annotations to PASCAL VOC XML annotations')
    parser.add_argument('--ann_file', required=True, help='Path to annotations file')
    parser.add_argument('--output_dir', required=True, help='Path to output directory where annotations are to be stored')
    args = parser.parse_args()
    
    coco2voc(ann_file=args.ann_file, output_dir=args.output_dir)
