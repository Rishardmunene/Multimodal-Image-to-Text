import yaml
from src.pipeline.caption_pipeline import CaptionPipeline
import os
from pycocotools.coco import COCO

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Initialize pipeline
    pipeline = CaptionPipeline(config)
    pipeline.initialize()
    
    # Initialize COCO API
    coco = COCO('data/annotations/captions_train2017.json')
    
    # Get image IDs
    img_ids = coco.getImgIds()
    
    # Process first 10 images (for testing)
    for img_id in img_ids[:10]:
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join('data/images', img_info['file_name'])
        
        # Get ground truth captions
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_captions = [ann['caption'] for ann in anns]
        
        print(f"Processing image: {img_info['file_name']}")
        print("Ground truth captions:")
        for cap in gt_captions:
            print(f"- {cap}")
            
        # Generate caption
        generated_caption = pipeline.generate_caption(image_path)
        print(f"Generated caption: {generated_caption}\n")

def download_sample_coco(num_images=1000):
    coco = COCO('data/annotations/captions_train2017.json')
    img_ids = coco.getImgIds()[:num_images]
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join('data/images', img_info['file_name'])
        if not os.path.exists(image_path):
            !wget -O {image_path} {img_info['coco_url']}

if __name__ == "__main__":
    main() 