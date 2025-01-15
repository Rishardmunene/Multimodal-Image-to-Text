import yaml
from src.pipeline.caption_pipeline import CaptionPipeline
import os
from pycocotools.coco import COCO
import urllib.request
import subprocess

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_sample_coco(num_images=1000):
    try:
        coco = COCO('data/annotations/captions_val2017.json')  # Changed to val2017
        print("Successfully loaded COCO annotations")
        
        img_ids = coco.getImgIds()[:num_images]
        print(f"Processing {len(img_ids)} images")
        
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            image_path = os.path.join('data/images', img_info['file_name'])
            if not os.path.exists(image_path):
                try:
                    urllib.request.urlretrieve(img_info['coco_url'], image_path)
                    print(f"Downloaded: {img_info['file_name']}")
                except Exception as e:
                    print(f"Error downloading {img_info['file_name']}: {str(e)}")
    except Exception as e:
        print(f"Error in download_sample_coco: {str(e)}")
def main():
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Initialize pipeline
    pipeline = CaptionPipeline(config)
    pipeline.initialize()
    
    # Download sample images if needed
    download_sample_coco(10)  # Start with 10 images for testing
    
    # Initialize COCO API with validation set
    try:
        coco = COCO('data/annotations/captions_val2017.json')  # Changed to val2017
        print("Successfully loaded COCO annotations")
    except Exception as e:
        print(f"Error loading COCO annotations: {str(e)}")
        return
    
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

if __name__ == "__main__":
    main()