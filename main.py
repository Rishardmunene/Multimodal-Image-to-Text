import yaml
from src.pipeline.caption_pipeline import CaptionPipeline

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Initialize pipeline
    pipeline = CaptionPipeline(config)
    pipeline.initialize()
    
    # Example usage
    image_path = "path/to/your/image.jpg"
    caption = pipeline.generate_caption(image_path)
    print(f"Generated caption: {caption}")

if __name__ == "__main__":
    main() 