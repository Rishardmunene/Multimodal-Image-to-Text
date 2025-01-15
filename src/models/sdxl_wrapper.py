from diffusers import StableDiffusionXLPipeline
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, BlipProcessor

class SDXLModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['model']['device'])
        self.pipeline = None
        self.caption_processor = None
        self.caption_model = None
        
    def load_model(self):
        """Load the SDXL model and caption generation model"""
        # Load BLIP model for caption generation
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        
        # Load SDXL (if needed for other purposes)
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config['model']['sdxl_model_path'],
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(self.device)
        
    def generate_caption(self, image):
        """Generate caption for the given image"""
        # Prepare image for the model
        inputs = self.caption_processor(image, return_tensors="pt").to(self.device)
        
        # Generate caption
        out = self.caption_model.generate(
            **inputs,
            max_length=self.config['pipeline']['max_length'],
            num_return_sequences=1
        )
        
        # Decode the generated caption
        caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
        
        return caption 