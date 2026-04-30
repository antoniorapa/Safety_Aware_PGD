import os
import torch
from abc import ABC, abstractmethod
from diffusers import DiffusionPipeline

class AbstractImageGenerator(ABC):
    def __init__(self, base_model_id, lora_model_id, model_name, local_dir="models"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.base_model_id = base_model_id
        self.lora_model_id = lora_model_id
        self.model_name = model_name
        self.local_dir = local_dir
        self.pipeline = self.load_model()

    def model_exists_locally(self, model_id):
        return os.path.exists(os.path.join('../src/models', model_id))

    def download_model(self, model_id):
        if not self.model_exists_locally(model_id):
            pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)
            pipeline.save_pretrained(os.path.join(self.local_dir, model_id))
        return DiffusionPipeline.from_pretrained(os.path.join('../src/models', model_id), torch_dtype=torch.float16).to(self.device)

    def load_model(self):
        # Load base model
        base_pipeline = self.download_model(self.base_model_id)
        # Load LoRA weights
        base_pipeline.load_lora_weights(self.lora_model_id)
        return base_pipeline.to(self.device)

    @abstractmethod
    def generate_image(self, prompt, num_images=1):
        pass


class SDXL(AbstractImageGenerator):
    def __init__(self, local_dir="models"):
        super().__init__(
            base_model_id="bonniebelle/juggernaut-xl-v5",
            lora_model_id="ehristoforu/dalle-3-xl",
            model_name="JuggernautXL",
            local_dir=local_dir
        )

    def generate_image(self, prompt, num_images=1,num_inferences=50):
        images = self.pipeline(prompt, num_inference_steps=num_inferences,num_images_per_prompt=num_images,disable_progress_bar=True)["images"]
        return images


class SDXL2(AbstractImageGenerator):
    def __init__(self, local_dir="models"):
        super().__init__(
            base_model_id="fluently/Fluently-XL-v2",
            lora_model_id="ehristoforu/dalle-3-xl-v2",
            model_name="FluentlyXL",
            local_dir=local_dir
        )

    def generate_image(self, prompt, num_images=1):
        images = self.pipeline(prompt, num_inference_steps=50, num_images_per_prompt=num_images)["images"]
        return images


class SD15(AbstractImageGenerator):
    def __init__(self, local_dir="models"):
        super().__init__(
            base_model_id="runwayml/stable-diffusion-v1-5",
            lora_model_id="Kvikontent/midjourney-v6",
            model_name="StableDiffusionV1_5",
            local_dir=local_dir
        )

    def generate_image(self, prompt, num_images=1):
        images = self.pipeline(prompt, num_inference_steps=50, num_images_per_prompt=num_images)["images"]
        return images
