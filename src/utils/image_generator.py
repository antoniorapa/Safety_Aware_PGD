import os
import torch
from abc import ABC, abstractmethod
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipelineSafe
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
import os
from dotenv import load_dotenv

load_dotenv()

HG_TOKEN = os.getenv('HG_TOKEN')

current_path = os.path.abspath(__file__)
models_root = os.path.abspath(os.path.join(current_path, "../../../models"))


class AbstractImageGenerator(ABC):
    def __init__(self, base_model_id, lora_model_id, model_name, local_dir=models_root):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_id = base_model_id
        self.lora_model_id = lora_model_id
        self.model_name = model_name
        self.local_dir = local_dir
        self.pipeline = self.load_model()

    def model_exists_locally(self, model_id):
        return os.path.exists(os.path.join(self.local_dir, model_id))

    def download_model(self, model_id):
        if model_id == 'AIML-TUDA/stable-diffusion-safe':
            if not self.model_exists_locally(model_id):
                pipeline = StableDiffusionPipelineSafe.from_pretrained(model_id, torch_dtype=torch.float16,
                                                                       safety_checker=None, use_auth_token=HG_TOKEN).to(
                    self.device)
                pipeline.save_pretrained(os.path.join(self.local_dir, model_id))
            return StableDiffusionPipelineSafe.from_pretrained(os.path.join(self.local_dir, model_id),
                                                               torch_dtype=torch.float16).to(self.device)
        else:
            if not self.model_exists_locally(model_id):
                pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,
                                                             safety_checker=None, ).to(self.device)
                pipeline.save_pretrained(os.path.join(self.local_dir, model_id))
            return DiffusionPipeline.from_pretrained(os.path.join(self.local_dir, model_id),
                                                     torch_dtype=torch.float16).to(self.device)

    def load_model(self):
        # Load base model
        base_pipeline = self.download_model(self.base_model_id)
        # Load LoRA weights
        if self.lora_model_id is not None:
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
        )

    def generate_image(self, prompt, num_images=1, num_inferences=50):
        generator = torch.Generator(device="cuda").manual_seed(0)
        images = self.pipeline(prompt, generator=generator, num_inference_steps=num_inferences,
                               num_images_per_prompt=num_images)["images"]
        return images

    def numpy_to_pil(self, images_numpy):
        return self.pipeline.numpy_to_pil(images_numpy)


class SDXL2(AbstractImageGenerator):
    def __init__(self):
        super().__init__(
            base_model_id="fluently/Fluently-XL-v2",
            lora_model_id="ehristoforu/dalle-3-xl-v2",
            model_name="FluentlyXL",
        )

    def generate_image(self, prompt, num_images=1):
        images = self.pipeline(prompt, num_inference_steps=50, num_images_per_prompt=num_images)["images"]
        return images



class SD14(AbstractImageGenerator):
    def __init__(self):
        super().__init__(
            base_model_id="CompVis/stable-diffusion-v1-4",
            lora_model_id=None,
            model_name="StableDiffusionV1_4",

        )

    def generate_image(self, prompt, num_images=1, num_inferences=50):
        generator = torch.Generator(device="cuda").manual_seed(0)
        images = \
            self.pipeline(prompt, generator=generator, num_inference_steps=num_inferences,
                          num_images_per_prompt=num_images)["images"]
        return images

    def numpy_to_pil(self, images_numpy):
        return self.pipeline.numpy_to_pil(images_numpy)


class SD15(AbstractImageGenerator):
    def __init__(self):
        super().__init__(
            base_model_id="runwayml/stable-diffusion-v1-5",
            lora_model_id=None,
            model_name="StableDiffusionV1_5",

        )

    def generate_image(self, prompt, num_images=1, num_inferences=50):
        generator = torch.Generator(device="cuda").manual_seed(0)
        images = \
            self.pipeline(prompt, generator=generator, num_inference_steps=num_inferences,
                          num_images_per_prompt=num_images)["images"]
        return images

    def numpy_to_pil(self, images_numpy):
        return self.pipeline.numpy_to_pil(images_numpy)


class SLD(AbstractImageGenerator):
    def __init__(self, config=SafetyConfig.MAX):
        self.config = config
        super().__init__(
            base_model_id="AIML-TUDA/stable-diffusion-safe",
            lora_model_id=None,
            model_name="SafeLatentDiffusion",

        )

    def generate_image(self, prompt, num_images=1, num_inferences=50, neg_prompt=None):
        generator = torch.Generator(device="cuda").manual_seed(0)
        images = self.pipeline(prompt, generator=generator, num_inference_steps=num_inferences,
                               num_images_per_prompt=num_images,
                               negative_prompt=neg_prompt, **self.config)["images"]
        return images

    def numpy_to_pil(self, images_numpy):
        return self.pipeline.numpy_to_pil(images_numpy)


class RealVisXL(AbstractImageGenerator):
    def __init__(self):
        super().__init__(
            base_model_id="SG161222/RealVisXL_V5.0",
            lora_model_id=None,
            model_name="RealVisXL_V5_0",
        )

    def generate_image(self, prompt, num_images=1, num_inferences=50):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(0)

        images = self.pipeline(
            prompt,
            generator=generator,
            num_inference_steps=num_inferences,
            num_images_per_prompt=num_images
        )["images"]

        return images

    def numpy_to_pil(self, images_numpy):
        return self.pipeline.numpy_to_pil(images_numpy)



    