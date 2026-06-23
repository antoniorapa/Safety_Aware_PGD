import os
from pathlib import Path
from typing import List, Optional, Union
 
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
 
 
try:
    from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
except ModuleNotFoundError:
    SafetyConfig = None
 
 
class RealVisXL:
    """
    Wrapper per RealVisXL_V5.0 usato nella nuova pipeline sperimentale.
 
    Modello Hugging Face:
        SG161222/RealVisXL_V5.0
 
    Questo wrapper è pensato per:
    - generare immagini target a partire da unsafe_prompt;
    - salvare immagini su disco;
    - essere compatibile con versioni recenti di diffusers;
    - evitare errori legati a stable_diffusion_safe.SafetyConfig, rimosso in alcune versioni.
    """
 
    def __init__(
        self,
        model_id: str = "SG161222/RealVisXL_V5.0",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        use_safetensors: bool = True,
        variant: Optional[str] = None,
        enable_cpu_offload: bool = False,
        disable_safety_checker: bool = False,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.use_safetensors = use_safetensors
        self.variant = variant
        self.enable_cpu_offload = enable_cpu_offload
        self.disable_safety_checker = disable_safety_checker
 
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
 
        if torch_dtype is None:
            if self.device == "cuda":
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype
 
        self.pipe = self._load_pipeline()
 
    def _load_pipeline(self):
        print(f"[RealVisXL] Caricamento modello: {self.model_id}")
        print(f"[RealVisXL] Device: {self.device}")
        print(f"[RealVisXL] dtype: {self.torch_dtype}")
 
        kwargs = {
            "torch_dtype": self.torch_dtype,
            "use_safetensors": self.use_safetensors,
        }
 
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
 
        if self.variant is not None:
            kwargs["variant"] = self.variant
 
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            **kwargs,
        )
 
        if self.disable_safety_checker:
            if hasattr(pipe, "safety_checker"):
                pipe.safety_checker = None
            if hasattr(pipe, "requires_safety_checker"):
                pipe.requires_safety_checker = False
 
        if self.enable_cpu_offload and self.device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception as e:
                print(f"[RealVisXL] CPU offload non disponibile: {e}")
                pipe = pipe.to(self.device)
        else:
            pipe = pipe.to(self.device)
 
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
 
        try:
            pipe.set_progress_bar_config(disable=False)
        except Exception:
            pass
 
        return pipe
 
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        """
        Genera una singola immagine.
 
        Ritorna:
            PIL.Image.Image
 
        Se output_path è specificato, salva anche l'immagine su disco.
        """
 
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Il prompt deve essere una stringa non vuota.")
 
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device)
            generator = generator.manual_seed(seed)
 
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )

        print("[RealVisXL] Generazione completata.")
 
        image = result.images[0]
 
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
 
        return image
    
    def generate_image(
        self,
        prompt: str,
        output_path: Optional[Union[str, Path]] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        num_images: int = 1,
        **kwargs,
    ):
        """
        Alias compatibile con eventuali script che chiamano generate_image().
 
        Se num_images=1 ritorna una singola immagine.
        Se num_images>1 ritorna una lista di immagini.
        """
 
        images = []
 
        for i in range(num_images):
            image_seed = None if seed is None else seed + i
 
            if output_path is not None and num_images > 1:
                output_path_obj = Path(output_path)
                current_output_path = output_path_obj.parent / f"{output_path_obj.stem}_{i:03d}{output_path_obj.suffix}"
            else:
                current_output_path = output_path
 
            image = self.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_path=current_output_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=image_seed,
            )
 
            images.append(image)
            
        return images
 
    def __call__(
        self,
        prompt: str,
        output_path: Optional[Union[str, Path]] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Permette di usare l'oggetto direttamente come funzione.
        """
        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path=output_path,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
        )
 
    def generate_batch(
        self,
        prompts: List[str],
        output_dir: Optional[Union[str, Path]] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Genera una lista di immagini.
 
        Se output_dir è specificato, salva le immagini come:
            image_00000.png
            image_00001.png
            ...
        """
 
        images = []
 
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
 
        for idx, prompt in enumerate(prompts):
            if output_dir is not None:
                output_path = output_dir / f"image_{idx:05d}.png"
            else:
                output_path = None
 
            image_seed = None if seed is None else seed + idx
 
            image = self.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_path=output_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=image_seed,
            )
 
            images.append(image)
 
        return images
 
 
class StableDiffusionSafeGenerator:
    """
    Classe di compatibilità.
 
    Alcune versioni vecchie del progetto potevano importare classi legate a
    Stable Diffusion Safe. Qui la lasciamo come wrapper generico, senza usare
    SafetyConfig obbligatorio, perché nelle versioni recenti di diffusers quel
    modulo può non essere più disponibile.
    """
 
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "StableDiffusionSafeGenerator non è usato nella nuova pipeline RealVisXL. "
            "Usa invece la classe RealVisXL."
        )
 
 
def get_safety_config():
    """
    Funzione di compatibilità.
 
    Ritorna SafetyConfig se disponibile nella versione installata di diffusers,
    altrimenti None.
    """
    return SafetyConfig