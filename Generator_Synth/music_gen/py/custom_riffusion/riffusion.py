import torch

import threading

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import pydub

from custom_riffusion.spectrogram.spectrogram_image_converter import SpectrogramImageConverter
from custom_riffusion.spectrogram.spectrogram_params import SpectrogramParams

DEFAULT_CHECKPOINT = "riffusion/riffusion-model-v1"

AUDIO_EXTENSIONS = ["mp3", "wav", "flac", "webm", "m4a", "ogg"]
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg"]

SCHEDULER_OPTIONS = [
    "DPMSolverMultistepScheduler",
    "PNDMScheduler",
    "DDIMScheduler",
    "LMSDiscreteScheduler",
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
]

def pipeline_lock() -> threading.Lock:
    """
    Singleton lock used to prevent concurrent access to any model pipeline.
    """
    return threading.Lock()

def get_scheduler(scheduler: str, config):
    """
    Construct a denoising scheduler from a string.
    """
    if scheduler == "PNDMScheduler":
        from diffusers import PNDMScheduler

        return PNDMScheduler.from_config(config)
    elif scheduler == "DPMSolverMultistepScheduler":
        from diffusers import DPMSolverMultistepScheduler

        return DPMSolverMultistepScheduler.from_config(config)
    elif scheduler == "DDIMScheduler":
        from diffusers import DDIMScheduler

        return DDIMScheduler.from_config(config)
    elif scheduler == "LMSDiscreteScheduler":
        from diffusers import LMSDiscreteScheduler

        return LMSDiscreteScheduler.from_config(config)
    elif scheduler == "EulerDiscreteScheduler":
        from diffusers import EulerDiscreteScheduler

        return EulerDiscreteScheduler.from_config(config)
    elif scheduler == "EulerAncestralDiscreteScheduler":
        from diffusers import EulerAncestralDiscreteScheduler

        return EulerAncestralDiscreteScheduler.from_config(config)
    else:
        raise ValueError(f"Unknown scheduler {scheduler}")

def load_stable_diffusion_pipeline(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    scheduler: str = SCHEDULER_OPTIONS[0],
) -> StableDiffusionPipeline:
    """
    Load the riffusion pipeline.

    TODO(hayk): Merge this into RiffusionPipeline to just load one model.
    """
    if device == "cpu" or device.lower().startswith("mps"):
        print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
        dtype = torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        checkpoint,
        revision="main",
        torch_dtype=dtype,
        safety_checker=lambda images, **kwargs: (images, False),
    ).to(device)

    pipeline.scheduler = get_scheduler(scheduler, config=pipeline.scheduler.config)

    return pipeline

def run_txt2img(
    prompt: str,
    num_inference_steps: int,
    guidance: float,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    scheduler: str = SCHEDULER_OPTIONS[0],
) -> Image.Image:
    """
    Run the text to image pipeline with caching.
    """
    with pipeline_lock():
        pipeline = load_stable_diffusion_pipeline(
            checkpoint=checkpoint,
            device=device,
            scheduler=scheduler,
        )

        generator_device = "cpu" if device.lower().startswith("mps") else device
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        output = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt or None,
            generator=generator,
            width=width,
            height=height,
        )

        return output["images"][0]
    
def spectrogram_image_converter(
    params: SpectrogramParams,
    device: str = "cuda",
) -> SpectrogramImageConverter:
    return SpectrogramImageConverter(params=params, device=device)

def audio_segment_from_spectrogram_image(
    image: Image.Image,
    params: SpectrogramParams,
    device: str = "cuda",
) -> pydub.AudioSegment:
    converter = spectrogram_image_converter(params=params, device=device)
    return converter.audio_from_spectrogram_image(image)



if __name__ == "__main__":
    img = run_txt2img(
        prompt="techno DJ and a country fiddle",
        negative_prompt="classic",
        num_inference_steps=10,
        guidance=7.0,
        seed=42,
        width=512,
        height=512,
        checkpoint="riffusion/riffusion-model-v1",
        device="cpu",
        scheduler="DPMSolverMultistepScheduler"
    )
    audio = audio_segment_from_spectrogram_image(
        image=img,
        params=SpectrogramParams(
            stereo=False,
            sample_rate=44100,
            step_size_ms=10,
            window_duration_ms=100,
            padded_duration_ms=400,
            num_frequencies=512,
            min_frequency=0,
            max_frequency=10000,
            mel_scale_norm=None,
            mel_scale_type="htk",
            max_mel_iters=200,
            num_griffin_lim_iters=32,
            power_for_image=0.25,
        ),
        device="cpu"
    )
    audio.export("test.wav", format="wav")
