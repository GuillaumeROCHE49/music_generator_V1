from custom_riffusion import riffusion
from custom_riffusion.spectrogram.spectrogram_params import SpectrogramParams
from util.classifier import Classifier

import torch
import numpy as np
import argparse as arg

parser = arg.ArgumentParser()

parser.add_argument("-p", "--prompt", help="positiv prompt to generate music.",
                    name_or_flags="prompt")
parser.add_argument("-n", "--negative_prompt", help="negative prompt to generate music.",
                    name_or_flags="negative_prompt", default="")
# -------------------------------------------------------------------------------
parser.add_argument("-o", "--output_dir", help="output directory to generate music.", 
                    name_or_flags="output_dir", default="music_gen\\output")
parser.add_argument("-a", "--name", help="name of the music to generate.",
                    name_or_flags="name", default="music")
parser.add_argument("-f", "--format", help="format to generate music.",
                    name_or_flags="format", default="wav")
# -------------------------------------------------------------------------------
parser.add_argument("-s", "--num_steps", help="num steps to generate music.",
                    name_or_flags="num_steps", default=20)
parser.add_argument("-e", "--seed", help="seed to generate music.",
                    name_or_flags="seed", default=42)
# -------------------------------------------------------------------------------
parser.add_argument("-i", "--image", help="export image to generate music.",
                    name_or_flags="image", default=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def generate_music(positiv_prompt: str, negative_prompt: str, output_path: str,
                   format :str, num_steps: int = 20, seed: int = 42) -> np.array:
    # Generate music using Riffusion (https://github.com/riffusion/riffusion)
    img = riffusion.run_txt2img(
        prompt=positiv_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance=7.0,
        seed=seed,
        width=512,
        height=512,
        checkpoint="riffusion/riffusion-model-v1",
        device=device,
        scheduler="DPMSolverMultistepScheduler"
    )
    audio = riffusion.audio_segment_from_spectrogram_image(
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
        device=device
    )
    # Save music to file
    audio.export(output_path, format=format)
    return np.array(img)

if __name__ == "__main__":
    args = parser.parse_args()

    path = args.output_dir + "\\" + args.name + "." + args.format
    img = generate_music(args.prompt, args.negative_prompt, path,
                         args.format, int(args.num_steps), int(args.seed))  # type: np.ndarray
    if args.image:
        img.save(args.output_path + "\\" + args.name + ".png")
    print("Music generated at: " + path)