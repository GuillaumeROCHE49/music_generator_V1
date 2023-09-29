import argparse as arg

parser = arg.ArgumentParser()

parser.add_argument("-env", "--environment", help="environment to use. (python path)", default=None, type=str)
# -------------------------------------------------------------------------------
parser.add_argument("-p", "--prompt", help="positiv prompt to generate music.", type=str)
parser.add_argument("-n", "--negative_prompt", help="negative prompt to generate music.",
                    default="", type=str)
# -------------------------------------------------------------------------------
parser.add_argument("-o", "--output_dir", help="output directory to export the generate music.", 
                    default="music_gen\\output", type=str)
parser.add_argument("-a", "--name", help="name of the music to generate.",
                    default="music", type=str)
parser.add_argument("-f", "--format", help="format to generate music. (wav, mp3, ogg, flac, ...)",
                    default="wav", type=str)
# -------------------------------------------------------------------------------
parser.add_argument("-s", "--num_steps", help="number of steps to generate music.",
                    default=20, type=int)
parser.add_argument("-e", "--seed", help="seed to generate music.",
                    default=42, type=int)
# -------------------------------------------------------------------------------
parser.add_argument("-i", "--image", help="export image of generate music ?",
                    default=True, type=bool)
# -------------------------------------------------------------------------------
parser.add_argument("-l", "--lyrics", help="generate lyrics ?", default=True, type=bool)
parser.add_argument("-k", "--key", help="openai api key.", default="", type=str)
# -------------------------------------------------------------------------------
parser.add_argument("-c", "--classify", help="classify music ?", default=False, type=bool)
# -------------------------------------------------------------------------------
parser.add_argument("-m", "--midi", help="generate midi file from lyrics ?", default=True, type=bool)


if __name__ == "__main__":
    args = parser.parse_args()

    print("Verify dependencies...")
    requirements = ["numpy", "openai", "torch", "torchaudio", "midiutil",
                    "librosa", "tensorflow", "tensorflow_hub", "csv", "pandas",
                    "IPython", "scipy", "diffusers", "pillow", "pydub"]
    env = args.environment
    if env is None:
        # Verify if it is python or python3
        import sys
        env = "python" if sys.version_info[0] == 3 else "python3"

    # Verify if pip is installed
    try:
        __import__("pip")
    except ImportError:
        # Install pip
        import subprocess
        subprocess.run(f"{env} -m ensurepip".split())
        print("Installed pip.")

    # Verify if all dependencies are installed
    for requirement in requirements:
        try:
            __import__(requirement)
        except ImportError:
            # Install missing dependencies
            process = f"{env} -m pip install {requirement}"
            import subprocess
            subprocess.run(process.split())
            print(f"Installed {requirement}.")
    print("Dependencies verified.")

    print("Verify output directory...")
    # Create output directory if not exists
    import os
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory at {args.output_dir}.")
    else:
        print(f"Output directory already exists at {args.output_dir}.")

    print("Importing libraries...")
    from custom_riffusion import riffusion
    from custom_riffusion.spectrogram.spectrogram_params import SpectrogramParams
    from util.classifier import Classifier
    from util.midi_generator import MidiGenerator

    import numpy as np
    import openai
    import torch
    print("Libraries imported.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading functions...")
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

    def classify_music(path: str) -> Classifier:
        # Classify music using YAMNet
        classifier = Classifier()
        classifier.classify_single(path)
        return classifier

    def create_lyrics(prompt:str, neg_prompt:str, spectrogram: np.ndarray) -> str:
        # Create lyrics on the spectrogram using GPT
        message = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a music expert and a singer."},
                {"role": "user", "content": f"Give me lyrics on a music with this prompt '{prompt}', this negativ prompt '{neg_prompt}' and this spectrogam:\n\n{spectrogram}"},
            ],
        )
        return message['choices'][0]['message']['content']

    def generate_lyrics_midi(sound_path: str, lyrics: str, output_path: str):
        # Delete everything that are in () or []
        import re
        lyrics = re.sub(r'\([^)]*\)', '', lyrics)
        lyrics = re.sub(r'\[[^)]*\]', '', lyrics)
        # Generate midi file from lyrics
        midi_generator = MidiGenerator(sound_path, lyrics)
        midi_generator.export_midi(output_path)
    print("Functions loaded.")

    print("----------------------------------------\nStarting...")
    print(f"Prompt: {args.prompt}")
    
    # Generate music
    path = args.output_dir + "\\" + args.name + "." + args.format
    img = generate_music(args.prompt, args.negative_prompt, path,
                            args.format, int(args.num_steps), int(args.seed))  # type: np.ndarray
    print("Music generated.")
    
    # Export image
    if args.image:
        img_path = args.output_dir + "\\" + args.name + ".png"
        import matplotlib.pyplot as plt
        plt.imsave(img_path, img)
        print("Spectrogram exported at: " + img_path)

    # Classify music
    if args.classify:
        classifier = classify_music(path)
        classifier.to_csv(args.output_dir + "\\" + args.name + ".csv")
        print("Music classified.")

    # Create lyrics
    if args.lyrics:
        if args.key == "":
            raise ConnectionError("You need to specify an openai api key.")
        
        openai.api_key = args.key

        spect_path = args.output_dir + "\\" + args.name + ".png"
        import matplotlib.pyplot as plt
        spectrogram = plt.imread(spect_path)
        
        lyrics = create_lyrics(args.prompt, args.negative_prompt, spectrogram)
        with open(args.output_dir + "\\" + args.name + ".txt", "w") as f:
            f.write(lyrics)
        print("Lyrics generated.")

        # Generate midi file from lyrics
        if args.midi:
            generate_lyrics_midi(path, lyrics, args.output_dir + "\\" + args.name + ".mid")
            print("Midi file generated.")

    print(f"Done at {args.output_dir}.\n----------------------------------------")
