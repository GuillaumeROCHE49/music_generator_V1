import openai
import numpy as np
import argparse as arg

parser = arg.ArgumentParser()

parser.add_argument("-p", "--prompt", type=str, help="prompt to create lyrics.", name_or_flags="prompt")
parser.add_argument("-n", "--neg_prompt", help="negativ prompt to create lyrics.",
                    type=str, default="", name_or_flags="neg_prompt")
parser.add_argument("-s", "--spectrogram", type=str, help="the path to the spectrogram. (the image)",
                    default="", name_or_flags="spectrogram")
parser.add_argument("-o", "--output", type=str, help="output path to create lyrics.",
                    default="music_gen\output\lyrics.txt", name_or_flags="output")
parser.add_argument("-k", "--key", type=str, help="openai api key.", name_or_flags="key")


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


if __name__ == "__main__":
    args = parser.parse_args()
    openai.api_key = args.key
    with open(args.spectrogram, "r") as f:
        spectrogram = f.read()
    lyrics = create_lyrics(args.prompt, args.neg_prompt, np.array(spectrogram))
    with open(args.output, "w") as f:
        f.write(lyrics)