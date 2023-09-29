'''
Sound classification using YAMNet.
'''
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import os
import pandas

from IPython.display import Audio
from scipy.io import wavfile
import scipy.signal

def class_names_from_csv(class_map_csv_text) -> list:
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000) -> tuple:
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

class Classifier:
    """
    Sound classifier using YAMNet.
    """
    def __init__(self):
        """ Initialize classifier. """
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = self.model.class_map_path().numpy()
        self.class_names = class_names_from_csv(class_map_path)
        self.datas = []
        self.spectrogram = []
    
    def get_class_names(self) -> list:
        """Returns list of class names corresponding to score vector."""
        return self.class_names
    
    def classify(self, wav_folder: str):
        """ Classify all wav files in a folder. """
        for wav_file_name in os.listdir(wav_folder):
            if not wav_file_name.endswith('.wav'):
                continue
            self.classify_single(wav_folder + '/' + wav_file_name)
    
    def classify_single(self, wav_file_path: str):
        """ Classify a single wav file. """
        if not wav_file_path.endswith('.wav'):
            raise Exception('File must be a wav file.')

        print(f'Processing {wav_file_path}...')
        sample_rate, wav_data = wavfile.read(wav_file_path, 'rb')
        # Convert to mono
        if len(wav_data.shape) > 1:
            wav_data = np.mean(wav_data, axis=1)
        sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
        
        # Listening to the wav file.
        Audio(wav_data, rate=sample_rate)

        waveform = wav_data / tf.int16.max

        # Run the model, check the output.
        scores, _, spectrogram = self.model(waveform)
        spectrogram = spectrogram.numpy() # type: np.ndarray

        scores_np = scores.numpy()
        infered_class = self.class_names[scores_np.mean(axis=0).argmax()]

        mean_scores = np.mean(scores, axis=0)
        top_n = 6
        top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
        sub_class = [self.class_names[x] for x in top_class_indices][1:top_n]

        self.datas.append({
            'name': os.path.basename(wav_file_path).split('.')[0], # 'music/ambient-classical-guitar-144998.wav' -> 'ambient-classical-guitar-144998
            'file_path': wav_file_path,
            'main_class': infered_class,
            'sub_class': sub_class,
            'spectrogram': spectrogram
        })
    
    def get_datas(self) -> list[dict]:
        """ Get all datas. """
        return self.datas

    def to_csv(self, csv_file_path: str):
        """ Save datas to csv file. """
        df = pandas.DataFrame(self.datas)
        df.to_csv(csv_file_path, index=False)

    def __str__(self) -> str:
        """ Get string representation of the classifier. """
        string = ""
        for data in self.datas:
            string += f'{data["file_path"]}:\n\t{data["main_class"]}\n\t{data["sub_class"]}\n\n'
        return string

if __name__ == '__main__':
    classifier = Classifier()
    classifier.classify('test_music')
    print(classifier)