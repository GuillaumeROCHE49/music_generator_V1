'''
Goal of this package is to place word of a given lyrics on a wave sound (by using its bpm and its frequency)
and then generate a midi file from it.
To normalize the datas, we used np.float32 for all datas type.
'''

from midiutil.MidiFile import MIDIFile
import numpy as np
import librosa

class MidiGenerator:
    def __init__(self, sound_path: str, lyrics: str) -> None:
        self.__sound, self.__sample_rate = librosa.load(sound_path, mono=True)
        self.__lyrics = lyrics.replace('\n', ' ').split(' ')

    def get_time(self) -> int:  # in seconds
        return int(np.round(self.__sound.size / self.__sample_rate))        

    def get_bpm(self) -> int:
        return int(np.round(librosa.beat.beat_track(y=self.__sound, sr=self.__sample_rate)[0]))
    
    def get_nb_beats(self) -> int:
        return int(np.round(self.get_time() * (self.get_bpm() / 60)))
    
    def get_nb_notes_in_a_beat(self) -> int:
        nb_notes = len(self.__sound)
        return int(np.round(nb_notes / self.get_nb_beats()))
    
    def get_frequency(self, first_time_code: int, second_time_code: int) -> int:
        '''
        Return the mean frequency of the sound between the two time codes.
        '''
        signal = self.__sound[first_time_code:second_time_code] # , 0]  # 0 for left channel
        return np.mean(librosa.feature.spectral_centroid(y=signal, sr=self.__sample_rate)[0], dtype=int)


    def export_midi(self, output_path: str):
        midi = MIDIFile(1)
        midi.addTempo(0, 0, self.get_bpm())

        nb_notes = self.get_nb_notes_in_a_beat()
        beat_duration = self.get_time() / self.get_nb_beats()
        text_version = ""
        for i in range(self.get_nb_beats()):
            frequency = self.get_frequency(i * nb_notes, (i + 1) * nb_notes)
            converted_frequency = int(librosa.hz_to_midi(frequency))
            print(f'beat {i}: {frequency} Hz -> {converted_frequency} midi')
            midi.addNote(
                0,  # track
                0,  # channel
                converted_frequency,  # pitch
                i * beat_duration,  # start
                beat_duration,  # duration
                100,  # volume
                annotation=self.__lyrics[i]
            )
            text_version += f"{0};{0};{frequency};{round(i * beat_duration, 2)};{round(beat_duration, 2)};{100};{self.__lyrics[i]}\n"
        
        with open(output_path, 'wb') as file:
            midi.writeFile(file)
        with open(output_path + '.txt', 'w') as file:
            file.write(text_version)

if __name__ == '__main__':
    midi_generator = MidiGenerator(
        r'C:\Users\groch\OneDrive\Documents\Dreamtonics\Generator_Synth\music_gen\output\music.wav',
        '''Late at night, feeling lost and alone,
The echoes of the past keep haunting my soul,
But through the static, I hear a familiar sound,
A lofi 80's melody that spins me around.''')
    
    midi_generator.export_midi(r'C:\Users\groch\OneDrive\Documents\Dreamtonics\Generator_Synth\music_gen\output\music.mid')