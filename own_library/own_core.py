import os
import numpy as np
import os.path
from pydub import AudioSegment
from own_library.feature_ext import YaafeMFCC
from own_library.normalization import ShortTermStandardization
from own_library.segment import SlidingWindow
from own_library.feature import SlidingWindowFeature


def get_audio_length(path_to_audio):
    audio_file = AudioSegment.from_wav(path_to_audio)
    return audio_file.duration_seconds


def run_feature_extraction(path, normalize=True, norm_window=2):

    feature_extractor = YaafeMFCC(**{"duration": 0.025,
                                     "step": 0.010,
                                     "stack": 1,
                                     "e": False,
                                     "coefs": 19,
                                     "De": True,
                                     "DDe": True,
                                     "D": True,
                                     "DD": True})
    # Call feature extractor
    features = feature_extractor(path)
    if normalize:
        Sliding_window = SlidingWindow(**{"duration": 0.025, "step": 0.01})
        Feature_sliding_window = SlidingWindowFeature(**{"data": features, "sliding_window": Sliding_window})
        Normalization = ShortTermStandardization(**{"duration": norm_window})
        Features_normalized = Normalization(Feature_sliding_window)
        features = Features_normalized.data

    return features


def build_speech_segment_list(speech_regions):
    '''
    Builds a list of speech segments from a pyannote object

    inputs:
        speech_regions: object, output from binarization of scores
    outputs:
        speech_segments_ms: list, contains lists with [[start time 1, end time 1],[start time 1, end time 1]]
                            of speech segments
    '''
    speech_segments_ms = []
    try:
        for segment in speech_regions:
            speech_segments_ms.append([float(format(segment[0], '.8f')), float(format(segment[1], '.8f'))])
        return speech_segments_ms
    except:
        print("Cannot build a list of speech segments from library object")


def convert_time_to_samples(speech_segments_ms, sampling_rate=16000, window_step_MFCC=0.01):
    '''
    Expresses speech segments in features (not miliseconds)

    inputs:
        speech_segments_ms: list, speech segments in seconds
        sample_rate: int, sample rate of the audio file, defaults to 16000
        window_step_MFCC: int,
    outputs:
        speech_segments_sample: list, speech segments in samples
    '''
    try:
        speech_segments_sample = []
        for segment in speech_segments_ms:
            speech_segments_sample.append([int(segment[0]/window_step_MFCC),
                                           int(segment[1]/window_step_MFCC)])
        return speech_segments_sample
    except:
        print("Cannot convert speech segments from seconds to features")

                
def remove_silence(speech_segments_ms, audio_path):
    song = AudioSegment.from_wav(audio_path)
    empty = AudioSegment.empty()
    for i in speech_segments_ms:
        # all in miliseconds
        segment = song[i[0]*1000:i[1]*1000]
        empty = empty + segment

    empty.export(audio_path, format="wav")


def remove_files():
    '''
    File removal: SAD and SCD scores
    The library is not producing new files if there exist new files
    '''
    list_files = ["feature-extraction/AMI/EN2001a.Mix-Headset.npy"]
    for file in list_files:
        try:
            os.remove(file)
        except:
            print("Cannot remove file", file, "after the process!")


def speech_segments_to_silence_segments(speech_segments, length_audio):
    silence_segments = []
    for i, segment in enumerate(speech_segments):
        if i == 0: 
            silence_segments.append([0, speech_segments[i][0]])
        elif i > 0 and i < len(speech_segments):
            silence_segments.append([speech_segments[i-1][1], speech_segments[i][0]])
    silence_segments.append([speech_segments[i][1], length_audio])
    return silence_segments


def changes_in_original(changes, silence_segments):
    new_changes = []
    for i, change in enumerate(changes):
        
        silence_segments_before = 0
        for i in range(len(silence_segments)):
            if change > silence_segments[i][0]:
                silence_segments_before += 1
        print(silence_segments_before)

        # Compute length of the silence up to change
        length = 0
        for i in range(silence_segments_before):
            length += float(silence_segments[i][1]-silence_segments[i][0])
        new_changes.append(change + length)
    return new_changes  



def get_change_detection(path):
    '''
    Main function:
    Precondition:
        Place an audio file in the directory specified in "feature-extraction/db.yml"
        Name the file: "EN2001a.Mix-Headset.wav"
    '''
    # Check if the audio file is correctly placed
    assert os.path.isfile(path), "Audio file not found!!!"
    # Run the process
    run_feature_extraction(path)
    compute_scores_speech_detection()
    speech_regions = activity_scores_to_segments()
    speech_segments_ms = build_speech_segment_list(speech_regions)
    remove_files()
    remove_silence(speech_segments_ms)
    run_feature_extraction(path)
    run_speaker_change_detection()
    hypotheses = binarize_change_detection_scores()
    length_audio = get_audio_length(path)
    speech_segments_to_silence_segments(speech_segments_ms, length_audio)
    remove_files()
    return hypotheses
