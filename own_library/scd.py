import numpy as np
import torch

from own_library.models import StackedRNN
from own_library.signal import Peak
from own_library.signal import Binarize
from own_library.segment import SlidingWindow
from own_library.feature import SlidingWindowFeature

#############################################################################
'''
    Inputs: 
        path_features: string, path to .npy features (possibly after silence removal)
        weigths_scd: string, path to .pt pytorch weigths of a pretrained StackedRNN
    Outputs:
        changes: list, absolute predicted changes [1.23, 2.45, ...]      
'''
############################################################################
'''
How to use:

    # Path to extracted features (using Yafeelib)
    path_features = "feature-extraction/AMI/IS1009a.Mix-Headset.npy"
    
    # Path to weigths for StackedRNN for speaker change detection
    weights_scd = 'own_library/weights/scd/0384.pt'

    data_scd = speaker_change_detection(path_features, weights_scd)
    changes = binarize(data_scd)
'''


def speaker_change_detection(features, weights_scd, window=320, step=80):
    # Load features from a file
    #features = np.load(path_features)
    
    # Load a pretrained model
    Network_scd = StackedRNN(n_features=35, n_classes=2, rnn='LSTM',
                       recurrent=[32,20], bidirectional=True,
                       linear=[40, 10], logsoftmax=True)
    # Load weights
    Network_scd.load_state_dict(torch.load(weights_scd))
    Network_scd.eval()
    
    # data[i] is the sum of all predictions for frame #i
    data = np.zeros((features.shape[0], 2), dtype=np.float32)

    # k[i] is the number of sequences that overlap with frame #i
    k = np.zeros((features.shape[0], 1), dtype=np.int8)

    #i = 0
    for i in range(0, features.shape[0], 80):
        #print(i)
    #while True:
        #try:
        break_ = False
        # indices of frames overlapped by subsequence
        indices = [i,i+window]
        try:
            feature_window = features[indices[0]:indices[1]]
        except:
            feature_window = features[i:]
            break_=True
        feature_window = np.expand_dims(feature_window, axis=0)

        # accumulate the outputs
        try:
            temp = Network_scd(torch.from_numpy(feature_window).float())
        except:
            pass
        temp = temp.detach()
        #print("temp shape: ", temp.shape)
        #print("temp.squeeze shape: ", np.squeeze(temp).shape)
        try:
            data[indices[0]:indices[1]] += np.squeeze(temp)
        except:
            print("Last could not be attached")
        # keep track of the number of overlapping sequence
        k[indices[0]:indices[1]] += 1

        if break_:
            break
        i = i+step
        #except:
         #   print("Array ended, no worries")
    # compute average embedding of each frame
    data = data / np.maximum(k, 1)
    
    return data
    binarizer_scd = Binarize()
def binarize_scd_peak(data, duration=2.24, step=0.1, alpha=0.5, min_duration=2, log_scale=True):
    # Build object sliding window
    sliding_window = SlidingWindow(**{"duration":duration, "step":step})
    # Build an object sliding window for features
    feature_slid_win = SlidingWindowFeature(**{"data": data, "sliding_window": sliding_window})
    # Initialize binarizer
    binarizer_scd = Peak(alpha=alpha, min_duration=min_duration, log_scale=log_scale)

    # hypothesis is an object
    hypothesis = binarizer_scd.apply(feature_slid_win, dimension=0)
    # Build a list of changes in seconds (absolute positions)
    changes = []
    for i, segment in enumerate(hypothesis):
        changes.append(round(segment[1], 3))
        
    return changes

def binarize_scd_binarize(data, duration=2.24, step=0.1, onset=0.5, offset=0.5, scale='absolute', log_scale=False,
                 pad_onset=0., pad_offset=0., min_duration_on=0., min_duration_off=0.):
    # Build object sliding window
    sliding_window = SlidingWindow(**{"duration":duration, "step":step})
    # Build an object sliding window for features
    feature_slid_win = SlidingWindowFeature(**{"data": data, "sliding_window": sliding_window})
    # Initialize binarizer
    binarizer_scd = Binarize(onset=onset, offset=offset, scale=scale, log_scale=log_scale,
                 pad_onset=pad_onset, pad_offset=pad_offset, min_duration_on=min_duration_on, min_duration_off=min_duration_off)
    # hypothesis is an object
    hypothesis = binarizer_scd.apply(feature_slid_win, dimension=0)
    # Build a list of changes in seconds (absolute positions)
    changes = []
    for i, segment in enumerate(hypothesis):
        changes.append(round(segment[1], 3))
        
    return changes
