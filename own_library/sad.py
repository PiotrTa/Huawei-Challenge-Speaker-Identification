import numpy as np
import torch

from own_library.models import StackedRNN
from own_library.signal import Binarize
from own_library.segment import SlidingWindow
from own_library.feature import SlidingWindowFeature

#############################################################################
'''
Inputs: 
    path_features: string, path to .npy features (possibly after silence removal)
    weigths_sad: string, path to .pt pytorch weigths of a pretrained StackedRNN
                    on speech activity detection (sad)
Outputs:
    hypothesis: object, contains segment objects with start and end time of speech segments     
'''
############################################################################
'''
How to use:

# Path to extracted features (using Yafeelib)
path_features = "feature-extraction/AMI/IS1009a.Mix-Headset.npy"

# Path to weigths for StackedRNN for speaker change detection
weights_sad = 'own_library/weights/sad/1078.pt'

data_sad = activity_detection(path_features, weights_sad)
hypothesis_sad = binarize_sad(data_sad)
'''


def activity_detection(features, weights_sad, window=320, step=80):
    # Load features from a file
    #features = np.load(path_features)
    
    # Load a pretrained model
    Network_sad = StackedRNN(n_features=35, n_classes=2, rnn='LSTM',
                       recurrent=[16,16], bidirectional=True, linear=[16,],
                       logsoftmax=True)
    # Load weights
    Network_sad.load_state_dict(torch.load(weights_sad))
    Network_sad.eval()
    
    # data[i] is the sum of all predictions for frame #i
    data = np.zeros((features.shape[0], 2), dtype=np.float32)

    # k[i] is the number of sequences that overlap with frame #i
    k = np.zeros((features.shape[0], 1), dtype=np.int8)

    i = 0
    while True:
        #try:
        break_ = False
        # indices of frames overlapped by subsequence
        indices = [i,i+step]
        try:
            feature_window = features[indices[0]:indices[1]]
        except:
            feature_window = features[i:]
            break_=True
        feature_window = np.expand_dims(feature_window, axis=0)

        # accumulate the outputs
        try:
            temp = Network_sad(torch.from_numpy(feature_window).float())
            temp = temp.detach()
            data[indices[0]:indices[1]] += np.squeeze(temp)

            # keep track of the number of overlapping sequence
            k[indices[0]:indices[1]] += 1
        except:
            break
        if break_:
            break
        i = i+step
        #except:
         #   print("Array ended, no worries")
    # compute average embedding of each frame
    data = data / np.maximum(k, 1)
    
    return data

def binarize_sad(data):
    # Build object sliding window
    sliding_window = SlidingWindow(**{"duration":3.2, "step":0.8})

    # Build an object sliding window for features
    feature_slid_win = SlidingWindowFeature(**{"data": data, "sliding_window": sliding_window})
    # Initialize binarizer
    binarizer_sad = Binarize(log_scale=True, scale='relative', onset=0.5, offset=0.5, pad_onset=0.,
                             pad_offset=0., min_duration_on=0.2, min_duration_off=0.1)
    # hypothesis is an object
    hypothesis = binarizer_sad.apply(feature_slid_win, dimension=1)
    
    return hypothesis
