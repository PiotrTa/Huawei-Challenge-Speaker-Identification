'''
Computes speech detection scores (higher level using pyannote)

Precodnitions:
    Extracted features are placed in "feature-extraction/" folder and named EN2001a.Mix-Headset.npy.
Outputs:
    Function saves a numpy array in the output folder "speech-activity-detection/sad"
    Shape of the array: (# features, 2) (may be different than features.shape)
'''
def compute_scores_speech_detection():

    # Link to a .yml that has a link to the dataset
    db_yml = "feature-extraction/db.yml"

    # Activity detection scores will be saved here
    output_dir = "speech-activity-detection/sad"

    # Point to a pre-trained model for speech activity detection
    model_pt = "speech-activity-detection/train/AMI.SpeakerDiarization.MixHeadset.train/weights/1078.pt"

    # Argument for the database plugin
    protocol_name = "AMI.SpeakerDiarization.MixHeadset"

    # Say if you have a GPU
    gpu = False
    device = torch.device('cuda') if gpu else torch.device('cpu')

    step = None
    if step is not None:
        step = float(step)

    batch_size = int(32)

    application = SpeechActivityDetection.from_model_pt(
        model_pt, db_yml=db_yml)
    application.device = device
    application.batch_size = batch_size
    application.apply(protocol_name, output_dir, step=step)
########################################################################################
########################################################################################
'''
Binarize predictions using onset/offset thresholding
Outputs an object with speech regions
Scores should be available in "speech-activity-detection/sad/AMI/EN2001a.Mix-Headset.npy"
Shape: (n,2)

Parameters of the binarizer (Crucial for the quality of speech segments)
----------
onset : float, optional
    Relative onset threshold. Defaults to 0.5.
offset : float, optional
    Relative offset threshold. Defaults to 0.5.
scale : {'absolute', 'relative', 'percentile'}
    Set to 'relative' to make onset/offset relative to min/max.
    Set to 'percentile' to make them relative 1% and 99% percentiles.
    Defaults to 'absolute'.
log_scale : bool, optional
    Set to True to indicate that binarized scores are log scaled.
    Will apply exponential first. Defaults to False.
'''

def activity_scores_to_segments():
    try:
        precomputed = Precomputed('speech-activity-detection/sad/')
        binarizer = Binarize(log_scale=True, scale='relative', onset=0.87, offset=0.87, pad_onset=0.,
                             pad_offset=0., min_duration_on=0.2, min_duration_off=0.1)

        protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset')
        for file in protocol.train():
            raw_scores = precomputed(file)
            speech_regions = binarizer.apply(raw_scores, dimension=1)
        return speech_regions
    except:
        print('Cannot convert activity scores to segments')
########################################################################################
########################################################################################
'''
Removes silence from feature array

inputs:
    speech_segments_sample: list of speech segments expressed in samples (not milisecond) (list of lists)
    features: array, shape(number of feature vectors, dimensionality of a feature vector)
output:
    features_no_silence: array shape(number of features without silence, dimensionality of a feature vector)
'''

def feature_silence_removal(speech_segments_sample):
    # Load feature file
    features = np.load("feature-extraction/AMI/EN2001a.Mix-Headset.npy")
    # Remove silence 
    try:
        for i, segment in enumerate(speech_segments_sample):
            if i == 0:
                features_no_silence = features[segment[0]:segment[1]]
            else:
                features_no_silence = np.concatenate((features_no_silence,
                                                      features[segment[0]:segment[1]]),
                                                      axis = 0)
    except:
        print("Cannot remove silence from feature array")
    # Delete the old file and save the new file
    try:
        os.remove("feature-extraction/AMI/EN2001a.Mix-Headset.npy")
        np.save("feature-extraction/AMI/EN2001a.Mix-Headset.npy", features_no_silence)
    except:
        print('Cannot remove or save feature file')
        
########################################################################################
########################################################################################
'''
Run speaker change detection on features with removed silence segments
Features have to be stored in a .npy format in: "feature-extraction/"
Features should have silence removed.
The scores for change detection will be stored in "change-detection/scd/AMI/"
They have to be binarized.
'''

def run_speaker_change_detection():

    db_yml = "feature-extraction/db.yml"
    output_dir = "change-detection/scd"
    model_pt = "change-detection/train/AMI.SpeakerDiarization.MixHeadset.train/weights/0065.pt"
    protocol_name = "AMI.SpeakerDiarization.MixHeadset"
    gpu = False # True if with GPU
    device = torch.device('cuda') if gpu else torch.device('cpu')
    step = None #arguments['--step']

    if step is not None:
        step = float(step)

    batch_size = int(32)
    application = SpeakerChangeDetection.from_model_pt(model_pt, db_yml=db_yml)
    application.device = device
    application.batch_size = batch_size
    application.apply(protocol_name, output_dir, step=step)
########################################################################################
########################################################################################

'''
Binarize change_detection_scores
We use log_scale = True because of the final log-softmax in the StackedRNN model

inputs:

outputs:
    hypotheses: list, speech segments, borders set the speaker changes
'''
def binarize_change_detection_scores(alpha=0.1, min_duration=1, log_scale=True):
    #try:
    protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset')
    precomputed = Precomputed('change-detection/scd')
    hypotheses = []
    for test_file in protocol.train():
        if test_file["uri"] == 'EN2001a.Mix-Headset':
            scd_scores = precomputed(test_file)
            peak = Peak(alpha=alpha, min_duration=min_duration, log_scale=log_scale)
            hypothesis = peak.apply(scd_scores, dimension=0)
    for i, segment in enumerate(hypothesis):
        hypotheses.append(hypothesis[i][1])
    return hypotheses
    #except:
     #   print("Cannot binarize change detection scores!")
        
###########################
