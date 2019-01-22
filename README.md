# Huawei Speaker Identification Challenge

Code for Huawei speaker identification challenge. In this repo, you can find two neural networks that are
trained to embed speech segments. The large Pytorch model is trained used pyannote.audio library. The second architecture is designed to be
run on Huawei Neural-Network processing unit (NPU) and is a lighter version of the large one. The training pipline is written 
from scratch in tensorflow.

- Models uploaded
- Notebooks for speech utterance embedding visualization added
- Notebook for speaker identification 


To do:
- Upload the training pipeline
- Documentation
- Clean the library

If you want to play around with the code: 
  - Clone the repo
  - Install dependencies (still need to be documented)
  - Create two folders: one for the speakers to be enrolled, and the other one for queries and place wave files with 16kHz sampling rate.
  - Open run_identification.ipynb and adjust the folder paths.
 
 You can also try visualizing the embeddings for your dataset. Take a look into embedding_visualization.ipynb

Models:
- large_model.pt (pytorch):
  Two bi-LSTM layers (2x512 hidden dimension) plus two fc-layers on top with tanh activation at the end.
- best_model.pb (tensorflow)
  static LSTM (1x128 hidden dimension) plus two fc-layers with tanh activation. Can be run on Huawei Kirin 970.
  
Training data included LibriSpeech subsets: “clean-train-100” and “clean-train-360”, both having together 1105 English speakers, Clarin Polish data set with 552 speakers, and Huawei data set containing 165 English speakers. 

Data preprocessing:

All audio data was converted to wave format and resampled to 16kHz using ffmpeg. Speech activity detection was done offline before training. For non-speech segment removal, a pre-trained support vector machine classifier is used. Speech removal reduced the overall size of the data set from 67.4GB to 55GB. 

Some code for the pytorch network from:
https://github.com/pyannote/pyannote-audio
