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
 
 You can also try visualizing the embeddings for your dataset. Take a look into speaker embedding visualization ipynb

