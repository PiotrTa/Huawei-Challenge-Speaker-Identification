from own_library.feature_ext import LibrosaMFCC
import numpy as np
import tensorflow as tf
import os
from own_library.models import TristouNet
import torch
from own_library.own_core import run_feature_extraction

class utils():
    def __init__(self, model):
        self.model = model

    @staticmethod
    def load_graph(frozen_model):
        '''
        Load frozen protobuf tf model

        Parameters:
            frozen_model: str, path to the .pb file

        '''
        with tf.gfile.GFile(frozen_model, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    @staticmethod
    def build_embedding(batch, dl_lib, average=True):
        '''
        For a batch of feature vectors it computes average embeddings
        '''
        embedding_size = 128 if dl_lib == "pytorch" else 32
        embedding = np.zeros((1, embedding_size), dtype=np.float)

        if dl_lib == "pytorch":
            model = TristouNet(**{'rnn': "LSTM", 'recurrent': [512, 512], 'n_features': 59,
                                  'bidirectional': True, 'pooling': "sum", 'linear': [512, 128]})
            model.load_state_dict(torch.load("models/large_model.pt", map_location=lambda storage, loc: storage))
            model.eval()
            embedding = model(torch.from_numpy(batch).float()).detach()

        else:
            graph = utils.load_graph("models/best_model.pb")

            x_input = graph.get_tensor_by_name('prefix/Placeholder:0')
            x_hid_state = graph.get_tensor_by_name('prefix/Placeholder_1:0')
            y = graph.get_tensor_by_name('prefix/Encoding/dense_1/Tanh:0')

            with tf.Session(graph=graph) as sess:

                for i in range(batch.shape[0]):
                    #batch_swap = np.swapaxes(batch[i], 1, 0)
                    embedding_temp = sess.run(y,
                                              feed_dict={x_input: batch[i],#, axis=0,
                                                         x_hid_state: np.zeros((1, 128))})
                    if i == 0:
                        embedding = embedding_temp
                    else:
                        embedding = np.vstack((embedding, embedding_temp))
        if average:
            embedding = np.average(embedding, axis=0)

        return embedding

    @staticmethod
    def tensorflow_features(batch):
        '''
        Converts
        features of size (batch, steps, feature_dim)
        to
        features of size (batch, 1, steps*feature_dim)
        while padding all axes to be divisible by 16
        '''

        def feature_padder(features):
            N, W, H = features.shape
            W_pad = (W % 16 > 0) * (16 - W % 16)
            H_pad = (H % 16 > 0) * (16 - H % 16)
            features = np.pad(features, ((0, 0), (0, W_pad), (0, H_pad)),
                              'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            return features

        batch_pad = feature_padder(batch)
        batch_reshaped = np.reshape(batch_pad, (batch_pad.shape[0], 1,
                                                batch_pad.shape[1] * batch_pad.shape[2]))

        return batch_reshaped

    @staticmethod
    def batch_builder(audio, seq_len, num_seq, model, normalize=True, tf_reshape=True):
        feature_stack = None

        for seq in range(num_seq):
            samples_starts = np.random.uniform(0, audio.duration_seconds - seq_len, 1)
            start = samples_starts[0]
            end = start + seq_len
            sample_audio = audio[1000 * start:1000 * end]
            # Save audio sample to file
            name = str(start) + ".wav"
            # TODO: Omit saving the file
            sample_audio.export(name, format="wav")
            # Compute features and add first "batch dimension=1"
            if model == "tensorflow":
                mobile_features = LibrosaMFCC()
                sample_features = np.expand_dims(mobile_features(name), axis=0)
            else:
                sample_features = np.expand_dims(run_feature_extraction(name, normalize, norm_window=2), axis=0)
            os.remove(name)
            if seq == 0:
                feature_stack = sample_features
            else:
                try:
                    feature_stack = np.vstack((feature_stack, sample_features))
                except:
                    print("wrong feature shape")
        if normalize and model == "tensorflow":
            feature_stack = utils.normalize_batch(feature_stack)
        else:
            pass

        if model == "tensorflow"and tf_reshape:

            feature_stack = np.reshape(feature_stack, (feature_stack.shape[0], 1,
                                                       feature_stack.shape[1] * feature_stack.shape[2]))

        return feature_stack

    @staticmethod
    def normalize_batch(batch: np.array) -> np.array:
        """
        Normalizes a batch of size N, W, H
        """
        mean = batch.mean(axis=(1, 2))
        std = batch.std(axis=(1, 2))
        N, W, H = batch.shape
        return np.reshape((np.reshape(batch, (W, H, N)) - mean) / std, (N, W, H))