from pydub import AudioSegment
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from own_library.utils import utils
import os
import glob


class SpeakerIdentificationTf(object):

    def __init__(self, enrollment_folder, data_folder, model="pytorch"):
        self.enrollment_model = []
        self.query_embeddings = []
        self.smallest_distances = []
        self.enrollment_folder = enrollment_folder
        self.data_folder = data_folder
        self.all_speakers = self.get_all_file_paths_labels(self.data_folder)
        self.model = model

    def enroll(self, seq_len=2, num_seq=50, normalize=True):
        """
        Enrollment: Building the speaker identification model by averaging the embeddings.
        :param seq_len:
        :param num_seq:
        :return:
        """
        for i, speaker in enumerate(sorted(os.listdir(self.enrollment_folder))):
            print("Enrolling speaker: ", speaker)
            audio = AudioSegment.from_wav(os.path.join(self.enrollment_folder, speaker))
            assert audio.duration_seconds > seq_len, "Audio file {} too short. Should be longer than {}"\
                .format(audio, seq_len)
            feature_batch = utils.batch_builder(audio, model=self.model, seq_len=seq_len,
                                                num_seq=num_seq, normalize=normalize)
            avg_emb = utils.build_embedding(feature_batch, self.model)
            self.enrollment_model.append({"Speaker": speaker, "Embedding": np.squeeze(avg_emb)})

        return self.enrollment_model

    def query(self, seq_len=2.23, num_seq=50, normalize=True):
        '''
            Calculates embeddings for query AND! enrolled speakers. Enrolled speakers will be sampled again.
        '''

        for speaker in self.all_speakers:
            print("Computing query embeddings for speaker: ", speaker[0])
            audio = AudioSegment.from_wav(speaker[0])
            assert audio.duration_seconds > seq_len, "Audio file {} too short. Should be longer than {}".format(audio, seq_len)
            feature_batch = utils.batch_builder(audio, model=self.model, seq_len=seq_len,
                                                num_seq=num_seq, normalize=normalize)
            avg_emb = utils.build_embedding(feature_batch, self.model)
            # TODO: Improve speaker naming
            self.query_embeddings.append({"Speaker": speaker[0], "Embedding": np.squeeze(avg_emb)})

    def compute_smallest_distances(self):
        for i, query in enumerate(self.query_embeddings):
            highest_similarity = -1e9
            for j, enrolled in enumerate(self.enrollment_model):

                query_vec = query["Embedding"].reshape(1, -1)
                enrolled_vec = enrolled["Embedding"].reshape(1, -1)

                similarity = cosine_similarity(query_vec, enrolled_vec)
                if similarity > highest_similarity:
                    highest_similarity = similarity
            self.smallest_distances.append({"Speaker": query["Speaker"],
                                            "Distance": round(float(np.squeeze(highest_similarity)), 4)})
            print("Final smallest distance", {"Speaker": query["Speaker"],
                                              "Distance": round(float(np.squeeze(highest_similarity)), 4)})

    def evaluate(self, thresholds,):
        # TODO: Finish evaluation
        scores = []
        # Build ground truth assignments
        y_true = []
        for speaker in sorted(os.listdir(self.enrollment_folder)):
            y_true.append(1)
        for speaker in sorted(os.listdir(os.path.join(self.data_folder, "query_folder"))):
            y_true.append(0)

        for threshold in thresholds:
            y_pred = []
            for speaker in self.smallest_distances:
                if speaker["Distance"] > threshold:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

            scores_thres_level = confusion_matrix(y_true, y_pred, [0, 1])
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=['Enrolled', 'Unknown'])
            temp_score_table ={"Threshold": threshold, "Confusion_matrix": scores_thres_level,
                               "Recall": recall, "Precision": precision, "Accuracy": accuracy, "F1-Score": f1,
                               "y_true": y_true, "y_pred": y_pred, "Classification_report": report}

            scores.append(temp_score_table)
            
        return scores

    @staticmethod
    def get_all_file_paths_labels(data_root: str) -> list:
        """
        Gets the paths of all wav-files in the data root directory plus there labels

        args:
            data_root: string holding name of root directory
        returns:
            all_files: list of lists. Each sub list contains two elements: path to file and label of that file
        """

        speaker_dirs = os.listdir(data_root)
        all_files = []
        i = 0
        for d in speaker_dirs:
            files = glob.iglob(data_root + '/' + d + '/**/*.wav', recursive=True)
            files = [[f, i] for f in files]
            all_files += files
            i += 1
        all_files = sorted(all_files, key=lambda x:x[0], reverse=False)

        return all_files
