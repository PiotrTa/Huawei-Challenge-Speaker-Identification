from own_library.speaker_identificationTF import SpeakerIdentificationTf


# TODO: Implement silence removal for enrollment and query
# TODO: Check if features are normalized

def main():

    identification = SpeakerIdentificationTf(**{"enrollment_folder": "libri_test/enrollment_folder",
                                                "data_folder": "libri_test",
                                                "model": "pytorch"})
    identification.enroll(seq_len=2, num_seq=10)
    identification.query(seq_len=2, num_seq=2)
    identification.compute_smallest_distances()

    #tensorflow
    #thresholds_cosine = [0.8, 0.85, 0.9, 0.95, 0.99, 0.991, 0.992, 0.995, 0.997, 0.998, 0.9985, 0.9987, 0.9995]
    #pytorch
    thresholds_cosine = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

    evaluation = identification.evaluate(thresholds_cosine)
    print(evaluation)


if __name__ == "__main__":
    main()
