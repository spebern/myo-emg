import numpy as np
import random
from enum import Enum
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from myo_emg_recorder import MyoEMGRecorder
from myo_emg_preprocessor import MyoEMGPreprocessor
import pickle
import fire


class HandGesture(Enum):
    RELAX = 0
    PINCH_1 = 1
    PINCH_2 = 2
    PINCH_3 = 3
    PINCH_4 = 4


def gen_labels(trials, label_types):
    """
    Generate an array of length trials containing a random
    sequence of label types.
    """
    labels = (trials // (len(label_types) - 1)) * label_types[1:]
    while len(labels) < trials:
        labels.append(random.choice(label_types[1:]))
    random.shuffle(labels)

    relaxtion_label = label_types[0]
    labels_with_relaxation = []
    for label in labels:
        for _ in range(5):
            labels_with_relaxation.append(relaxtion_label)
        for _ in range(5):
            labels_with_relaxation.append(label)
    return labels_with_relaxation


class App(object):
    def __init__(self):
        recorder = MyoEMGRecorder()
        preprocessor = MyoEMGPreprocessor()

        self._recorder = recorder
        self._preprocessor = preprocessor

    def __del__(self):
        self._recorder.stop()

    def train(self):
        npzfile = np.load("training_data/data.npz")

        clf = LinearDiscriminantAnalysis()
        clf.fit(npzfile["features"], npzfile["labels"])

        with open("trained_classifiers/clf.pkl", "wb") as fid:
            pickle.dump(clf, fid)

    def record(self, trials=10):
        preprocessor = self._preprocessor
        recorder = self._recorder

        recorder.start_recording()

        labels = gen_labels(trials, list(HandGesture))
        features = np.zeros(
            (len(labels), preprocessor.num_channels * preprocessor.num_features * 2)
        )
        for i, label in enumerate(labels):
            print(label.name)
            # skip one second
            _ = recorder.get()
            # now classify
            signals = recorder.get()
            features[i, :] = preprocessor.extract_features(signals)
        labels = [label.value for label in labels]
        np.savez("training_data/data.npz", features=features, labels=labels)

    def evaluate(self, trials=10):
        self._recorder.start_recording()

        with open("trained_classifiers/clf.pkl", "rb") as fid:
            clf = pickle.load(fid)
        for _ in range(trials):
            signals = self._recorder.get()
            features = self._preprocessor.extract_features(signals)
            label = clf.predict([features])[0]
            print(HandGesture(label))


def main():
    fire.Fire(App)


if __name__ == "__main__":
    main()
