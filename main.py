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


def train_and_save_classifier(clf_name, features, labels):
    clf = LinearDiscriminantAnalysis()
    clf.fit(features, labels)

    with open("trained_classifiers/{}".format(clf_name), "wb") as fid:
        pickle.dump(clf, fid)


class App(object):
    def __init__(self):
        preprocessor = MyoEMGPreprocessor()
        self._preprocessor = preprocessor
        self._recorder = None

    def _init_recorder(self):
        recorder = MyoEMGRecorder()
        recorder.start_recording()
        self._recorder = recorder

    def __del__(self):
        if self._recorder is not None:
            self._recorder.stop()

    def train(self):
        npzfile = np.load("training_data/data.npz")

        labels = npzfile["labels"]
        features = npzfile["features"]

        # train relaxation detector
        relaxation_labels = (labels < 1).astype(int)
        # train gesture detector
        gesture_indexes = np.argwhere(labels > 0).flatten()
        gesture_features = features[gesture_indexes]
        gesture_labels = labels[gesture_indexes]

        train_and_save_classifier("relaxation_clf.pkl", features, relaxation_labels)
        train_and_save_classifier("gesture_clf.pkl", gesture_features, gesture_labels)

    def record(self, trials=10):
        self._init_recorder()
        preprocessor = self._preprocessor
        recorder = self._recorder

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
        self._init_recorder()

        with open("trained_classifiers/relaxation_clf.pkl", "rb") as fid:
            relaxation_clf = pickle.load(fid)

        with open("trained_classifiers/gesture_clf.pkl", "rb") as fid:
            gesture_clf = pickle.load(fid)

        for _ in range(trials):
            signals = self._recorder.get()
            features = self._preprocessor.extract_features(signals)

            label = relaxation_clf.predict([features])[0]
            if label == HandGesture.RELAX:
                print(HandGesture(label))
            else:
                label = gesture_clf.predict([features])[0]
                print(HandGesture(label))


def main():
    fire.Fire(App)


if __name__ == "__main__":
    main()
