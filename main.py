import numpy as np
import random
from enum import Enum
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from myo_emg_recorder import MyoEMGRecorder
from myo_emg_preprocessor import MyoEMGPreprocessor
import pickle
import fire


class HandGesture(Enum):
    OPEN = 0
    CLOSE = 1


def gen_labels(trials, label_types):
    """
    Generate an array of length trials containing a random
    sequence of label types.
    """
    labels = (trials // len(label_types)) * label_types
    while len(labels) < trials:
        labels.append(random.choice(label_types))
    random.shuffle(labels)
    labels = [label.value for label in labels]
    return labels


def run(trials=10, train=False):
    """
    Return EMG classification for HandGestures using the myo armband.

    train : bool
        Train the classifier.
    trials : int
        Number of trials to train or predict.
    """

    recorder = MyoEMGRecorder()
    recorder.start_recording()
    preprocessor = MyoEMGPreprocessor()

    if train is True:
        labels = gen_labels(trials, list(HandGesture))
        features = np.zeros(
            (len(labels), preprocessor.num_channels * preprocessor.num_features * 2)
        )
        print(labels)
        return
        for i, label in enumerate(labels):
            print(label.name)
            # skip one second
            _ = recorder.get()
            # now classify
            signals = recorder.get()
            features[i, :] = preprocessor.extract_features(signals)

        clf = LinearDiscriminantAnalysis()
        clf.fit(features, labels)

        with open("trained_classifiers/clf.pkl", "wb") as fid:
            pickle.dump(clf, fid)
    else:
        with open("trained_classifiers/clf.pkl", "rb") as fid:
            clf = pickle.load(fid)
        for _ in range(trials):
            signals = recorder.get()
            features = preprocessor.extract_features(signals)
            label = clf.predict([features])[0]
            print(HandGesture(label))
    recorder.stop()


def main():
    fire.Fire(run)


if __name__ == "__main__":
    main()
