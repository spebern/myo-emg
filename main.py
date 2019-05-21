from myo_raw import MyoRaw
import numpy as np
import random
from scipy.signal import butter, filtfilt, iirnotch
from enum import Enum
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from myo_emg_recorder import MyoEMGRecorder
from myo_emg_preprocessor import MyoEMGPreprocessor
import pickle


num_trials_per_gesture = 3
num_channels = 8


class Hand(Enum):
    OPEN = 0
    CLOSE = 1


train = False

recorder = MyoEMGRecorder()
recorder.start_recording()
preprocessor = MyoEMGPreprocessor()

if train is True:
    cues = num_trials_per_gesture * [Hand.OPEN, Hand.CLOSE]
    random.shuffle(cues)

    labels = [cue.value for cue in cues]
    print(preprocessor.num_features)
    features = np.zeros((len(cues), num_channels * preprocessor.num_features * 2))
    print(features.shape)
    for i, cue in enumerate(cues):
        print(cue.name)
        # skip one second
        _ = recorder.get()
        # now classify
        signals = recorder.get()
        features[i, :] = preprocessor.extract_features(signals)

    clf = LinearDiscriminantAnalysis()
    clf.fit(features, labels)

    with open("trained_classifiers/clf.pkl", "wb") as fid:
        pickle.dump(clf, fid)

    recorder.stop()
else:
    with open("trained_classifiers/clf.pkl", "rb") as fid:
        clf = pickle.load(fid)
    while True:
        signals = recorder.get()
        features = preprocessor.extract_features(signals)
        label = clf.predict([features])[0]
        print(Hand(label))
