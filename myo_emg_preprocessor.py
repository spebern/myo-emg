import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


class BandPassFilter:
    def __init__(self, lowcut=20, highcut=90, fs=200, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(order, [low, high], btype="band")

    def __call__(self, signal):
        return filtfilt(self.b, self.a, signal)


class BandStopFilter:
    def __init__(self, w0=0.9, Q=50):
        self.b, self.a = iirnotch(w0, Q)

    def __call__(self, signal):
        return filtfilt(self.b, self.a, signal)


class MyoEMGPreprocessor:
    def __init__(
        self,
        fs=200,
        num_channels=8,
        duration=1,
        win_size=0.2,
        win_shift=0.05,
        bandpass_filter=BandPassFilter(),
        bandstop_filter=BandStopFilter(),
    ):
        self.fs = fs
        self.num_channels = num_channels
        self.duration = duration
        self.win_size = win_size
        self.win_shift = win_shift
        self.bandpass_filter = bandpass_filter
        self.bandstop_filter = bandstop_filter
        self.num_features = int(self.duration / (self.win_size - self.win_shift))

    def _filter(self, signals):
        for i in range(self.num_channels):
            signals[:, i] = self.bandpass_filter(signals[:, i])
            signals[:, i] = self.bandstop_filter(signals[:, i])

    def _extract_features_from_channel(self, signal):
        mav_features = np.zeros(self.num_features)
        wl_features = np.zeros(self.num_features)

        for i in range(self.num_features):
            start = int((i * (self.win_size - self.win_shift)) * self.fs)
            end = int(((i + 1) * self.win_size - i * self.win_shift) * self.fs)

            mav = np.mean(np.abs(signal[start:end]))
            wl = np.sum(np.abs(np.diff(signal[start:end])))

            wl_features[i] = wl
            mav_features[i] = mav

        return wl_features, mav_features

    def extract_features(self, signals):
        features = np.zeros(self.num_channels * self.num_features * 2)

        self._filter(signals)
        for i in range(self.num_channels):
            wl_features, mav_features = self._extract_features_from_channel(
                signals[:, i]
            )
            features[
                i * self.num_features * 2 : (i + 1) * self.num_features * 2
            ] = np.concatenate((wl_features, mav_features))

        return features
