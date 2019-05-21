import numpy as np
from myo_raw import MyoRaw
import threading
import queue


class MyoEMGRecorder:
    def __init__(self, fs=200, win_duration=1, num_channels=8):
        self._stop = False

        myo = MyoRaw(None)

        num_samples = win_duration * fs
        emg_win = np.zeros((num_samples, num_channels))

        i = 0

        signals_chan = queue.Queue()

        def collect_data(emg, _moving):
            nonlocal i
            if i == num_samples:
                signals_chan.put(np.copy(emg_win))
                i = 0
            emg_win[i] = emg
            i += 1

        myo.add_emg_handler(collect_data)
        myo.connect()

        self.signals_chan = signals_chan
        self.myo = myo

    def start_recording(self):
        """
        Start a thread for recording.
        """
        threading.Thread(target=self._record).start()

    def stop(self):
        """
        Terminate the recording thread.
        """
        self._stop = True

    def get(self):
        """
        Get data for the duratoin of the previously defined win_duration.
        """
        return self.signals_chan.get()

    def _record(self):
        while self._stop is not True:
            self.myo.run(1)
