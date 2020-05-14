"""
This is the module to perform operations needed to measure heart rate from the TROIKA dataset.

Steps to run the algorithm:

1. Run the script `prepare_regressor.py`. This script will produce a regression model in the form of a [Pickle](https://docs.python.org/3/library/pickle.html) object that we can load and use to perform predictions on new data.
2. Run the code cell below to check the performance of the algorithm on the training data.
3. Testing data are available in the directory `datasets/troika/testing_data` if you'd like to see the performance of the model on new data.
"""

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
import scipy.signal
import glob
import pickle


class Log():
    """
    A class to help us log important variables.

    `prepare()` function to set temporary variables,
    then commit to storing the reference by the `commit()` function. 
    """
    temp = {}
    
    def __init__(self):
        pass
    
    def prepare(self, **kwargs):
        """
        Prepare a variable

        Prepared variable is stored in the temporary dictionary.
        """
        for key in kwargs:
            self.temp[key] = kwargs[key]
    
    def commit(self):
        """
        Attach the variables in the temporary dictionary into the class instance.
        """
        for key in self.temp: 
            setattr(self, key, self.temp[key])


def LoadTroikaDataset(path):
    """
    Retrieve the .mat filenames for the troika dataset.

    Review the README in ./datasets/troika/ to understand the organization of the .mat files.

    Returns:
        data_fls: Names of the .mat files that contain signal data
        ref_fls: Names of the .mat files that contain reference data
        <data_fls> and <ref_fls> are ordered correspondingly, so that ref_fls[5] is the 
            reference data for data_fls[5], etc...
    """
    data_dir = path
    data_fls = sorted(glob.glob(data_dir + "/DATA_*.mat"))
    ref_fls = sorted(glob.glob(data_dir + "/REF_*.mat"))
    return data_fls, ref_fls


def get_idxs(n_sigs, n_targets, fs=125, window_len_s=8, window_shift_s=2):
    """
    Get start and end ids to be used to iterate over a set of signals
    Args:
        n_sigs: Number of signals
        n_targets: Number of targets
        fs: Sampling frequency in Hz
        window_len_s: Window length in seconds
        window_shift_s: Overlap between each window
    """
    n = n_sigs
    if n_targets < n:
        n = n_targets
    start_idxs = (np.cumsum(np.ones(n) * fs * window_shift_s) - fs * window_shift_s).astype(int)
    end_idxs = start_idxs + window_len_s * fs
    return (start_idxs, end_idxs)


def bandpass_filter(signal, pass_band=(40/60.0, 240/60.0), fs=125):
    """
    Bandpass Filter
    
    Args:
        signal: (np.array) The input signal
        pass_band: (tuple) The pass band. Frequency components outside 
            the two elements in the tuple will be removed.
        fs: (number) The sampling rate of <signal>
        
    Returns:
        (np.array) The filtered signal
    """
    b, a = scipy.signal.butter(3, pass_band, btype='bandpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)