import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
import scipy.signal
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle


class TroikaPredictor():
    """
    A class that contains our algorithm
    """
    reg = None
    log = None
    def __init__(self, pickle_path,
                 fs=125, pass_band=(40/60.0, 240/60.0), multiplier=4,
                 log_vars=False):
        """
        Initialize an instance of TroikaPredictor class
        """
        self.pickle_path = pickle_path
        self.reg = load_regressor(pickle_path)

        self.fs = fs
        self.pass_band = pass_band
        self.multiplier = multiplier

        self.log_vars = log_vars
        if self.log_vars:
            self.log = Log()

    def predict(self, ppg1, ppg2, accx, accy, accz):
        """
        Predict heart rate (in BPM) and confidence rate
        Args:
            ppg1 (numpy.array): PPG signals channel 1
            ppg2 (numpy.array): PPG signals channel 2
            accx (numpy.array): IMU signals axis x
            accy (numpy.array): IMU signals axis y
            accz (numpy.array): IMU signals axis z
        Returns:
            Heart Rate (float)
            Confidence Rate (float)
        """
        fs = self.fs
        pass_band = self.pass_band
        multiplier = self.multiplier
        feature = featurize(ppg1, ppg2, accx, accy, accz,
                            fs, pass_band, multiplier)

        est = self.reg.predict(np.reshape(feature, (1, -1)))[0]

        # Confidence Calculation
        # ----------------------
        ppg = np.mean(np.vstack([ppg1, ppg2]), axis=0)

        if self.log_vars:
            self.log.prepare(ppg1=ppg1, ppg2=ppg2, ppg=ppg,
                             accx=accx, accy=accy, accz=accz)
            
        # Bandpass Filter
        ppg = bandpass_filter(ppg, pass_band, fs)
        ppg1 = bandpass_filter(ppg1, pass_band, fs)
        ppg2 = bandpass_filter(ppg2, pass_band, fs)
        accx = bandpass_filter(accx, pass_band, fs)
        accy = bandpass_filter(accy, pass_band, fs)
        accz = bandpass_filter(accz, pass_band, fs)

        if self.log_vars:
            self.log.prepare(ppg1_bp=ppg1, ppg2_bp=ppg2, ppg_bp=ppg,
                             accx_bp=accx, accy_bp=accy, accz_bp=accz)

        n = len(ppg) * multiplier
        freqs = np.fft.rfftfreq(n, 1/fs)
        fft = np.abs(np.fft.rfft(ppg, n))
        fft[freqs <= pass_band[0]] = 0.0
        fft[freqs >= pass_band[1]] = 0.0
        fs_window = 5 / 60.0
        est_fs = est / 60.0
        est_fs_window = (freqs >= est_fs - fs_window) & (freqs <= est_fs + fs_window)
        conf = np.sum(fft[est_fs_window]) / np.sum(fft)

        # END - Confidence Calculation
        # ----------------------

        return (est, conf)


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


def featurize(ppg1, ppg2, accx, accy, accz,
              fs=125, pass_band=(40/60.0, 240/60.0), multiplier=4):
    """
    Create features based on some inputs.
    
    Args:
        ppg1 (numpy.array) PPG signals channel 1
        ppg2 (numpy.array) PPG signals channel 2
        accx (numpy.array) IMU signals axis x
        accy (numpy.array) IMU signals axis y
        accz (numpy.array) IMU signals axis z
        fs: (int) Sampling frequency in Hz
        pass_band: ((float, float)) min and max pass bands tuple
        multiplier: (int) The number of frequencies should be multiplied by this

    Returns:
        List of features
    """

    ppg1 = bandpass_filter(ppg1, pass_band, fs)
    ppg2 = bandpass_filter(ppg2, pass_band, fs)
    accx = bandpass_filter(accx, pass_band, fs)
    accy = bandpass_filter(accy, pass_band, fs)
    accz = bandpass_filter(accz, pass_band, fs)

    ppg = np.mean(np.vstack([ppg1, ppg2]), axis=0)

    n = len(ppg) * multiplier
    freqs = np.fft.rfftfreq(n, 1/fs)

    # Get PPG power spectrums
    fft = np.abs(np.fft.rfft(ppg, n))
    fft[freqs <= pass_band[0]] = 0.0
    fft[freqs >= pass_band[1]] = 0.0

    # Get L2-norms of accelerations
    acc_l2 = np.sqrt(accx ** 2 + accy ** 2 + accz ** 2)

    # Get acceleration power spectrums
    acc_fft = np.abs(np.fft.rfft(acc_l2, n))
    acc_fft[freqs <= pass_band[0]] = 0.0
    acc_fft[freqs >= pass_band[1]] = 0.0

    # Get max magnitude's frequency as one of the features
    ppg_fs = freqs[np.argmax(fft)]

    # Get max magnitude's acc_l2 as one of the features
    acc_fs = freqs[np.argmax(acc_fft)]

    return [ppg_fs, acc_fs]


def error_score(y_test, y_pred):
    """
    Calculate error score of a prediction
    """
    return mean_squared_error(y_test, y_pred)


def prepare_regressor(features, targets, subjects, print_log=True, pickle_path=None):
    """ Prepare a regression model.

    Returns:
        regression object (sklearn.model)
        error scores during validation (numpy.array))
    """
    features = np.array(features)
    targets = np.array(targets)
    # reg = RandomForestRegressor(n_estimators=150,
    #                             max_depth=2,
    #                             random_state=42)
    reg = LinearRegression()

    logo = LeaveOneGroupOut()
    val_scores = []
    splits = logo.split(features, targets, subjects)
    for i, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = features[train_idx], targets[train_idx]
        X_test, y_test = features[test_idx], targets[test_idx]
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        score = error_score(y_test, y_pred)
        val_scores.append(score)

        if print_log:
            print("Iter {} score: {}".format(i, score))

    if pickle_path is not None:
        reg_pkl = open(pickle_path, 'wb')
        pickle.dump(reg, reg_pkl)
        print("Regressor stored at {}".format(pickle_path))


    return reg, val_scores


def load_regressor(pickle_path):
    """
    Load a regressor (pickle file)

    Args:
        pickle_path: (string) Path to the pickle file
    """
    file = open(pickle_path, 'rb')
    return pickle.load(file)


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