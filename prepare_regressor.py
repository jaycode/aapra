from lib import troika
import scipy.io
from tqdm import tqdm
import os

FS = 125
WINDOW_LEN_S = 8
WINDOW_SHIFT_S = 2
PASS_BAND = (40/60.0, 240/60.0)
MULTIPLIER = 4

def load_data(path, fs=125, window_len_s=8,
              window_shift_s=2, pass_band=(40/60.0, 240/60.0),
              multiplier=4):
    """
    Loads Troika data

    Args:
        path: (string) Path to data folder
        fs: (int) Sampling frequency
        window_len_s: (int) Window length in seconds
        window_shift_s: (int) Overlap between each window
        pass_band: ((float, float)) min and max pass bands tuple
        multiplier: (int) The number of frequencies should be multiplied by this.
    """
    data_fls, ref_fls = troika.LoadTroikaDataset(path)
    
    pbar = tqdm(list(zip(data_fls, ref_fls)), desc="Data Preparation")

    targets, subjects, features = [], [], []

    for data_fl, ref_fl in pbar:
        sigs = scipy.io.loadmat(data_fl)['sig']
        refs = scipy.io.loadmat(ref_fl)['BPM0'].reshape(-1)
        subject_name = os.path.basename(data_fl).split('.')[0]

        # Bandpass Filter
        

        start_idxs, end_idxs = troika.get_idxs(sigs.shape[1], len(refs),
                                   fs=fs,
                                   window_len_s=window_len_s,
                                   window_shift_s=window_shift_s)
        for i, start_idx in enumerate(start_idxs):
            end_idx = end_idxs[i]

            # ECG-related
            ecg = sigs[0, start_idx:end_idx]
            ref = refs[i]

            # Extract features
            ppg = sigs[0, start_idx: end_idx]
            accx = sigs[3, start_idx:end_idx]
            accy = sigs[4, start_idx:end_idx]
            accz = sigs[5, start_idx:end_idx]

            feature = troika.featurize(ppg, accx, accy, accz,
                                       fs, pass_band, multiplier)
            
            targets.append(ref)
            subjects.append(subject_name)
            features.append(feature)

    return (targets, subjects, features)


if __name__ == "__main__":
    targets, subjects, features = \
        load_data(
            "./part_1/datasets/troika/training_data",
            fs=FS,
            window_len_s=WINDOW_LEN_S,
            window_shift_s=WINDOW_SHIFT_S,
            pass_band=PASS_BAND,
            multiplier=MULTIPLIER)

    reg, val_scores = troika.prepare_regressor(features, targets, subjects,
                                               print_log=True,
                                               pickle_path='reg.pkl')
