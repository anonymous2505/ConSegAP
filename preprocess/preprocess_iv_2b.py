import mne
import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat

# An example to get the BCI competition IV datasets 2b, only for reference
# Data from: http://www.bbci.de/competition/iv/
# using open-source MNE-Python: https://mne.tools/stable/index.html
# Just an example, you should change as you need.

# get processed T data
def process(subject_index, id ,session_type='T'):
    # BioSig Get the data
    # T data
    # subject_index = 1; #1-9
    dir = f"./data/raw/BCICIV_2b_gdf/B0{subject_index}0{id}{session_type}.gdf"   
    raw = mne.io.read_raw_gdf(dir, preload=True)
    s = raw.get_data(picks=['EEG:C3', 'EEG:Cz', 'EEG:C4']).T  # Shape: (samples, 3)

    # Label
    labeldir = f"./data/raw/true_labels/B0{subject_index}0{id}{session_type}.mat"
    label = loadmat(labeldir)['classlabel'].flatten()

    # construct sample - data Section 1000*3*120
    annotations = raw.annotations
    Typ = annotations.description.astype(str)  # Event types
    Pos = (annotations.onset * raw.info['sfreq']).astype(int)  # Event positions in samples
    event_mask = Typ == '768'  # Filter for Typ == 768
    Pos = Pos[event_mask]
    Typ = Typ[event_mask]

    k = 0
    data_1 = np.zeros((1000, 3, len(Typ)))
    for j in range(len(Typ)):
        if Typ[j] == "768":
            k += 1
            data_1[:, :, k-1] = s[Pos[j-1]+750:Pos[j-1]+1750, :]  # 1000 samples

    # wipe off NaN
    data_1[np.isnan(data_1)] = 0

    data = data_1
    pindex = np.random.permutation(120)
    data = data[:, :, pindex]
    label = label[pindex]

    # 4-40 Hz
    fc = 250
    fb_data = np.zeros((1000, 3, 120))

    Wl = 4
    Wh = 40  
    Wn = [Wl*2/fc, Wh*2/fc]
    b, a = signal.cheby2(6, 60, Wn, btype='bandpass')
    for j in range(120):
        fb_data[:, :, j] = signal.filtfilt(b, a, data[:, :, j], axis=0)

    eeg_mean = np.mean(fb_data, axis=2)
    eeg_std = np.std(fb_data, axis=2)
    fb_data = (fb_data - eeg_mean[:, :, np.newaxis]) / eeg_std[:, :, np.newaxis]

    data = fb_data

    saveDir = f"./data/bci_iv_2b/B0{subject_index}0{id}{session_type}.mat"
    savemat(saveDir, {'data': data, 'label': label})

    return data


for i in range(1, 10):
    for j in range(1, 6):
        if j <= 3:
            process(i, j, 'T')
        else:
            process(i, j, 'E')
