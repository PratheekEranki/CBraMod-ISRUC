import numpy as np
import matplotlib.pyplot as plt

# 🔁 Load one EEG and label pair
seq_path = r'D:\projects\CBraMod\CBraMod-ISRUC\ISRUC\precessed_filter_35\seq\ISRUC-group1-1\ISRUC-group1-1-0.npy'
label_path = r'D:\projects\CBraMod\CBraMod-ISRUC\ISRUC\precessed_filter_35\labels\ISRUC-group1-1\ISRUC-group1-1-0.npy'

eeg = np.load(seq_path)     # shape: (20, 6, 6000)
labels = np.load(label_path)  # shape: (20,)

# Label mapping 
id2label = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}

# 📊 Plot first 3 epochs 
epochs_to_plot = 20
channel_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']

for i in range(epochs_to_plot):
    fig, axs = plt.subplots(6, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Epoch {i} - Sleep Stage: {id2label[labels[i]]}", fontsize=14)

    for ch in range(6):
        axs[ch].plot(eeg[i][ch])
        axs[ch].set_ylabel(channel_names[ch])
        axs[ch].grid(True)

    axs[-1].set_xlabel("Time (samples @ 200 Hz)")
    plt.tight_layout()
    plt.show()
