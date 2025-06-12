# %%
from mne.io import concatenate_raws
from edf import read_raw_edf
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler


dir_path = r'd:/CBraMod-ISRUC/ISRUC/Subgroup_1'
seq_dir = r'd:/CBraMod-ISRUC/ISRUC/precessed_filter_35/seq'
label_dir = r'd:/CBraMod-ISRUC/ISRUC/precessed_filter_35/labels'

psg_f_names = []
label_f_names = []
for i in range(1, 101):
    numstr = str(i)
    psg_f_names.append(f'{dir_path}/{numstr}/{numstr}.rec')
    label_f_names.append(f'{dir_path}/{numstr}/{numstr}_1.txt')

# psg_f_names.sort()
# label_f_names.sort()
label2id = {'0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '5': 4,}
print(label2id)

# psg_label_f_pairs = []
# for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
#     if psg_f_name[:-4] == label_f_name[:-6]:
#         psg_label_f_pairs.append((psg_f_name, label_f_name))
# for item in psg_label_f_pairs:
#     print(item)

# Collect valid file pairs
psg_label_f_pairs = []
for i in range(1, 101):
    subj_dir = os.path.join(dir_path, str(i))
    psg_path = os.path.join(subj_dir, f'{i}.rec')
    label_path = os.path.join(subj_dir, f'{i}_1.txt')
    if os.path.exists(psg_path) and os.path.exists(label_path):
        psg_label_f_pairs.append((psg_path, label_path))
for item in psg_label_f_pairs:
    print(item)
# %%
# signal_name = ['LOC-A2', 'F4-A1']
n = 0
num_seqs = 0
num_labels = 0

for psg_f_name, label_f_name in tqdm(psg_label_f_pairs):
    n += 1
    labels_list = []

    # Load and preprocess EDF/REC file
    raw = read_raw_edf(psg_f_name, preload=True)
    raw.filter(0.3, 35, fir_design='firwin')
    raw.notch_filter(50)

    # Convert to numpy array
    psg_array = raw.to_data_frame().values
    psg_array = psg_array[:, 2:8]  # Select 6 EEG channels

    # Clip for full 30s epochs (assuming 200 Hz)
    i = psg_array.shape[0] % (30 * 200)
    if i > 0:
        psg_array = psg_array[:-i, :]
    psg_array = psg_array.reshape(-1, 30 * 200, 6)

    # Clip for 20-epoch sequences
    a = psg_array.shape[0] % 20
    if a > 0:
        psg_array = psg_array[:-a, :, :]
    psg_array = psg_array.reshape(-1, 20, 30 * 200, 6)
    epochs_seq = psg_array.transpose(0, 1, 3, 2)  # shape: [seqs, 20, 6, 6000]

    # Read labels
    with open(label_f_name) as f:
        for line in f:
            line_str = line.strip()
            if line_str != '':
                labels_list.append(label2id[line_str])
    labels_array = np.array(labels_list)
    if a > 0:
        labels_array = labels_array[:-a]
    labels_seq = labels_array.reshape(-1, 20)

    # Save sequences
    seq_subdir = os.path.join(seq_dir, f'ISRUC-group1-{n}')
    os.makedirs(seq_subdir, exist_ok=True)
    for seq in epochs_seq:
        seq_name = os.path.join(seq_subdir, f'ISRUC-group1-{n}-{num_seqs}.npy')
        np.save(seq_name, seq)
        num_seqs += 1

    # Save labels
    label_subdir = os.path.join(label_dir, f'ISRUC-group1-{n}')
    os.makedirs(label_subdir, exist_ok=True)
    for label in labels_seq:
        label_name = os.path.join(label_subdir, f'ISRUC-group1-{n}-{num_labels}.npy')
        np.save(label_name, label)
        num_labels += 1