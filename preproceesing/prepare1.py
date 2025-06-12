import os
import numpy as np
from tqdm import tqdm
from edf import read_raw_edf  
import mne

# Set the correct root path
dir_path = r'D:\projects\CBraMod\CBraMod-ISRUC\ISRUC\Subgroup_1'

# Output directories
seq_dir = r'D:\projects\CBraMod\CBraMod-ISRUC\ISRUC\precessed_filter_35\seq'
label_dir = r'D:\projects\CBraMod\CBraMod-ISRUC\ISRUC\precessed_filter_35\labels'




# Label dictionary
label2id = {'0': 0, '1': 1, '2': 2, '3': 3, '5': 4}
print(label2id)

# Collect matching file pairs
psg_label_f_pairs = []
for i in range(1, 101):
    subj_dir = os.path.join(dir_path, str(i))
    psg_file = os.path.normpath(os.path.join(subj_dir, f'{i}.rec'))
    label_file = os.path.normpath(os.path.join(subj_dir, f'{i}_1.txt'))

    if os.path.exists(psg_file) and os.path.exists(label_file):
        psg_label_f_pairs.append((psg_file, label_file))
    else:
        print(f"[SKIP] Missing files for subject {i}:")
        if not os.path.exists(psg_file):
            print(f"    Missing PSG: {psg_file}")
        if not os.path.exists(label_file):
            print(f"    Missing Label: {label_file}")


print(f"✅ Found {len(psg_label_f_pairs)} valid PSG-label file pairs.")

# Process files
n = 0
num_seqs = 0
num_labels = 0

for psg_f_name, label_f_name in tqdm(psg_label_f_pairs, desc="Processing subjects"):
    n += 1
    labels_list = []

    try:
        raw = read_raw_edf(psg_f_name, preload=True)
        raw.filter(0.3, 35, fir_design='firwin')
        raw.notch_filter(50)

        psg_array = raw.to_data_frame().values
        psg_array = psg_array[:, 2:8]  # Select channels 2 to 7 (6 channels)

        # Crop to full 30s epochs
        i = psg_array.shape[0] % (30 * 200)
        if i > 0:
            psg_array = psg_array[:-i, :]
        psg_array = psg_array.reshape(-1, 30 * 200, 6)

        # Crop to full 20-epoch sequences
        a = psg_array.shape[0] % 20
        if a > 0:
            psg_array = psg_array[:-a, :, :]
        psg_array = psg_array.reshape(-1, 20, 30 * 200, 6)
        epochs_seq = psg_array.transpose(0, 1, 3, 2)  # [N, 20, 6, 6000]

        # Read label file
        with open(label_f_name) as f:
            for line in f:
                line_str = line.strip()
                if line_str in label2id:
                    labels_list.append(label2id[line_str])
        labels_array = np.array(labels_list)
        if a > 0:
            labels_array = labels_array[:-a]
        labels_seq = labels_array.reshape(-1, 20)

        # Save sequences
        seq_subdir = os.path.join(seq_dir, f'ISRUC-group1-{n}')
        os.makedirs(seq_subdir, exist_ok=True)
        for seq in epochs_seq:
            seq_path = os.path.join(seq_subdir, f'ISRUC-group1-{n}-{num_seqs}.npy')
            np.save(seq_path, seq)
            num_seqs += 1

        # Save labels
        label_subdir = os.path.join(label_dir, f'ISRUC-group1-{n}')
        os.makedirs(label_subdir, exist_ok=True)
        for label in labels_seq:
            label_path = os.path.join(label_subdir, f'ISRUC-group1-{n}-{num_labels}.npy')
            np.save(label_path, label)
            num_labels += 1

    except Exception as e:
        print(f"     Error processing {psg_f_name}: {e}")
