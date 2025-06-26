import os
import pickle
import lmdb
import numpy as np
from tqdm import tqdm

def split_epoch_to_patches(epoch, patch_size=200):
    """
    Convert [6, 6000] epoch into [6, 30, 200] for CBraMod
    """
    ch, total_len = epoch.shape
    assert total_len % patch_size == 0, f"Epoch must divide evenly by patch_size ({patch_size})"
    return epoch.reshape(ch, total_len // patch_size, patch_size)

def convert_npy_to_lmdb(seq_root, lmdb_path):
    print(f"🔄 Converting [20, 6, 6000] .npy files from {seq_root} into LMDB at {lmdb_path}")

    # Collect file paths
    npy_paths = []
    for group_dir in os.listdir(seq_root):
        group_path = os.path.join(seq_root, group_dir)
        if os.path.isdir(group_path):
            for fname in os.listdir(group_path):
                if fname.endswith('.npy'):
                    npy_paths.append(os.path.join(group_path, fname))

    print(f"✅ Found {len(npy_paths)} .npy files.")
    map_size = 40 * 1024 * 1024 * 1024  # 40 GB

    env = lmdb.open(lmdb_path, map_size=map_size)
    keys = []
    idx = 0

    with env.begin(write=True) as txn:
        for path in tqdm(npy_paths, desc="Writing to LMDB"):
            data = np.load(path)  # shape: [20, 6, 6000]
            if data.ndim != 3 or data.shape[1:] != (6, 6000):
                raise ValueError(f"Invalid shape {data.shape} in file {path}")

            for i in range(data.shape[0]):  # 20 epochs per file
                epoch = data[i]  # shape: [6, 6000]
                patch = split_epoch_to_patches(epoch)  # → [6, 30, 200]
                key = f"patch-{idx:06d}"
                txn.put(key.encode(), pickle.dumps(patch))
                keys.append(key)
                idx += 1

        txn.put('__keys__'.encode(), pickle.dumps(keys))

    print(f"🎉 LMDB complete: {idx} entries saved")

if __name__ == "__main__":
    seq_root = r"D:\projects\CBraMod\CBraMod-ISRUC\ISRUC\precessed_filter_35\seq"
    lmdb_path = r"D:\projects\CBraMod\CBraMod-ISRUC\dataset\pretrain_lmdb"

    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    convert_npy_to_lmdb(seq_root, lmdb_path)
