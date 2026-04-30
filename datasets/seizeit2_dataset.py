import torch
from torch.utils.data import Dataset
import mne
import os
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

class SeizeIT2Dataset(Dataset):
    def __init__(self, data_dir, mode='train', preictal_min=30, channels=2, max_subjects=20):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.preictal_min = preictal_min * 60   # convert minutes to seconds
        self.channels = channels
        self.max_subjects = max_subjects
        self.samples = []
        self.labels = []
        self.subject_ids = []
        self._load_data()

    def _load_data(self):
        subjects = [d for d in os.listdir(self.data_dir) if d.startswith('sub-')]
        random.seed(42)
        random.shuffle(subjects)
        subjects = subjects[:self.max_subjects]

        preictal_list = []
        interictal_list = []
        pre_subject_list = []
        inter_subject_list = []

        for sub_idx, sub in enumerate(tqdm(subjects, desc=f"Loading {self.mode} (preictal)")):
            eeg_dir = os.path.join(self.data_dir, sub, 'eeg')
            if not os.path.exists(eeg_dir):
                continue

            edf_files = [f for f in os.listdir(eeg_dir) if f.endswith('_eeg.edf')]
            for edf_file in edf_files:
                edf_path = os.path.join(eeg_dir, edf_file)
                events_path = edf_path.replace('_eeg.edf', '_events.tsv')
                try:
                    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                    raw.pick_channels(raw.ch_names[:self.channels])
                    raw.filter(l_freq=0.5, h_freq=40.0, fir_design='firwin', verbose=False)
                    data = raw.get_data()
                    sfreq = int(raw.info['sfreq'])
                    if sfreq != 256:
                        continue

                    # parse seizure onsets
                    onset_seconds = []
                    if os.path.exists(events_path):
                        events = pd.read_csv(events_path, sep='\t')
                        if 'onset' in events.columns:
                            onset_seconds = events['onset'].values
                        elif 'sample' in events.columns:
                            onset_seconds = events['sample'].values

                    window_size = 256 * 30      # 30 seconds
                    step = 256 * 10             # 10 seconds step
                    for start in range(0, data.shape[1] - window_size + 1, step):
                        segment = data[:, start:start + window_size]
                        if segment.shape[1] < window_size:
                            continue

                        # resample to 200 Hz (linear interpolation)
                        t_old = np.linspace(0, 1, window_size)
                        t_new = np.linspace(0, 1, 6000)       # 30*200
                        segment = np.array([np.interp(t_new, t_old, ch) for ch in segment])
                        segment = segment.reshape(self.channels, 30, 200)

                        # per‑channel per‑window z‑score
                        segment = (segment - segment.mean(axis=(1,2), keepdims=True)) / \
                                  (segment.std(axis=(1,2), keepdims=True) + 1e-8)

                        # determine label
                        window_center_sec = (start + window_size // 2) / sfreq
                        is_preictal = any(
                            onset_sec - self.preictal_min <= window_center_sec < onset_sec
                            for onset_sec in onset_seconds
                        )

                        if is_preictal:
                            preictal_list.append(segment)
                            pre_subject_list.append(sub_idx)
                        else:
                            interictal_list.append(segment)
                            inter_subject_list.append(sub_idx)

                except Exception as e:
                    continue

        # Balance to 1:1 by random undersampling of interictal samples
        target_count = len(preictal_list)
        if len(interictal_list) > target_count:
            random.seed(42)
            idx = random.sample(range(len(interictal_list)), target_count)
            interictal_list = [interictal_list[i] for i in idx]
            inter_subject_list = [inter_subject_list[i] for i in idx]

        self.samples = preictal_list + interictal_list
        self.labels = [1] * len(preictal_list) + [0] * len(interictal_list)
        self.subject_ids = pre_subject_list + inter_subject_list

        # shuffle
        random.seed(42)
        idx = list(range(len(self.samples)))
        random.shuffle(idx)
        self.samples = [self.samples[i] for i in idx]
        self.labels = [self.labels[i] for i in idx]
        self.subject_ids = [self.subject_ids[i] for i in idx]

        print(f"Dataset loaded: {len(self.samples)} samples "
              f"(preictal={self.labels.count(1)}, interictal={self.labels.count(0)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long),
                torch.tensor(self.subject_ids[idx], dtype=torch.long))

    @staticmethod
    def collate(batch):
        inputs, labels, subject_ids = zip(*batch)
        return torch.stack(inputs), torch.tensor(labels), torch.tensor(subject_ids)