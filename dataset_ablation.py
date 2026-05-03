import torch
from torch.utils.data import Dataset
import mne
import os
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

class SeizeIT2DatasetAblation(Dataset):
    def __init__(self, data_dir, mode='train', preictal_min=30, channels=2,
                 max_subjects=20,
                 norm_type='window',
                 low_freq=0.5, high_freq=40.0,
                 window_sec=30, step_sec=10,
                 target_sr=200,
                 balance=True,
                 seed=42,
                 **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.preictal_min = preictal_min * 60
        self.channels = channels
        self.max_subjects = max_subjects
        self.norm_type = norm_type
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.target_sr = target_sr
        self.balance = balance
        self.seed = seed
        self.samples = []
        self.labels = []
        self.subject_ids = []
        self._load_data()

    def _load_data(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

        subjects = [d for d in os.listdir(self.data_dir) if d.startswith('sub-')]
        random.shuffle(subjects)
        subjects = subjects[:self.max_subjects]

        preictal_list, interictal_list = [], []
        pre_subject_list, inter_subject_list = [], []

        for sub_idx, sub in enumerate(tqdm(subjects, desc=f"Loading {self.mode}")):
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
                    raw.filter(l_freq=self.low_freq, h_freq=self.high_freq,
                               fir_design='firwin', verbose=False)
                    data = raw.get_data()
                    sfreq = int(raw.info['sfreq'])
                    if sfreq != 256:
                        continue

                    onset_seconds = []
                    if os.path.exists(events_path):
                        events = pd.read_csv(events_path, sep='\t')
                        if 'onset' in events.columns:
                            onset_seconds = events['onset'].values
                        elif 'sample' in events.columns:
                            onset_seconds = events['sample'].values

                    window_size_raw = sfreq * self.window_sec
                    step_raw = sfreq * self.step_sec

                    for start in range(0, data.shape[1] - window_size_raw + 1, step_raw):
                        segment = data[:, start:start + window_size_raw]
                        if segment.shape[1] < window_size_raw:
                            continue

                        target_len = self.window_sec * self.target_sr
                        t_old = np.linspace(0, 1, window_size_raw)
                        t_new = np.linspace(0, 1, target_len)
                        segment = np.array([np.interp(t_new, t_old, ch) for ch in segment])
                        segment = segment.reshape(self.channels, self.window_sec, self.target_sr)

                        if self.norm_type == 'window':
                            mean = segment.mean(axis=(1, 2), keepdims=True)
                            std = segment.std(axis=(1, 2), keepdims=True)
                            segment = (segment - mean) / (std + 1e-8)
                        elif self.norm_type == 'global':
                            data_rs = np.array([np.interp(np.linspace(0, 1, int(data.shape[1]*(self.target_sr/sfreq))),
                                                         np.linspace(0, 1, data.shape[1]), ch) for ch in data])
                            mean = data_rs.mean(axis=1, keepdims=True)
                            std = data_rs.std(axis=1, keepdims=True)
                            segment = (segment - mean[:, np.newaxis, np.newaxis]) / (std[:, np.newaxis, np.newaxis] + 1e-8)
                        elif self.norm_type == 'none':
                            pass
                        else:
                            raise ValueError(f"Unknown norm_type: {self.norm_type}")

                        window_center_sec = (start + window_size_raw // 2) / sfreq
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

                except Exception:
                    continue

        if self.balance:
            target_count = len(preictal_list)
            if len(interictal_list) > target_count:
                random.seed(self.seed)
                idx = random.sample(range(len(interictal_list)), target_count)
                interictal_list = [interictal_list[i] for i in idx]
                inter_subject_list = [inter_subject_list[i] for i in idx]

        self.samples = preictal_list + interictal_list
        self.labels = [1] * len(preictal_list) + [0] * len(interictal_list)
        self.subject_ids = pre_subject_list + inter_subject_list

        random.seed(self.seed)
        idx = list(range(len(self.samples)))
        random.shuffle(idx)
        self.samples = [self.samples[i] for i in idx]
        self.labels = [self.labels[i] for i in idx]
        self.subject_ids = [self.subject_ids[i] for i in idx]

        print(f"Dataset: {len(self.samples)} samples (preictal={sum(self.labels)}, "
              f"interictal={len(self.labels)-sum(self.labels)})")

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