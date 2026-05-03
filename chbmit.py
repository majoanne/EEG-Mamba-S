import argparse
import torch
import torch.nn as nn
import os
import mne
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def parse_chbmit_summary(summary_path):
    seizure_info = {}
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, 'r') as f:
        content = f.read()
        for block in content.split("File Name: ")[1:]:
            lines = block.split('\n')
            edf_name = lines[0].strip()
            num_seizures = 0
            for line in lines:
                if "Number of Seizures in File:" in line:
                    num_seizures = int(line.split(":")[1].strip())
            seizure_info[edf_name] = []
            if num_seizures > 0:
                for i in range(1, num_seizures+1):
                    s_line = [l for l in lines if f"Seizure {i} Start Time" in l or ("Seizure Start Time" in l and i==1)]
                    e_line = [l for l in lines if f"Seizure {i} End Time" in l or ("Seizure End Time" in l and i==1)]
                    if s_line and e_line:
                        s = int(re.findall(r'\d+', s_line[0].split(":")[1])[0])
                        e = int(re.findall(r'\d+', e_line[0].split(":")[1])[0])
                        seizure_info[edf_name].append((s, e))
    return seizure_info

def get_clean_2ch_data(raw):
    mne.set_log_level('ERROR')
    target_ch = ['T7-P7', 'T8-P8']
    raw.load_data().filter(l_freq=0.5, h_freq=75, verbose=False)
    raw.notch_filter(freqs=50, verbose=False)
    all_chs = raw.ch_names
    picked = []
    for tc in target_ch:
        found = [c for c in all_chs if tc.replace('-', '').upper() in c.replace('-', '').upper()]
        picked.append(found[0] if found else all_chs[0])
    raw.pick(picked)
    data = raw.get_data()
    for i in range(data.shape[0]):
        data[i] = (data[i] - np.mean(data[i])) / (np.std(data[i]) + 1e-6)
    return data

class CHBMITFullDataset(Dataset):
    def __init__(self, data_dir, patient_list, preictal_min=30, neg_ratio=0.1):
        super().__init__()
        self.samples, self.labels = [], []
        for patient in tqdm(patient_list, desc="Building CHB-MIT"):
            p_dir = os.path.join(data_dir, patient)
            summary_path = os.path.join(p_dir, f"{patient}-summary.txt")
            summary = parse_chbmit_summary(summary_path)
            for edf_name, seizures in summary.items():
                path = os.path.join(p_dir, edf_name)
                if not os.path.exists(path):
                    continue
                try:
                    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
                    data = get_clean_2ch_data(raw)
                    sfreq = int(raw.info['sfreq'])
                    win = sfreq * 30
                    if seizures:
                        for (s, e) in seizures:
                            p_start, p_end = max(0, s - preictal_min*60), s
                            for start in range(int(p_start*sfreq), int(p_end*sfreq)-win, sfreq*5):
                                seg = data[:, start:start+win]
                                if sfreq != 200:
                                    t_new = np.linspace(0, 1, 6000)
                                    seg = np.array([np.interp(t_new, np.linspace(0, 1, win), ch) for ch in seg])
                                self.samples.append(seg.reshape(2, 30, 200))
                                self.labels.append(1)
                    else:
                        for start in range(0, data.shape[1]-win, sfreq*30):
                            if np.random.rand() < neg_ratio:
                                seg = data[:, start:start+win]
                                if sfreq != 200:
                                    t_new = np.linspace(0, 1, 6000)
                                    seg = np.array([np.interp(t_new, np.linspace(0, 1, win), ch) for ch in seg])
                                self.samples.append(seg.reshape(2, 30, 200))
                                self.labels.append(0)
                except:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

class Trainer:
    def __init__(self, params, train_loader, test_loader, model):
        self.params = params
        self.model = model.cuda(params.cuda)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = FocalLoss(alpha=0.5, gamma=3.0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=params.lr, weight_decay=5e-2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.epochs)

    def train(self):
        final_auc = 0.0
        for epoch in range(self.params.epochs):
            self.model.train()
            total_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.cuda(self.params.cuda), labels.cuda(self.params.cuda)
                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
            current_auc = self.evaluate()
            print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(self.train_loader):.4f} | Test AUC: {current_auc:.4f}")
            final_auc = current_auc
        return final_auc

    def evaluate(self):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                logits = self.model(inputs.cuda(self.params.cuda))
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                trues.extend(targets.numpy())
        return roc_auc_score(trues, preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--chbmit_dir', type=str, default="/root/autodl-tmp/chb-mit_data")
    parser.add_argument('--weights_path', type=str, default="")
    args = parser.parse_args()

    all_pts = [f"chb{i:02d}" for i in range(1, 21)]
    full_dataset = CHBMITFullDataset(args.chbmit_dir, all_pts, preictal_min=30)

    seeds = [3407, 42, 1234]
    results = []

    for seed in seeds:
        torch.manual_seed(seed)
        print(f"\nCHB-MIT experiment | Seed: {seed}")
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_ds, test_ds = random_split(full_dataset, [train_size, test_size],
                                         generator=torch.Generator().manual_seed(seed))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        from models.model_for_seizure import Model
        model = Model(argparse.Namespace(dropout=0.1, channels=2))
        print("Random initialization (no pretrained weights).")

        trainer = Trainer(args, train_loader, test_loader, model)
        final_auc = trainer.train()
        results.append(final_auc)

    print(f"\n{'='*30}\nFinal Results: {np.mean(results):.4f} ± {np.std(results):.4f}\n{'='*30}")

if __name__ == "__main__":
    main()