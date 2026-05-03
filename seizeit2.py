import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from datasets.seizeit2_dataset import SeizeIT2Dataset
from models.model_for_seizure import Model
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class Trainer:
    def __init__(self, params, data_loader, model):
        self.params = params
        self.model = model.cuda(params.cuda)
        self.train_loader = data_loader['train']
        self.test_loader = data_loader['test']
        self.criterion = FocalLoss(alpha=0.5, gamma=3.0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=params.lr, weight_decay=5e-2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.epochs)
        self.model_dir = params.model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self):
        final_auc = 0.0
        for epoch in range(self.params.epochs):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.params.epochs} (seed={self.params.seed})")
            for inputs, labels, _ in pbar:
                inputs = inputs.cuda(self.params.cuda)
                labels = labels.float().cuda(self.params.cuda)
                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            self.scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
            torch.save(self.model.state_dict(), os.path.join(self.model_dir, f"epoch_{epoch+1}.pth"))
            current_auc = self.evaluate()
            if epoch == self.params.epochs - 1:
                final_auc = current_auc
        print(f"Seed {self.params.seed} finished. Final ROC-AUC: {final_auc:.4f}")
        return final_auc

    def evaluate(self):
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets, _ in self.test_loader:
                inputs = inputs.cuda(self.params.cuda)
                logits = self.model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(targets.cpu().numpy())
        auc = roc_auc_score(all_targets, all_preds)
        pr_auc = average_precision_score(all_targets, all_preds)
        print(f"ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")
        return auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--datasets_dir', type=str, default="/root/seizeit2_data/1")
    args = parser.parse_args()

    seeds = [3407, 42, 1234]
    auc_list = []

    for seed in seeds:
        torch.manual_seed(seed)
        print(f"\n{'='*100}\nRunning experiment | Seed = {seed}\n{'='*100}")

        full_dataset = SeizeIT2Dataset(
            data_dir=args.datasets_dir,
            mode='train',
            channels=2,
            preictal_min=30,
            max_subjects=20
        )

        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(
            full_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        data_loader = {
            'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                collate_fn=full_dataset.collate, num_workers=4, drop_last=True),
            'test': DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                               collate_fn=full_dataset.collate, num_workers=4)
        }

        model_params = argparse.Namespace(dropout=0.1, channels=2)
        model = Model(model_params)

        run_params = argparse.Namespace(
            epochs=args.epochs,
            lr=args.lr,
            cuda=args.cuda,
            seed=seed,
            model_dir=f"./multi_seed_best_seed_{seed}"
        )

        trainer = Trainer(run_params, data_loader, model)
        final_auc = trainer.train()
        auc_list.append(final_auc)

    print(f"\n{'='*100}\nAll seeds finished. ROC-AUC values: {[round(x, 4) for x in auc_list]}")
    print(f"Mean ± std: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}\n{'='*100}")

if __name__ == "__main__":
    main()