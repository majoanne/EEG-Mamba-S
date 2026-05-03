import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset_ablation import SeizeIT2DatasetAblation
from model_ablation import FlexibleModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def get_criterion(loss_type, alpha, gamma):
    if loss_type == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

class Trainer:
    def __init__(self, args, train_loader, test_loader, model):
        self.args = args
        self.model = model.cuda(args.cuda)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = get_criterion(args.loss_type, args.alpha, args.gamma)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=5e-2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.model_dir = args.model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self):
        final_auc = 0.0
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            for inputs, labels, _ in pbar:
                inputs = inputs.cuda(self.args.cuda)
                labels = labels.float().cuda(self.args.cuda)
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
            current_auc = self.evaluate()
            if epoch == self.args.epochs - 1:
                final_auc = current_auc
        print(f"Final ROC-AUC: {final_auc:.4f}")
        return final_auc

    def evaluate(self):
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets, _ in self.test_loader:
                inputs = inputs.cuda(self.args.cuda)
                logits = self.model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(targets.cpu().numpy())
        auc = roc_auc_score(all_targets, all_preds)
        pr_auc = average_precision_score(all_targets, all_preds)
        print(f"ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")
        return auc

def main():
    parser = argparse.ArgumentParser(description="EEG-Mamba-S Ablation Study")
    parser.add_argument('--data_dir', type=str, default="/root/seizeit2_data/1")
    parser.add_argument('--max_subjects', type=int, default=20)
    parser.add_argument('--norm_type', type=str, default='window', choices=['window', 'global', 'none'])
    parser.add_argument('--low_freq', type=float, default=0.5)
    parser.add_argument('--high_freq', type=float, default=40.0)
    parser.add_argument('--window_sec', type=int, default=30)
    parser.add_argument('--step_sec', type=int, default=10)
    parser.add_argument('--target_sr', type=int, default=200)
    parser.add_argument('--balance', action='store_true', default=True)
    parser.add_argument('--no_balance', dest='balance', action='store_false')

    parser.add_argument('--d_model', type=int, default=200)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_residual', action='store_true', default=True)
    parser.add_argument('--no_residual', dest='use_residual', action='store_false')
    parser.add_argument('--fusion_mode', type=str, default='interleave',
                        choices=['interleave', 'concat_feat', 'parallel'])
    parser.add_argument('--pooling', type=str, default='gap', choices=['gap', 'last'])
    parser.add_argument('--classifier_depth', type=int, default=3, choices=[1,2,3])

    parser.add_argument('--loss_type', type=str, default='focal', choices=['focal', 'bce'])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=3.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--pretrained_weights', type=str, default='')
    parser.add_argument('--seeds', type=int, nargs='+', default=[3407, 42, 1234])
    parser.add_argument('--model_dir', type=str, default='./ablation_results')

    args = parser.parse_args()
    auc_list = []

    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")

        full_dataset = SeizeIT2DatasetAblation(
            data_dir=args.data_dir,
            mode='train',
            preictal_min=30,
            channels=2,
            max_subjects=args.max_subjects,
            norm_type=args.norm_type,
            low_freq=args.low_freq,
            high_freq=args.high_freq,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
            target_sr=args.target_sr,
            balance=args.balance,
            seed=seed
        )

        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(
            full_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=full_dataset.collate, num_workers=4, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=full_dataset.collate, num_workers=4)

        if args.fusion_mode == 'concat_feat':
            model_d_model = args.d_model * 2
        else:
            model_d_model = args.d_model

        model = FlexibleModel(
            d_model=model_d_model,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_residual=args.use_residual,
            fusion_mode=args.fusion_mode,
            pooling=args.pooling,
            classifier_depth=args.classifier_depth
        )

        if args.pretrained_weights and os.path.exists(args.pretrained_weights):
            state = torch.load(args.pretrained_weights, map_location='cpu')
            model.load_state_dict(state, strict=False)
            print("Pretrained weights loaded (strict=False).")

        trainer = Trainer(args, train_loader, test_loader, model)
        final_auc = trainer.train()
        auc_list.append(final_auc)

    print(f"\n{'='*60}\nAblation Results: Mean ROC-AUC = {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}\n{'='*60}")

if __name__ == "__main__":
    main()