import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score


class ChenMLP(nn.Module):
    """
    MLP backbone from Chen et al. 2026:
    512 -> 256 -> 128 -> 1, ReLU + Dropout, un modelo por par especie-antibiótico.
    """
    def __init__(self, input_dim=6000, dropout=0.5):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class ChenMLP_Extended(ChenMLP):
    def __init__(
        self,
        input_dim=6000,
        dropout=0.5,
        epochs=200,
        lr=1e-3,
        patience=20,
    ):
        super().__init__(input_dim=input_dim, dropout=dropout)

        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.loss_during_training = []

    def _class_weights(self, y_train):
        """
        w+ = N / (2 * N+)
        w- = N / (2 * N-)
        Chen et al. eq. (1-2)
        """
        N = len(y_train)
        N_pos = y_train.sum()
        N_neg = N - N_pos
        w_pos = N / (2.0 * N_pos) if N_pos > 0 else 1.0
        w_neg = N / (2.0 * N_neg) if N_neg > 0 else 1.0
        return float(w_pos), float(w_neg)

    def trainloop(self, trainloader, validloader, device, finetune=False):
        lr = 1e-4 if finetune else self.lr

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Compute class weights from training labels
        all_y = np.concatenate([a.numpy() for _, a in trainloader])
        w_pos, w_neg = self._class_weights(all_y)

        self.to(device)
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.train()
            tr_loss = 0.0

            for x, y in trainloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                logits = self.forward(x)
                w = torch.where(y == 1,
                                torch.tensor(w_pos, device=device),
                                torch.tensor(w_neg, device=device))
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, y, weight=w, reduction="mean"
                )
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()

            tr_loss /= max(len(trainloader), 1)

            self.eval()
            val_loss = 0.0

            with torch.no_grad():
                for x, y in validloader:
                    x, y = x.to(device), y.to(device)
                    logits = self.forward(x)
                    w = torch.where(y == 1,
                                    torch.tensor(w_pos, device=device),
                                    torch.tensor(w_neg, device=device))
                    val_loss += nn.functional.binary_cross_entropy_with_logits(
                        logits, y, weight=w, reduction="mean"
                    ).item()

            val_loss /= max(len(validloader), 1)
            self.loss_during_training.append((tr_loss, val_loss))
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | LR={current_lr:.2e} | "
                    f"[Train] Loss={tr_loss:.4f} || [Val] Loss={val_loss:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.load_state_dict(best_state)

    def evaluate(self, X, y_true, device, batch_size=512):
        self.eval()
        self.to(device)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        all_logits = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i: i + batch_size].to(device)
                all_logits.append(self.forward(batch).cpu())

        logits = torch.cat(all_logits).numpy()
        probs = 1 / (1 + np.exp(-logits))

        y_true = np.array(y_true).astype(int)

        if len(np.unique(y_true)) < 2:
            return None

        return {
            "auc":    roc_auc_score(y_true, probs),
            "pr_auc": average_precision_score(y_true, probs),
            "n":      len(y_true),
        }
    