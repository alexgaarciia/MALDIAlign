import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score


class ChenMLP_MultiHead(nn.Module):
    """
    Arquitectura Chen et al. 2026 con múltiples heads (uno por antibiótico).
    Backbone compartido: 512 → 256 → 128, ReLU + Dropout.
    Comparable directamente con el VAE multi-head.
    """
    def __init__(self, input_dim=6000, n_antibiotics=1, dropout=0.5):
        super().__init__()
        self.n_antibiotics = n_antibiotics

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
        )
        self.amr_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(n_antibiotics)
        ])

    def forward(self, x):
        h = self.backbone(x)
        return torch.cat([head(h) for head in self.amr_heads], dim=1)


class ChenMLP_MultiHead_Extended(ChenMLP_MultiHead):
    def __init__(
        self,
        input_dim=6000,
        n_antibiotics=1,
        antibiotic_names=None,
        dropout=0.5,
        epochs=200,
        lr=1e-3,
        patience=20,
    ):
        super().__init__(input_dim=input_dim, n_antibiotics=n_antibiotics, dropout=dropout)
        self.antibiotic_names = antibiotic_names or [f"ab_{j}" for j in range(n_antibiotics)]
        self.epochs   = epochs
        self.lr       = lr
        self.patience = patience
        self.loss_during_training = []

    def _class_weights_per_ab(self, all_y):
        """
        Compute inverse-frequency weights per antibiotic (Chen et al. eq. 1-2).
        all_y: np.ndarray (N, n_antibiotics), may contain NaN.
        Returns list of (w_pos, w_neg) per antibiotic.
        """
        weights = []
        for j in range(self.n_antibiotics):
            y_j   = all_y[:, j]
            valid = ~np.isnan(y_j)
            y_v   = y_j[valid]
            N     = len(y_v)
            N_pos = y_v.sum()
            N_neg = N - N_pos
            w_pos = N / (2.0 * N_pos) if N_pos > 0 else 1.0
            w_neg = N / (2.0 * N_neg) if N_neg > 0 else 1.0
            weights.append((float(w_pos), float(w_neg)))
        return weights

    def trainloop(self, trainloader, validloader, device, finetune=False):
        lr        = 1e-4 if finetune else self.lr
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Compute class weights from full training set
        all_y = np.concatenate([a.numpy() for _, a in trainloader], axis=0)
        if all_y.ndim == 1:
            all_y = all_y[:, None]
        class_weights = self._class_weights_per_ab(all_y)

        self.to(device)
        best_val_loss    = float("inf")
        patience_counter = 0
        best_state       = None

        for epoch in range(self.epochs):
            # ── TRAIN ──────────────────────────────────────────
            self.train()
            tr_loss = 0.0

            for x, y in trainloader:
                x = x.to(device)
                y = y.to(device)
                if y.ndim == 1:
                    y = y.unsqueeze(1)

                optimizer.zero_grad()
                logits = self.forward(x)

                loss    = torch.tensor(0.0, device=device)
                n_tasks = 0
                for j in range(self.n_antibiotics):
                    mask_j = ~torch.isnan(y[:, j])
                    if mask_j.sum() == 0:
                        continue
                    l_j = logits[mask_j, j]
                    y_j = y[mask_j, j]
                    w_pos, w_neg = class_weights[j]
                    w = torch.where(
                        y_j == 1,
                        torch.tensor(w_pos, device=device),
                        torch.tensor(w_neg, device=device),
                    )
                    loss    += nn.functional.binary_cross_entropy_with_logits(
                        l_j, y_j, weight=w, reduction="mean"
                    )
                    n_tasks += 1

                if n_tasks > 0:
                    loss = loss / n_tasks
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item()

            tr_loss /= max(len(trainloader), 1)

            # ── VALIDATION ─────────────────────────────────────
            self.eval()
            val_loss = 0.0

            with torch.no_grad():
                for x, y in validloader:
                    x = x.to(device)
                    y = y.to(device)
                    if y.ndim == 1:
                        y = y.unsqueeze(1)

                    logits      = self.forward(x)
                    batch_loss  = torch.tensor(0.0, device=device)
                    n_tasks     = 0

                    for j in range(self.n_antibiotics):
                        mask_j = ~torch.isnan(y[:, j])
                        if mask_j.sum() == 0:
                            continue
                        l_j = logits[mask_j, j]
                        y_j = y[mask_j, j]
                        w_pos, w_neg = class_weights[j]
                        w = torch.where(
                            y_j == 1,
                            torch.tensor(w_pos, device=device),
                            torch.tensor(w_neg, device=device),
                        )
                        batch_loss += nn.functional.binary_cross_entropy_with_logits(
                            l_j, y_j, weight=w, reduction="mean"
                        )
                        n_tasks += 1

                    if n_tasks > 0:
                        val_loss += (batch_loss / n_tasks).item()

            val_loss /= max(len(validloader), 1)
            self.loss_during_training.append((tr_loss, val_loss))
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | LR={current_lr:.2e} | "
                    f"[Train] Loss={tr_loss:.4f} || [Val] Loss={val_loss:.4f}"
                )

            # ── EARLY STOPPING ─────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_state       = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.load_state_dict(best_state)

    def evaluate(self, X, amr_labels, device, batch_size=512):
        self.eval()
        self.to(device)

        X_tensor   = torch.tensor(X, dtype=torch.float32)
        all_logits = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i: i + batch_size].to(device)
                all_logits.append(self.forward(batch).cpu())

        all_logits = torch.cat(all_logits, dim=0).numpy()
        probs      = 1 / (1 + np.exp(-all_logits))

        if amr_labels.ndim == 1:
            amr_labels = amr_labels[:, None]

        results = {}
        for j, atb_name in enumerate(self.antibiotic_names):
            y_true = amr_labels[:, j]
            valid  = ~np.isnan(y_true)
            y_v    = y_true[valid].astype(int)
            y_p    = probs[valid, j]

            if len(y_v) < 10 or len(np.unique(y_v)) < 2:
                continue

            results[atb_name] = {
                "auc":    roc_auc_score(y_v, y_p),
                "pr_auc": average_precision_score(y_v, y_p),
                "n":      int(valid.sum()),
            }

        return results
    