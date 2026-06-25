import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score, average_precision_score


class SimpleAMRMLP(nn.Module):
    def __init__(self, input_dim, latent_dim, n_antibiotics):
        super().__init__()

        self.n_antibiotics = n_antibiotics
        
        self.encoder_part = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim), 
            nn.ReLU(), 
        )

        self.amr_trunk = nn.Sequential(
            nn.LayerNorm(latent_dim),
            
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.LayerNorm(32)
        )

        self.amr_heads = nn.ModuleList([
            nn.Linear(32, 1) for _ in range(n_antibiotics)
        ])

    def forward(self, x):
        z = self.encoder_part(x)
        h = self.amr_trunk(z)
        return torch.cat([head(h) for head in self.amr_heads], dim=1)


class SimpleAMRMLP_Extended(SimpleAMRMLP):
    def __init__(self, input_dim, latent_dim, n_antibiotics, antibiotic_names, epochs=100, lr=1e-3, patience=15):
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, n_antibiotics=n_antibiotics)

        self.antibiotic_names = antibiotic_names
        self.epochs = epochs
        self.lr = lr
        self.patience = patience

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.loss_during_training = []
            
    def trainloop(self, trainloader, validloader, device):
        self.to(device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # =======================
            # TRAIN
            # =======================
            self.train()
            tr_loss = 0.0

            for x, a in trainloader:
                x, a = x.to(device), a.to(device)
                
                self.optimizer.zero_grad()
                logits = self.forward(x)
                
                loss, n_tasks = 0, 0
                for j in range(self.n_antibiotics):
                    mask = ~torch.isnan(a[:, j])
                    if mask.sum() == 0:
                        continue
                    loss += nn.functional.binary_cross_entropy_with_logits(
                        logits[mask, j], a[mask, j], reduction="mean"
                    )
                    n_tasks += 1
                
                if n_tasks > 0:
                    loss = loss / n_tasks
                    loss.backward()
                    self.optimizer.step()
                    
                    tr_loss += loss.item()

            tr_loss /= max(len(trainloader), 1)

            # =======================
            # VALIDATION
            # =======================
            self.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for x, a in validloader:
                    x, a = x.to(device), a.to(device)
                    logits = self.forward(x)
                    
                    batch_loss, n_tasks = 0.0, 0
                    for j in range(self.n_antibiotics):
                        mask = ~torch.isnan(a[:, j])
                        if mask.sum() == 0:
                            continue
                        batch_loss += nn.functional.binary_cross_entropy_with_logits(
                            logits[mask, j], a[mask, j], reduction="mean"
                        ).item()
                        n_tasks += 1
                    
                    if n_tasks > 0:
                        val_loss += batch_loss / n_tasks

            val_loss /= max(len(validloader), 1)

            self.loss_during_training.append((tr_loss, val_loss))
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | LR={current_lr:.2e} | "
                    f"[Train] Loss={tr_loss:.4f} || "
                    f"[Val] Loss={val_loss:.4f}"
                )

            # =======================
            # EARLY STOPPING
            # =======================
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

    def evaluate(self, X, amr_labels, device, batch_size=512):
        self.eval()
        self.to(device)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        all_logits = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i : i + batch_size].to(device)
                logits = self.forward(batch)
                all_logits.append(logits.cpu())
            
        all_logits = torch.cat(all_logits, dim=0).numpy()
        probs = 1 / (1 + np.exp(-all_logits))  
        
        results = {}
        for j, atb_name in enumerate(self.antibiotic_names):
            y_true = amr_labels[:, j]
            valid  = ~np.isnan(y_true)
            y_true_clean = y_true[valid].astype(int)
            y_prob = probs[valid, j]
            
            if len(y_true_clean) < 10 or len(np.unique(y_true_clean)) < 2:
                continue
                
            results[atb_name] = {
                "auc":    roc_auc_score(y_true_clean, y_prob),
                "pr_auc": average_precision_score(y_true_clean, y_prob),
                "n":      int(valid.sum()),
            }
            
        return results
    