import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.deep.networks import ConditionalPrior
from models.deep.MultiVAEAMRHead import MultiVAE_Bernoulli

from src.evaluation.metrics import compute_multilabel_auc, compute_per_antibiotic_auc


class MultiVAE_Bernoulli_SpeciesPrior_AMR_Head(MultiVAE_Bernoulli):
    """
    Multi-decoder VAE with species-conditional latent prior.
    
    Encoder: q(z | x)
    Prior:   p(z | species)
    Decoder: p(x | z, domain)
    """

    def __init__(self, input_dim, latent_dim, num_domains, n_species, n_antibiotics, lambda_amr, pos_weight=None, antibiotic_names=None):
        super().__init__(input_dim, latent_dim, num_domains)

        self.n_species = n_species
        self.prior = ConditionalPrior(n_species=n_species, latent_dim=latent_dim)
        self.lambda_amr = lambda_amr
        self.n_antibiotics = n_antibiotics
        self.pos_weight = pos_weight
        self.antibiotic_names = antibiotic_names

        # AMR classification head
        self.amr_trunk = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Dropout(0.4)
        )

        # One head per antibiotic
        self.amr_heads = nn.ModuleList([
            nn.Linear(latent_dim, 1) for _ in range(n_antibiotics)
        ])

    def elbo_loss(self, x, mu, logvar, z, domain_id, species_id, beta=1.0):
        # Log-likelihood p(x|z,d)
        RE = self.decoder.log_prob(x, z, domain_id)

        # Build species one-hot
        species_onehot = torch.nn.functional.one_hot(
            species_id, num_classes=self.n_species
        ).float()

        # Prior p(z | species)
        mu_p, logvar_p = self.prior(species_onehot)

        # KL(q(z|x) || p(z|species))
        KL = -0.5 * torch.sum(
            1
            + (logvar - logvar_p)
            - ((mu - mu_p) ** 2 + logvar.exp()) / logvar_p.exp(),
            dim=1,
        )

        NLL = -(RE - beta * KL)
        return NLL.mean(), (-RE).mean(), KL.mean()

    def total_loss(self, x, mu, logvar, z, domain_id, species_id, amr_logits, amr_labels, beta=1.0):
        loss, recon, kl = self.elbo_loss(
            x, mu, logvar, z,
            domain_id=domain_id,
            species_id=species_id,
            beta=beta,
        )

        amr_loss = torch.tensor(0.0, device=x.device)
        n_tasks = 0

        per_antibiotic_losses = {}
        per_antibiotic_counts = {}

        if amr_logits is not None and amr_labels is not None:
            if amr_labels.ndim == 1:
                amr_labels = amr_labels.view(-1, 1)

            for j in range(amr_logits.shape[1]):
                mask_j = ~torch.isnan(amr_labels[:, j])

                if mask_j.sum() == 0:
                    per_antibiotic_losses[j] = np.nan
                    per_antibiotic_counts[j] = 0
                    continue

                logits_j = amr_logits[mask_j, j]
                labels_j = amr_labels[mask_j, j]

                if self.pos_weight is not None:
                    if self.pos_weight.ndim == 0 or len(self.pos_weight) == 1:
                        pos_weight_j = self.pos_weight.to(x.device)
                    else:
                        pos_weight_j = self.pos_weight[j].to(x.device)
                else:
                    pos_weight_j = None

                loss_j = F.binary_cross_entropy_with_logits(
                    logits_j,
                    labels_j,
                    pos_weight=pos_weight_j,
                    reduction="mean"
                )

                amr_loss += loss_j
                n_tasks += 1

                per_antibiotic_losses[j] = loss_j.item()
                per_antibiotic_counts[j] = int(mask_j.sum().item())

            if n_tasks > 0:
                amr_loss = amr_loss / n_tasks
                loss = loss + self.lambda_amr * amr_loss

        return loss, recon, kl, amr_loss, per_antibiotic_losses, per_antibiotic_counts

class MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_Extended(MultiVAE_Bernoulli_SpeciesPrior_AMR_Head):
    def __init__(self, input_dim, latent_dim, num_domains, n_species, n_antibiotics, lambda_amr=0.1, pos_weight=None, antibiotic_names=None, epochs=100, lr=1e-4, annealing_epochs=50, patience=20):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_domains=num_domains,
            n_species=n_species,
            n_antibiotics=n_antibiotics,
            lambda_amr=lambda_amr,
            pos_weight=pos_weight,
            antibiotic_names=antibiotic_names
        )

        self.epochs = epochs
        self.lr = lr
        self.annealing_epochs = annealing_epochs
        self.patience = patience

        self.optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=1e-4
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            min_lr=1e-6
        )

        self.loss_during_training = []
        self.reconstruc_during_training = []
        self.KL_during_training = []
        self.AMR_during_training = []
        self.AUC_during_training = []
            
    def trainloop(self, trainloader, validloader, device):
        self.to(device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            beta = 1

            # =======================
            # TRAIN
            # =======================
            self.train()
            tr_loss, tr_recon, tr_kl, tr_amr = 0.0, 0.0, 0.0, 0.0
            tr_logits_all, tr_labels_all = [], []

            tr_ab_loss_sum = {j: 0.0 for j in range(self.n_antibiotics)}
            tr_ab_loss_num = {j: 0 for j in range(self.n_antibiotics)}
            tr_ab_count_sum = {j: 0 for j in range(self.n_antibiotics)}

            for x, domain_id, species_id, amr_labels in trainloader:
                x = x.to(device)
                domain_id = domain_id.to(device)
                species_id = species_id.to(device)
                amr_labels = amr_labels.to(device)

                self.optimizer.zero_grad()

                mu, logvar, z, amr_logits = self.forward(x, domain_id)
                loss, recon, kl, amr, batch_ab_losses, batch_ab_counts = self.total_loss(
                    x, mu, logvar, z,
                    domain_id=domain_id,
                    species_id=species_id,
                    amr_logits=amr_logits,
                    amr_labels=amr_labels,
                    beta=beta,
                )

                loss.backward()
                self.optimizer.step()

                tr_loss += loss.item()
                tr_recon += recon.item()
                tr_kl += kl.item()
                tr_amr += amr.item()

                tr_logits_all.append(amr_logits.detach().cpu())
                tr_labels_all.append(amr_labels.detach().cpu())

                for j in range(self.n_antibiotics):
                    if not np.isnan(batch_ab_losses[j]):
                        tr_ab_loss_sum[j] += batch_ab_losses[j]
                        tr_ab_loss_num[j] += 1
                    tr_ab_count_sum[j] += batch_ab_counts[j]

            tr_loss /= len(trainloader)
            tr_recon /= len(trainloader)
            tr_kl /= len(trainloader)
            tr_amr /= len(trainloader)

            tr_logits_all = torch.cat(tr_logits_all, dim=0)
            tr_labels_all = torch.cat(tr_labels_all, dim=0)
            tr_auc = compute_multilabel_auc(tr_logits_all, tr_labels_all)
            tr_auc_per_ab = compute_per_antibiotic_auc(
                tr_logits_all,
                tr_labels_all,
                antibiotic_names=self.antibiotic_names
            )

            tr_ab_loss_mean = {}
            for j in range(self.n_antibiotics):
                if tr_ab_loss_num[j] > 0:
                    tr_ab_loss_mean[j] = tr_ab_loss_sum[j] / tr_ab_loss_num[j]
                else:
                    tr_ab_loss_mean[j] = np.nan

            # =======================
            # VALIDATION
            # =======================
            self.eval()
            val_loss, val_recon, val_kl, val_amr = 0.0, 0.0, 0.0, 0.0
            val_logits_all, val_labels_all = [], []

            val_ab_loss_sum = {j: 0.0 for j in range(self.n_antibiotics)}
            val_ab_loss_num = {j: 0 for j in range(self.n_antibiotics)}
            val_ab_count_sum = {j: 0 for j in range(self.n_antibiotics)}

            with torch.no_grad():
                for x, domain_id, species_id, amr_labels in validloader:
                    x = x.to(device)
                    domain_id = domain_id.to(device)
                    species_id = species_id.to(device)
                    amr_labels = amr_labels.to(device)

                    mu, logvar, z, amr_logits = self.forward(x, domain_id)
                    loss, recon, kl, amr, batch_ab_losses, batch_ab_counts = self.total_loss(
                        x, mu, logvar, z,
                        domain_id=domain_id,
                        species_id=species_id,
                        amr_logits=amr_logits,
                        amr_labels=amr_labels,
                        beta=beta,
                    )

                    val_loss += loss.item()
                    val_recon += recon.item()
                    val_kl += kl.item()
                    val_amr += amr.item()

                    val_logits_all.append(amr_logits.detach().cpu())
                    val_labels_all.append(amr_labels.detach().cpu())

                    for j in range(self.n_antibiotics):
                        if not np.isnan(batch_ab_losses[j]):
                            val_ab_loss_sum[j] += batch_ab_losses[j]
                            val_ab_loss_num[j] += 1
                        val_ab_count_sum[j] += batch_ab_counts[j]

            val_loss /= len(validloader)
            val_recon /= len(validloader)
            val_kl /= len(validloader)
            val_amr /= len(validloader)

            val_logits_all = torch.cat(val_logits_all, dim=0)
            val_labels_all = torch.cat(val_labels_all, dim=0)
            val_auc = compute_multilabel_auc(val_logits_all, val_labels_all)
            val_auc_per_ab = compute_per_antibiotic_auc(val_logits_all,val_labels_all,antibiotic_names=self.antibiotic_names)

            val_ab_loss_mean = {}
            for j in range(self.n_antibiotics):
                if val_ab_loss_num[j] > 0:
                    val_ab_loss_mean[j] = val_ab_loss_sum[j] / val_ab_loss_num[j]
                else:
                    val_ab_loss_mean[j] = np.nan

            self.loss_during_training.append((tr_loss, val_loss))
            self.reconstruc_during_training.append((tr_recon, val_recon))
            self.KL_during_training.append((tr_kl, val_kl))
            self.AMR_during_training.append((tr_amr, val_amr))
            self.AUC_during_training.append((tr_auc, val_auc))


            # =======================
            # SCHEDULER STEP
            # =======================
            self.scheduler.step(val_loss)
            
            # Log del LR actual
            current_lr = self.optimizer.param_groups[0]['lr']

            if (epoch + 1) % 10 == 0:
                tr_auc_str = f"{tr_auc:.4f}" if not np.isnan(tr_auc) else "nan"
                val_auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "nan"

                print(
                    f"Epoch {epoch+1}/{self.epochs} | LR={current_lr:.2e} | "
                    f"[Train] Loss={tr_loss:.4f} | Recon={tr_recon:.4f} | "
                    f"KL={tr_kl:.4f} | AMR={tr_amr:.4f} | AUC={tr_auc_str} || "
                    f"[Val] Loss={val_loss:.4f} | Recon={val_recon:.4f} | "
                    f"KL={val_kl:.4f} | AMR={val_amr:.4f} | AUC={val_auc_str}"
                )

                print("Train antibiotic AUC:", {k: round(v, 4) if not np.isnan(v) else None for k, v in tr_auc_per_ab.items()})
                print("Val antibiotic AUC:", {k: round(v, 4) if not np.isnan(v) else None for k, v in val_auc_per_ab.items()})
                print("")
                
            # =======================
            # EARLY STOPPING
            # =======================
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.load_state_dict(best_state)
