import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.deep.networks import ConditionalPrior
from src.evaluation.metrics import compute_multilabel_auc, compute_per_antibiotic_auc


# ============================================================
#  ENCODER — arquitectura Chen et al. 2026
#  input_dim → 512 → 256 → 128 → (mu, logvar)
# ============================================================
class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.mu     = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h      = self.net(x)
        mu     = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -6, 6)
        return mu, logvar
    

# ============================================================
#  DECODER — simétrico a Chen
#  latent_dim → 128 → 256 → 512 → output_dim  (por dominio)
# ============================================================
class BernoulliDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, num_domains: int = 1):
        super().__init__()
        self.num_domains = num_domains
        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim),
                nn.Sigmoid(),
            )
            for _ in range(num_domains)
        ])

    def forward(self, z, domain_id=None):
        if self.num_domains == 1:
            return self.net[0](z)
        if domain_id is None:
            raise ValueError("domain_id required when num_domains > 1")
        x_recon = torch.zeros(z.size(0), self.net[0][-2].out_features, device=z.device)
        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.any():
                x_recon[mask] = self.net[d](z[mask])
        return x_recon

    def log_prob(self, x, z, domain_id=None):
        theta = self.forward(z, domain_id)
        if torch.any(theta < 0) or torch.any(theta > 1) or torch.isnan(theta).any():
            raise ValueError("Decoder output out of [0,1]")
        if torch.any(x < 0) or torch.any(x > 1) or torch.isnan(x).any():
            raise ValueError("Input x must be in [0,1]")
        return -F.binary_cross_entropy(theta, x, reduction="none").sum(dim=1)


# ============================================================
#  BASE VAE — encoder + reparametrize + decoder
# ============================================================
class MultiVAE_Bernoulli(nn.Module):
    def __init__(self, input_dim, latent_dim, num_domains):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = BernoulliDecoder(latent_dim, input_dim, num_domains)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, domain_id):
        mu, logvar = self.encoder(x)
        z          = self.reparametrize(mu, logvar)
        amr_logits = None  # se sobreescribe en subclase
        return mu, logvar, z, amr_logits


# ============================================================
#  VAE CON PRIOR CONDICIONAL + AMR HEAD
# ============================================================
class MultiVAE_Bernoulli_SpeciesPrior_AMR_HeadZ(MultiVAE_Bernoulli):
    """
    VAE con:
      - Encoder/Decoder arquitectura Chen et al. 2026
      - Prior condicional por especie p(z | species)  [o N(0,1) fijo]
      - AMR head: dropout + linear directo sobre z
      - Weighted BCE inverse-frequency (Chen et al.)
    """
    def __init__(
        self,
        input_dim,
        latent_dim,
        num_domains,
        n_species,
        n_antibiotics,
        lambda_amr,
        antibiotic_names=None,
        use_fixed_prior=False,
    ):
        super().__init__(input_dim, latent_dim, num_domains)
        self.n_species        = n_species
        self.lambda_amr       = lambda_amr
        self.n_antibiotics    = n_antibiotics
        self.antibiotic_names = antibiotic_names
        self.use_fixed_prior  = use_fixed_prior
        self.n_amr_samples    = 1

        self.prior = None if use_fixed_prior else ConditionalPrior(
            n_species=n_species, latent_dim=latent_dim
        )

        # AMR head — dropout + linear directo sobre z (sin trunk)
        self.amr_drop  = nn.Dropout(0.3)
        self.amr_heads = nn.ModuleList([
            nn.Linear(latent_dim, 1) for _ in range(n_antibiotics)
        ])

    def forward(self, x, domain_id):
        mu, logvar  = self.encoder(x)
        z           = self.reparametrize(mu, logvar)
        h           = self.amr_drop(z)
        amr_logits  = torch.cat([head(h) for head in self.amr_heads], dim=1)
        return mu, logvar, z, amr_logits

    def elbo_loss(self, x, mu, logvar, z, domain_id, species_id, beta=1.0):
        RE = self.decoder.log_prob(x, z, domain_id)

        if self.use_fixed_prior:
            KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        else:
            species_onehot = F.one_hot(species_id, num_classes=self.n_species).float()
            mu_p, logvar_p = self.prior(species_onehot)
            KL = -0.5 * torch.sum(
                1
                + (logvar - logvar_p)
                - ((mu - mu_p) ** 2 + logvar.exp()) / logvar_p.exp(),
                dim=1,
            )

        NLL = -(RE - beta * KL)
        return NLL.mean(), (-RE).mean(), KL.mean()

    def total_loss(self, x, mu, logvar, z, domain_id, species_id, amr_logits, amr_labels, beta=1.0):
        loss, recon, kl = self.elbo_loss(x, mu, logvar, z, domain_id, species_id, beta)

        amr_loss             = torch.tensor(0.0, device=x.device)
        per_antibiotic_losses = {}
        per_antibiotic_counts = {}

        if amr_logits is not None and amr_labels is not None:
            if amr_labels.ndim == 1:
                amr_labels = amr_labels.view(-1, 1)

            std          = torch.exp(0.5 * logvar)
            sample_losses = []

            for k in range(self.n_amr_samples):
                z_k      = z if k == 0 else (mu + std * torch.randn_like(std))
                h_k      = self.amr_drop(z_k)
                logits_k = torch.cat([head(h_k) for head in self.amr_heads], dim=1)

                task_loss_k = torch.tensor(0.0, device=x.device)
                n_tasks_k   = 0
                ab_losses_k = {}
                ab_counts_k = {}

                for j in range(logits_k.shape[1]):
                    mask_j = ~torch.isnan(amr_labels[:, j])
                    if mask_j.sum() == 0:
                        ab_losses_k[j] = np.nan
                        ab_counts_k[j] = 0
                        continue

                    logits_j = logits_k[mask_j, j]
                    labels_j = amr_labels[mask_j, j]

                    # Chen et al. inverse-frequency weighting
                    N     = mask_j.sum().float()
                    N_pos = labels_j.sum().float()
                    N_neg = N - N_pos
                    w_pos = N / (2.0 * N_pos) if N_pos > 0 else torch.tensor(1.0, device=x.device)
                    w_neg = N / (2.0 * N_neg) if N_neg > 0 else torch.tensor(1.0, device=x.device)
                    sample_weights = torch.where(labels_j == 1, w_pos, w_neg)

                    loss_j = F.binary_cross_entropy_with_logits(
                        logits_j, labels_j, weight=sample_weights, reduction="mean"
                    )
                    task_loss_k       += loss_j
                    n_tasks_k         += 1
                    ab_losses_k[j]     = loss_j.item()
                    ab_counts_k[j]     = int(mask_j.sum().item())

                if n_tasks_k > 0:
                    sample_losses.append(task_loss_k / n_tasks_k)
                if k == 0:
                    per_antibiotic_losses = ab_losses_k
                    per_antibiotic_counts = ab_counts_k

            if sample_losses:
                amr_loss = torch.stack(sample_losses).mean()
                loss     = loss + self.lambda_amr * amr_loss

        return loss, recon, kl, amr_loss, per_antibiotic_losses, per_antibiotic_counts


# ============================================================
#  EXTENDED — trainloop completo
# ============================================================
class MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZChinos(
    MultiVAE_Bernoulli_SpeciesPrior_AMR_HeadZ
):
    def __init__(
        self,
        input_dim,
        latent_dim,        # recomendado: 128
        num_domains,
        n_species,
        n_antibiotics,
        lambda_amr=0.1,
        antibiotic_names=None,
        epochs=100,
        lr=1e-4,
        annealing_epochs=50,
        patience=20,
        use_fixed_prior=False,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_domains=num_domains,
            n_species=n_species,
            n_antibiotics=n_antibiotics,
            lambda_amr=lambda_amr,
            antibiotic_names=antibiotic_names,
            use_fixed_prior=use_fixed_prior,
        )
        self.epochs          = epochs
        self.lr              = lr
        self.annealing_epochs = annealing_epochs
        self.patience        = patience

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-6, cooldown=10
        )

        self.loss_during_training    = []
        self.reconstruc_during_training = []
        self.KL_during_training      = []
        self.AMR_during_training     = []
        self.AUC_during_training     = []

    def trainloop(self, trainloader, validloader, device):
        self.to(device)
        best_val_loss = float("inf")
        best_val_auc    = -float("inf")
        patience_counter = 0
        best_state      = None

        prior_mode  = "N(0,1) fixed" if self.use_fixed_prior else "learned prior"
        anneal_mode = (
            f"annealing ({self.annealing_epochs} epochs)"
            if self.annealing_epochs is not None else "no annealing (beta=1)"
        )
        print(f"Training mode: VAE-Chinos | Prior: {prior_mode} | {anneal_mode}")

        for epoch in range(self.epochs):
            beta = (
                1.0 if self.annealing_epochs is None
                else min(0.1, (epoch + 1) / self.annealing_epochs)
            )

            # ── TRAIN ─────────────────────────────────────────────
            self.train()
            tr_loss = tr_recon = tr_kl = tr_amr = 0.0
            tr_logits_all, tr_labels_all = [], []
            tr_ab_loss_sum = {j: 0.0 for j in range(self.n_antibiotics)}
            tr_ab_loss_num = {j: 0   for j in range(self.n_antibiotics)}

            for x, domain_id, species_id, amr_labels in trainloader:
                x, domain_id, species_id, amr_labels = (
                    x.to(device), domain_id.to(device),
                    species_id.to(device), amr_labels.to(device)
                )
                self.optimizer.zero_grad()
                mu, logvar, z, amr_logits = self.forward(x, domain_id)
                loss, recon, kl, amr, batch_ab_losses, _ = self.total_loss(
                    x, mu, logvar, z, domain_id, species_id,
                    amr_logits, amr_labels, beta=beta,
                )
                if loss.requires_grad:
                    loss.backward()
                    self.optimizer.step()

                tr_loss  += loss.item()
                tr_recon += recon.item()
                tr_kl    += kl.item()
                tr_amr   += amr.item()
                tr_logits_all.append(amr_logits.detach().cpu())
                tr_labels_all.append(amr_labels.detach().cpu())
                for j in range(self.n_antibiotics):
                    if not np.isnan(batch_ab_losses.get(j, np.nan)):
                        tr_ab_loss_sum[j] += batch_ab_losses[j]
                        tr_ab_loss_num[j] += 1

            n_tr = len(trainloader)
            tr_loss /= n_tr; tr_recon /= n_tr; tr_kl /= n_tr; tr_amr /= n_tr
            tr_logits_all = torch.cat(tr_logits_all, dim=0)
            tr_labels_all = torch.cat(tr_labels_all, dim=0)
            tr_auc        = compute_multilabel_auc(tr_logits_all, tr_labels_all)
            tr_auc_per_ab = compute_per_antibiotic_auc(
                tr_logits_all, tr_labels_all, antibiotic_names=self.antibiotic_names
            )

            # ── VALIDATION ────────────────────────────────────────
            self.eval()
            val_loss = val_recon = val_kl = val_amr = 0.0
            val_logits_all, val_labels_all = [], []

            with torch.no_grad():
                for x, domain_id, species_id, amr_labels in validloader:
                    x, domain_id, species_id, amr_labels = (
                        x.to(device), domain_id.to(device),
                        species_id.to(device), amr_labels.to(device)
                    )
                    mu, logvar, z, amr_logits = self.forward(x, domain_id)
                    loss, recon, kl, amr, _, _ = self.total_loss(
                        x, mu, logvar, z, domain_id, species_id,
                        amr_logits, amr_labels, beta=beta,
                    )
                    val_loss  += loss.item()
                    val_recon += recon.item()
                    val_kl    += kl.item()
                    val_amr   += amr.item()
                    val_logits_all.append(amr_logits.detach().cpu())
                    val_labels_all.append(amr_labels.detach().cpu())

            n_vl = len(validloader)
            val_loss /= n_vl; val_recon /= n_vl; val_kl /= n_vl; val_amr /= n_vl
            val_logits_all = torch.cat(val_logits_all, dim=0)
            val_labels_all = torch.cat(val_labels_all, dim=0)
            val_auc        = compute_multilabel_auc(val_logits_all, val_labels_all)
            val_auc_per_ab = compute_per_antibiotic_auc(
                val_logits_all, val_labels_all, antibiotic_names=self.antibiotic_names
            )

            self.loss_during_training.append((tr_loss,  val_loss))
            self.reconstruc_during_training.append((tr_recon, val_recon))
            self.KL_during_training.append((tr_kl,   val_kl))
            self.AMR_during_training.append((tr_amr,  val_amr))
            self.AUC_during_training.append((tr_auc,  val_auc))

            self.scheduler.step(val_auc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | LR={current_lr:.2e} | "
                    f"[Train] Loss={tr_loss:.4f} Recon={tr_recon:.4f} "
                    f"KL={tr_kl:.4f} AMR={tr_amr:.4f} AUC={tr_auc:.4f} || "
                    f"[Val] Loss={val_loss:.4f} Recon={val_recon:.4f} "
                    f"KL={val_kl:.4f} AMR={val_amr:.4f} AUC={val_auc:.4f}"
                )
                print("  Train AUC/ab:", {
                    k: round(v, 4) if not np.isnan(v) else None
                    for k, v in tr_auc_per_ab.items()
                })
                print("  Val   AUC/ab:", {
                    k: round(v, 4) if not np.isnan(v) else None
                    for k, v in val_auc_per_ab.items()
                })

            # ── EARLY STOPPING por AUC ────────────────────────────
            if val_auc > best_val_auc:
                best_val_auc     = val_auc
                best_state       = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # # ── EARLY STOPPING por VAL ────────────────────────────
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     best_state = copy.deepcopy(self.state_dict())
            #     patience_counter = 0
            # else:
            #     patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1} | best val AUC={best_val_auc:.4f}")
                break

        if best_state is not None:
            self.load_state_dict(best_state)