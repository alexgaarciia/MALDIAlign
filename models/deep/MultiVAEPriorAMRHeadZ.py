import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.deep.networks import ConditionalPrior
from models.deep.MultiVAEAMRHeadZ import MultiVAE_Bernoulli

from src.evaluation.metrics import compute_multilabel_auc, compute_per_antibiotic_auc


class MultiVAE_Bernoulli_SpeciesPrior_AMR_HeadZ(MultiVAE_Bernoulli):
    """
    Multi-decoder VAE with species-conditional latent prior and per-antibiotic AMR heads.

    Encoder:   q(z | x)
    Prior:     p(z | species) (or N(0, I) if `use_fixed_prior`)
    Decoder:   p(x | z, domain)
    AMR heads: one linear head per antibiotic, predicting resistance from z.
    """

    def __init__(self, input_dim, latent_dim, num_domains, n_species, n_antibiotics, lambda_amr, pos_weight=None, antibiotic_names=None, use_fixed_prior=False):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input data x.
        latent_dim : int
            Dimensionality of the latent space z.
        num_domains : int
            Number of domains (one decoder per domain).
        n_species : int
            Number of species (conditions the latent prior).
        n_antibiotics : int
            Number of antibiotics (one AMR head per antibiotic).
        lambda_amr : float
            Weight of the AMR loss term relative to the ELBO in `total_loss`.
        pos_weight : torch.Tensor, optional
            Positive-class weight(s) for the AMR binary cross-entropy loss.
            Either a scalar/length-1 tensor (shared across antibiotics) or
            one weight per antibiotic.
        antibiotic_names : list of str, optional
            Names of the antibiotics, in the same order as the AMR heads.
            Used for per-antibiotic AUC reporting.
        use_fixed_prior : bool, default=False
            If True, use a fixed N(0, I) prior on z instead of the
            species-conditional prior.
        """
        super().__init__(input_dim, latent_dim, num_domains)

        self.n_species = n_species
        self.prior = ConditionalPrior(n_species=n_species, latent_dim=latent_dim)
        self.lambda_amr = lambda_amr
        self.n_antibiotics = n_antibiotics
        self.pos_weight = pos_weight
        self.antibiotic_names = antibiotic_names
        self.use_fixed_prior  = use_fixed_prior
        self.n_amr_samples = 1

        if not use_fixed_prior:
            self.prior = ConditionalPrior(n_species=n_species, latent_dim=latent_dim)

        self.amr_heads = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in range(n_antibiotics)])

    def elbo_loss(self, x, mu, logvar, z, domain_id, species_id, beta=1.0):
        """
        Computes the ELBO loss against either a fixed or species-conditional prior.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        mu : torch.Tensor
            Mean of q(z | x), as returned by `forward`.
        logvar : torch.Tensor
            Log-variance of q(z | x), as returned by `forward`.
        z : torch.Tensor
            Sampled latent representation, as returned by `forward`.
        domain_id : torch.Tensor
            Domain identifiers selecting the decoder used for the
            reconstruction term.
        species_id : torch.Tensor
            Per-sample species identifiers used to condition the prior
            p(z | species). Unused if `self.use_fixed_prior` is True.
        beta : float, default=1.0
            Weight applied to the KL divergence term (beta-VAE annealing).

        Returns
        -------
        loss : torch.Tensor
            Mean negative ELBO (NLL) over the batch.
        recon_loss : torch.Tensor
            Mean reconstruction loss (negative log-likelihood) over the
            batch.
        kl_loss : torch.Tensor
            Mean KL divergence between q(z | x) and the prior over the
            batch.
        """
        # Log-likelihood p(x|z,d)
        RE = self.decoder.log_prob(x, z, domain_id)

        if self.use_fixed_prior:
            # KL contra N(0,1) estándar
            KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        else:
            # Build species one-hot
            species_onehot = torch.nn.functional.one_hot(species_id, num_classes=self.n_species).float()

            # Prior p(z | species)
            mu_p, logvar_p = self.prior(species_onehot)

            # KL(q(z|x) || p(z|species))
            KL = -0.5 * torch.sum(1 + (logvar - logvar_p) - ((mu - mu_p) ** 2 + logvar.exp()) / logvar_p.exp(), dim=1)

        NLL = -(RE - beta * KL)
        return NLL.mean(), (-RE).mean(), KL.mean()

    def total_loss(self, x, mu, logvar, z, domain_id, species_id, amr_logits, amr_labels, beta=1.0):
        """
        Computes the ELBO loss plus an averaged multi-antibiotic AMR loss.

        The AMR loss is a mean binary cross-entropy over antibiotics
        (skipping antibiotics with no labeled samples in the batch), with
        labels smoothed by 0.9 * y + 0.05. If `self.n_amr_samples > 1`,
        it is averaged over that many z samples drawn from q(z | x)
        (`z` is used for the first sample, fresh reparameterized samples
        for the rest). The final loss is
        `elbo_loss + self.lambda_amr * amr_loss`.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        mu : torch.Tensor
            Mean of q(z | x), as returned by `forward`.
        logvar : torch.Tensor
            Log-variance of q(z | x), as returned by `forward`.
        z : torch.Tensor
            Sampled latent representation, as returned by `forward`.
        domain_id : torch.Tensor
            Domain identifiers selecting the decoder used for the
            reconstruction term.
        species_id : torch.Tensor
            Per-sample species identifiers used to condition the prior.
        amr_logits : torch.Tensor or None
            AMR prediction logits, shape (batch_size, n_antibiotics). If
            None (together with `amr_labels`), the AMR loss is skipped.
        amr_labels : torch.Tensor or None
            AMR ground-truth labels, shape (batch_size, n_antibiotics) or
            (batch_size,) for a single antibiotic. May contain NaN for
            missing labels, which are excluded from the loss.
        beta : float, default=1.0
            Weight applied to the KL divergence term (beta-VAE annealing).

        Returns
        -------
        loss : torch.Tensor
            Total loss: ELBO loss plus the weighted AMR loss.
        recon_loss : torch.Tensor
            Mean reconstruction loss over the batch.
        kl_loss : torch.Tensor
            Mean KL divergence over the batch.
        amr_loss : torch.Tensor
            Mean multi-antibiotic AMR loss (0.0 if no AMR labels were
            available).
        per_antibiotic_losses : dict of int -> float
            BCE loss per antibiotic index (from the first z sample), NaN
            for antibiotics with no labeled samples in the batch.
        per_antibiotic_counts : dict of int -> int
            Number of labeled samples per antibiotic index in the batch.
        """
        loss, recon, kl = self.elbo_loss(x, mu, logvar, z, domain_id=domain_id, species_id=species_id, beta=beta)

        amr_loss = torch.tensor(0.0, device=x.device)
        per_antibiotic_losses = {}
        per_antibiotic_counts = {}

        if amr_logits is not None and amr_labels is not None:
            if amr_labels.ndim == 1:
                amr_labels = amr_labels.view(-1, 1)

            std = torch.exp(0.5 * logvar)
            sample_losses = []

            for k in range(self.n_amr_samples):
                z_k = z if k == 0 else (mu + std * torch.randn_like(std))
                logits_k = torch.cat([head(z_k) for head in self.amr_heads], dim=1)

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

                    if self.pos_weight is not None:
                        pos_weight_j = self.pos_weight.to(x.device) if (self.pos_weight.ndim == 0 or len(self.pos_weight) == 1) else self.pos_weight[j].to(x.device)
                    else:
                        pos_weight_j = None

                    smoothed_labels = labels_j * 0.9 + 0.05
                    loss_j = F.binary_cross_entropy_with_logits(logits_j, smoothed_labels, pos_weight=pos_weight_j, reduction="mean")
                    task_loss_k += loss_j
                    n_tasks_k   += 1
                    ab_losses_k[j] = loss_j.item()
                    ab_counts_k[j] = int(mask_j.sum().item())

                if n_tasks_k > 0:
                    sample_losses.append(task_loss_k / n_tasks_k)

                if k == 0:
                    per_antibiotic_losses = ab_losses_k
                    per_antibiotic_counts = ab_counts_k

            if sample_losses:
                amr_loss = torch.stack(sample_losses).mean()
                loss = loss + self.lambda_amr * amr_loss

        return loss, recon, kl, amr_loss, per_antibiotic_losses, per_antibiotic_counts


class MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ(MultiVAE_Bernoulli_SpeciesPrior_AMR_HeadZ):
    """
    Extended AMR-head VAE with optimizer, LR scheduler, training loop and early stopping.

    Early stopping and LR scheduling are both driven by validation AUC
    (`val_auc`) rather than validation loss.
    """

    def __init__(self, input_dim, latent_dim, num_domains, n_species, n_antibiotics, lambda_amr=0.1, pos_weight=None, antibiotic_names=None, epochs=100, lr=1e-4, annealing_epochs=50, patience=20, use_fixed_prior=False):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input data x.
        latent_dim : int
            Dimensionality of the latent space z.
        num_domains : int
            Number of domains (one decoder per domain).
        n_species : int
            Number of species (conditions the latent prior).
        n_antibiotics : int
            Number of antibiotics (one AMR head per antibiotic).
        lambda_amr : float, default=0.1
            Weight of the AMR loss term relative to the ELBO in `total_loss`.
        pos_weight : torch.Tensor, optional
            Positive-class weight(s) for the AMR binary cross-entropy loss.
        antibiotic_names : list of str, optional
            Names of the antibiotics, used for per-antibiotic AUC reporting.
        epochs : int, default=100
            Maximum number of training epochs.
        lr : float, default=1e-4
            Learning rate for the Adam optimizer.
        annealing_epochs : int or None, default=50
            Number of epochs over which beta is linearly annealed up to
            0.1. If None, beta is fixed at 1.0 (no annealing).
        patience : int, default=20
            Number of epochs without validation-AUC improvement before
            early stopping is triggered.
        use_fixed_prior : bool, default=False
            If True, use a fixed N(0, I) prior on z instead of the
            species-conditional prior.
        """
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, num_domains=num_domains, n_species=n_species, n_antibiotics=n_antibiotics, lambda_amr=lambda_amr, pos_weight=pos_weight, antibiotic_names=antibiotic_names, use_fixed_prior=use_fixed_prior)

        self.epochs = epochs
        self.lr = lr
        self.annealing_epochs = annealing_epochs
        self.patience = patience

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler_mode = 'min' if self.lambda_amr == 0 else 'max'
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=scheduler_mode, factor=0.5, patience=20, min_lr=1e-6, cooldown=10)

        self.loss_during_training = []
        self.reconstruc_during_training = []
        self.KL_during_training = []
        self.AMR_during_training = []
        self.AUC_during_training = []

    def trainloop(self, trainloader, validloader, device):
        """
        Trains the model with beta-annealing, AMR supervision and AUC-based early stopping.

        Runs up to `self.epochs` epochs, evaluating on `validloader` after
        each one. Beta is linearly annealed (if `self.annealing_epochs`
        is set). The LR scheduler and early stopping both track
        multilabel AMR AUC (higher is better) rather than validation
        loss, and the model reverts to the best-validation-AUC weights
        found if early stopping triggers before the last epoch. Per-epoch
        (train, val) tuples for total loss, reconstruction loss, KL, AMR
        loss and AUC are appended to `self.loss_during_training`,
        `self.reconstruc_during_training`, `self.KL_during_training`,
        `self.AMR_during_training` and `self.AUC_during_training`
        respectively.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            Yields (x, domain_id, species_id, amr_labels) batches.
        validloader : torch.utils.data.DataLoader
            Same batch format as `trainloader`, used for validation,
            LR scheduling and early stopping.
        device : torch.device or str
            Device to move the model and batches to.

        Returns
        -------
        None
            The model is updated in place; on early stopping it is
            reloaded with the best validation-AUC weights found.
        """
        self.to(device)

        best_val_loss = float("inf")
        best_val_auc = -float("inf")
        patience_counter = 0
        best_state = None

        prior_mode = "N(0,1) fixed" if self.use_fixed_prior else "learned prior"
        anneal_mode = f"annealing ({self.annealing_epochs} epochs)" if self.annealing_epochs is not None else "no annealing (beta=1)"
        print(f"Training mode: VAE | Prior: {prior_mode} | {anneal_mode}")

        for epoch in range(self.epochs):
            if self.annealing_epochs is None:
                beta = 1.0
            else:
                beta = min(0.1, (epoch + 1) / self.annealing_epochs)

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
                loss, recon, kl, amr, batch_ab_losses, batch_ab_counts = self.total_loss(x, mu, logvar, z, domain_id=domain_id, species_id=species_id, amr_logits=amr_logits, amr_labels=amr_labels, beta=beta)

                if loss.requires_grad:
                    loss.backward()
                    self.optimizer.step()
                else:
                    pass

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
            tr_auc_per_ab = compute_per_antibiotic_auc(tr_logits_all, tr_labels_all, antibiotic_names=self.antibiotic_names)

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
                    loss, recon, kl, amr, batch_ab_losses, batch_ab_counts = self.total_loss(x, mu, logvar, z, domain_id=domain_id, species_id=species_id, amr_logits=amr_logits, amr_labels=amr_labels, beta=beta)

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
            val_auc_per_ab = compute_per_antibiotic_auc(val_logits_all, val_labels_all, antibiotic_names=self.antibiotic_names)

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

            self.scheduler.step(val_loss if self.lambda_amr == 0 else val_auc)
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
            if self.lambda_amr == 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
        if best_state is not None:
            self.load_state_dict(best_state)
