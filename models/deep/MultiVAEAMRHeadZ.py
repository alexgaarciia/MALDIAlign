import torch
import torch.nn as nn
import torch.optim as optim
from models.deep.networks import Encoder, BernoulliDecoder


class MultiVAE_Bernoulli(nn.Module):
    """
    Multi-decoder Variational Autoencoder with a Bernoulli likelihood.

    A single encoder learns q(z | x), while the decoder consists of multiple
    domain-specific Bernoulli decoders p(x | z, domain).
    """

    def __init__(self, input_dim, latent_dim, num_domains):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input data x.
        latent_dim : int
            Dimensionality of the latent space z.
        num_domains : int
            Number of domains (one decoder per domain).
        """
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = BernoulliDecoder(latent_dim, input_dim, num_domains)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample z ~ q(z | x).

        Parameters
        ----------
        mu : torch.Tensor
            Mean of q(z | x).
        logvar : torch.Tensor
            Log-variance of q(z | x).

        Returns
        -------
        torch.Tensor
            Sampled latent representation z = mu + eps * std, with
            eps ~ N(0, I).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, domain_id):
        """
        Forward pass of the MultiVAE.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch_size, input_dim).
        domain_id : torch.Tensor
            Domain identifiers selecting the decoder for each sample,
            shape (batch_size,).

        Returns
        -------
        mu : torch.Tensor
            Mean of q(z | x).
        logvar : torch.Tensor
            Log-variance of q(z | x).
        z : torch.Tensor
            Sampled latent representation.
        amr_logits : torch.Tensor or None
            AMR prediction logits obtained by running `z` through
            `self.amr_heads`, one per head, concatenated along dim=1.
            None if the model has no `amr_heads` attribute attached.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        x_recon = torch.zeros_like(x)
        for d in range(self.decoder.num_domains):
            mask = domain_id == d
            if mask.any():
                x_recon[mask] = self.decoder.net[d](z[mask])
        
        amr_logits = None
        if hasattr(self, "amr_heads"):
            amr_logits = torch.cat([head(z) for head in self.amr_heads], dim=1)

        return mu, logvar, z, amr_logits

    def elbo_loss(self, x, mu, logvar, z, domain_id, species_id=None, species_weights=None, beta=1.0):
        """
        Computes the (negative) ELBO loss with optional species-dependent weighting.

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
        species_id : torch.Tensor, optional
            Per-sample species identifiers used to index into
            `species_weights`. If None, no per-species weighting is applied.
        species_weights : torch.Tensor, optional
            Per-species loss weights, indexed by `species_id`. Ignored if
            `species_id` is None.
        beta : float, default=1.0
            Weight applied to the KL divergence term (beta-VAE annealing).

        Returns
        -------
        loss : torch.Tensor
            Mean negative ELBO (NLL) over the batch, optionally weighted
            by species.
        recon_loss : torch.Tensor
            Mean reconstruction loss (negative log-likelihood) over the
            batch.
        kl_loss : torch.Tensor
            Mean KL divergence between q(z | x) and the prior over the
            batch.
        """
        RE = self.decoder.log_prob(x, z, domain_id)
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        NLL_per_sample = -(RE - beta * KL)

        if species_id is not None and species_weights is not None:
            weights = species_weights[species_id]
            NLL_per_sample = NLL_per_sample * weights

        return NLL_per_sample.mean(), (-RE).mean(), KL.mean()


class MultiVAE_Bernoulli_Extended(MultiVAE_Bernoulli):
    """
    MultiVAE_Bernoulli extended with a training loop.

    Adds an Adam optimizer, KL-annealing, early stopping on validation
    loss, and per-epoch training/validation metric tracking on top of
    the base MultiVAE_Bernoulli model.
    """

    def __init__(self, input_dim, latent_dim, num_domains, epochs=100, lr=1e-4, annealing_epochs=50, patience=20):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input data x.
        latent_dim : int
            Dimensionality of the latent space z.
        num_domains : int
            Number of domains (one decoder per domain).
        epochs : int, default=100
            Maximum number of training epochs.
        lr : float, default=1e-4
            Learning rate for the Adam optimizer.
        annealing_epochs : int or None, default=50
            Number of epochs over which beta is linearly annealed up to
            0.1. If None, beta is fixed at 1.0 (no annealing).
        patience : int, default=20
            Number of epochs without validation-loss improvement before
            early stopping is triggered.
        """
        super().__init__(input_dim, latent_dim, num_domains)
        self.epochs = epochs
        self.lr = lr
        self.annealing_epochs = annealing_epochs
        self.patience = patience

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

        self.loss_during_training = []
        self.reconstruc_during_training = []
        self.KL_during_training = []

    def trainloop(self, trainloader, validloader, device, species_weights=None):
        """
        Trains the model with beta-annealing and early stopping.

        Runs up to `self.epochs` epochs, evaluating on `validloader` after
        each one. Beta is linearly annealed (if `self.annealing_epochs`
        is set) and the model reverts to the best-validation-loss weights
        found if early stopping triggers before the last epoch. Per-epoch
        (train, val) tuples for total loss, reconstruction loss, and KL
        are appended to `self.loss_during_training`,
        `self.reconstruc_during_training`, and `self.KL_during_training`
        respectively.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            Yields batches of either (x, domain_id) or
            (x, domain_id, species_id).
        validloader : torch.utils.data.DataLoader
            Same batch format as `trainloader`, used for validation and
            early stopping.
        device : torch.device or str
            Device to move the model and batches to.
        species_weights : torch.Tensor, optional
            Per-species loss weights indexed by species_id, passed to
            `elbo_loss`. Ignored if batches don't include species_id.

        Returns
        -------
        None
            The model is updated in place; on early stopping it is
            reloaded with the best validation-loss weights found.
        """
        self.to(device)
        species_weight_tensor = species_weights.to(device) if species_weights is not None else None

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            if self.annealing_epochs is None:
                beta = 1.0
            else:
                beta = min(0.1, (epoch + 1) / self.annealing_epochs)

            # =======================
            #        TRAIN
            # =======================
            self.train()
            train_total_loss, train_recon_loss, train_kl_loss = 0, 0, 0

            for batch in trainloader:
                if len(batch) == 2:
                    x, domain_id = batch
                    species_id = None
                else:
                    x, domain_id, species_id = batch

                x, domain_id = x.to(device), domain_id.to(device)
                if species_id is not None:
                    species_id = species_id.to(device)

                self.optimizer.zero_grad()
                mu, logvar, z, _ = self.forward(x, domain_id)
                loss, recon, kl = self.elbo_loss(
                    x, mu, logvar, z,
                    domain_id,
                    beta=beta,
                    species_id=species_id,
                    species_weights=species_weight_tensor,
                )
                loss.backward()
                self.optimizer.step()

                train_total_loss += loss.item()
                train_recon_loss += recon.item()
                train_kl_loss += kl.item()

            train_total_loss /= len(trainloader)
            train_recon_loss /= len(trainloader)
            train_kl_loss /= len(trainloader)

            # =======================
            #      VALIDATION
            # =======================
            self.eval()
            val_loss, val_recon, val_kl = 0.0, 0.0, 0.0

            with torch.no_grad():
                for batch in validloader:
                    if len(batch) == 2:
                        x, domain_id = batch
                        species_id = None
                    else:
                        x, domain_id, species_id = batch

                    x, domain_id = x.to(device), domain_id.to(device)
                    if species_id is not None:
                        species_id = species_id.to(device)

                    mu, logvar, z, _ = self.forward(x, domain_id)
                    loss, recon, kl = self.elbo_loss(
                        x, mu, logvar, z,
                        domain_id,
                        beta=beta,
                        species_id=species_id,
                        species_weights=species_weight_tensor,
                    )
                    val_loss += loss.item()
                    val_recon += recon.item()
                    val_kl += kl.item()

            val_loss /= len(validloader)
            val_recon /= len(validloader)
            val_kl /= len(validloader)

            self.loss_during_training.append((train_total_loss, val_loss))
            self.reconstruc_during_training.append((train_recon_loss, val_recon))
            self.KL_during_training.append((train_kl_loss, val_kl))

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"[Train] Loss: {train_total_loss:.4f} | Recon: {train_recon_loss:.4f} | KL: {train_kl_loss:.4f} || "
                    f"[Val] Loss: {val_loss:.4f} | Recon: {val_recon:.4f} | KL: {val_kl:.4f}"
                )

            # =======================
            #     EARLY STOPPING
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
