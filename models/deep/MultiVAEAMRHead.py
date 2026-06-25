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
            Input data.
        domain_id : torch.Tensor
            Domain identifiers selecting the decoder.

        Returns
        -------
        mu : torch.Tensor
            Mean of q(z | x).
        logvar : torch.Tensor
            Log-variance of q(z | x).
        z : torch.Tensor
            Sampled latent representation.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        x_recon = torch.zeros_like(x)
        for d in range(self.decoder.num_domains):
            mask = domain_id == d
            if mask.any():
                x_recon[mask] = self.decoder.net[d](z[mask])
        
        # AMR predicion
        amr_logits = None
        if hasattr(self, "amr_heads"):
            h = self.amr_trunk(mu)
            amr_logits = torch.cat([head(h) for head in self.amr_heads], dim=1)

        return mu, logvar, z, amr_logits

    def elbo_loss(self, x, mu, logvar, z, domain_id, species_id=None, species_weights=None, beta=1.0):
        """
        Computes the ELBO loss with optional species-dependent weighting.
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
    Extended MultiVAE including optimizer, training and validation loops,
    KL annealing, early stopping, and loss tracking.
    """

    def __init__(self, input_dim, latent_dim, num_domains, epochs=100, lr=1e-4, annealing_epochs=50, patience=20):
        super().__init__(input_dim, latent_dim, num_domains)
        self.epochs = epochs
        self.lr = lr
        self.annealing_epochs = annealing_epochs
        self.patience = patience

        self.optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=1e-5
        )

        self.loss_during_training = []
        self.reconstruc_during_training = []
        self.KL_during_training = []

    def trainloop(self, trainloader, validloader, device, species_weights=None):
        """
        Training loop with validation, KL annealing and early stopping.

        Supports both (x, domain_id) and (x, domain_id, species_id) batches.
        """
        self.to(device)
        species_weight_tensor = species_weights.to(device) if species_weights is not None else None

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            beta = min(1.0, (epoch + 1) / self.annealing_epochs)

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
                mu, logvar, z = self.forward(x, domain_id)
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

                    mu, logvar, z = self.forward(x, domain_id)
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
