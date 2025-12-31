import torch
import torch.nn as nn
import torch.optim as optim
from models.deep.networks import Encoder, BernoulliDecoder


class VAE_Bernoulli(nn.Module):
    """
    Standard Variational Autoencoder (VAE) with a Bernoulli likelihood.

    The encoder learns q(z | x) and the decoder models p(x | z),
    optionally conditioned on a domain identifier.
    """

    def __init__(self, input_dim, latent_dim, num_domains=1):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input data x.
        latent_dim : int
            Dimensionality of the latent space z.
        num_domains : int, optional
            Number of domains for optional domain conditioning in the decoder.
        """
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = BernoulliDecoder(latent_dim, input_dim, num_domains=num_domains)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample z ~ q(z | x).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, domain_id=None):
        """
        Forward pass of the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        domain_id : torch.Tensor or None, optional
            Domain identifiers used by the decoder.

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
        _ = self.decoder(z, domain_id=domain_id)
        return mu, logvar, z

    def elbo_loss(self, x, mu, logvar, z, beta=1.0, domain_id=None):
        """
        Computes the ELBO loss.

        Parameters
        ----------
        beta : float, optional
            Weight of the KL divergence term.
        """
        RE = self.decoder.log_prob(x, z, domain_id=domain_id)
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        NLL = -(RE - beta * KL)
        return NLL.mean(), (-RE).mean(), KL.mean()


class VAE_Bernoulli_Extended(VAE_Bernoulli):
    """
    Extended VAE including optimizer, training and validation loops,
    KL annealing, early stopping and loss tracking.
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        num_domains=1,
        lr=1e-3,
        epochs=100,
        patience=10,
        annealing_epochs=50,
    ):
        super().__init__(input_dim, latent_dim, num_domains=num_domains)
        self.lr = lr
        self.epochs = epochs
        self.annealing_epochs = annealing_epochs
        self.patience = patience

        self.optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=1e-5
        )

        self.loss_during_training = []
        self.reconstruc_during_training = []
        self.KL_during_training = []

    def trainloop(self, trainloader, validloader, device):
        """
        Training loop with validation, KL annealing and early stopping.
        """
        self.to(device)
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            beta = min(1.0, (epoch + 1) / self.annealing_epochs)

            # =======================
            #        TRAIN
            # =======================
            self.train()
            total_loss, total_recon, total_kl = 0, 0, 0

            for x, domains in trainloader:
                x, domains = x.to(device), domains.to(device)
                self.optimizer.zero_grad()
                mu, logvar, z = self.forward(x, domain_id=domains)
                loss, recon, kl = self.elbo_loss(
                    x, mu, logvar, z, beta, domain_id=domains
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()

            train_loss = total_loss / len(trainloader)
            train_recon = total_recon / len(trainloader)
            train_kl = total_kl / len(trainloader)

            # =======================
            #      VALIDATION
            # =======================
            self.eval()
            val_loss, val_recon, val_kl = 0.0, 0.0, 0.0

            with torch.no_grad():
                for x, domains in validloader:
                    x, domains = x.to(device), domains.to(device)
                    mu, logvar, z = self.forward(x, domain_id=domains)
                    loss, recon, kl = self.elbo_loss(
                        x, mu, logvar, z, beta, domain_id=domains
                    )
                    val_loss += loss.item()
                    val_recon += recon.item()
                    val_kl += kl.item()

            val_loss /= len(validloader)
            val_recon /= len(validloader)
            val_kl /= len(validloader)

            self.loss_during_training.append((train_loss, val_loss))
            self.reconstruc_during_training.append((train_recon, val_recon))
            self.KL_during_training.append((train_kl, val_kl))

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"[Train] Loss: {train_loss:.4f} | Recon: {train_recon:.4f} | KL: {train_kl:.4f} || "
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
