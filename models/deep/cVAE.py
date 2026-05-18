import torch
import torch.nn as nn
import torch.optim as optim
from models.deep.networks import ConditionalEncoder, ConditionalDecoder
    

class ConditionalVAE_Bernoulli(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) with a Bernoulli likelihood.

    This model learns a conditional latent representation z given an input x
    and a conditioning variable c (e.g., domain or hospital), following the
    standard VAE formulation with the reparameterization trick.

    The decoder is assumed to model p(x | z, c) as a Bernoulli distribution.
    """

    def __init__(self, input_dim, latent_dim, num_domains, emb_dim=8):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input data x.
        latent_dim : int
            Dimensionality of the latent space z.
        num_domains : int
            Dimensionality of the conditioning variable c
            (e.g., number of domains for one-hot encoding).
        """
        super().__init__()
        self.domain_emb = nn.Embedding(num_domains, emb_dim)
        self.encoder = ConditionalEncoder(input_dim, latent_dim, emb_dim)
        self.decoder = ConditionalDecoder(latent_dim, input_dim, emb_dim)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from q(z | x, c).

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the approximate posterior.
        logvar : torch.Tensor
            Log-variance of the approximate posterior.

        Returns
        -------
        z : torch.Tensor
            Sampled latent variable.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, domain_id):
        """
        Forward pass through the CVAE.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        c : torch.Tensor
            Conditioning variable (e.g., one-hot encoded domain).

        Returns
        -------
        mu : torch.Tensor
            Mean of q(z | x, c).
        logvar : torch.Tensor
            Log-variance of q(z | x, c).
        z : torch.Tensor
            Sampled latent representation.
        """
        c = self.domain_emb(domain_id)
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z, c

    def elbo_loss(self, x, mu, logvar, z, c, beta=1.0):
        """
        Computes the Evidence Lower Bound (ELBO) loss.

        ELBO = E_q[log p(x | z, c)] - beta * KL(q(z | x, c) || p(z))

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        mu : torch.Tensor
            Mean of the approximate posterior.
        logvar : torch.Tensor
            Log-variance of the approximate posterior.
        z : torch.Tensor
            Sampled latent variable.
        c : torch.Tensor
            Conditioning variable.
        beta : float, optional
            Weight for the KL divergence term (default: 1.0).

        Returns
        -------
        loss : torch.Tensor
            Mean negative ELBO.
        recon_loss : torch.Tensor
            Mean reconstruction loss (negative log-likelihood).
        kl_loss : torch.Tensor
            Mean KL divergence.
        """
        RE = self.decoder.log_prob(x, z, c)
        KL = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        )
        NLL = -(RE - beta * KL)
        return NLL.mean(), (-RE).mean(), KL.mean()


class ConditionalVAE_Bernoulli_Extended(ConditionalVAE_Bernoulli):
    def __init__(self, input_dim, latent_dim, num_domains, lr=1e-4, epochs=100, patience=20, annealing_epochs=50 ):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input data.
        latent_dim : int
            Dimensionality of the latent space.
        num_domains : int
            Dimensionality of the conditioning variable.
        lr : float, optional
            Learning rate for the optimizer (default: 1e-4).
        epochs : int, optional
            Maximum number of training epochs (default: 100).
        patience : int, optional
            Number of epochs without validation improvement before early stopping.
        annealing_epochs : int, optional
            Number of epochs over which beta is linearly annealed to 1.
        """
        super().__init__(input_dim, latent_dim, num_domains)

        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.annealing_epochs = annealing_epochs

        self.optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=1e-5
        )

        self.loss_during_training = []
        self.reconstruc_during_training = []
        self.KL_during_training = []

    def trainloop(self, trainloader, validloader, device):
        """
        Full training loop with validation, KL annealing and early stopping.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            DataLoader for the training set.
        validloader : torch.utils.data.DataLoader
            DataLoader for the validation set.
        device : torch.device
            Device on which to run the model (CPU or CUDA).
        """
        self.to(device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            beta = 1

            # =======================
            #        TRAIN
            # =======================
            self.train()
            total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

            for batch in trainloader:
                if len(batch) == 3:
                    x, domain_id, _ = batch
                else:
                    x, domain_id = batch

                x, domain_id = x.to(device), domain_id.to(device)

                self.optimizer.zero_grad()

                mu, logvar, z, c = self.forward(x, domain_id)
                loss, recon, kl = self.elbo_loss(x, mu, logvar, z, c, beta)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()

            train_loss = total_loss / len(trainloader)
            train_recon = total_recon / len(trainloader)
            train_kl = total_kl / len(trainloader)

            # =======================
            #       VALIDATION
            # =======================
            self.eval()
            val_loss, val_recon, val_kl = 0.0, 0.0, 0.0

            with torch.no_grad():
                for batch in validloader:
                    if len(batch) == 3:
                        x, domain_id, _ = batch
                    else:
                        x, domain_id = batch

                    x = x.to(device)
                    domain_id = domain_id.to(device)

                    mu, logvar, z, c = self.forward(x, domain_id)
                    loss, recon, kl = self.elbo_loss(x, mu, logvar, z, c, beta)

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
                    f"[Train] Loss={train_loss:.4f} | Recon={train_recon:.4f} | KL={train_kl:.4f} || "
                    f"[Val] Loss={val_loss:.4f} | Recon={val_recon:.4f} | KL={val_kl:.4f}"
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
