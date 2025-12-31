import torch
import torch.nn as nn
import torch.optim as optim
from models.deep.MultiVAE import MultiVAE_Bernoulli


class MultiVAE_CORAL(MultiVAE_Bernoulli):
    """
    Multi-decoder VAE with CORAL regularization in the latent space.

    Extends MultiVAE_Bernoulli by adding a CORAL loss term that aligns
    the second-order statistics of latent representations across domains.
    """

    def __init__(self, input_dim, latent_dim, num_domains, epochs=100, lr=1e-4, annealing_epochs=50, patience=20):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input data.
        latent_dim : int
            Dimensionality of the latent space.
        num_domains : int
            Number of domains (one decoder per domain).
        epochs : int, optional
            Number of training epochs.
        lr : float, optional
            Learning rate.
        annealing_epochs : int, optional
            Number of epochs for KL annealing.
        patience : int, optional
            Patience for early stopping.
        """
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
        self.coral_loss_during_training = []

    def compute_coral(self, z_a, z_b):
        """
        Computes the CORAL loss between two sets of latent representations.

        CORAL aligns covariance matrices between domains.
        """
        z_a = z_a - z_a.mean(0)
        z_b = z_b - z_b.mean(0)

        cov_a = (z_a.T @ z_a) / (z_a.shape[0] - 1)
        cov_b = (z_b.T @ z_b) / (z_b.shape[0] - 1)

        coral_loss = torch.mean((cov_a - cov_b) ** 2)
        coral_loss = coral_loss / (4 * (z_a.shape[1] ** 2))
        return coral_loss

    def trainloop(self, trainloader, validloader, device):
        """
        Training loop with CORAL regularization, KL annealing,
        validation and early stopping.
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
            train_coral_total = 0.0
            train_total_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0

            for x, domain_id in trainloader:
                x, domain_id = x.to(device), domain_id.to(device)
                self.optimizer.zero_grad()

                mu, logvar, z = self.forward(x, domain_id)
                loss, recon, kl = self.elbo_loss(x, mu, logvar, z, domain_id, beta)

                mask_A = domain_id == 0
                mask_D = domain_id == 1

                if mask_A.any() and mask_D.any():
                    coral = self.compute_coral(z[mask_A], z[mask_D])
                    lambda_coral = 1e-3
                    loss = loss + lambda_coral * coral
                    train_coral_total += coral.item()

                loss.backward()
                self.optimizer.step()

                train_total_loss += loss.item()
                train_recon_loss += recon.item()
                train_kl_loss += kl.item()

            train_coral_avg = train_coral_total / len(trainloader)
            train_total_loss /= len(trainloader)
            train_recon_loss /= len(trainloader)
            train_kl_loss /= len(trainloader)

            # =======================
            #      VALIDATION
            # =======================
            self.eval()
            val_total_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_coral_total = 0.0

            with torch.no_grad():
                for x, domain_id in validloader:
                    x, domain_id = x.to(device), domain_id.to(device)

                    mu, logvar, z = self.forward(x, domain_id)
                    loss, recon, kl = self.elbo_loss(x, mu, logvar, z, domain_id, beta)

                    mask_A = domain_id == 0
                    mask_D = domain_id == 1
                    if mask_A.any() and mask_D.any():
                        coral = self.compute_coral(z[mask_A], z[mask_D])
                        lambda_coral = 5e-3
                        loss = loss + lambda_coral * coral
                        val_coral_total += coral.item()

                    val_total_loss += loss.item()
                    val_recon_loss += recon.item()
                    val_kl_loss += kl.item()

            val_total_loss /= len(validloader)
            val_recon_loss /= len(validloader)
            val_kl_loss /= len(validloader)
            val_coral_avg = val_coral_total / len(validloader)

            self.loss_during_training.append((train_total_loss, val_total_loss))
            self.reconstruc_during_training.append((train_recon_loss, val_recon_loss))
            self.KL_during_training.append((train_kl_loss, val_kl_loss))
            self.coral_loss_during_training.append((train_coral_avg, val_coral_avg))

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"[Train] Loss: {train_total_loss:.4f} | Recon: {train_recon_loss:.4f} | "
                    f"KL: {train_kl_loss:.4f} | CORAL: {train_coral_avg:.6f} || "
                    f"[Val] Loss: {val_total_loss:.4f} | Recon: {val_recon_loss:.4f} | "
                    f"KL: {val_kl_loss:.4f} | CORAL: {val_coral_avg:.6f}"
                )

            # =======================
            #     EARLY STOPPING
            # =======================
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                best_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.load_state_dict(best_state)
