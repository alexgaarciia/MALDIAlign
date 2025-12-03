import torch
import torch.nn as nn
import torch.optim as optim
from models.deep.networks import Encoder, Decoder


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


class VAE_Extended(VAE):
    def __init__(
        self,
        input_dim,
        latent_dim,
        lr=1e-3,
        epochs=100,
        annealing_epochs=50,
        patience=10,
    ):
        super().__init__(input_dim, latent_dim)

        self.lr = lr
        self.epochs = epochs
        self.annealing_epochs = annealing_epochs
        self.patience = patience

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

        self.loss_during_training = []
        self.reconstruc_during_training = []
        self.KL_during_training = []

    def loss_function(self, x, x_recon, mu, logvar, beta):
        recon_loss = self.criterion(x_recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def trainloop(self, trainloader, validloader, device):
        self.to(device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            beta = min(1.0, (epoch + 1) / self.annealing_epochs)

            # TRAIN
            self.train()
            total_loss, total_recon, total_kl = 0, 0, 0
            for x, _ in trainloader:
                x = x.to(device)
                self.optimizer.zero_grad()
                x_recon, mu, logvar = self.forward(x)
                loss, recon, kl = self.loss_function(x, x_recon, mu, logvar, beta=beta)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()

            train_loss = total_loss / len(trainloader)
            train_recon = total_recon / len(trainloader)
            train_kl = total_kl / len(trainloader)

            # VALIDATION
            self.eval()
            val_loss, val_recon, val_kl = 0.0, 0.0, 0.0

            with torch.no_grad():
                for x, _ in validloader:
                    x = x.to(device)
                    x_recon, mu, logvar = self.forward(x)
                    loss, recon, kl = self.loss_function(
                        x, x_recon, mu, logvar, beta=beta
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

            # EARLY STOPPING
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            self.load_state_dict(best_state)
