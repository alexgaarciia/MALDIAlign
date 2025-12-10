import torch
import torch.nn as nn
import torch.optim as optim
from models.deep.networks import Encoder, ConditionalDecoder


class InvariantCVAE_Bernoulli(nn.Module):
    def __init__(self, input_dim, latent_dim, cond_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, input_dim, cond_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def elbo_loss(self, x, mu, logvar, z, c, beta=1.0):
        RE = self.decoder.log_prob(x, z, c)
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        NLL = -(RE - beta * KL)
        return NLL.mean(), (-RE).mean(), KL.mean()
    

class InvariantCVAE_Bernoulli_Extended(InvariantCVAE_Bernoulli):
    def __init__(self, input_dim, latent_dim, cond_dim, lr=1e-4, epochs=100,
                 patience=20, annealing_epochs=50):
        super().__init__(input_dim, latent_dim, cond_dim)
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.annealing_epochs = annealing_epochs
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

        self.loss_during_training = []
        self.reconstruc_during_training = []
        self.KL_during_training = []

    def trainloop(self, trainloader, validloader, device):
        self.to(device)
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            beta = min(1.0, (epoch + 1) / self.annealing_epochs)
            self.train()
            total_loss, total_recon, total_kl = 0, 0, 0

            for x, domain_id in trainloader:
                x, domain_id = x.to(device), domain_id.to(device)
                c = nn.functional.one_hot(domain_id, num_classes=2).float().to(device)

                self.optimizer.zero_grad()
                mu, logvar, z = self.forward(x, c)
                loss, recon, kl = self.elbo_loss(x, mu, logvar, z, c, beta)
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
            val_loss, val_recon, val_kl = 0, 0, 0
            with torch.no_grad():
                for x, domain_id in validloader:
                    x, domain_id = x.to(device), domain_id.to(device)
                    c = nn.functional.one_hot(domain_id, num_classes=2).float().to(device)
                    mu, logvar, z = self.forward(x, c)
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
                    f"[Train] Loss: {train_loss:.4f} | Recon: {train_recon:.4f} | KL: {train_kl:.4f} || "
                    f"[Val] Loss: {val_loss:.4f} | Recon: {val_recon:.4f} | KL: {val_kl:.4f}"
                )

            # Early stopping
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
            