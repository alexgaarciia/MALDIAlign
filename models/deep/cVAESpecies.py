import torch
import torch.nn as nn
import torch.optim as optim
from models.deep.networks import ConditionalEncoder, ConditionalDecoder


class SpeciesCVAE_Bernoulli(nn.Module):
    def __init__(self, input_dim, latent_dim, n_species):
        super().__init__()
        self.n_species = n_species

        self.encoder = ConditionalEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            cond_dim=n_species
        )

        self.decoder = ConditionalDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            cond_dim=n_species
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, species_onehot):
        mu, logvar = self.encoder(x, species_onehot)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def elbo_loss(self, x, mu, logvar, z, species_onehot, beta=1.0):
        RE = self.decoder.log_prob(x, z, species_onehot)
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        NLL = -(RE - beta * KL)
        return NLL.mean(), (-RE).mean(), KL.mean()

class SpeciesCVAE_Bernoulli_Extended(SpeciesCVAE_Bernoulli):
    def __init__(
        self,
        input_dim,
        latent_dim,
        n_species,
        lr=1e-4,
        epochs=200,
        patience=40,
        annealing_epochs=100,
    ):
        super().__init__(input_dim, latent_dim, n_species)

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
        self.to(device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            beta = min(1.0, (epoch + 1) / self.annealing_epochs)

            # ---------- TRAIN ----------
            self.train()
            total_loss, total_recon, total_kl = 0, 0, 0

            for x, domain_id, species_id in trainloader:
                x = x.to(device)
                species_id = species_id.to(device)

                c = nn.functional.one_hot(
                    species_id, num_classes=self.n_species
                ).float()

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

            # ---------- VALID ----------
            self.eval()
            val_loss, val_recon, val_kl = 0, 0, 0

            with torch.no_grad():
                for x, domain_id, species_id in validloader:
                    x = x.to(device)
                    species_id = species_id.to(device)

                    c = nn.functional.one_hot(
                        species_id, num_classes=self.n_species
                    ).float()

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

            # ---------- EARLY STOP ----------
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

