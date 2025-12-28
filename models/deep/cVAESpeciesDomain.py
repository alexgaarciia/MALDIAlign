import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SpeciesConditionalEncoder(nn.Module):
    """
    q(z | x, species)
    """
    def __init__(self, input_dim, latent_dim, species_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim + species_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x, species_onehot):
        h = torch.cat([x, species_onehot], dim=1)
        h = self.net(h)

        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -6, 6)

        return mu, logvar

class SpeciesDomainConditionalDecoder(nn.Module):
    """
    p(x | z, species, domain)
    """
    def __init__(self, latent_dim, output_dim, species_dim, domain_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim + species_dim + domain_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z, species_onehot, domain_onehot):
        h = torch.cat([z, species_onehot, domain_onehot], dim=1)
        return self.net(h)

    def log_prob(self, x, z, species_onehot, domain_onehot):
        theta = self.forward(z, species_onehot, domain_onehot)

        if torch.any(theta < 0) or torch.any(theta > 1) or torch.isnan(theta).any():
            raise ValueError("Decoder output out of [0,1]")

        if torch.any(x < 0) or torch.any(x > 1) or torch.isnan(x).any():
            raise ValueError("Input x must be in [0,1] for Bernoulli decoder")

        log_prob = -F.binary_cross_entropy(theta, x, reduction="none").sum(dim=1)
        return log_prob

class CVAE_SpeciesDomain_Bernoulli(nn.Module):
    """
    Encoder: q(z | x, species)
    Decoder: p(x | z, species, domain)
    """
    def __init__(
        self,
        input_dim,
        latent_dim,
        species_dim,
        domain_dim
    ):
        super().__init__()

        self.encoder = SpeciesConditionalEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            species_dim=species_dim
        )

        self.decoder = SpeciesDomainConditionalDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            species_dim=species_dim,
            domain_dim=domain_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, species_onehot):
        mu, logvar = self.encoder(x, species_onehot)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def elbo_loss(
        self,
        x,
        mu,
        logvar,
        z,
        species_onehot,
        domain_onehot,
        beta=1.0
    ):
        RE = self.decoder.log_prob(
            x, z, species_onehot, domain_onehot
        )

        KL = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1
        )

        NLL = -(RE - beta * KL)
        return NLL.mean(), (-RE).mean(), KL.mean()

class CVAE_SpeciesDomain_Bernoulli_Extended(CVAE_SpeciesDomain_Bernoulli):
    def __init__(
        self,
        input_dim,
        latent_dim,
        species_dim,
        domain_dim,
        lr=1e-4,
        epochs=100,
        patience=20,
        annealing_epochs=50
    ):
        super().__init__(
            input_dim,
            latent_dim,
            species_dim,
            domain_dim
        )

        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.annealing_epochs = annealing_epochs

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-5
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

            # ================= TRAIN =================
            self.train()
            total_loss, total_recon, total_kl = 0, 0, 0

            for x, species_id, domain_id in trainloader:
                x = x.to(device)
                species_id = species_id.to(device)
                domain_id = domain_id.to(device)

                species_onehot = F.one_hot(
                    species_id, num_classes=self.encoder.net[0].in_features - x.shape[1]
                ).float()

                domain_onehot = F.one_hot(
                    domain_id, num_classes=self.decoder.net[0].in_features -
                    self.encoder.mu.out_features - species_onehot.shape[1]
                ).float()

                species_onehot = species_onehot.to(device)
                domain_onehot = domain_onehot.to(device)

                self.optimizer.zero_grad()

                mu, logvar, z = self.forward(x, species_onehot)
                loss, recon, kl = self.elbo_loss(
                    x, mu, logvar, z,
                    species_onehot,
                    domain_onehot,
                    beta
                )

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()

            train_loss = total_loss / len(trainloader)
            train_recon = total_recon / len(trainloader)
            train_kl = total_kl / len(trainloader)

            # ================= VALID =================
            self.eval()
            val_loss, val_recon, val_kl = 0, 0, 0

            with torch.no_grad():
                for x, species_id, domain_id in validloader:
                    x = x.to(device)
                    species_id = species_id.to(device)
                    domain_id = domain_id.to(device)

                    species_onehot = F.one_hot(
                        species_id, num_classes=species_onehot.shape[1]
                    ).float().to(device)

                    domain_onehot = F.one_hot(
                        domain_id, num_classes=domain_onehot.shape[1]
                    ).float().to(device)

                    mu, logvar, z = self.forward(x, species_onehot)
                    loss, recon, kl = self.elbo_loss(
                        x, mu, logvar, z,
                        species_onehot,
                        domain_onehot,
                        beta
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
