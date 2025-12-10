import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        hidden_rep = self.net(x)
        mu = self.mu(hidden_rep)
        logvar = self.logvar(hidden_rep)
        logvar = torch.clamp(logvar, -6, 6) # Clamp logvar to [-6, 6] to avoid extreme variances and stabilize training
        return mu, logvar


class BernoulliDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_domains=1):
        super().__init__()
        self.num_domains = num_domains

        self.net = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, output_dim),
                    nn.Sigmoid()
                )
                for _ in range(num_domains)
            ]
        )

    def forward(self, z, domain_id=None):
        if self.num_domains == 1:
            return self.net[0](z)

        if domain_id is None:
            raise ValueError("domain_id must be provided when num_domains > 1")

        x_recon = torch.zeros(z.shape[0], self.net[0][-2].out_features, device=z.device)
        for d in range(self.num_domains):
            mask = (domain_id == d)
            if mask.any():
                x_recon[mask] = self.net[d](z[mask])
        return x_recon
        
    def log_prob(self, x, z, domain_id=None):
        theta = self.forward(z, domain_id)

        if torch.any(theta < 0) or torch.any(theta > 1) or torch.isnan(theta).any():
            raise ValueError(f"[ERROR] theta out of bounds: min={theta.min()}, max={theta.max()}")
        if torch.any(x < 0) or torch.any(x > 1) or torch.isnan(x).any():
            raise ValueError("Input x must be in [0,1] for Bernoulli log_prob computation.")

        log_prob = -F.binary_cross_entropy(theta, x, reduction='none').sum(dim=1)
        return log_prob


class ConditionalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, cond_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1028),
            nn.LeakyReLU(),
            nn.Linear(1028, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x, c):
        h = torch.cat([x, c], dim=1)
        hidden_rep = self.net(h)
        mu = self.mu(hidden_rep)
        logvar = self.logvar(hidden_rep)
        logvar = torch.clamp(logvar, -6, 6)
        return mu, logvar


class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, cond_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, output_dim),
        )

    def forward(self, z, c):
        h = torch.cat([z, c], dim=1)
        x_recon = self.net(h)
        return x_recon
