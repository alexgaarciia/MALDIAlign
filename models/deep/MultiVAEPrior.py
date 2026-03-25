import torch
import torch.optim as optim
from models.deep.networks import ConditionalPrior
from models.deep.MultiVAE import MultiVAE_Bernoulli


class MultiVAE_Bernoulli_SpeciesPrior(MultiVAE_Bernoulli):
    """
    Multi-decoder VAE with species-conditional latent prior.
    
    Encoder: q(z | x)
    Prior:   p(z | species)
    Decoder: p(x | z, domain)
    """

    def __init__(self, input_dim, latent_dim, num_domains, n_species):
        super().__init__(input_dim, latent_dim, num_domains)

        self.n_species = n_species
        self.prior = ConditionalPrior(
            n_species=n_species,
            latent_dim=latent_dim
        )

    def elbo_loss(self, x, mu, logvar, z, domain_id, species_id, beta=1.0):
        # Log-likelihood p(x|z,d)
        RE = self.decoder.log_prob(x, z, domain_id)

        # Build species one-hot
        species_onehot = torch.nn.functional.one_hot(
            species_id, num_classes=self.n_species
        ).float()

        # Prior p(z | species)
        mu_p, logvar_p = self.prior(species_onehot)

        # KL(q(z|x) || p(z|species))
        KL = -0.5 * torch.sum(
            1
            + (logvar - logvar_p)
            - ((mu - mu_p) ** 2 + logvar.exp()) / logvar_p.exp(),
            dim=1,
        )

        NLL = -(RE - beta * KL)
        return NLL.mean(), (-RE).mean(), KL.mean()

class MultiVAE_Bernoulli_SpeciesPrior_Extended(MultiVAE_Bernoulli_SpeciesPrior):
    def __init__(self, input_dim, latent_dim, num_domains, n_species, epochs=100, lr=1e-4, annealing_epochs=50, patience=20):
        super().__init__(input_dim, latent_dim, num_domains, n_species)

        self.epochs = epochs
        self.lr = lr
        self.annealing_epochs = annealing_epochs
        self.patience = patience

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
            beta = 1

            # =======================
            #        TRAIN
            # =======================
            self.train()
            tr_loss, tr_recon, tr_kl = 0, 0, 0

            for x, domain_id, species_id in trainloader:
                x = x.to(device)
                domain_id = domain_id.to(device)
                species_id = species_id.to(device)

                self.optimizer.zero_grad()

                mu, logvar, z = self.forward(x, domain_id)
                loss, recon, kl = self.elbo_loss(
                    x, mu, logvar, z,
                    domain_id=domain_id,
                    species_id=species_id,
                    beta=beta,
                )

                loss.backward()
                self.optimizer.step()

                tr_loss += loss.item()
                tr_recon += recon.item()
                tr_kl += kl.item()

            tr_loss /= len(trainloader)
            tr_recon /= len(trainloader)
            tr_kl /= len(trainloader)

            # =======================
            #      VALIDATION
            # =======================
            self.eval()
            val_loss, val_recon, val_kl = 0, 0, 0

            with torch.no_grad():
                for x, domain_id, species_id in validloader:
                    x = x.to(device)
                    domain_id = domain_id.to(device)
                    species_id = species_id.to(device)

                    mu, logvar, z = self.forward(x, domain_id)
                    loss, recon, kl = self.elbo_loss(
                        x, mu, logvar, z,
                        domain_id=domain_id,
                        species_id=species_id,
                        beta=beta,
                    )

                    val_loss += loss.item()
                    val_recon += recon.item()
                    val_kl += kl.item()

            val_loss /= len(validloader)
            val_recon /= len(validloader)
            val_kl /= len(validloader)

            self.loss_during_training.append((tr_loss, val_loss))
            self.reconstruc_during_training.append((tr_recon, val_recon))
            self.KL_during_training.append((tr_kl, val_kl))

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"[Train] Loss={tr_loss:.4f} | Recon={tr_recon:.4f} | KL={tr_kl:.4f} || "
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
