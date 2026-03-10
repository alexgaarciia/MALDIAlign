import torch
import torch.nn as nn
import torch.optim as optim
from models.deep.networks import ConditionalPrior
from models.deep.MultiVAEAMR import MultiVAE_Bernoulli


class MultiVAE_Bernoulli_SpeciesPrior_AMR(MultiVAE_Bernoulli):
    """
    Multi-decoder VAE with species-conditional latent prior.
    
    Encoder: q(z | x)
    Prior:   p(z | species)
    Decoder: p(x | z, domain)
    """

    def __init__(self, input_dim, latent_dim, num_domains, n_species, lambda_amr):
        super().__init__(input_dim, latent_dim, num_domains)

        self.n_species = n_species
        self.prior = ConditionalPrior(
            n_species=n_species,
            latent_dim=latent_dim
        )
        self.lambda_amr = lambda_amr

        # AMR classification had
        self.amr_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
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

    def total_loss(self, x, mu, logvar, z, domain_id, species_id, amr_logits, amr_labels, beta=1.0):
        loss, recon, kl = self.elbo_loss(
            x, mu, logvar, z,
            domain_id=domain_id,
            species_id=species_id,
            beta=beta,
        )

        amr_loss = torch.tensor(0.0, device=x.device)
        if amr_labels is not None and amr_logits is not None:
            if self.pos_weight is not None:
                bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(x.device))
            else:
                bce = nn.BCEWithLogitsLoss()
            amr_loss = bce(amr_logits.view(-1), amr_labels.float())
            loss = loss + self.lambda_amr * amr_loss

        return loss, recon, kl, amr_loss

class MultiVAE_Bernoulli_SpeciesPrior_AMR_Extended(MultiVAE_Bernoulli_SpeciesPrior_AMR):
    def __init__(self, input_dim, latent_dim, num_domains, n_species, lambda_amr=0.1, epochs=100, lr=1e-4, annealing_epochs=50, patience=20):
        super().__init__(input_dim, latent_dim, num_domains, n_species, lambda_amr)

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
        self.AMR_during_training = []

    def trainloop(self, trainloader, validloader, device):
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
            tr_loss, tr_recon, tr_kl, tr_amr = 0, 0, 0, 0

            for x, domain_id, species_id, amr_labels in trainloader:
                x = x.to(device)
                domain_id = domain_id.to(device)
                species_id = species_id.to(device)
                amr_labels = amr_labels.to(device)

                self.optimizer.zero_grad()

                mu, logvar, z, amr_logits = self.forward(x, domain_id)
                loss, recon, kl, amr = self.total_loss(
                    x, mu, logvar, z,
                    domain_id=domain_id,
                    species_id=species_id,
                    amr_logits=amr_logits,
                    amr_labels=amr_labels,
                    beta=beta,
                )

                loss.backward()
                self.optimizer.step()

                tr_loss += loss.item()
                tr_recon += recon.item()
                tr_kl += kl.item()
                tr_amr += amr.item()

            tr_loss /= len(trainloader)
            tr_recon /= len(trainloader)
            tr_kl /= len(trainloader)
            tr_amr /= len(trainloader)

            # =======================
            #      VALIDATION
            # =======================
            self.eval()
            val_loss, val_recon, val_kl, val_amr = 0, 0, 0, 0

            with torch.no_grad():
                for x, domain_id, species_id, amr_labels in validloader:
                    x = x.to(device)
                    domain_id = domain_id.to(device)
                    species_id = species_id.to(device)
                    amr_labels = amr_labels.to(device)

                    mu, logvar, z, amr_logits = self.forward(x, domain_id)
                    loss, recon, kl, amr = self.total_loss(
                        x, mu, logvar, z,
                        domain_id=domain_id,
                        species_id=species_id,
                        amr_logits=amr_logits,
                        amr_labels=amr_labels,
                        beta=beta,
                    )

                    val_loss += loss.item()
                    val_recon += recon.item()
                    val_kl += kl.item()
                    val_amr += amr.item()

            val_loss /= len(validloader)
            val_recon /= len(validloader)
            val_kl /= len(validloader)
            val_amr /= len(validloader)

            self.loss_during_training.append((tr_loss, val_loss))
            self.reconstruc_during_training.append((tr_recon, val_recon))
            self.KL_during_training.append((tr_kl, val_kl))
            self.AMR_during_training.append((tr_amr, val_amr))

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"[Train] Loss={tr_loss:.4f} | Recon={tr_recon:.4f} | KL={tr_kl:.4f} | AMR={tr_amr:.4f} || "
                    f"[Val] Loss={val_loss:.4f} | Recon={val_recon:.4f} | KL={val_kl:.4f} | AMR={val_amr:.4f} "
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
