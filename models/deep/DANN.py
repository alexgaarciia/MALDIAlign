import copy
import torch
import torch.nn as nn
import torch.optim as optim
import math
from models.deep.networks import EncoderDANN, SpeciesClassifier, DomainClassifier, grad_reverse


class DANNFull(nn.Module):
    """
    Full Domain-Adversarial Neural Network (DANN).
    """

    def __init__(self, input_dim, latent_dim, n_species, n_domains=2):
        super().__init__()

        self.encoder = EncoderDANN(input_dim, latent_dim)
        self.species_clf = SpeciesClassifier(latent_dim, n_species)
        self.domain_clf = DomainClassifier(latent_dim, n_domains)

    def forward(self, x, lambda_=0.0):
        z = self.encoder(x)
        species_logits = self.species_clf(z)
        z_rev = grad_reverse(z, lambda_)
        domain_logits = self.domain_clf(z_rev)
        return species_logits, domain_logits


class DANNFull_Extended(DANNFull):
    """
    DANN with integrated training loop, following the exact structure
    of the *_Extended VAE classes.

    Expected batch format (from prepare_data / dataloader):
        (x, domain_id, species_id)

    Domain labels for adversarial head are derived as:
        is_source = (domain_id == source_domain_id)
        domain_y = (~is_source).long()  # source=0, target=1
    """

    def __init__(self, input_dim, latent_dim, n_species, n_domains=2, source_domain_id=0, epochs=50, lr=1e-5, lambda_domain=0.01, patience=20, grad_clip=1.0,
    ):
        super().__init__(input_dim, latent_dim, n_species, n_domains)

        self.source_domain_id = int(source_domain_id)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.lambda_domain = float(lambda_domain)
        self.patience = int(patience)
        self.grad_clip = float(grad_clip)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.criterion_species = nn.CrossEntropyLoss()
        self.criterion_domain = nn.CrossEntropyLoss()

        self.loss_during_training = []
        self.species_loss_during_training = []
        self.domain_loss_during_training = []

    def trainloop(self, trainloader, validloader, device):
        self.to(device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            p = (epoch+1) / self.epochs
            lambda_eff = self.lambda_domain * (
                2 / (1 + math.exp(-10*p)) - 1
            )

            # =======================
            #        TRAIN
            # =======================
            self.train()
            tr_loss, tr_sp, tr_dom = 0.0, 0.0, 0.0
            used=0

            for x, domain_id, species_id in trainloader:
                x = x.to(device).float()
                domain_id = domain_id.to(device)
                species_id = species_id.to(device)

                is_source = domain_id == self.source_domain_id

                # Species loss SOLO en source
                if is_source.sum() == 0:
                    continue

                self.optimizer.zero_grad()

                species_logits, domain_logits = self.forward(
                    x, lambda_=lambda_eff
                )

                loss_species = self.criterion_species(
                    species_logits[is_source],
                    species_id[is_source],
                )

                # Domain loss on ALL samples (paper convention: source=0, target=1)
                domain_y = (~is_source).long()
                loss_domain = self.criterion_domain(
                    domain_logits,
                    domain_y,
                )

                loss = loss_species + loss_domain

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), self.grad_clip
                )

                self.optimizer.step()

                tr_loss += loss.item()
                tr_sp += loss_species.item()
                tr_dom += loss_domain.item()
                used += 1

            tr_loss /= max(used, 1)
            tr_sp   /= max(used, 1)
            tr_dom  /= max(used, 1)

            # =======================
            #      VALIDATION
            # =======================
            self.eval()
            val_loss, val_sp, val_dom = 0.0, 0.0, 0.0
            used_val = 0

            with torch.no_grad():
                for x, domain_id, species_id in validloader:
                    x = x.to(device).float()
                    domain_id = domain_id.to(device)
                    species_id = species_id.to(device)

                    is_source = domain_id == self.source_domain_id

                    if is_source.sum() == 0:
                        continue

                    species_logits, domain_logits = self.forward(
                        x, lambda_=0.0
                    )

                    loss_species = self.criterion_species(
                        species_logits[is_source],
                        species_id[is_source],
                    )

                    domain_y = (~is_source).long()
                    loss_domain = self.criterion_domain(
                        domain_logits,
                        domain_y,
                    )

                    loss = loss_species + loss_domain

                    val_loss += loss.item()
                    val_sp += loss_species.item()
                    val_dom += loss_domain.item()
                    used_val += 1

            val_loss /= max(used_val, 1)
            val_sp   /= max(used_val, 1)
            val_dom  /= max(used_val, 1)

            self.loss_during_training.append((tr_loss, val_loss))
            self.species_loss_during_training.append((tr_sp, val_sp))
            self.domain_loss_during_training.append((tr_dom, val_dom))

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"[Train] Loss={tr_loss:.4f} | Sp={tr_sp:.4f} | Dom={tr_dom:.4f} || "
                    f"[Val] Loss={val_loss:.4f} | Sp={val_sp:.4f} | Dom={val_dom:.4f}"
                )

            # =======================
            #     EARLY STOPPING
            # =======================
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.load_state_dict(best_state)
