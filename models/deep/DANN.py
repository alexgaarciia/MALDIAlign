import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from models.deep.networks import EncoderDANN, SpeciesClassifier, DomainClassifier, grad_reverse


class DANNFull(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN).

    This model learns a latent representation z from input spectra x such that:
    - z is predictive of species labels
    - z is invariant to acquisition domain

    The encoder is trained jointly with:
    - a species classifier
    - a multi-class domain discriminator connected through a Gradient Reversal Layer (GRL)
    """

    def __init__(self, input_dim, latent_dim, n_species, n_domains):
        super().__init__()

        self.encoder = EncoderDANN(input_dim, latent_dim)
        self.species_clf = SpeciesClassifier(latent_dim, n_species)
        self.domain_clf = DomainClassifier(latent_dim, n_domains=n_domains)

    def forward(self, x: torch.Tensor, lambda_: float = 0.0):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input spectra of shape (batch_size, input_dim)
        lambda_ : float, optional
            Strength of gradient reversal for the domain branch

        Returns
        -------
        species_logits : torch.Tensor
            Logits for species classification
        domain_logits : torch.Tensor
            Logits for multi-class domain classification
        z : torch.Tensor
            Latent representation
        """
        z = self.encoder(x)
        species_logits = self.species_clf(z)
        z_rev = grad_reverse(z, lambda_)
        domain_logits = self.domain_clf(z_rev)
        return species_logits, domain_logits, z


class DANNFull_Extended(DANNFull):
    """
    DANN with integrated training loop.

    Expected batch format:
        (x, domain_id, species_id)

    Notes
    -----
    - Species supervision is applied to all samples.
    - Domain supervision is multi-class and uses the original domain labels.
    - The encoder is encouraged to learn domain-invariant features through GRL.
    """

    def __init__(self, input_dim, latent_dim, n_species, n_domains, epochs: int = 50, lr = 1e-5, patience = 20, grad_clip = 1.0):
        super().__init__(input_dim, latent_dim, n_species, n_domains)

        self.epochs = int(epochs)
        self.lr = float(lr)
        self.patience = int(patience)
        self.grad_clip = float(grad_clip)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
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
            p = (epoch + 1) / self.epochs
            lambda_eff = 2 / (1 + math.exp(-10*p)) - 1

            # =======================
            #        TRAIN
            # =======================
            self.train()
            tr_loss, tr_sp, tr_dom = 0.0, 0.0, 0.0
            used = 0

            for batch in trainloader:
                if len(batch) != 3:
                    raise ValueError(
                        "DANN expects batches of the form (x, domain_id, species_id)"
                    )

                x, domain_id, species_id = batch
                x = x.to(device).float()
                domain_id = domain_id.to(device).long()
                species_id = species_id.to(device).long()

                self.optimizer.zero_grad()

                species_logits, domain_logits, _ = self.forward(x, lambda_=lambda_eff)

                loss_species = self.criterion_species(species_logits, species_id)
                loss_domain = self.criterion_domain(domain_logits, domain_id)
                loss = loss_species + loss_domain

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                self.optimizer.step()

                tr_loss += loss.item()
                tr_sp += loss_species.item()
                tr_dom += loss_domain.item()
                used += 1

            tr_loss /= max(used, 1)
            tr_sp /= max(used, 1)
            tr_dom /= max(used, 1)

            # =======================
            #      VALIDATION
            # =======================
            self.eval()
            val_loss, val_sp, val_dom = 0.0, 0.0, 0.0
            used_val = 0

            with torch.no_grad():
                for batch in validloader:
                    if len(batch) != 3:
                        raise ValueError("DANN expects batches of the form (x, domain_id, species_id)")

                    x, domain_id, species_id = batch
                    x = x.to(device).float()
                    domain_id = domain_id.to(device).long()
                    species_id = species_id.to(device).long()

                    species_logits, domain_logits, _ = self.forward(x, lambda_=0.0)

                    loss_species = self.criterion_species(species_logits, species_id)
                    loss_domain = self.criterion_domain(domain_logits, domain_id)
                    loss = loss_species + loss_domain

                    val_loss += loss.item()
                    val_sp += loss_species.item()
                    val_dom += loss_domain.item()
                    used_val += 1

            val_loss /= max(used_val, 1)
            val_sp /= max(used_val, 1)
            val_dom /= max(used_val, 1)

            self.loss_during_training.append((tr_loss, val_loss))
            self.species_loss_during_training.append((tr_sp, val_sp))
            self.domain_loss_during_training.append((tr_dom, val_dom))

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"[Train] Loss={tr_loss:.4f} | Sp={tr_sp:.4f} | Dom={tr_dom:.4f} || "
                    f"[Val] Loss={val_loss:.4f} | Sp={val_sp:.4f} | Dom={val_dom:.4f} | "
                    f"lambda={lambda_eff:.4f}"
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
