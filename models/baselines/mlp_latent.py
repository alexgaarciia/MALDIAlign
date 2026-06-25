import copy
import torch
import torch.nn as nn
import torch.optim as optim

class LinearProbe(nn.Module):
    def __init__(self, latent_dim=64, n_species=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_species = n_species
        self.net = nn.Linear(latent_dim, n_species)

    def forward(self, x):
        x = self.net(x)
        return x

class LinearProbe_Extended(LinearProbe):
    def __init__(self, latent_dim=64, n_species=6, epochs=100, lr=1e-3, patience=10):
        super().__init__(latent_dim, n_species)
        self.epochs = epochs
        self.lr = lr
        self.patience = patience

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()

        self.loss_during_training = []
        self.val_loss_during_training = []

    def trainloop(self, trainloader, validloader, device):
        self.to(device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            train_loss, val_loss = 0, 0

            # =======================
            #        TRAIN
            # =======================
            self.train()
            for batch in trainloader:
                x, species_id = batch
                x, species_id = x.to(device), species_id.to(device)

                self.optimizer.zero_grad()
                pred = self.forward(x)
                loss = self.loss(pred, species_id)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
        
            train_loss /= len(trainloader)

            # =======================
            #      VALIDATION
            # =======================
            self.eval()
            
            with torch.no_grad():
                for batch in validloader:
                    x, species_id = batch
                    x, species_id = x.to(device), species_id.to(device)
                    pred = self.forward(x)
                    loss = self.loss(pred, species_id)
                    val_loss += loss.item()
            
            val_loss /= len(validloader)

            self.loss_during_training.append(train_loss)
            self.val_loss_during_training.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {train_loss} | Validation Loss: {val_loss}")

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
