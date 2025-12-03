import torch
import torch.nn as nn
import torch.optim as optim
from models.deep.networks import MultiVAE

class MultiVAE_CORAL(MultiVAE):
  def __init__(self, input_dim, latent_dim, num_domains, epochs=100, lr=1e-4, annealing_epochs=50, patience=20):
    super().__init__(input_dim, latent_dim, num_domains)

    self.epochs = epochs
    self.lr = lr
    self.annealing_epochs = annealing_epochs
    self.patience = patience

    self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
    self.criterion = nn.MSELoss()

    self.loss_during_training = []
    self.reconstruc_during_training = []
    self.KL_during_training = []
    self.coral_loss_during_training = []

  def loss_function(self, x, x_recon, mu, logvar, beta):
    recon_loss = self.criterion(x_recon, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

  def compute_coral(self, z_a, z_b):
    z_a = z_a - z_a.mean(0)
    z_b = z_b - z_b.mean(0)

    cov_a = (z_a.T @ z_a) / (z_a.shape[0]-1)
    cov_b = (z_b.T @ z_b) / (z_b.shape[0]-1)

    coral_loss = torch.mean((cov_a - cov_b)**2)
    coral_loss = coral_loss / (4*(z_a.shape[1]**2))

    return coral_loss

  def trainloop(self, trainloader, validloader, device):
    self.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(self.epochs):
      beta = min(1.0, (epoch + 1) / self.annealing_epochs)

      # TRAIN
      self.train()
      train_total_loss, train_recon_loss, train_kl_loss, train_coral_loss = 0, 0, 0, 0

      for x, domain_id in trainloader:
        x = x.to(device)
        domain_id = domain_id.to(device)
        self.optimizer.zero_grad()
        x_recon, mu, logvar = self.forward(x, domain_id)
        loss, recon, kl = self.loss_function(x, x_recon, mu, logvar, beta=beta)

        # CORAL
        mask_A = (domain_id == 0)
        mask_D = (domain_id == 1)
        if mask_A.any() and mask_D.any():
            z = self.reparameterize(mu, logvar)
            coral_loss = self.compute_coral(z[mask_A], z[mask_D])
            lambda_coral = 1e-3
            loss = loss + lambda_coral * coral_loss
            train_coral_loss += coral_loss.item()

        loss.backward()
        self.optimizer.step()
        train_total_loss += loss.item()
        train_recon_loss += recon.item()
        train_kl_loss += kl.item()

      train_total_loss = train_total_loss / len(trainloader)
      train_recon_loss = train_recon_loss / len(trainloader)
      train_kl_loss = train_kl_loss / len(trainloader)
      train_coral_loss = train_coral_loss / len(trainloader)

      # VALIDATION
      self.eval()
      val_loss, val_recon, val_kl, val_coral = 0, 0, 0, 0

      with torch.no_grad():
        for x, domain_id in validloader:
          x = x.to(device)
          domain_id = domain_id.to(device)
          x_recon, mu, logvar = self.forward(x, domain_id)
          loss, recon, kl = self.loss_function(x, x_recon, mu, logvar, beta=beta)

        # CORAL
          mask_A = (domain_id == 0 )
          mask_D = (domain_id == 1)
          if mask_A.any() and mask_D.any():
              z = self.reparameterize(mu, logvar)
              coral_loss = self.compute_coral(z[mask_A], z[mask_D])
              lambda_coral = 5e-3
              loss = loss + lambda_coral * coral_loss
              val_coral += coral_loss.item()

          val_loss += loss.item()
          val_recon += recon.item()
          val_kl += kl.item()

      val_loss /= len(validloader)
      val_recon /= len(validloader)
      val_kl /= len(validloader)
      val_coral /= len(validloader)
    
      self.loss_during_training.append((train_total_loss, val_loss))
      self.reconstruc_during_training.append((train_recon_loss, val_recon))
      self.KL_during_training.append((train_kl_loss, val_kl))
      self.coral_loss_during_training.append((train_coral_loss, val_coral))

      if (epoch+1) % 5 == 0:
        print(
          f"Epoch {epoch+1}/{self.epochs} | "
          f"[Train] Loss: {train_total_loss:.4f} | Recon: {train_recon_loss:.4f} | KL: {train_kl_loss:.4f} | CORAL: {train_coral_loss:.5f} || "
          f"[Val] Loss: {val_loss:.4f} | Recon: {val_recon:.4f} | KL: {val_kl:.4f} | CORAL: {val_coral:.5f}"
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
        