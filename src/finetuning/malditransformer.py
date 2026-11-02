import numpy as np

import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader


MZ_AXIS = torch.linspace(2000, 20000, steps=6000) 


class FTDataset(Dataset):
    def __init__(self, data_matrix, labels_enc, n_peaks=200):
        self.data = data_matrix
        self.labels = labels_enc
        self.n_peaks = n_peaks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bin_vec = torch.from_numpy(self.data[idx]).float()
        intensity, indices = torch.topk(bin_vec, k=self.n_peaks)
        mz = MZ_AXIS[indices]
        return {"mz": mz, "intensity": intensity, "species": int(self.labels[idx])}

def collate_ft(batch):
    return {
        "mz": torch.stack([b["mz"] for b in batch]),
        "intensity": torch.stack([b["intensity"] for b in batch]),
        "species": torch.tensor([b["species"] for b in batch]),
    }

class FTDataModule(L.LightningDataModule):
    def __init__(self, train_ds, val_ds, batch_size=256, n_workers=4):
        super().__init__()
        self.train_ds, self.val_ds = train_ds, val_ds
        self.batch_size, self.n_workers = batch_size, n_workers

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, collate_fn=collate_ft, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, collate_fn=collate_ft, pin_memory=True)

def get_transformer_latent(data_matrix, model, n_peaks=200, batch_size=256, device=None):
    model.eval()
    model.to(device)
    mz_axis = torch.linspace(2000, 20000, steps=6000).to(device)
    all_z = []

    for i in range(0, len(data_matrix), batch_size):
        batch_bin = torch.from_numpy(data_matrix[i:i+batch_size]).float().to(device)

        intensities, indices = torch.topk(batch_bin, k=n_peaks, dim=1)
        mzs = mz_axis[indices]

        batch_dict = {"mz": mzs, "intensity": intensities}

        with torch.no_grad():
            z_seq = model.transformer(batch_dict)
            z_cls = z_seq[:, 0, :]
            all_z.append(z_cls.cpu().numpy())

    return np.vstack(all_z)
