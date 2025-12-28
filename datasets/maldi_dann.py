from torch.utils.data import Dataset

class MaldiDANNDataset(Dataset):
    def __init__(self, x, labels, meta, source_domain):
        self.x = x
        self.labels = labels
        self.meta = meta
        self.source_domain = source_domain
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        x_i = self.x[idx]
        y_i = self.labels[idx]
        hospital_i = self.meta["hospital"][idx]
        is_source = (hospital_i == self.source_domain)

        return x_i, y_i, hospital_i, is_source
    