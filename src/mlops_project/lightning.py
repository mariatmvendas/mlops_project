import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# Define your Lightning Module
class MyModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Dataset and DataLoader
def create_dataloaders():
    # Example dataset
    X = torch.randn(1000, 10)
    Y = torch.randn(1000, 1)
    dataset = TensorDataset(X, Y)
    train, val = random_split(dataset, [800, 200])
    train_loader = DataLoader(train, batch_size=32)
    val_loader = DataLoader(val, batch_size=32)
    return train_loader, val_loader

# Main Training Loop
if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders()
    model = MyModel(input_dim=10, output_dim=1)

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, train_loader, val_loader)

    
logger = TensorBoardLogger("logs/", name="my_model")

trainer = pl.Trainer(max_epochs=5, logger=logger)


