import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# 🧩 Denoising Autoencoder Definition
# ==========================================================

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder:
    Learns a compressed representation of the input while reconstructing it.
    Adds Gaussian noise during training to improve robustness.
    """

    def __init__(self, input_dim: int, latent_dim: int = 128, noise_factor: float = 0.1):
        super(DenoisingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.noise_factor = noise_factor

        # Encoder Network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, latent_dim)
        )

        # Decoder Network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, input_dim)
        )

    def forward(self, x: torch.Tensor, add_noise: bool = True):
        """
        Forward pass with optional denoising.
        """
        if self.training and add_noise:
            noise = torch.randn_like(x) * self.noise_factor
            x = x + noise
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z


# ==========================================================
# ⚙️ Utility Functions for Training & Feature Extraction
# ==========================================================

def train_autoencoder(
    model: nn.Module,
    data_loader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Trains the autoencoder on input modality data.
    Uses MSE loss + optional contractive regularization.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            inputs = batch.to(device).float()
            optimizer.zero_grad()
            reconstructed, z = model(inputs, add_noise=True)

            # Reconstruction loss
            loss = criterion(reconstructed, inputs)

            # Optional contractive loss (Jacobian regularization)
            # Encourages robustness to small input perturbations
            if hasattr(model, 'encoder'):
                W = model.encoder[0].weight  # first linear layer
                contractive_loss = torch.sum(W ** 2)
                loss += 1e-4 * contractive_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        print(f"[Epoch {epoch+1:03d}] Autoencoder Loss: {avg_loss:.6f}")

    print("[INFO] Autoencoder training completed.")
    return model


def extract_latent_features(model: nn.Module, data_loader, device=None):
    """
    Encodes input data into latent vectors using the trained autoencoder.
    Returns a tensor of latent embeddings for downstream Transformer fusion.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    all_latents = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch.to(device).float()
            _, z = model(inputs, add_noise=False)
            all_latents.append(z.cpu())

    embeddings = torch.cat(all_latents, dim=0)
    print(f"[INFO] Extracted latent embeddings shape: {embeddings.shape}")
    return embeddings


# ==========================================================
# 🚀 Example Usage
# ==========================================================
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np

    # Example synthetic dataset for demonstration
    X = np.random.rand(500, 100)  # 500 samples, 100 features
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Instantiate and train autoencoder
    model = DenoisingAutoencoder(input_dim=100, latent_dim=64)
    trained_model = train_autoencoder(model, loader, num_epochs=10)

    # Extract latent features
    latent_vectors = extract_latent_features(trained_model, loader)
