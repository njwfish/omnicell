import numpy as np
import scanpy as sc

import torch
import torch.nn.functional as F
from torch import nn

from omnicell.constants import *
from omnicell.processing.utils import to_dense
from omnicell.models.utils.datamodules import get_dataloader

from omnicell.models.scot.sampling_utils import batch_pert_sampling
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

import logging
logger = logging.getLogger(__name__)

def sliced_wasserstein_distance(X1, X2, n_projections=100, p=2):
    """
    Computes the Sliced Wasserstein Distance (SWD) between two batches using POT.

    Args:
        X1: Tensor of shape (N, d) - First batch of points.
        X2: Tensor of shape (M, d) - Second batch of points.
        n_projections: Number of random projections (default: 100).
        p: Power of distance metric (default: 2).

    Returns:
        SWD (scalar tensor).
    """
    device = X1.device
    d = X1.shape[1]  # Feature dimension

    # Generate random projection vectors
    projections = torch.randn((n_projections, d), device=device)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)  # Normalize

    # Project both distributions onto 1D subspaces
    X1_proj = X1 @ projections.T  # Shape: (N, n_projections)
    X2_proj = X2 @ projections.T  # Shape: (M, n_projections)

    # Sort projections along each 1D slice
    X1_proj_sorted, _ = torch.sort(X1_proj, dim=0)
    X2_proj_sorted, _ = torch.sort(X2_proj, dim=0)

    # Compute 1D Wasserstein distance per projection (L_p norm)
    SW_dist = torch.mean(torch.abs(X1_proj_sorted - X2_proj_sorted) ** p) ** (1/p)

    return SW_dist

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU()
        ])
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MeanPooledFC(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.fc = MLP(2 * in_dim, hidden_dim, out_dim, layers=layers)

    def forward(self, x):
        pooled_rep = x.mean(dim=0)
        pooled_rep = torch.tile(pooled_rep, (x.shape[0], 1))
        x = torch.cat([x, pooled_rep], dim=1)
        return self.fc(x)


class FCGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(FCGNN, self).__init__()
        self.predictor = nn.Sequential(
            torch.nn.Linear(in_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            MeanPooledFC(2 * hidden_dim, hidden_dim, hidden_dim, 1),
            nn.LeakyReLU(),
            MeanPooledFC(hidden_dim, hidden_dim, hidden_dim, 1),
            nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z):
        return self.predictor(z)


class SCOT(torch.nn.Module):
    def __init__(self, adata, pert_embedding, cell_embed_dim, gene_embed_dim, hidden_dim, max_epochs=10):
        super(SCOT, self).__init__()
        self.total_adata = adata
        self.total_genes = adata.shape[1]
        self.pert_embedding = pert_embedding
        self.cell_embedder = MLP(self.total_genes, 2 * cell_embed_dim, cell_embed_dim, layers=3)
        self.gene_embedder = torch.nn.Embedding(self.total_genes, gene_embed_dim)
        self.gnn = FCGNN(gene_embed_dim + cell_embed_dim + 3, hidden_dim, 1)
        self.max_epochs = max_epochs

    def forward(self, ctrl, mean_shift, gene_indices=None):
        device = ctrl.device
        if gene_indices is None:
            gene_indices = torch.arange(self.total_genes).to(device)
        
        num_cells = torch.tensor(ctrl.shape[0]).to(device)
        num_genes = gene_indices.shape[0]

        cell_embed = self.cell_embedder(ctrl)
        cell_embed = torch.tile(cell_embed[:, None, :], (1, num_genes, 1))

        ctrl = ctrl[:, gene_indices]
        mean_shift = mean_shift[gene_indices]

        gene_embed = self.gene_embedder(gene_indices)
        gene_embed = torch.tile(gene_embed, (num_cells, 1, 1))

        shift = torch.tile(mean_shift[None, :, None], (num_cells, 1, 1))

        num_cells = torch.tile(num_cells, (num_cells, num_genes, 1))

        ctrl_and_embed = torch.cat([ctrl[:, :, None], cell_embed, gene_embed, shift, num_cells], dim=2)
        output = torch.vmap(self.gnn, in_dims=1)(ctrl_and_embed)
        output = torch.transpose(output, 0, 1).squeeze()
        weighted_dist = torch.nn.Softmax(dim=0)(output)
        return weighted_dist
    
    def loss(self, ctrl, pert, n_projections=1000, negative_penalty=100):
        device = ctrl.device
        batch_size = torch.tensor(ctrl.shape[0]).to(device)
    
        shift_vec = pert.sum(axis=0) - ctrl.sum(axis=0)
        weighted_dist = self.forward(ctrl, shift_vec / batch_size)

        pred_pert = (ctrl + (weighted_dist * shift_vec))

        loss = sliced_wasserstein_distance(pred_pert, pert, n_projections=n_projections) 
        loss -= negative_penalty * ((pred_pert < 0) * pred_pert).sum() / (batch_size * self.total_genes)
        return loss
    
    def train(self, adata, model_savepath, dl=None):
        if dl is None:
            _, dl = get_dataloader(
                adata, pert_ids=np.array(adata.obs[PERT_KEY].values), offline=False, pert_map=self.pert_embedding, collate=None
            )

        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, verbose=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        num_epochs = self.max_epochs
        logger.info(f"Training for {num_epochs} epochs")

        best_loss = float('inf')
        plateau_count = 0
        for epoch in range(num_epochs):
            running_loss = 0.0
            current_lr = optimizer.param_groups[0]['lr']
            
            # Create progress bar for this epoch
            pbar = tqdm.tqdm(dl, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=True)
            
            for batch_idx, (ctrl, pert, pert_id) in enumerate(pbar):
                ctrl, pert = ctrl.to(device), pert.to(device)
                
                # Forward pass and loss calculation
                loss = self.loss(ctrl, pert)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update running loss and progress bar
                running_loss += loss.item()
                avg_loss = running_loss / (batch_idx + 1)
                
                # Update progress bar description with current loss and lr
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'best': f'{best_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            # Calculate epoch average loss
            epoch_loss = running_loss / (batch_idx + 1)
            
            # Update learning rate based on loss plateau
            scheduler.step(epoch_loss)
            
            # Update best loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                plateau_count = 0
            else:
                plateau_count += 1
            
            # Log epoch results
            logger.info(f'Epoch {epoch+1}/{num_epochs} - Avg Loss: {epoch_loss:.4f} - Best Loss: {best_loss:.4f} - LR: {current_lr:.2e}')
    
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        X_ctrl = to_dense(self.total_adata[(self.total_adata.obs[PERT_KEY] == CONTROL_PERT) & (self.total_adata.obs[CELL_KEY] == cell_type)].X.toarray())
        X_pert = to_dense(self.total_adata[(self.total_adata.obs[PERT_KEY] == pert_id) & (self.total_adata.obs[CELL_KEY] == cell_type)].X.toarray())
        return batch_pert_sampling(self, X_ctrl, X_pert)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
