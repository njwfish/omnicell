import numpy as np
import scanpy as sc

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data

from omnicell.constants import *
from omnicell.processing.utils import to_dense
from omnicell.models.distribute_shift import sample_pert
from omnicell.models.datamodules import get_dataloader


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

class FCGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(FCGNN, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, z):
        x = self.fc1(z)
        x = F.elu(x)
        pooled_rep = x.mean(dim=0)
        pooled_rep = torch.tile(pooled_rep, (x.shape[0], 1))
        # x = x + pooled_rep
        x = torch.cat([x, pooled_rep], dim=1)
        x = self.fc2(x)
        pooled_rep = x.mean(dim=0)
        pooled_rep = torch.tile(pooled_rep, (x.shape[0], 1))
        # x = x + pooled_rep
        x = torch.cat([x, pooled_rep], dim=1)
        x = F.elu(x)
        x = self.fc3(x)
        return x

class CellEmbedding(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(CellEmbedding, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        return x

class SCOT(torch.nn.Module):
    def __init__(self, adata, pert_rep_map, cell_embed_dim, gene_embed_dim, hidden_dim, max_epochs=10):
        super(SCOT, self).__init__()
        self.total_adata = adata
        self.total_genes = adata.shape[1]
        self.pert_rep_map = pert_rep_map
        self.cell_embedder = CellEmbedding(self.total_genes, hidden_dim, cell_embed_dim)
        self.gene_embedder = torch.nn.Embedding(self.total_genes, gene_embed_dim)
        self.gnn = FCGNN(gene_embed_dim + cell_embed_dim + 3, hidden_dim, 1)
        self.max_epochs = max_epochs

    def forward(self, ctrl, shift_vec, gene_indices=None):
        device = ctrl.device
        if gene_indices is None:
            gene_indices = torch.arange(self.total_genes).to(device)
        
        num_cells = torch.tensor(ctrl.shape[0]).to(device)
        num_genes = gene_indices.shape[0]

        cell_embed = self.cell_embedder(ctrl)
        cell_embed = torch.tile(cell_embed[:, None, :], (1, num_genes, 1))

        ctrl = ctrl[:, gene_indices]
        shift_vec = shift_vec[gene_indices]

        gene_embed = self.gene_embedder(gene_indices)
        gene_embed = torch.tile(gene_embed, (num_cells, 1, 1))

        shift = torch.tile(shift_vec[None, :, None], (num_cells, 1, 1))

        num_cells = torch.tile(num_cells, (num_cells, num_genes, 1))
        # print(ctrl.shape, cell_embed.shape, gene_embed.shape, shift.shape, num_cells.shape)

        ctrl_and_embed = torch.cat([ctrl[:, :, None], cell_embed, gene_embed, shift, num_cells], dim=2)
        output = torch.vmap(self.gnn, in_dims=1)(ctrl_and_embed)
        output = torch.transpose(output, 0, 1).squeeze()
        weighted_dist = torch.nn.Softmax(dim=0)(output)
        return weighted_dist
    
    def loss(self, ctrl, pert, n_projections=1000, negative_penalty=100):
        device = ctrl.device
        batch_size = torch.tensor(ctrl.shape[0]).to(device)
    
        shift_vec = pert.sum(axis=0) - ctrl.sum(axis=0)

        weighted_dist = self.forward(ctrl, shift_vec)
        pred_pert = (ctrl + (weighted_dist * shift_vec))
        loss =  sliced_wasserstein_distance(pred_pert, pert, n_projections=n_projections) 
        loss -=negative_penalty * ((pred_pert < 0) * pred_pert).sum() / (batch_size * self.total_genes)
        return loss
    
    def sample_pert(self, ctrl, pert, max_rejections=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_batches = ctrl.shape[0] / pert.shape[0]
        preds = []
        for ctrl_data_batch in np.array_split(ctrl, n_batches):
            mean_shift = pert.mean(axis=0) - ctrl_data_batch.mean(axis=0)
            shift_vec = np.round(mean_shift * ctrl_data_batch.shape[0])
            ctrl_data_batch = torch.tensor(ctrl_data_batch).to(device)
            shift_vec_batch = torch.tensor(shift_vec).to(device)
            # no grad
            with torch.no_grad():
                weighted_dist = self.forward(ctrl_data_batch, shift_vec_batch).cpu()
            weighted_dist = weighted_dist.numpy().astype(np.float64)
            weighted_dist /= weighted_dist.sum(axis=0)
            preds.append(
                sample_pert(
                    ctrl_data_batch.cpu().numpy(), weighted_dist, mean_shift, 
                    max_rejections=max_rejections
                )
            )

        preds = np.concatenate(preds, axis=0)
        return preds
    
    def train(self, adata):
        dset, ns, dl = get_dataloader(
            adata, pert_ids=np.array(adata.obs[PERT_KEY].values), pert_map=self.pert_rep_map, collate='ot'
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        from tqdm.notebook import tqdm
        pbar = tqdm(total=len(dl), desc='Training')

        num_epochs = self.max_epochs
        for epoch in range(num_epochs):
            losses = []
            for i, (ctrl, pert, pert_id) in enumerate(dl):
                # generate random indices for ctrl and pert
                
                ctrl, pert = ctrl.to(device), pert.to(device)
                loss = self.loss(ctrl, pert)
                
                losses.append(loss.item())
                # Update progress bar with loss
                pbar.set_postfix({'loss': f'{np.mean(losses[-100:]):.4f}'})
                pbar.update(1)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        pbar.close()



    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        X_ctrl = to_dense(self.total_adata[(self.total_adata.obs[PERT_KEY] == CONTROL_PERT) & (self.total_adata.obs[CELL_KEY] == cell_type)].X.toarray())
        X_pert = to_dense(self.total_adata[(self.total_adata.obs[PERT_KEY] == pert_id) & (self.total_adata.obs[CELL_KEY] == cell_type)].X.toarray())

        return self.sample_pert(X_ctrl, X_pert)