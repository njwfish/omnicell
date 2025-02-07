
import scanpy as sc
import numpy as np

from omnicell.constants import *

def distribute_shift(ctrl_cells, mean_shift):
    """
    Distribute the global per-gene difference (sum_diff[g]) across cells in proportion
    to the cell's existing counts for that gene. 
    """ 
    ctrl_cells = ctrl_cells.copy()
    sum_shift = (mean_shift * ctrl_cells.shape[0]).astype(int)

    n_cells, n_genes = ctrl_cells.shape


    #Its a matrix right now
    sum_shift = np.squeeze(np.array(sum_shift))

    #For each gene, distribute sum_diff[g] using a single multinomial draw
    for g in range(n_genes):
        diff = int(sum_shift[g])
        if diff == 0:
            continue  

        # Current counts for this gene across cells
        gene_counts = ctrl_cells[:, g].astype(np.float64)

        current_total = gene_counts.sum().astype(np.float64)
        

        # Probabilities ~ gene_counts / current_total
        p = gene_counts / current_total


        if diff > 0:
            # We want to add `diff` counts
            draws = np.random.multinomial(diff, p)  # shape: (n_cells,)
            
            ctrl_cells[:, g] = gene_counts + draws
        else:
            if current_total <= 0:
                continue

            # We want to remove `abs(diff)` counts
            amt_to_remove = abs(diff)

            to_remove = min(amt_to_remove, current_total)
            draws = np.random.multinomial(to_remove, p)
            # Subtract, then clamp
            updated = gene_counts - draws
            updated[updated < 0] = 0
            ctrl_cells[:, g] = updated

    return ctrl_cells


"""
Predictor to test the rest of the pipeline. 

It takes the entire dataset at construction and then returns a prediction based on the mean shift distributed on the control population.
"""
class MeanShiftDistributionPredictor():

    def __init__(self, adata: sc.AnnData):
        self.model = None
        self.total_adata = adata

    def train(self, adata: sc.AnnData, model_savepath: Path):
        """Does nothing because we are going to cheat"""
        pass

    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        
        adata_ctrl = self.total_adata[(self.total_adata.obs[PERT_KEY] == pert_id) & (self.total_adata.obs[CELL_KEY] == cell_type)]
        adata_pert = self.total_adata[(self.total_adata.obs[PERT_KEY] == pert_id) & (self.total_adata.obs[CELL_KEY] == cell_type)]

        mean_ctrl = adata_ctrl.X.mean(axis=0)
        mean_pert = adata_pert.X.mean(axis=0)
        
        mean_shift = mean_pert - mean_ctrl
        logger.debug(f"Mean shift shape: {mean_shift.shape}")

        preds = distribute_shift(adata_ctrl.X, mean_shift)
        return preds

         
