
import scanpy as sc
import numpy as np

from omnicell.constants import *


"""
Returns the Sparsisty pattern of the ground truth
"""
class SparsityGroundTruthPredictor():

    def __init__(self, adata: sc.AnnData):
        self.model = None
        self.total_adata = adata

    def train(self, adata: sc.AnnData, model_savepath: Path):
        """Does nothing because we are going to cheat"""
        pass

    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        ground_truth = self.total_adata[(self.total_adata.obs[PERT_KEY] == pert_id) & (self.total_adata.obs[CELL_KEY] == cell_type)]
        return (ground_truth.X > 0).astype(np.float32)
    
