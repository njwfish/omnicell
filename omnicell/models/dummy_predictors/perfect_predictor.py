
import scanpy as sc
import numpy as np

from omnicell.constants import *


"""
Predictor to test the rest of the pipeline. 

It takes the entire dataset at construction and then uses the ground truth to make predictions. Thus it is a perfect predictor.
"""
class PerfectPredictor():

    def __init__(self, adata: sc.AnnData):
        self.model = None
        self.total_adata = adata

    def train(self, adata: sc.AnnData, model_savepath: Path):
        """Does nothing because we are going to cheat"""
        pass

    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        ground_truth = self.total_adata[(self.total_adata.obs[PERT_KEY] == pert_id) & (self.total_adata.obs[CELL_KEY] == cell_type)]
        return ground_truth.X
    
