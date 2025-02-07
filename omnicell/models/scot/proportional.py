import sys
import yaml
import torch
import logging
from pathlib import Path

# Add the path to the directory containing the omnicell package
# Assuming the omnicell package is in the parent directory of your notebook
sys.path.append('..')  # Adjust this path as needed

import numpy as np
import scanpy as sc
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT, logger
from omnicell.models.distribute_shift import sample_pert, get_proportional_weighted_dist
from omnicell.processing.utils import to_dense

class ProportionalSCOT():

    def __init__(self, adata: sc.AnnData):
        self.model = None
        self.total_adata = adata

    def train(self, adata: sc.AnnData):
        """Does nothing because we are going to cheat"""
        pass

    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        X_ctrl = to_dense(self.total_adata[(self.total_adata.obs[PERT_KEY] == CONTROL_PERT) & (self.total_adata.obs[CELL_KEY] == cell_type)].X.toarray())
        X_pert = to_dense(self.total_adata[(self.total_adata.obs[PERT_KEY] == pert_id) & (self.total_adata.obs[CELL_KEY] == cell_type)].X.toarray())
        mean_ctrl = X_ctrl.mean(axis=0)
        mean_pert = X_pert.mean(axis=0)
        
        mean_shift = mean_pert - mean_ctrl
        logger.debug(f"Mean shift shape: {mean_shift.shape}")

        weighted_dist = get_proportional_weighted_dist(X_ctrl)
        
        preds = sample_pert(X_ctrl, weighted_dist, mean_shift, max_rejections=100)
        return preds
    