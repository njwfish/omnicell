import sys
import yaml
import torch
import logging
from pathlib import Path
import time

# Add the path to the directory containing the omnicell package
# Assuming the omnicell package is in the parent directory of your notebook

import numpy as np
import scanpy as sc
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT, logger
from omnicell.models.distribute_shift import sample_pert, get_proportional_weighted_dist
from omnicell.processing.utils import to_dense

from omnicell.models.scot.sampling_utils import batch_pert_sampling, sample_pert_from_model_numpy

import logging

logger = logging.getLogger(__name__)
class ProportionalSCOT():

    def __init__(self, adata: sc.AnnData):
        self.model = None
        self.total_adata = adata
        self.numpy_model = True

    def train(self, adata: sc.AnnData, save_path: str):
        """Does nothing because we are going to cheat"""
        pass

    def __call__(self, ctrl, mean_shift):
        return get_proportional_weighted_dist(ctrl)

    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        X_ctrl = self.total_adata[(self.total_adata.obs[PERT_KEY] == CONTROL_PERT) & (self.total_adata.obs[CELL_KEY] == cell_type)].X
        X_pert = self.total_adata[(self.total_adata.obs[PERT_KEY] == pert_id) & (self.total_adata.obs[CELL_KEY] == cell_type)].X
        
        return batch_pert_sampling(self, X_ctrl, X_pert)
            