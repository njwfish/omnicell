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
from omnicell.models.utils.distribute_shift import sample_pert, get_proportional_weighted_dist
from omnicell.processing.utils import to_dense
from omnicell.models.utils.datamodules import get_dataloader



from omnicell.models.scot.sampling_utils import batch_pert_sampling, sample_pert_from_model_numpy, generate_batched_counterfactuals

import logging

logger = logging.getLogger(__name__)
class ProportionalSCOT():


    #TODO: FIX THIS, IT SHOULD NOT TAKE A PERT EMBEDDING, but we give it one because data loader needs it, but in fact we are not even using the data loader
    def __init__(self, total_adata: sc.AnnData, pert_embedding: str, model_params: dict):
        self.model = None
        self.numpy_model = True
        self.batch_size = model_params["batch_size"]
        self.pert_embeddding = pert_embedding
        self.total_adata = total_adata

    def train(self, adata: sc.AnnData, save_path: str):
        """Does nothing because we are going to cheat"""
        pass


    def generate_synthetic(self, adata: sc.AnnData):
        """Yields stratified synthetic data for each stratum in the training adata that is passed"""

        logger.info(f"Fenerating synthetic data for Proportional scot, with - # control samples: {len(adata[(adata.obs[PERT_KEY] == CONTROL_PERT)])} - # perturbations {adata.obs[PERT_KEY].nunique()}")
        
        dset, _ = get_dataloader(adata, pert_ids=np.array(adata.obs[PERT_KEY].values), offline=False, pert_embedding=self.pert_embeddding, collate='ot')

        for x in generate_batched_counterfactuals(self, dset, batch_size=self.batch_size):
            yield x



    def __call__(self, ctrl, mean_shift):
        """Returns a weighted distribution over the given control data."""
        return get_proportional_weighted_dist(ctrl)

    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        X_ctrl = self.total_adata[(self.total_adata.obs[PERT_KEY] == CONTROL_PERT) & (self.total_adata.obs[CELL_KEY] == cell_type)].X
        X_pert = self.total_adata[(self.total_adata.obs[PERT_KEY] == pert_id) & (self.total_adata.obs[CELL_KEY] == cell_type)].X
        
        return batch_pert_sampling(self, X_ctrl, X_pert)
            