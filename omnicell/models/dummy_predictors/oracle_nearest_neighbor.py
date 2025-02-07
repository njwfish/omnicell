
from turtle import distance
import scanpy as sc
import pandas as pd
import numpy as np

from omnicell.constants import *

import logging

logger = logging.getLogger(__name__)
"""
Predictor to test the rest of the pipeline. 

It takes the entire dataset at construction and then uses the ground truth to make predictions. Thus it is a perfect predictor.
"""
class OracleNNPredictor():

    def __init__(self, adata: sc.AnnData, model_config: dict):

        assert "DEGs" in adata.uns.keys(), "DEGs not found in adata.uns"

        self.model = None
        self.total_adata = adata

        self.DEGs = adata.uns["DEGs"].copy()
        self.number_top_DEGs_overlap = model_config["number_top_DEGs_overlap"]

        for cell in self.DEGs:
            for pert in self.DEGs[cell]:
                df = pd.DataFrame.from_dict(self.DEGs[cell][pert], orient='index')
                df = df[df['pvals_adj'] < model_config['p_threshold']]
                self.DEGs[cell][pert] = df


    

    def train(self, adata: sc.AnnData, model_savepath: Path):
        """Does nothing because we are going to cheat"""
        pass



    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:

        DEGs_cell_type = self.DEGs[cell_type]

        non_target_perts_in_cell = [p for p in DEGs_cell_type.keys() if p != pert_id]

        DEG_overlaps = {}
        DEG_target = DEGs_cell_type[pert_id]

        for p in non_target_perts_in_cell:
            DEG_p = DEGs_cell_type[p]
            DEG_overlaps[p] = len(DEG_target.index[:self.number_top_DEGs_overlap].intersection(DEG_p.index[:self.number_top_DEGs_overlap]))

        closest_pert = max(DEG_overlaps, key=DEG_overlaps.get)

        
        logger.info(f"Selected perturbation {closest_pert} as closest to {pert_id} in cell type {cell_type}")

        nn_population = self.total_adata[(self.total_adata.obs[PERT_KEY] == closest_pert) & (self.total_adata.obs[CELL_KEY] == cell_type)]

        #sampled_gt = sc.pp.subsample(ground_truth, n_obs=number_preds, replace=True, copy=True)

        return nn_population.X
    
