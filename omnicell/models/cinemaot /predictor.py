import scanpy as sc
from pathlib import Path
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT
from omnicell.models.cinemaot.cinemaot_model import CinemaOTPredictor
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CinemaOTPredictorWrapper:

    def __init__(self, device, model_config: dict):
        self.device = device
        self.model_config = model_config
        self.cinemaot_model = None

    def train(self, adata: sc.AnnData, path: Path):
        # Ensure the data is in the correct format
        adata.obs["perturbation"] = adata.obs[PERT_KEY]
        adata.obs["cell_type"] = adata.obs[CELL_KEY]
        adata.var["gene_name"] = adata.var_names

        # Initialize and train CINEMA-OT
        self.cinemaot_model = CinemaOTPredictor(self.model_config, device=self.device)
        self.cinemaot_model.train(adata)

    def make_predict(self, ctrl_data, pert_id, cell_id=None):
        if self.cinemaot_model is None:
            raise ValueError("Model not trained. Call .train() first.")
        
        # Return treatment-effect matrix
        return self.cinemaot_model.make_predict(ctrl_data, pert_id, cell_id)
