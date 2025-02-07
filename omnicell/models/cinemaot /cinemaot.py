#Mohcene Maher Bouceneche, Abudayyeh-Gootenberg Laboratory.
#please make sure these libs are installed: numpy pandas scanpy scikit-learn scipy statsmodels anndata
import os
import pickle
import numpy as np
import scanpy as sc
import torch
from cinemaotlib.cinemaot import cinemaot_unweighted, cinemaot_weighted, synergy, attribution_scatter
from cinemaotlib import benchmark  # optional for evaluation routines
# from cinemaotlib import utils # can also import functions from utils if u want downstream visualizations

class CinemaOTModel:
    def __init__(self, config):
        """
        Initialize the model using parameters from YAML configuration.
        Expected configuration parameters:
          - name: identifier (should be "cinemaot")
          - mode: "parametric" (or "non_parametric")
          - dim: number of independent components (e.g., 20)
          - thres: threshold for confounder selection (e.g., 0.15)
          - smoothness: smoothness parameter for entropy regularization (e.g., 1e-4)
          - eps: convergence tolerance for the Sinkhorn-Knopp OT computation (e.g., 1e-3)
          - preweight_label: an optional column name (e.g., "cell_type0528") for weighting
          - weighted: boolean flag (if true, use cinemaot_weighted; otherwise, use unweighted)
        """
        self.name = config.name
        self.mode = config.parameters.get("mode", "parametric")
        self.dim = config.parameters.get("dim", 20)
        self.thres = config.parameters.get("thres", 0.15)
        self.smoothness = config.parameters.get("smoothness", 1e-4)
        self.eps = config.parameters.get("eps", 1e-3)
        self.preweight_label = config.parameters.get("preweight_label", None)
        self.weighted = config.parameters.get("weighted", False)
        
        self.cf = None   # Confounder embedding (numpy array)
        self.ot = None   # Optimal transport matrix (numpy array)
        self.de = None   # Differential expression matrix (as an AnnData object)
        #store additional outputs(synergy weights) if needed.

    def train(self, adata, **kwargs):
        """
        Run the analysis on AnnData object:
          - Selects the control (e.g., "ctrl") and experimental (e.g., "IFNb") groups.
          - Computes the confounder embedding, OT matrix, and differential expression matrix.
          
        Parameters:
          adata: pre-processed AnnData object.
          kwargs: Optional overrides such as:
                  - ref_label (default "ctrl")
                  - expr_label (default "IFNb")
                  - use_rep (if using weighted version)
                  
        Returns:
          A dictionary with keys "cf", "ot", and "de".
        """
        assert "pert" in adata.obs, "Missing perturbation labels in adata.obs"
        assert adata.obsm['embedding'].shape[1] == self.dim, "Embedding dimension mismatch"
        
        if not self.config.etl_config.log1p:
            logger.warning("CinemaOT expects log-normalized data. Check log1p in ETL config.")
        ref_label = kwargs.get("ref_label", "ctrl")
        expr_label = kwargs.get("expr_label", "IFNb")
        
        if self.weighted:
            self.cf, self.ot, self.de, self.weights = cinemaot_weighted(
                adata,
                obs_label="pert",
                ref_label=ref_label,
                expr_label=expr_label,
                use_rep=kwargs.get("use_rep", None),
                dim=self.dim,
                thres=self.thres,
                smoothness=self.smoothness,
                eps=self.eps,
                mode=self.mode,
                preweight_label=self.preweight_label
            )
        else:
            self.cf, self.ot, self.de = cinemaot_unweighted(
                adata,
                obs_label="pert",
                ref_label=ref_label,
                expr_label=expr_label,
                dim=self.dim,
                thres=self.thres,
                smoothness=self.smoothness,
                eps=self.eps,
                mode=self.mode,
                preweight_label=self.preweight_label
            )
        return {"cf": self.cf, "ot": self.ot, "de": self.de}

    def predict(self, adata, **kwargs):
        """
        Apply the learned OT mapping to transform new data.
        Try to transform cells in a condition by applying the OT matrix to reference cells from the "ctrl" condition.
        """
        assert adata.shape[0] == self.ot.shape[0], "Cell count mismatch"
        if self.cf is None or self.ot is None:
            raise RuntimeError("[X] Model has not been trained. Run train() first.")
        
        adata_new = adata.copy()
        condition = kwargs.get("condition", "IFNb")
        idx = adata_new.obs["pert"] == condition
        ref_cf = self.cf[adata_new.obs["pert"] == "ctrl", :]
        ot_norm = self.ot / np.sum(self.ot, axis=1)[:, None]
        adata_new.obsm["cf_transformed"] = self.cf.copy()
        adata_new.obsm["cf_transformed"][idx, :] = np.matmul(ot_norm, ref_cf)
        adata_new.X = adata_new.obsm["cf_transformed"]
        return adata_new.X  # or adata_new.obsm["cf_transformed"] #return adata_new

    def make_predict(self, ctrl_data, pert_id, cell_id):
        """just to be used by train.py"""
        return self.predict(ctrl_data)

    def save(self, filepath):
        """Save the model state (cf, ot, de, configuration)."""
        state = {
            "cf": self.cf,
            "ot": self.ot,
            "de": self.de,
            "config": {
                "name": self.name,
                "mode": self.mode,
                "dim": self.dim,
                "thres": self.thres,
                "smoothness": self.smoothness,
                "eps": self.eps,
                "preweight_label": self.preweight_label,
                "weighted": self.weighted,
            }
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.cf = state["cf"]
        self.ot = state["ot"]
        self.de = state["de"]

