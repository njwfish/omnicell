import pertpy as pt
import scanpy as sc
#from omnicell.models.base import SJIKMSKM  

class CinemaOTModel():    
    def __init__(self, config, pert_key="perturbation", control="No stimulation", device=None):
        super().__init__(config)
        self.cot = pt.tl.Cinemaot()
        self.pert_key = pert_key
        self.control = control
        self.device = device  # Not used but kept for interface consistency
        
        self.thres = config.get("thres", 0.5)
        self.smoothness = config.get("smoothness", 1e-5)
        self.eps = config.get("eps", 1e-3)
        self.solver = config.get("solver", "Sinkhorn")
        self.preweight_label = config.get("preweight_label", None)
        
        self.de = None  # Treatment-effect AnnData
        self.ad = None  # Subsampled AnnData (for CINEMA-OT-W)

    def train(self, adata, **kwargs):
        # run CINEMA-OT causal effect analysis.
        # PCA (required by CINEMA-OT)
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata)
        
        # causal effect analysis
        if self.preweight_label:
            # Supervised mode with cell-type labels
            self.de = self.cot.causaleffect(
                adata,
                pert_key=self.pert_key,
                control=self.control,
                return_matching=True,
                thres=self.thres,
                smoothness=self.smoothness,
                eps=self.eps,
                solver=self.solver,
                preweight_label=self.preweight_label
            )
        else:
            # Unsupervised CINEMA-OT-W
            self.ad, self.de = self.cot.causaleffect_weighted(
                adata,
                pert_key=self.pert_key,
                control=self.control,
                return_matching=True,
                thres=self.thres,
                smoothness=self.smoothness,
                eps=self.eps,
                solver=self.solver
            )
        return self.de

    def predict(self, adata, **kwargs):
        #Return treatment-effect matrix (de.X)
        if self.de is None:
            raise ValueError("Model not trained. Call .train() first.")
        return self.de.X 

    def make_predict(self, ctrl_data, pert_id, cell_id=None):
        # ctrl_data: Control AnnData (unused here since CINEMA-OT handles pairing)
        return self.de.X  #or return ctrl_data.X + self.de.X

    def save(self, path):
        if self.de is not None:
            self.de.write(f"{path}/treatment_effect.h5ad")
        if self.ad is not None:
            self.ad.write(f"{path}/confounder_data.h5ad")

    def load(self, path):
        try:
            self.de = sc.read_h5ad(f"{path}/treatment_effect.h5ad")
            return True
        except FileNotFoundError:
            return False

    def synergy_effect(self, adata, treatment_A, treatment_B, combo_name):
        return self.cot.synergy(
            adata,
            pert_key=self.pert_key,
            base=self.control,
            A=treatment_A,
            B=treatment_B,
            AB=combo_name,
            thres=self.thres,
            smoothness=self.smoothness,
            eps=self.eps,
            solver=self.solver
        )
