import scanpy as sc
from pathlib import Path
from omnicell.constants import *
from omnicell.models.gears.gears import GEARS
from omnicell.models.gears.pertdata import PertData
import numpy as np
logger = logging.getLogger(__name__)

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

class GEARSPredictor():

    def __init__(self, device, model_config: dict):
        self.seen_cells = []
        self.model_config = model_config
        self.device = device
        self.gears_model = None

    

    def train(self, adata: sc.AnnData, path: Path):

        self.seen_cells = adata.obs[CELL_KEY].unique()
        assert len(self.seen_cells) == 1, "Only one cell type is allowed in the dataset for model GEARS"

        data_path = (path / "data").resolve()
        model_path = (path / "model").resolve()





        adata.obs["condition"] = adata.obs[PERT_KEY]
        adata.obs["cell_type"] = adata.obs[CELL_KEY]
        perts = [p for p in adata.obs["condition"].unique() if p != CONTROL_PERT]
        adata.obs["condition"] = adata.obs["condition"].replace({CONTROL_PERT:"ctrl"})
        adata.obs["condition"] = adata.obs["condition"].replace({p:p+"+ctrl" for p in perts})
        adata.var["gene_name"] = adata.var_names



        pert_data = PertData(data_path.resolve()) # specific saved folder

        logger.debug(f"Preparing new data process with skip calc de")

        #C.f. Implementation Readme on why 
        pert_data.new_data_process(dataset_name = "gears", adata = adata, skip_calc_de=True) # specific dataset name and adata object
        
        logger.debug(f"Preparing split with seed")
        pert_data.prepare_split(split = 'no_test', seed = 1) # get data split with seed

        logger.debug(f"Preparing dataloader")
        pert_data.get_dataloader(batch_size = self.model_config['batch_size'], test_batch_size = self.model_config['test_batch_size']) # prepare data loader



        gears_model = GEARS(pert_data, device = self.device, 
                                weight_bias_track = self.model_config['weight_bias_track'], 
                                proj_name = 'pertnet', 
                                exp_name = 'pertnet')

        gears_model.model_initialize(hidden_size = self.model_config['hidden_size'],)

        gears_model.tunable_parameters()


        gears_model.train(epochs = self.model_config['epochs'], lr = self.model_config['lr'])


        self.gears_model = gears_model


        #We don't need to save the model for now
        #gears_model.save_model(model_path)
        #gears_model.load_pretrained(model_path)






    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:

        prediction_cell_types = adata.obs[CELL_KEY].unique()
        assert all(prediction_cell_types == self.seen_cells), "The cell type in the prediction data is not the same as the training data, model GEARS does not support multiple cell types"


        #GEARs returns bulk predictions, we transform those in single cell predictions
        bulk_pred = self.gears_model.predict([[pert_id]])[pert_id]
        bulk_pred = np.array(bulk_pred)


        control_cells = adata[(adata.obs[PERT_KEY] == CONTROL_PERT) & (adata.obs[CELL_KEY] == cell_type)].X

        mean_control = control_cells.mean(axis=0)
        mean_shift = bulk_pred - mean_control

        res = None
        if self.model_config['distribution_strategy'] == 'jason':
            res = distribute_shift(control_cells, mean_shift)
        else: 
            raise ValueError(f"Invalid distribution strategy: {self.model_config['distribution_strategy']}")


        return res 




