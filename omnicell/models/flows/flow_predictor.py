import os 
import scanpy as sc
import torch

import numpy as np

from omnicell.models.utils.datamodules import get_dataloader

from omnicell.models.flows.arch import CMHA, CMLP, CFM, CFMC, CMLPC
from omnicell.models.flows.flow_utils import compute_conditional_flow
from pytorch_lightning.callbacks import TQDMProgressBar
import pytorch_lightning as pl
from pathlib import Path

import logging

from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY

logger = logging.getLogger(__name__)

class FlowPredictor():
    def __init__(self, config, input_size, pert_embedding):
        self.model_config = config['model'] if config['model'] is not None else {}
        self.trainig_config = config['training'] if config['training'] is not None else {}


        pert_keys = list(pert_embedding.keys())
        pert_rep = np.array([pert_embedding[k] for k in pert_keys])
        self.pert_map = {k: i for i, k in enumerate(pert_keys)}

        self.max_epochs = self.trainig_config['max_epochs']

        # self.pert_rep = pert_rep

        
        if config['arch'] == 'mlp':
            self.model = CMLP(training_module=CFM, feat_dim=input_size, cond_dim=pert_rep.shape[1], time_varying=True, **self.model_config)
        elif config['arch'] == 'mlpc':
            self.model = CMLPC(training_module=CFMC, feat_dim=input_size, cond_dim=pert_rep.shape[1], time_varying=True, **self.model_config)
        else:
            raise NotImplementedError(f"Model architecture {self.model_config['arch']} not implemented")
        

    #Should take care of saving the model under some results/model/checkpoints in 
    #BTW I think hidden dirs fuck with with the cluster, so don't call it .checkpoint
    def train(self, adata, model_savepath: Path):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.debug(f"Adata obsm keys: {adata.obsm}")

        #TODO: Will this copy the data again? - We are already getting oom errors
        self.pert_ids = np.array(adata.obs[PERT_KEY].values) # .map(self.pert_map | {'NT': -1}).values.astype(int) 
        # adata.obsm['embedding'] = torch.Tensor(adata['embedding']).type(torch.float32)

        dset, ns, dl = get_dataloader(adata, pert_ids=self.pert_ids, pert_map=self.pert_map, collate='cfm')

        logger.info(f"Training model")
        # Train the model
        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=1,
            max_epochs=self.max_epochs,
            logger=True)
        
        self.model = self.model.to(device)            
        trainer.fit(self.model, dl)

    def save(self, path):
        torch.save(self.model.state_dict(), f"{path}/model.pth")

    def load(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(f"{path}/model.pth"))
            return True
        return False
    

    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        cell_types = adata.obs[CELL_KEY].values
        control_eval = adata[cell_types == cell_type].X
        traj = compute_conditional_flow(
            self.model, 
            control_eval, 
            np.repeat(0, control_eval.shape[0]), 
            self.pert_map[pert_id][None, :],
            n_batches = 5 
        )  
        return traj[-1, :, :]
