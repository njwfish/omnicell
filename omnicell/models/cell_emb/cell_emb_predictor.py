import os 
import scanpy as sc
import torch

import numpy as np
import pickle 

from omnicell.models.utils.datamodules import get_dataloader

from omnicell.models.cell_emb.cell_emb import *
from omnicell.models.flows.flow_utils import compute_conditional_flow
import pytorch_lightning as pl
from tqdm import tqdm

import logging

from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY

logger = logging.getLogger(__name__)

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Create a custom Dataset class
class TensorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

import math
import torch
import numpy as np
from torchdyn.core import NeuralODE
from omnicell.models.flows.flow_utils import torch_wrapper

def compute_conditional_flow(
    model, control, pert_ids, pert_mat, batch_size=32, num_steps=400, n_batches=1e8, true_bin=None
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    node = NeuralODE(
        torch_wrapper(model.flow).to(device), solver="dopri5", sensitivity="adjoint"
    )
    model = model.to(device)
    n_samples = min(control.shape[0], pert_ids.shape[0])
    n_batches = min(math.ceil(n_samples / batch_size), n_batches)
    preds = np.zeros((min(n_batches * batch_size, n_samples), control.shape[1]))
    with torch.no_grad():
        for i in range(n_batches):
            control_batch = control[batch_size*i:min(batch_size*(i+1), n_samples)]
            pert_batch = pert_mat[pert_ids][batch_size*i:min(batch_size*(i+1), n_samples)]# [:control_batch.shape[0]]
            model.flow.cond = pert_batch.to(device)
            inp = control_batch.float()
            inp = inp.to(device)
            
            idx = torch.arange(control.shape[1]).to(device)
            cell_embedding = model(inp)
            
            outp = node.trajectory(
                cell_embedding,
                t_span=torch.linspace(0, 1, num_steps)
            )
            outp = outp[-1, :, :]
            outp, outb = model.recon(cell_embedding, idx)
            if true_bin:
                outb = true_bin.to(device)
            outp = model.sparsify(outp, outb)
            outp = outp.cpu()
            preds[batch_size*i:batch_size*(i+1), :] = outp.squeeze()
            
    return preds

class CellEmbPredictor():
    def __init__(self, config, input_size, device):
        # self.model_config = config['model'] if config['model'] is not None else {}
        # self.trainig_config = config['training'] if config['training'] is not None else {}

        # self.max_epochs = self.trainig_config['max_epochs']

        self.model = MAE(
            input_size, 
            emb_dim=256, 
            encoder_layer=4,
            ff_dim=256
        )
        self.device = device
                
    #Should take care of saving the model under some results/model/checkpoints in 
    #BTW I think hidden dirs fuck with with the cluster, so don't call it .checkpoint
    def train(self, adata, model_savepath: Path):

        device = self.device

        logger.debug(f"Adata obsm keys: {adata.obsm}")

        gene_map = {k: i for i, k in enumerate(adata.var['gene'])}
        gene_map = gene_map | {CONTROL_PERT: max(gene_map.values()) + 1}
        # print(gene_map)
        self.gene_map = gene_map
        gene_unmap = {gene_map[k]: k for k in gene_map}
        perts = adata.obs[PERT_KEY].unique().map(gene_map)
        adata.obs['pert_type'] = adata.obs[PERT_KEY].map(gene_map)
        self.pert_ids = np.array(adata.obs['pert_type'])
        # print(self.pert_ids)
        # print(adata.obs[PERT_KEY])
        self.pert_mat = np.arange(self.pert_ids.max() + 1)[:, None]

        #TODO: Will this copy the data again? - We are already getting oom errors
        # adata.obsm['embedding'] = torch.Tensor(adata.obsm['embedding']).type(torch.float32)
        # adata.obsm['embedding'] = adata.obsm['embedding'].toarray()
        # adata.obsm['embedding'] = adata.obsm['embedding'] / adata.obsm['embedding'].sum(axis=1)[:, None]
        # adata.obsm["standard"] = adata.obsm['embedding']

        # Create an instance of the dataset
        dataset = TensorDataset(adata.obsm['embedding'])
        # ae_dl = DataLoader(dataset, batch_size=128, shuffle=True)
        paired_dl = get_dataloader(adata, pert_ids=self.pert_ids, pert_reps=self.pert_mat, collate='ot')

        base_learning_rate = 2e-4
        total_epoch = 1000
        warmup_epoch = 5
        optim = torch.optim.Adam(self.model.parameters(), lr=base_learning_rate)
        lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-5), 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

        
        self.model = self.model.to(device)

        lr_step = 32
        minibatch_size = 64

        step_count = 0
        optim.zero_grad()

        # Create a DataLoader
        ae_dl = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

        for e in range(10):
            self.model.train()
            losses = {'loss': [], 'control': [], 'pert': [], 'flow': []}
            for batch in (pbar := tqdm(iter(ae_dl))):
                step_count += 1
                batch = batch.to(device)
                batch_emb = self.model(batch)
                idx = torch.arange(batch.shape[1]).to(device)
                loss, batch_recon, batch_bin_recon = self.model.ae_loss(batch_emb, batch, idx, return_recon=True)
                loss.backward()
                optim.step()
                optim.zero_grad()
                with torch.no_grad():
                    batch_recon = self.model.sparsify(batch_recon, batch_bin_recon)
                    recon_loss = torch.mean((batch_recon - batch)**2)
                losses['loss'].append(recon_loss.item())
                if step_count % lr_step == 0:
                    lr_scheduler.step()
                pbar.set_description(
                    f"loss: {np.array(losses['loss'])[-lr_step:].mean():.3f}"
                )
            
            avg_loss = sum(losses['loss']) / len(losses['loss'])
            # torch.save(model, f"{save_dir}{e}")
            # writer.add_scalar('mae_loss', avg_loss, global_step=e)
            print(f'In epoch {e}, average traning loss is {avg_loss}.')

        # step_count = 0
        optim.zero_grad()
        minibatch_size = 32
        for e in range(5):
            step_count = 0
            losses = {'loss': [], 'control': [], 'pert': [], 'flow': []}
            for (bcontrol, bpert, bpert_index) in (pbar := tqdm(iter(paired_dl))):
                bcontrol, bpert, bpert_index = bcontrol.squeeze(), bpert.squeeze(), bpert_index.reshape(-1, 1)# , # bpert_expr.squeeze()
                curr_batch_size = bcontrol.shape[0]
                for i in range(curr_batch_size // minibatch_size):
                    ctrl = bcontrol[(i * minibatch_size):((i + 1) * minibatch_size)]
                    if ctrl.shape[0] == 0:
                        continue
                    pert = bpert[(i * minibatch_size):((i + 1) * minibatch_size)]
                    pert_index = bpert_index[(i * minibatch_size):((i + 1) * minibatch_size)]
                    pert_index = pert_index.squeeze()
                    
                    ctrl = ctrl.float().to(device)
                    pert = pert.float().to(device)
                    
                    idx = torch.arange(ctrl.shape[1]).to(device)
                    
                    step_count += 1
                    
                    ctrl_emb = self.model(ctrl)
                    ctrl_loss = self.model.ae_loss(ctrl_emb, ctrl, idx)
                    
                    pert_emb = self.model(pert)
                    pert_loss = self.model.ae_loss(pert_emb, pert, idx)
                    
                    cond = self.model.gene_embedding.pos[:, pert_index][0]
                    flow_loss = self.model.flow_loss(ctrl_emb, pert_emb, cond)
                    
                    loss = ctrl_loss + pert_loss + flow_loss
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
                    losses['loss'].append(loss.item())
                    losses['control'].append(ctrl_loss.item())
                    losses['pert'].append(pert_loss.item())
                    losses['flow'].append(flow_loss.item())
                    if step_count % lr_step == 0:
                        lr_scheduler.step()
                    pbar.set_description(
                        f"loss: {np.array(losses['loss'])[-lr_step:].mean():.3f}, tv: {np.array(losses['control'])[-lr_step:].mean():.3f}, ptv: {np.array(losses['pert'])[-lr_step:].mean():.3f}, flow: {np.array(losses['flow'])[-lr_step:].mean():.3f}"
                    )
                if step_count > 2_000:
                    break
            avg_loss = sum(losses['control']) / len(losses['control'])
            # torch.save(model, f"{save_dir}{e}")
            # writer.add_scalar('mae_loss', avg_loss, global_step=e)
            print(f'In epoch {e}, average traning loss is {avg_loss}.')

    def save(self, path):
        torch.save(self.model.state_dict(), f"{path}/model.pth")
        with open(f'{path}/gene_map.pkl', 'wb') as handle:
            pickle.dump(self.gene_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(f"{path}/model.pth"))
            with open(f'{path}/gene_map.pkl', 'rb') as handle:
                self.gene_map = pickle.load(handle)
            return True
        return False
    
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        # self.model = self.model.to(device)
        cell_types = adata.obs[CELL_KEY].values
        pert_types = adata.obs[PERT_KEY]
        control_eval = torch.tensor(adata[(cell_types == cell_type) & (pert_types == CONTROL_PERT)].obsm['embedding'])# .float().to(device)
        pred = compute_conditional_flow(
            self.model,
            control_eval, 
            torch.tensor(np.repeat(self.gene_map[pert_id], control_eval.shape[0])), 
            self.model.gene_embedding.pos[0]
        )  
        return pred
