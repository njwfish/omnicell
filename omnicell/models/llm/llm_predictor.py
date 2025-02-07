import scanpy as sc
import torch
import yaml
import numpy as np
import math 
from omnicell.models.llm.llm import MAE
from omnicell.models.datamodules import get_dataloader

from torchcfm.conditional_flow_matching import *
import scanpy as sc

from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY
from tqdm import tqdm

class LLMPredictor():


    def __init__(self, config, input_size, device, pert_ids):
        self.model_config = config['model']
        self.trainig_config = config['training']
        self.device = device

        base_learning_rate = 5e-5

        self.model = model = MAE(
            input_size, 
            **self.model_config
        )


        #TODO: This should be in the config, not hardcoded, at least even if these are not changed we can see them in the run config
        base_learning_rate = 5e-5
        weight_decay = 0.0
        total_epoch = 10
        warmup_epoch = 10
        self.minibatch_size = 128 

        self.use_sparsity_loss = self.trainig_config['use_sparsity_loss']
        self.use_mask_task = self.trainig_config['use_mask_task']
        self.use_active_weights = self.trainig_config['use_active_weights']
        self.lr_step = self.trainig_config['lr_step']

        self.optim = torch.optim.AdamW(model.parameters(), lr=base_learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
        self.lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-5), 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=self.lr_func, verbose=True)
        # model = torch.load(f"...")
        # device = 'cpu'
        self.model = model.to(device)

        self.pert_ids = pert_ids

        print(pert_ids)
        

    #Should take care of saving the model under some results/model/checkpoints in 
    #BTW I think hidden dirs fuck with with the cluster, so don't call it .checkpoint
    def train(self, adata, model_savepath: Path):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        gene_map = {k: i for i, k in enumerate(adata.var.index)}
        gene_map = gene_map | {'NT': max(gene_map.values()) + 1}
        adata.obs['pert_type'] = adata.obs.gene.map(gene_map)
        pert_ids = np.array(adata.obs['pert_type'])
        pert_mat = np.arange(pert_ids.max() + 1)[:, None]

        adata.X = adata.X.toarray()
        adata.X = adata.X / adata.X.sum(axis=1)[:, None]
        adata.obsm["standard"] = adata.X

        dl = get_dataloader(adata, pert_reps=pert_mat, pert_ids=pert_ids)

        self.use_sparsity_loss = True
        self.use_mask_task = False
        self.use_active_weights = True
        self.lr_step = 32

        step_count = 0
        self.optim.zero_grad()
        pert_task = 0
        for e in range(20):
            self.model.train()
            losses = {'control': [], 'pert': []}
            for (bcontrol, bpert, bpert_index) in (pbar := tqdm(iter(dl))):
                bcontrol, bpert, bpert_index = bcontrol.squeeze(), bpert.squeeze(), bpert_index.reshape(-1, 1)# , # bpert_expr.squeeze()
                curr_batch_size = bcontrol.shape[0]
                for i in range(curr_batch_size // self.minibatch_size):
                    control = bcontrol[(i * self.minibatch_size):((i + 1) * self.minibatch_size)]
                    if control.shape[0] == 0:
                        continue
                    pert = bpert[(i * self.minibatch_size):((i + 1) * self.minibatch_size)]
                    pert_index = bpert_index[(i * self.minibatch_size):((i + 1) * self.minibatch_size)]
                    pert_index = pert_index.squeeze()
                    # print(pert_index)
                    active_weights = 100 * (control > 0).float().to(device) + 1 if self.use_active_weights else 1
                    pert_active_weights = 100 * (pert > 0).float().to(device) + 1 if self.use_active_weights else 1
                    control = control.float().to(device)
                    pert = pert.float().to(device)
                    step_count += 1

                    pert_expr = pert[torch.arange(pert.size(0)), pert_index, None]
                    control_results, pert_results = self.model(
                        control, pert_expr=pert_expr, pert_index=pert_index, mask=self.use_mask_task, recon_and_pert=True
                    )
                    
                    control_recon, pert_recon = control_results[0], pert_results[0]
                    control_loss = torch.sum(active_weights * torch.abs(control_recon - control)) / self.minibatch_size
                    pert_loss = torch.sum(pert_active_weights * torch.abs(pert_recon - pert)) / self.minibatch_size
                    
                    mean_pert, mean_pert_recon = pert.mean(axis=0), pert_recon.mean(axis=0)
                    mean_pert_loss = 10 * torch.sum(torch.abs(mean_pert - mean_pert_recon)) 
                    
                    std_pert, std_pert_recon = pert.std(axis=0), pert_recon.std(axis=0)
                    std_pert_loss = 10 * torch.sum(torch.abs(std_pert - std_pert_recon)) 
                    
                    if self.use_sparsity_loss and len(control_results) == 3:
                        control_sparsity, pert_sparsity = control_results[1], pert_results[1]
                        control_loss += torch.sum(active_weights * torch.abs(control_sparsity - (control > 0).float())) / self.minibatch_size
                        pert_loss += torch.sum(pert_active_weights * torch.abs(pert_sparsity - (pert > 0).float())) / self.minibatch_size

                    loss = (pert_loss + control_loss + mean_pert_loss + std_pert_loss)
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                    losses['control'].append(control_loss.item())
                    losses['pert'].append(pert_loss.item())
                    if step_count % self.lr_step == 0:
                        self.lr_scheduler.step()
                    pbar.set_description(
                        f"loss: {loss:.3f}, tv: {np.array(losses['control'])[-self.lr_step:].mean():.3f}, ptv: {np.array(losses['pert'])[-self.lr_step:].mean():.3f}"
                    )
            
            avg_loss = sum(losses['control']) / len(losses['control'])
            # torch.save(self.model, f"{save_dir}{e}")
            # writer.add_scalar('mae_loss', avg_loss, global_step=e)
            print(f'In epoch {e}, average traning loss is {avg_loss}.')
    

    #I mean to we need to evaluate anything? 
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        X = adata.obsm["standard"]
        X = X.toarray()
        X = X / X.sum(axis=1)[:, None]

        cell_type_names = adata.obs[CELL_KEY]
        pert_type_names = adata.obs[PERT_KEY]

        control_eval =  torch.tensor(X[cell_type_names == cell_type]).float()# .to(device)
        true_pert= torch.tensor(X[(pert_type_names == pert_id) & (cell_type_names == cell_type)]).float()# .to(device)
        curr_batch_size = min(control_eval.shape[0], true_pert.shape[0])
        pred_perts = []
        for i in range(curr_batch_size // self.minibatch_size + 1):
            print(i)
            control = control_eval[(i * self.minibatch_size):min(curr_batch_size, ((i + 1) * self.minibatch_size))].to(device)
            pert = true_pert[(i * self.minibatch_size):min(curr_batch_size, ((i + 1) * self.minibatch_size))].to(device)
            pert_expr = pert[torch.arange(pert.shape[0]), pert_id, None]
            pred_pert, _, = self.model(
                control, pert_expr=pert_expr, pert_index=pert_id, mask=False
            )
            pred_pert = control.cpu()
            control.cpu().numpy()
            
            pert.cpu().numpy()
            pred_perts.append(pred_pert.cpu().detach().numpy())
            torch.cuda.empty_cache()
        pred_pert = np.vstack(pred_perts)
        return pred_pert
