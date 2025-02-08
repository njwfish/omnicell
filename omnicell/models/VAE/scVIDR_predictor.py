from torch import nn, optim
import numpy as np
import scanpy as sc
import torch
import logging
from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY
from omnicell.models.VAE.vae import Net
from omnicell.models.utils.early_stopping import EarlyStopper
from omnicell.processing.utils import to_dense

logger = logging.getLogger(__name__)


"""
So We don't have to import sklearn
"""
class LinearRegression(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
    
    def forward(self, x):
        return self.linear(x)


class ScVIDRPredictor():
    """
    
    """
    def __init__(self, config, input_dim, device, pert_ids):
        self.training_config = config['training']
        self.model_config = config['model']
        self.input_dim = input_dim
        self.device = device
        self.model = Net(input_dim, self.model_config['hidden_dim'],
                         self.model_config['latent_dim'], 
                         self.training_config['alpha'],
                         self.training_config['dropout_rate'])
        
        self.epochs = self.training_config['epochs']
        self.delta_predictor_epochs = self.training_config['delta_predictor_epochs']
        self.batsize = self.training_config['batsize']
        self.model.to(device)

        #Stores the latent space shift associated which each perturbation
        self.perts = pert_ids
        self.deltas =  None
        self.perts_to_idx = {pert: idx for idx, pert in enumerate(pert_ids)}

        logger.debug(f"VAE predictor initialized, perturbations: {self.perts}, perts to idx: {self.perts_to_idx}")
        self.cell_ids = None
        self.invalid_perts = set()


    #Note this model needs the entire data or sth like that. 
    #The mean operations are computed on the entire dataset.
    def train(self, adata, model_savepath: Path):
        self.cell_ids = adata.obs[CELL_KEY].unique()
        device = self.device
        epochs = self.epochs
        batsize = self.batsize

        #This excludes any cell type that does not have a perturbation
        cell_types_with_pert = adata.obs[adata.obs[PERT_KEY] != CONTROL_PERT][CELL_KEY].unique()

        #TODO bad if we start passing batches to the model, it will see only part of the data and the len of the data will be wrong.
        datalen = len(adata)
        indices = np.random.permutation(datalen)

        train = adata  # 90% training data 

        #Compute the deltas for each perturbation
        cell_means = []

        for cell in cell_types_with_pert:
            cell_ctrl_adata = train[(train.obs[CELL_KEY] == cell) & (train.obs[PERT_KEY] == CONTROL_PERT)]
            logger.debug(f"Shape of ctrl data for cell {cell}: {cell_ctrl_adata.shape}")
            cell_data = torch.from_numpy(to_dense(cell_ctrl_adata.X)).to(device)
            cell_mean = cell_data.mean(axis=0).cpu().detach()
            cell_means.append(cell_mean)

        cell_means = torch.stack(cell_means)
        logger.debug(f"Cell means shape: {cell_means.shape}")

        #Computing mean of means --> equal weighting per class
        cell_mean = cell_means.mean(axis=0)
        logger.debug(f"Cell mean shape: {cell_mean.shape}")
        logger.debug(f"Cell mean norm: {torch.norm(cell_mean)}")

        self.regressors = []

        # Altough scgen theoretically only uses one perturbation, we will use multiple for the sake of generality
        # On Kang the number of perts is 1
        for i in range(len(self.perts)):
            model = LinearRegression(self.model_config['latent_dim'], self.model_config['latent_dim'])
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            #Contains [cell_type_mean, delta] tuples for each pair on which this perturbation is applied
            x_and_y_data = []
            for j, cell in enumerate(self.cell_ids):
                logger.debug(f'Computing mean for perturbation {self.perts[i]} and cell type {cell}')
                cell_pert_adata = adata[(adata.obs[CELL_KEY] == cell) & (adata.obs[PERT_KEY] == self.perts[i])]
                #Due to iid mode we might not have all perturbations for all cell types
                if len(cell_pert_adata) > 0:
                    pert_data = torch.from_numpy(to_dense(cell_pert_adata.X)).to(device)
                    pert_mean = pert_data.mean(axis=0).cpu().detach()
                    delta = pert_mean - cell_mean
                    x_and_y_data.append((cell_mean, delta))
                else:
                    logger.info(f'Cell {cell} does not have perturbation {self.perts[i]}')
               
            if len(x_and_y_data) == 0:
                self.invalid_perts.add(self.perts[i])
                logger.warning(f'Perturbation {self.perts[i]} is not applied to any cell in the training data - Delta linear model will be None and trying to transfer this perturbation to unseen cells will cause an undefined state.')
                self.regressors.append(None)
            else:
                logger.info(f"Training regressor to predict delta for perturbation {self.perts[i]}")
                for _ in range(self.delta_predictor_epochs):
                    logger.debug(f"Epoch {_+1}/{self.delta_predictor_epochs}")
                    #Batch size is 1
                    for X, y in x_and_y_data:
                        outputs = model(X)
                        loss = criterion(outputs, y)
                    
                        # Backward pass and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                self.regressors.append(model)

    #Predicting perturbations --> How do we compute the means? 
    #--> We need a delta for each perturbation
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        assert len(adata.obs[CELL_KEY].unique()) == 1, 'Input data contains multiple cell types'
        assert len(adata.obs[PERT_KEY].unique()) == 1, 'Input data contains multiple perturbations'
        assert adata.obs[CELL_KEY].unique() == [cell_type], f'Cell type {cell_type} not in the provided data'
        assert adata.obs[PERT_KEY].unique() == [CONTROL_PERT], 'Input data contains non control perturbations'  
        
        if pert_id in self.invalid_perts:
            raise ValueError(f'Perturbation {pert_id} is not applied to any cell in the training data - Delta is NaN')

        logger.info(f'Predicting seen perturbation {pert_id} for unseen cell type {cell_type}')

        data = to_dense(adata.X)
        data = torch.from_numpy(data.astype(np.float32)).to(self.device)
        # mu, logvar = self.model.encode(data)

        logger.debug(f"Pert idx = {self.perts_to_idx[pert_id]}")

        with torch.no_grad():
            pert_predictor = self.regressors[self.perts_to_idx[pert_id]]
            pert_predictor.eval()
            pert_delta = pert_predictor(mu.cpu())

        pert_mu = mu + pert_delta.to(self.device)

        z = self.model.reparameterize(pert_mu, logvar)

        return self.model.decode(z).cpu().detach().numpy()

