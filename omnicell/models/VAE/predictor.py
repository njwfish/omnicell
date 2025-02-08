from torch import nn, optim
import numpy as np
import scanpy as sc
import torch
import logging
from typing import Optional
from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY
from omnicell.models.VAE.vae import Net
from omnicell.models.utils.early_stopping import EarlyStopper
from omnicell.processing.utils import to_dense

logger = logging.getLogger(__name__)


class VAEPredictor():
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
    def train(self, adata, **kwargs):

        self.cell_ids = adata.obs[CELL_KEY].unique()
        device = self.device
        epochs = self.epochs
        batsize = self.batsize

        #This excludes any cell type that does not have a perturbation
        cell_types_with_pert = adata.obs[adata.obs[PERT_KEY] != CONTROL_PERT][CELL_KEY].unique()

        #TODO bad if we start passing batches to the model, it will see only part of the data and the len of the data will be wrong.
        datalen = len(adata)
        indices = np.random.permutation(datalen)

        train = adata[indices[:np.int32(datalen*0.9)]]  # 90% training data
        valid = adata[indices[np.int32(datalen*0.9):]]         

        trainlen = train.shape[0]
        validlen = valid.shape[0]

        logger.debug(f"Train data shape: {train.shape}")
        logger.debug(f"Validation data shape: {valid.shape}")

        net = self.model
        net.to(device)  # Model for training
    
        optimizer = optim.Adam(net.parameters(), lr=self.training_config['learning_rate'])  # Adam optimizer
        running_loss = 0


        train_X = to_dense(train.X)
        valid_X = to_dense(valid.X)

        train_X = torch.from_numpy(train_X.astype(np.float32))  # Convert train data to torch tensor and move to device
        valid_X = torch.from_numpy(valid_X.astype(np.float32))  # Convert validation data to torch tensor and move to device

        logger.debug(f"Training data shape: {train_X.shape}")
        logger.debug(f"Validation data shape: {valid_X.shape}")
        logger.debug(f"Batch size: {batsize}")
        logger.info(f'Training VAE model for {epochs} epochs')

        #Training loop
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            running_loss = 0

            net.train()  # Set model to training mode
        
            for lower in range(0, trainlen, batsize):
                upper = min(lower + batsize, trainlen)
                lower = min(trainlen - batsize, lower)
                batch = train_X[lower:upper, :].to(device)
                optimizer.zero_grad()  
                out, mu, logvar = net(batch)  # Forward pass
                
                #Printing batch details once per epoch.
                if lower == 0:
                    logger.debug(f'Batch shape: {batch.shape}')
                    logger.debug(f'Output shape: {out.shape} - mu shape: {mu.shape} - logvar shape: {logvar.shape}')
                    
                loss = net.loss_function(out, batch, mu, logvar) 
                loss.backward() 
                running_loss += loss.item()
                optimizer.step() 
            logger.info(f'Train loss: {running_loss/1000000}')
        
            running_loss = 0

            net.eval()  # Set model to evaluation mode

            # Validation loop
            with torch.no_grad():
                for lower in range(0, validlen, batsize):
                    upper = min(lower + batsize, validlen)
                    lower = min(validlen - batsize, lower)
                    batch = valid_X[lower:upper, :].to(device)
                    out, mu, logvar = net(batch)
                    loss = net.loss_function(out, batch, mu, logvar)
                    running_loss += loss.item()
                logger.info(f'Valid loss: {running_loss/1000000}')

            #Early stopping
            early_stopper = EarlyStopper(patience=10, min_delta=0.01)

            if early_stopper.early_stop(loss.item()):
                logger.info(f'Early stopping after {e+1} epochs')
                break


        
        net.eval()  # Set model to evaluation mode
        #Compute the deltas for each perturbation
        cell_means = []

        for cell in cell_types_with_pert:
            cell_ctrl_adata = train[(train.obs[CELL_KEY] == cell) & (train.obs[PERT_KEY] == CONTROL_PERT)]
            logger.debug(f"Shape of ctrl data for cell {cell}: {cell_ctrl_adata.shape}")
            cell_data = torch.from_numpy(to_dense(cell_ctrl_adata.X)).to(device)
            mu, _ = net.encode(cell_data)
            cell_mean = mu.mean(axis=0).cpu().detach()
            cell_means.append(cell_mean)

        cell_means = torch.stack(cell_means)
        logger.debug(f"Cell means shape: {cell_means.shape}")



        #Computing mean of means --> equal weighting per class
        cell_mean = cell_means.mean(axis=0)
        logger.debug(f"Cell mean shape: {cell_mean.shape}")
        logger.debug(f"Cell mean norm: {torch.norm(cell_mean)}")


        self.deltas = []

        #Altough scgen theoretically only uses one perturbation, we will use multiple for the sake of generality
        #On Kang the number of perts is 1
        for i in range(len(self.perts)):
            pert_means = []
            
            for j, cell in enumerate(self.cell_ids):

                logger.debug(f'Computing mean for perturbation {self.perts[i]} and cell type {cell}')
                cell_pert_adata = adata[(adata.obs[CELL_KEY] == cell) & (adata.obs[PERT_KEY] == self.perts[i])]
                #Due to iid mode we might not have all perturbations for all cell types
                if len(cell_pert_adata) > 0:
                    cell_data = torch.from_numpy(to_dense(cell_pert_adata.X)).to(device)
                    pert_mu, _ = net.encode(cell_data)
                    pert_mean = pert_mu.mean(axis=0).cpu().detach()
                    pert_means.append(pert_mean)

                else:
                    logger.info(f'Cell {cell} does not have perturbation {self.perts[i]}')
                    

               
            if len(pert_means) == 0:
                self.invalid_perts.add(self.perts[i])
                logger.warning(
                    f'Perturbation {self.perts[i]} is not applied to any cell in the training data - Delta will be \
                        NaN and trying to transfer this perturbation to unseen cells will cause an undefined state.'
                )
                
                nan_delta = torch.full((self.model_config['latent_dim'],), np.nan)
                self.deltas.append(nan_delta)


            else:
                logger.debug(f"Number of perturbations for {self.perts[i]}: {len(pert_means)}")
                #Computing the mean of the perturbation means, equal weighting per class
                pert_means = torch.stack(pert_means)
                pert_mean  = pert_means.mean(axis=0)

                logger.debug(f"Pert mean norm: {torch.norm(pert_mean)}")    

                pert_delta = pert_mean - cell_mean
                logger.debug(f"Pert delta shape: {pert_delta.shape}")
                logger.debug(f"Pert: {self.perts[i]}")
                logger.debug(f"Pert delta norm: {torch.norm(pert_delta)}")
                self.deltas.append(pert_delta)

        logger.debug(f"Deltas norms, {torch.norm(torch.stack(self.deltas), dim=1)}")

        self.deltas = torch.stack(self.deltas)

        logger.debug(f"VAE training done - Deltas shape: {self.deltas.shape}")



    #Predicting perturbations --> How do we compute the means? 
    #--> We need a delta for each perturbation
    def make_predict(self, adata: sc.AnnData, pert_embedding: Optional[dict], pert_id: str, cell_type: str) -> np.ndarray:

        assert len(adata.obs[CELL_KEY].unique()) == 1, 'Input data contains multiple cell types'
        assert len(adata.obs[PERT_KEY].unique()) == 1, 'Input data contains multiple perturbations'
        assert adata.obs[CELL_KEY].unique() == [cell_type], f'Cell type {cell_type} not in the provided data'
        assert adata.obs[PERT_KEY].unique() == [CONTROL_PERT], f'Input data contains non control perturbations'  
        
        if pert_id in self.invalid_perts:
            
            raise ValueError(f'Perturbation {pert_id} is not applied to any cell in the training data - Delta is NaN')

        logger.info(f'Predicting seen perturbation {pert_id} for unseen cell type {cell_type}')

        data = to_dense(adata.X)

        data = torch.from_numpy(data.astype(np.float32)).to(self.device)

        mu, logvar = self.model.encode(data)


        logger.debug(f"Pert idx = {self.perts_to_idx[pert_id]}")


        pert_delta = self.deltas[self.perts_to_idx[pert_id]]

        logger.debug(f"Pert delta shape: {pert_delta.shape}")
        logger.debug(f"Mu shape: {mu.shape}")

        pert_mu = mu + pert_delta.to(self.device)

        z = self.model.reparameterize(pert_mu, logvar)

        return self.model.decode(z).cpu().detach().numpy()



