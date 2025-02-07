

import os

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import genfromtxt
import numpy as np
import logging

from omnicell.processing.utils import to_dense
from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY

logger = logging.getLogger(__name__)


class Net(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim, alpha, dropout_rate):
            super().__init__()
            self.input_dim = input_dim  
            self.latent_dim = latent_dim  # Dimension of the latent space
            self.dropout_rate = dropout_rate  
            self.alpha = alpha  # Weight for the KL divergence term in loss

            # Encoder network definitions
            self.encoder_fc1 = nn.Linear(input_dim, hidden_dim, bias=False) 
            self.encoder_bn1 = nn.BatchNorm1d(hidden_dim)  
            self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False) 
            self.encoder_bn2 = nn.BatchNorm1d(hidden_dim) 
            self.fc_mu = nn.Linear(hidden_dim, latent_dim, bias=True) 
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim, bias=True) 
            
            # Decoder network definitions
            self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim, bias=False) 
            self.decoder_bn1 = nn.BatchNorm1d(hidden_dim)  
            self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  
            self.decoder_bn2 = nn.BatchNorm1d(hidden_dim)  
            self.decoder_fc3 = nn.Linear(hidden_dim, input_dim, bias=True) 
            
        def encode(self, x):
            h = F.leaky_relu(self.encoder_bn1(self.encoder_fc1(x)))
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            h = F.leaky_relu(self.encoder_bn2(self.encoder_fc2(h)))
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h = self.decoder_fc1(z)
            h = self.decoder_bn1(h)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            h = self.decoder_fc2(h)
            h = self.decoder_bn2(h)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            h = self.decoder_fc3(h)
            h = torch.relu(h)  # Using ReLU as the final activation
            return h

        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, self.input_dim))
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

        def loss_function(self, recon_x, x, mu, logvar):
            recon_loss = F.mse_loss(recon_x, x.view(-1, self.input_dim), reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + self.alpha * KLD

# Train a Variational Autoencoder (VAE) based on a numpy array where rows correspond to samples (cells)
# and columns to features (genes). This function includes parameter definitions and training logic.
def Train_VAE(
    net, train, valid, epochs=300, batsize=32, 
    latent_dim=100, hidden_dim=800, dropout_rate=0.2, learning_rate=0.001, alpha=0.001
):
    # Set device to GPU if available, else use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    trainlen = train.shape[0]
    validlen = valid.shape[0]
    num_genes = train.shape[1]

    net.to(device)  # Model for training

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    running_loss = 0

    train = torch.from_numpy(train.astype(np.float32)).to(device)  # Convert train data to torch tensor and move to device
    valid = torch.from_numpy(valid.astype(np.float32)).to(device)  # Convert validation data to torch tensor and move to device
    
    # Training loop
    for e in range(epochs):
        running_loss = 0

        net.train()
    
        for lower in range(0, trainlen, batsize):
            upper = min(lower + batsize, trainlen)
            lower = min(trainlen - batsize, lower)
            batch = train[lower:upper, :]
            optimizer.zero_grad()  
            out, mu, logvar = net(batch)  # Forward pass
            loss = net.loss_function(out, batch, mu, logvar) 
            loss.backward() 
            running_loss += loss.item()
            optimizer.step() 
        logger.debug(f'Epoch {e+1}/{epochs}')
        logger.debug(f'Train loss: {running_loss/1000000}')
    
        running_loss = 0

        net.eval() 
        
        # Validation loop
        with torch.no_grad():
            for lower in range(0, validlen, batsize):
                upper = min(lower + batsize, validlen)
                lower = min(validlen - batsize, lower)
                batch = valid[lower:upper, :]
                out, mu, logvar = net(batch)
                loss = net.loss_function(out, batch, mu, logvar)
                running_loss += loss.item()
            logger.debug(f'Valid loss: {running_loss/1000000}')
    

    params = [epochs, batsize, latent_dim, hidden_dim, dropout_rate, learning_rate, alpha, num_genes]

    return net, params

def Encode(net, train_params, inp):
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the VAE model, load the pre-trained weights, and set it to evaluation mode
    net.eval()

    epochs, batsize, latent_dim, hidden_dim, dropout_rate, learning_rate, alpha, num_genes = train_params
    
    # Prepare input data
    inplen = inp.shape[0]
    inp = torch.from_numpy(inp.astype(np.float32))
    
    # Output tensors for storing encoded means and log variances
    out = torch.zeros((inplen, latent_dim), device=device)
    outvar = torch.zeros((inplen, latent_dim), device=device)

    # Process each batch and store the encoded outputs
    for lower in range(0, inplen, batsize):
        upper = min(lower + batsize, inplen)
        lower = min(inplen - batsize, lower)
        batch = inp[lower:upper, :].to(device)
        mu, logvar = net.encode(batch)
        out[lower:upper, :] = mu
        outvar[lower:upper, :] = logvar

    # Convert outputs back to numpy arrays
    out = out.cpu().detach().numpy()
    outvar = outvar.cpu().detach().numpy()
    
    return out, outvar

def Decode(net, train_params, inp, inpvar):
    # Configure the device (GPU or CPU) based on availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    epochs, batsize, latent_dim, hidden_dim, dropout_rate, learning_rate, alpha, num_genes = train_params

    # Load the model, set to evaluation mode and load weights
    net.eval()
    
    # Prepare input data and output storage
    inp = torch.from_numpy(inp.astype(np.float32)).to(device)
    inpvar = torch.from_numpy(inpvar.astype(np.float32)).to(device)
    out = torch.zeros((inp.shape[0], num_genes), device=device)

    # Process each batch
    for lower in range(0, inp.shape[0], batsize):
        upper = min(lower + batsize, inp.shape[0])
        lower = min(inp.shape[0] - batsize, lower)
        batch = inp[lower:upper, :]
        batchvar = inpvar[lower:upper, :]
        reparam = net.reparameterize(batch, batchvar)  # Reparameterize to get latent variables
        decod = net.decode(reparam)  # Decode latent variables to reconstruct data
        out[lower:upper, :] = decod  # Store reconstructed data

    # Convert tensor back to numpy array and return
    out = out.cpu().detach().numpy()
    return out


class VAE():
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

        self.train_params = None
     
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

        train = adata[indices[:np.int32(datalen*0.9)]]  # 90% training data
        valid = adata[indices[np.int32(datalen*0.9):]]   

        train = to_dense(train.obsm['embedding'])
        valid = to_dense(valid.obsm['embedding'])

        self.model, self.train_params = Train_VAE(
            self.model, train, valid, epochs=epochs, batsize=batsize, 
            latent_dim=self.model_config['latent_dim'], hidden_dim=self.model_config['hidden_dim'], 
            dropout_rate=self.training_config['dropout_rate'], learning_rate=self.training_config['learning_rate'], 
            alpha=self.training_config['alpha']
        )

    def save(self, savepath):
        if self.train_params is None:
            raise ValueError("Model has not been trained yet")

        torch.save(self.model.state_dict(), f'{savepath}/state_dict.pt')
        with open(f'{savepath}/train_params.pkl', 'wb') as fp:
            pickle.dump(self.train_params, fp)


    def load(self, savepath):
        if not os.path.exists(f'{savepath}/trained_model'):
            return False
        with open(f'{savepath}/train_params.pkl', 'rb') as fp:
            self.train_params = pickle.load(fp)
        self.model.load_state_dict(torch.load(f'{savepath}/trained_model'))
        self.model.to(self.device)
        return True

    def encode(self, adata): 
        X = to_dense(adata.obsm['embedding'])
        if self.train_params is None:
            raise ValueError("Model has not been trained yet")
        return Encode(self.model, self.train_params, X)
    
    def decode(self, latent, latent_var):
        if self.train_params is None:
            raise ValueError("Model has not been trained yet")
        return Decode(self.model, self.train_params, latent, latent_var)

