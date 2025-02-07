from venv import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, x_dim, p_dim, large: bool, latent_dim = 30, hidden_dim = 512):
        super(Net, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Encoder_x = Encoder(input_dim=x_dim,large =large, hidden_dim=hidden_dim, latent_dim=latent_dim).to(self.device)
        self.Encoder_p = Encoder(input_dim=p_dim, large=large, hidden_dim=hidden_dim, latent_dim=latent_dim, VAE=False).to(self.device)
        self.Decoder_x = Decoder(latent_dim=latent_dim, large=large, hidden_dim = hidden_dim, output_dim = x_dim).to(self.device)
        self.Decoder_p = Decoder(latent_dim=latent_dim, large=large, hidden_dim = hidden_dim, output_dim = p_dim).to(self.device)
        self.MINE = MINE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(self.device)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon # reparameterization trick
        return z

    def forward(self, x, p):
        mean_z, log_var_z = self.Encoder_x(x)
        z = self.reparameterization(mean_z, torch.exp(0.5 * log_var_z)) # takes exponential function (log var -> var)
        s = self.Encoder_p(p)
        x_hat = self.Decoder_x(z+s)
        p_hat = self.Decoder_p(s)
        
        return x_hat, p_hat, mean_z, log_var_z, s



class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, VAE=True, large=False):
        super(Encoder, self).__init__()
        self.VAE = VAE
        self.large = large
        
        if large:
            logger.debug("Using large encoder")
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim*8),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim*8, hidden_dim*2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim*2, hidden_dim)

            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        if VAE:
            self.fc_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = self.layers(x)
        mean = self.fc_mean(h)
        if self.VAE:
            log_var = self.fc_var(h)
            return mean, log_var
        return mean

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim, large=False):
        super(Decoder, self).__init__()
        
        if large:
            logger.debug("Using large decoder")
            self.layers = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim*2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim*2, hidden_dim*8),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim*8, output_dim)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, output_dim)
            )
    
    def forward(self, x):
        return self.layers(x)


class MINE(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(MINE, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim*2, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, 1)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, z, s):
        h = self.LeakyReLU(self.FC_hidden(torch.cat((z, s), 1)))
        h = self.LeakyReLU(self.FC_hidden2(h))
        T = self.FC_output(h)
        return torch.clamp(T, min=-50.0, max=50.0)
