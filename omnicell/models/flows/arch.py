import torch
import torch.nn as nn
import pytorch_lightning as pl

class MLP(pl.LightningModule):
    def __init__(self, training_module, dim, out_dim=None, w=128, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )
        self.training_module = training_module

    def forward(self, x):
        return self.net(x)

    def _shared_step(self, batch, batch_idx):
        return self.training_module._shared_step(self, batch, batch_idx)

    def training_step(self, batch, batch_idx):
        return self.training_module.training_step(self, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.training_module.validation_step(self, batch, batch_idx)

    def configure_optimizers(self):
        return self.training_module.configure_optimizers(self)


class CMLP(pl.LightningModule):
    def __init__(self, training_module, feat_dim, cond_dim, out_dim=None, w1=128, w2=128, n_combo_layer=3, n_cond_layer=3, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = feat_dim
            
        self.combo_net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim + (1 if time_varying else 0) + w2, w1), torch.nn.SELU(),
            *([torch.nn.Linear(w1, w1), torch.nn.SELU()] * n_combo_layer),
            torch.nn.Linear(w1, out_dim)
        )
        self.cond_net = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, w2), torch.nn.SELU(),
            *([torch.nn.Linear(w2, w2), torch.nn.SELU()] * n_cond_layer),
            torch.nn.Linear(w2, w2)
        )
        self.cond = None
        self.training_module = training_module
        

    def forward(self, x, cond=None):
        if cond is None:
            cond = self.cond
        cond = self.cond_net(cond)
        return self.combo_net(torch.cat([x, cond], dim=-1))
    
    def _shared_step(self, batch, batch_idx):
        return self.training_module._shared_step(self, batch, batch_idx)

    def training_step(self, batch, batch_idx):
        return self.training_module.training_step(self, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.training_module.validation_step(self, batch, batch_idx)

    def configure_optimizers(self):
        return self.training_module.configure_optimizers(self)

class CMLPC(pl.LightningModule):
    def __init__(self, training_module, feat_dim, cond_dim, out_dim=None, w1=128, w2=128, n_combo_layer=3, n_cond_layer=3, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = feat_dim
            
        self.combo_net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim + (1 if time_varying else 0) + w2 + w2, w1), torch.nn.SELU(),
            *([torch.nn.Linear(w1, w1), torch.nn.SELU()] * n_combo_layer),
            torch.nn.Linear(w1, out_dim)
        )
        self.cond_net = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, w2), torch.nn.SELU(),
            *([torch.nn.Linear(w2, w2), torch.nn.SELU()] * n_cond_layer),
            torch.nn.Linear(w2, w2)
        )
        self.control_net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, w2), torch.nn.SELU(),
            *([torch.nn.Linear(w2, w2), torch.nn.SELU()] * n_cond_layer),
            torch.nn.Linear(w2, w2)
        )
        self.cond = None
        self.control = None
        self.training_module = training_module
        

    def forward(self, x, cond=None, control=None):
        if cond is None:
            cond = self.cond
        if control is None:
            control = self.control
        cond = self.cond_net(cond)
        control = self.control_net(control).mean(dim=1)
        # tile control to match the shape of cond
        control = control.unsqueeze(1).repeat(1, cond.shape[1], 1)
        return self.combo_net(torch.cat([x, cond, control], dim=-1))
    
    def _shared_step(self, batch, batch_idx):
        return self.training_module._shared_step(self, batch, batch_idx)

    def training_step(self, batch, batch_idx):
        return self.training_module.training_step(self, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.training_module.validation_step(self, batch, batch_idx)

    def configure_optimizers(self):
        return self.training_module.configure_optimizers(self)



class CMHA(pl.LightningModule):
    def __init__(self, training_module, feat_dim, cond_dim, out_dim=None,
                 w1=128, w2=128, num_heads=3, n_combo_layer=2, n_cond_layer=2, n_feat_layer=2, num_mhas=3, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = feat_dim
        self.feat_net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim + (1 if time_varying else 0) + w2, w1), torch.nn.SELU(),
            *([torch.nn.Linear(w2, w2), torch.nn.SELU()] * n_feat_layer),
            torch.nn.Linear(w1, w1 * num_heads)
        )
        self.cond_net = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, w2), torch.nn.SELU(),
            *([torch.nn.Linear(w2, w2), torch.nn.SELU()] * n_cond_layer),
            torch.nn.Linear(w2, w2)
        )
        self.mhas = nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=w1 * num_heads, num_heads=num_heads)] * num_mhas)
        self.combo_net = torch.nn.Sequential(
            torch.nn.Linear(w1 * num_heads, w1), torch.nn.SELU(),
            *([torch.nn.Linear(w2, w2), torch.nn.SELU()] * n_combo_layer),
            torch.nn.Linear(w1, out_dim)
        )
        self.cond = None
        self.training_module = training_module
        

    def forward(self, x, cond=None):
        if cond is None:
            cond = self.cond
        cond = self.cond_net(cond)
        combo_rep = torch.cat([x, cond], dim=-1)
        x = self.feat_net(combo_rep)
        z = torch.zeros_like(x)
        for mha in self.mhas:
            z, w = mha(z + x, z + x, z + x)
        return self.combo_net(z)
    
    def _shared_step(self, batch, batch_idx):
        return self.training_module._shared_step(self, batch, batch_idx)

    def training_step(self, batch, batch_idx):
        return self.training_module.training_step(self, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.training_module.validation_step(self, batch, batch_idx)

    def configure_optimizers(self):
        return self.training_module.configure_optimizers(self)

class FM(pl.LightningModule):
    @staticmethod
    def _shared_step(self, batch, batch_idx):
        t, xt, ut = batch
        inp = torch.cat([xt, t[:, None]], dim=-1)
        vt = self(inp)
        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss

    @staticmethod
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    @staticmethod
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)

    @staticmethod
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)
    
class CFM(pl.LightningModule):
    @staticmethod
    def _shared_step(self, batch, batch_idx):
        t, xt, ut, pt = batch
        inp = torch.cat([xt, t[:, None]], dim=-1)
        vt = self(inp, pt)
        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss

    @staticmethod
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    @staticmethod
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)

    @staticmethod
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)

class CFM(pl.LightningModule):
    @staticmethod
    def _shared_step(self, batch, batch_idx):
        t, xt, ut, pt = batch
        inp = torch.cat([xt, t[:, None]], dim=-1)
        vt = self(inp, pt)
        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss

    @staticmethod
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    @staticmethod
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)

    @staticmethod
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)

class CFMC(pl.LightningModule):
    @staticmethod
    def _shared_step(self, batch, batch_idx):
        t, xt, ut, pt, control = batch
        inp = torch.cat([xt, t[:, None]], dim=-1)
        vt = self(inp, pt, control)
        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss

    @staticmethod
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    @staticmethod
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)

    @staticmethod
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)

class MSE(pl.LightningModule):
    @staticmethod
    def _shared_step(self, batch, batch_idx):
        x0, x1, pt = batch
        x1_pred = self(x0, pt)
        loss = torch.nn.functional.mse_loss(x1_pred, x1)
        return loss

    @staticmethod
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    @staticmethod
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)

    @staticmethod
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)