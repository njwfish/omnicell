import sys

from omnicell.models.gears.pertdata import PertData
from omnicell.data.catalogue import Catalogue
from omnicell.models.gears.gears import GEARS
import scanpy as sc
import torch
import numpy as np

DATASET_NAME = 'satija_IFNB_raw'
dd = Catalogue.get_dataset_details(DATASET_NAME)
adata = sc.read(dd.path)

adata.obs["condition"] = adata.obs["gene"]
perts = [p for p in adata.obs["condition"].unique() if p != dd.control]
adata.obs["condition"] = adata.obs["condition"].replace({dd.control:"ctrl"})
adata.obs["condition"] = adata.obs["condition"].replace({p:p+"+ctrl" for p in perts})
adata.var["gene_name"] = adata.var_names


print(f"Data loaded from {dd.path}")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata,n_top_genes=5000, subset=True)
print(f"Data normalized and log transformed")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


pert_data = PertData('./data') # specific saved folder
print(f"Saving data in ./data")

pert_data.new_data_process(dataset_name = DATASET_NAME, adata = adata, skip_calc_de=False) # specific dataset name and adata object
print(f"Data processed and saved in {pert_data.data_path}")
pert_data.prepare_split(split = 'no_test', seed = 1) # get data split with seed
print(f"Data split with seed 1")
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128) # prepare data loader
print(f"Data loader prepared")


gears_model = GEARS(pert_data, device = device, 
                        weight_bias_track = False, 
                        proj_name = 'pertnet', 
                        exp_name = 'pertnet')

gears_model.model_initialize(hidden_size = 64)

gears_model.tunable_parameters()

print(f"Training model")
gears_model.train(epochs = 20, lr = 1e-3)


gears_model.save_model('test_model')
gears_model.load_pretrained('test_model')


res = gears_model.predict([['RPL15']])

np.savez('res.npz', res)

