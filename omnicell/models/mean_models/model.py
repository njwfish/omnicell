from omnicell.constants import PERT_KEY, GENE_EMBEDDING_KEY, CONTROL_PERT, CELL_KEY

import scanpy as sc
import pandas as pd 
import numpy as np
import scipy

# After the existing imports, add:
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from pathlib import Path

from omnicell.models.distribute_shift import sample_pert, get_proportional_weighted_dist


def fit_supervised_model(X, Y, model_type='linear', **kwargs):
    """
    Fit a supervised model based on the specified model type.
    
    Args:
        X: Input features (gene embeddings)
        Y: Target values (perturbation effects)
        model_type: Type of model to fit ('linear', 'ridge', 'lasso', 'elastic_net', 'rf', 'svr')
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        fitted model, training MSE, R2 score
    """
    models = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
        'rf': RandomForestRegressor,
        'svr': SVR
    }
    
    if model_type not in models:
        raise ValueError(f"Model type {model_type} not supported. Choose from {list(models.keys())}")
    
    model = models[model_type](**kwargs)
    model.fit(X, Y)
    
    # Make predictions and calculate metrics
    Y_pred = model.predict(X)
    mse = mean_squared_error(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)
    
    return model, mse, r2

def compute_cell_type_means(adata, cell_type):
    """Compute perturbation effect embeddings for a specific cell type"""
    
    # Filter data for this cell type
    cell_type_data = adata[adata.obs[CELL_KEY] == cell_type]
    
    # Compute control mean for this cell type
    ctrl_mean = np.mean(
        cell_type_data[cell_type_data.obs[PERT_KEY] == CONTROL_PERT].X, axis=0
    )
    
    # Convert to dense array if sparse
    X = cell_type_data.X
    
    # Create dataframe
    df = pd.DataFrame(X, index=cell_type_data.obs.index)
    df['perturbation'] = cell_type_data.obs[PERT_KEY].values
    
    # Compute means per perturbation
    pert_means = df.groupby('perturbation').mean()
    
    # Compute deltas from control
    pert_deltas = pd.DataFrame(pert_means.values - ctrl_mean, index=pert_means.index)
    pert_deltas_dict = {
        pert: np.array(means) 
        for pert, means in pert_deltas.iterrows() 
        if pert != CONTROL_PERT
    }
    
    return ctrl_mean, pert_deltas_dict

class MeanPredictor():

    def __init__(self, model_config: dict, pert_embedding: dict):
        self.model = None
        self.model_type = model_config['model_type']
        self.pca_pert_embeddings = model_config['pca_pert_embeddings']
        self.pca_pert_embeddings_components = model_config['pca_pert_embeddings_components']
        self.pert_embedding = pert_embedding

    def train(self, adata: sc.AnnData, model_savepath: Path):
        if self.pca_pert_embeddings:
            pca = PCA(n_components=self.pca_pert_embeddings_components)
            pert_emb_temp = pca.fit_transform(np.array(list(self.pert_embedding.values())))
            self.pert_embedding = {pert : pert_emb_temp[i] for i, pert in enumerate(self.pert_embedding.keys())}

        # Get unique cell types
        cell_types = adata.obs[CELL_KEY].unique()

        # Compute embeddings for each cell type
        Xs = []
        Ys = []
        for cell_type in cell_types:
            ctrl_mean, pert_deltas_dict = compute_cell_type_means(adata, cell_type)
            
            # Get perturbation IDs for this cell type
            idxs = pert_deltas_dict.keys()
            
            # Create feature matrix X and target matrix Y
            Y = np.array([pert_deltas_dict[pert] for pert in idxs])
            X = np.array([self.pert_embedding[g] for g in idxs])

            # Store the embeddings
            Xs.append(X)
            Ys.append(Y)

        # Now you can train a model for each cell type
        X = np.concatenate(Xs)
        Y = np.concatenate(Ys)
        self.model, mse, r2 = fit_supervised_model(X, Y, model_type=self.model_type)
        
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        ctrl_cells = adata[(adata.obs[PERT_KEY] == CONTROL_PERT) & (adata.obs[CELL_KEY] == cell_type)].X
        X_new = np.array(self.pert_embedding[pert_id].reshape(1, -1))
        mean_shift_pred = np.array(self.model.predict(X_new)).flatten()
        weighted_dist = get_proportional_weighted_dist(ctrl_cells)
        samples = sample_pert(ctrl_cells, weighted_dist, mean_shift_pred)
        return samples
