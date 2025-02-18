import torch
import numpy as np

from omnicell.models.utils.distribute_shift import sample_pert
import logging
import time


logger = logging.getLogger(__name__)

def sample_pert_from_model_numpy(model, ctrl, pert, max_rejections=100):
    mean_shift = pert.mean(axis=0) - ctrl.mean(axis=0)
    weighted_dist = model(ctrl, mean_shift)

    return sample_pert(ctrl, weighted_dist, mean_shift, max_rejections, num_threads=2)

def sample_pert_from_model(model, ctrl, pert, max_rejections=100, device=None):
    # if model hasattr numpy_model, use numpy model
    if hasattr(model, 'numpy_model') and model.numpy_model:
        return sample_pert_from_model_numpy(model, ctrl, pert, max_rejections)
    
    #For models that use GPU acceleration for inference
    mean_shift = pert.mean(axis=0) - ctrl.mean(axis=0)
    ctrl_tensor = torch.tensor(ctrl).to(device)
    mean_shift_tensor = torch.tensor(mean_shift).to(device)
    with torch.no_grad():
        weighted_dist = model(ctrl_tensor, mean_shift_tensor).cpu()
    weighted_dist = weighted_dist.numpy()
    return sample_pert(ctrl, weighted_dist, mean_shift, max_rejections)

def batch_pert_sampling(model, ctrl, pert, max_rejections=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_batches = ctrl.shape[0] / pert.shape[0]
    n_batches = max(1, int(np.ceil(ctrl.shape[0] / pert.shape[0])))
    preds = []
    for ctrl_data_batch in np.array_split(ctrl, n_batches):
        pred = sample_pert_from_model(model, ctrl_data_batch, pert, max_rejections, device)
        
        preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds

def generate_batched_counterfactuals(model, dset, batch_size=256, max_rejections=100):

    
    for stratum in dset.strata:
        logger.info(f"Generating synthetic data for stratum {stratum}")
        stratum_start = time.time()
        source_batch = {} 
        synthetic_counterfactual_batch = {}

        num_ctrls = dset.source[stratum].shape[0]

        for i in range(0, num_ctrls, batch_size):   
            batch_start = time.time()
            source_batch[stratum] = X_ctrl = dset.source[stratum][i:i+batch_size]
            synthetic_counterfactual_batch[stratum] = {}
            
            for j, pert in enumerate(dset.unique_pert_ids):
                pert_start = time.time()
                
                X_pert = dset.target[stratum][pert]
                
                # Time the sample_pert call
                preds = batch_pert_sampling(model, X_ctrl, X_pert, max_rejections)
                
                synthetic_counterfactual_batch[stratum][pert] = preds.astype(np.int16)
                
                pert_time = time.time() - pert_start

                logger.debug(f"Stratum {stratum} - Batch {i} - Perturbation {j}/{len(dset.unique_pert_ids)} took: {pert_time:.2f}s")
            
            batch_time = time.time() - batch_start
            logger.debug(f"Stratum {stratum} - Batch {i} took: {batch_time:.2f}s")

            # Save timing data along with results
            data_dict = {
                'synthetic_counterfactuals': synthetic_counterfactual_batch,
                'source': source_batch,
                'unique_pert_ids': dset.unique_pert_ids,
                'strata': dset.strata,
            }
            yield data_dict

   