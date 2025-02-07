from os import listdir
from scipy.sparse import issparse
import anndata
import scanpy as sc
import numpy as np
import pandas as pd

from scipy.stats import pearsonr

import logging

logger = logging.getLogger(__name__)

def r2_mse_filename(pert, cell):
    return f'r2_and_mse_{pert}_{cell}.json'

def c_r_filename(pert, cell):
    return f'c_r_results_{pert}_{cell}.json'

def DEGs_overlap_filename(pert, cell):   
    return f'DEGs_overlaps_{pert}_{cell}.json'


def get_DEG_with_direction(gene, score):
    if score > 0:
        return(f'{gene}+')
    else:
        return(f'{gene}-')
        
def to_dense(X):
    if issparse(X):
        return X.toarray()
    else:
        return np.asarray(X)

def get_DEGs(control_adata, target_adata):
    temp_concat = anndata.concat([control_adata, target_adata], label = 'batch')
    sc.tl.rank_genes_groups(
        temp_concat, 'batch', method='wilcoxon', 
        groups = ['1'], ref = '0', rankby_abs = True, tie_correct=True
    )

    rankings = temp_concat.uns['rank_genes_groups']
    result_df = pd.DataFrame({'scores': rankings['scores']['1'],
                     'pvals_adj': rankings['pvals_adj']['1'],
                     'lfc': rankings['logfoldchanges']['1']},
                    index = rankings['names']['1'])
    return result_df

def get_eval(ctrl_adata, true_adata, pred_adata, DEGs, DEG_vals, pval_threshold, lfc_threshold):
        
    results_dict =  {}
    
    logger.debug(f"Computing R, R2, and MSE metrics")
    ctrl_X = to_dense(ctrl_adata.X)
    true_X = to_dense(true_adata.X)
    pred_X = to_dense(pred_adata.X)

    ctrl_mean = ctrl_X.mean(axis = 0)

    true_mean = true_X.mean(axis = 0)
    true_var = true_X.var(axis = 0)
    
    pred_mean = pred_X.mean(axis = 0)
    pred_var = pred_X.var(axis = 0)
    
    true_corr_mtx = np.corrcoef(true_X, rowvar=False).flatten()
    true_cov_mtx = np.cov(true_X, rowvar=False).flatten()
        
    pred_corr_mtx = np.corrcoef(pred_X, rowvar=False).flatten()
    pred_cov_mtx = np.cov(pred_X, rowvar=False).flatten()

    true_sub_diff = true_mean - ctrl_mean
    pred_sub_diff = pred_mean - ctrl_mean

    true_diff = np.expm1(true_mean) - np.expm1(ctrl_mean)
    pred_diff = np.expm1(pred_mean) - np.expm1(ctrl_mean)

    results_dict['all_genes_mean_sub_diff_R'] = pearsonr(true_sub_diff, pred_sub_diff)[0]
    results_dict['all_genes_mean_sub_diff_R2'] = pearsonr(true_sub_diff, pred_sub_diff)[0]**2
    results_dict['all_genes_mean_sub_diff_MSE'] = (np.square(true_sub_diff - pred_sub_diff)).mean(axis=0)

    results_dict['all_genes_mean_fold_diff_R'] = pearsonr(true_diff, pred_diff)[0]
    results_dict['all_genes_mean_fold_diff_R2'] = pearsonr(true_diff, pred_diff)[0]**2
    results_dict['all_genes_mean_fold_diff_MSE'] = (np.square(true_diff - pred_diff)).mean(axis=0)
    
    results_dict['all_genes_mean_R'] = pearsonr(true_mean, pred_mean)[0]
    results_dict['all_genes_mean_R2'] = pearsonr(true_mean, pred_mean)[0]**2
    results_dict['all_genes_mean_MSE'] = (np.square(true_mean - pred_mean)).mean(axis=0)

    results_dict['all_genes_var_R'] = pearsonr(true_var, pred_var)[0]
    results_dict['all_genes_var_R2'] = pearsonr(true_var, pred_var)[0]**2
    results_dict['all_genes_var_MSE'] = (np.square(true_var - pred_var)).mean(axis=0)

    results_dict['all_genes_corr_mtx_R'] = pearsonr(true_corr_mtx.flatten(), pred_corr_mtx.flatten())[0]
    results_dict['all_genes_corr_mtx_R2'] = pearsonr(true_corr_mtx.flatten(), pred_corr_mtx.flatten())[0]**2
    results_dict['all_genes_corr_mtx_MSE'] = (np.square(true_corr_mtx.flatten() - pred_corr_mtx.flatten())).mean(axis=0)

    results_dict['all_genes_cov_mtx_R'] = pearsonr(true_cov_mtx.flatten(), pred_cov_mtx.flatten())[0]
    results_dict['all_genes_cov_mtx_R2'] = pearsonr(true_cov_mtx.flatten(), pred_cov_mtx.flatten())[0]**2
    results_dict['all_genes_cov_mtx_MSE'] = (np.square(true_cov_mtx.flatten() - pred_cov_mtx.flatten())).mean(axis=0)

    if lfc_threshold:   
        significant_DEGs = DEGs[(DEGs['pvals_adj'] < pval_threshold) & (abs(DEGs) > lfc_threshold)]
    else:
        significant_DEGs = DEGs[DEGs['pvals_adj'] < pval_threshold]
    num_DEGs = len(significant_DEGs)
    DEG_vals.insert(0, num_DEGs)


    logger.debug(f"Significant DEGs {significant_DEGs}")
    
    for val in DEG_vals:

        logger.debug(f"Computing R, R2, and MSE metrics for top {val} DEGs")

        #If val == 1 we can't
        if ((val > num_DEGs) or (val == 0) or (val == 1)):
            results_dict[f'Top_{val}_DEGs_sub_diff_mean_R'] = None
            results_dict[f'Top_{val}_DEGs_sub_diff_mean_R2'] = None
            results_dict[f'Top_{val}_DEGs_sub_diff_mean_MSE'] = None
            
            results_dict[f'Top_{val}_DEGs_fold_diff_mean_R'] = None
            results_dict[f'Top_{val}_DEGs_fold_diff_mean_R2'] = None
            results_dict[f'Top_{val}_DEGs_fold_diff_mean_MSE'] = None
            
            results_dict[f'Top_{val}_DEGs_mean_R'] = None
            results_dict[f'Top_{val}_DEGs_mean_R2'] = None
            results_dict[f'Top_{val}_DEGs_mean_MSE'] = None

            results_dict[f'Top_{val}_DEGs_var_R'] = None
            results_dict[f'Top_{val}_DEGs_var_R2'] = None
            results_dict[f'Top_{val}_DEGs_var_MSE'] = None
            
            results_dict[f'Top_{val}_DEGs_corr_mtx_R'] = None
            results_dict[f'Top_{val}_DEGs_corr_mtx_R2'] = None
            results_dict[f'Top_{val}_DEGs_corr_mtx_MSE'] = None
            
            results_dict[f'Top_{val}_DEGs_cov_mtx_R'] = None
            results_dict[f'Top_{val}_DEGs_cov_mtx_R2'] = None
            results_dict[f'Top_{val}_DEGs_cov_mtx_MSE'] = None
        
        else:
            top_DEGs = significant_DEGs[0:val].index.map(int)

            logger.debug(f"Top DEGs: {top_DEGs}")


            #Reshape --> If there is a single gene, the shape is (1,) and we need to reshape it to (1,1)

            ctrl_mean = to_dense(ctrl_X[:,top_DEGs]).mean(axis = 0)
            true_mean = to_dense(true_X[:,top_DEGs]).mean(axis = 0)

            logger.debug(f"Shape ctrl_adata with top DEGs: {ctrl_adata[:,top_DEGs].X.shape}, shape true_adata with top DEGs: {true_adata[:,top_DEGs].X.shape}")


            true_var = to_dense(true_X[:,top_DEGs]).var(axis = 0)
            true_corr_mtx = np.corrcoef(to_dense(true_X[:,top_DEGs]), rowvar=False).flatten()
            true_cov_mtx = np.cov(to_dense(true_X[:,top_DEGs]), rowvar=False).flatten()

            pred_mean = to_dense(pred_X[:,top_DEGs]).mean(axis = 0)
            logger.debug(f"Shape of true_mean shape: {true_mean.shape}, ctrl_mean shape: {ctrl_mean.shape}, pred_mean shape: {pred_mean.shape}")

            pred_var = to_dense(pred_X[:,top_DEGs]).var(axis = 0)
            pred_corr_mtx = np.corrcoef(to_dense(pred_X[:,top_DEGs]), rowvar=False).flatten()
            pred_cov_mtx = np.cov(to_dense(pred_X[:,top_DEGs]), rowvar=False).flatten()

            logger.debug(f"Shape of true_var shape: {true_var.shape}, pred_var shape: {pred_var.shape}")

            true_sub_diff = true_mean - ctrl_mean
            pred_sub_diff = pred_mean - ctrl_mean
        
            # inverse log1p to get sub change
            true_diff = np.expm1(true_mean) - np.expm1(ctrl_mean)
            pred_diff = np.expm1(pred_mean) - np.expm1(ctrl_mean)

            results_dict[f'Top_{val}_DEGs_sub_diff_R'] = pearsonr(true_sub_diff, pred_sub_diff)[0]
            results_dict[f'Top_{val}_DEGs_sub_diff_R2'] = pearsonr(true_sub_diff, pred_sub_diff)[0]**2
            results_dict[f'Top_{val}_DEGs_sub_diff_MSE'] = (np.square(true_sub_diff - pred_sub_diff)).mean(axis=0)
        
            results_dict[f'Top_{val}_DEGs_fold_diff_R'] = pearsonr(true_diff, pred_diff)[0]
            results_dict[f'Top_{val}_DEGs_fold_diff_R2'] = pearsonr(true_diff, pred_diff)[0]**2
            results_dict[f'Top_{val}_DEGs_fold_diff_MSE'] = (np.square(true_diff - pred_diff)).mean(axis=0)
    
            results_dict[f'Top_{val}_DEGs_mean_R'] = pearsonr(true_mean, pred_mean)[0]
            results_dict[f'Top_{val}_DEGs_mean_R2'] = pearsonr(true_mean, pred_mean)[0]**2
            results_dict[f'Top_{val}_DEGs_mean_MSE'] = (np.square(true_mean - pred_mean)).mean(axis=0)

            results_dict[f'Top_{val}_DEGs_var_R'] = pearsonr(true_var, pred_var)[0]
            results_dict[f'Top_{val}_DEGs_var_R2'] = pearsonr(true_var, pred_var)[0]**2
            results_dict[f'Top_{val}_DEGs_var_MSE'] = (np.square(true_var - pred_var)).mean(axis=0)

            results_dict[f'Top_{val}_DEGs_corr_mtx_R'] = pearsonr(true_corr_mtx.flatten(), pred_corr_mtx.flatten())[0]
            results_dict[f'Top_{val}_DEGs_corr_mtx_R2'] = pearsonr(true_corr_mtx.flatten(), pred_corr_mtx.flatten())[0]**2
            results_dict[f'Top_{val}_DEGs_corr_mtx_MSE'] = (np.square(true_corr_mtx.flatten() - pred_corr_mtx.flatten())).mean(axis=0)

            results_dict[f'Top_{val}_DEGs_cov_mtx_R'] = pearsonr(true_cov_mtx.flatten(), pred_cov_mtx.flatten())[0]
            results_dict[f'Top_{val}_DEGs_cov_mtx_R2'] = pearsonr(true_cov_mtx.flatten(), pred_cov_mtx.flatten())[0]**2
            results_dict[f'Top_{val}_DEGs_cov_mtx_MSE'] = (np.square(true_cov_mtx.flatten() - pred_cov_mtx.flatten())).mean(axis=0)

    return results_dict

def get_DEG_Coverage_Recall(true_DEGs, pred_DEGs, p_cutoff):
    sig_true_DEGs = true_DEGs[true_DEGs['pvals_adj'] < p_cutoff]
    true_DEGs_with_direction = [get_DEG_with_direction(gene,score) for gene, score in zip(sig_true_DEGs.index, sig_true_DEGs['scores'])]
    sig_pred_DEGs = pred_DEGs[pred_DEGs['pvals_adj'] < p_cutoff]
    pred_DEGs_with_direction = [get_DEG_with_direction(gene,score) for gene, score in zip(sig_pred_DEGs.index, sig_pred_DEGs['scores'])]
    num_true_DEGs = len(true_DEGs_with_direction)
    num_pred_DEGs = len(pred_DEGs_with_direction)
    num_overlapping_DEGs = len(set(true_DEGs_with_direction).intersection(set(pred_DEGs_with_direction)))
    if num_true_DEGs > 0: 
        COVERAGE = num_overlapping_DEGs/num_true_DEGs
    else:
        COVERAGE = None
    if num_pred_DEGs > 0:
        RECALL = num_overlapping_DEGs/num_pred_DEGs
    else:
        RECALL = None
    return COVERAGE, RECALL

def get_DEGs_overlaps(true_DEGs, pred_DEGs, DEG_vals, pval_threshold, lfc_threshold):
    if lfc_threshold:
        significant_true_DEGs = true_DEGs[(true_DEGs['pvals_adj'] < pval_threshold) & (abs(true_DEGs['lfc']) > lfc_threshold)]
        significant_pred_DEGs = pred_DEGs[(pred_DEGs['pvals_adj'] < pval_threshold) & (abs(pred_DEGs['lfc']) > lfc_threshold)]
    else:
        significant_true_DEGs = true_DEGs[true_DEGs['pvals_adj'] < pval_threshold]
        significant_pred_DEGs = pred_DEGs[pred_DEGs['pvals_adj'] < pval_threshold]

    true_DEGs_for_comparison = [get_DEG_with_direction(gene,score) for gene, score in zip(significant_true_DEGs.index, significant_true_DEGs['scores'])]   
    pred_DEGs_for_comparison = [get_DEG_with_direction(gene,score) for gene, score in zip(significant_pred_DEGs.index, significant_pred_DEGs['scores'])]
    
    logger.debug(f"Computing DEG overlaps, # of significant DEGs in true data: {len(true_DEGs_for_comparison)}, # of significant DEGs in pred data: {len(pred_DEGs_for_comparison)}")
    num_DEGs = len(significant_true_DEGs)
    DEG_vals.insert(0, num_DEGs)
    
    results = {}
    for val in DEG_vals:
        if val > num_DEGs:
            results[f'Overlap_in_top_{val}_DEGs'] = None
        else:
            results[f'Overlap_in_top_{val}_DEGs'] = len(set(true_DEGs_for_comparison[0:val]).intersection(set(pred_DEGs_for_comparison[0:val])))

    intersection = len(set(true_DEGs_for_comparison).intersection(set(pred_DEGs_for_comparison)))
    union = len(set(true_DEGs_for_comparison).union(set(pred_DEGs_for_comparison)))
    if union > 0:
        results['Jaccard'] = intersection/union
    else:
        results['Jaccard'] = None
    
    return results