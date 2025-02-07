import argparse
import scanpy as sc
import yaml
import json
from pathlib import Path
import sys
from scanpy import AnnData as AnnData
from os import listdir
from omnicell.evaluation.utils import get_DEGs, get_eval, get_DEG_Coverage_Recall, get_DEGs_overlaps, c_r_filename, DEGs_overlap_filename, r2_mse_filename
from omnicell.data.utils import prediction_filename
from omnicell.config.config import Config
from omnicell.processing.utils import to_dense
from statistics import mean
import scipy.sparse as sparse
from utils.encoder import NumpyTypeEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import yaml
import logging
import random


logger = logging.getLogger(__name__)

random.seed(42)


def generate_evaluation(dir, args):
    
    with open(f'{dir}/config.yaml') as f:
        config = yaml.load(f, yaml.SafeLoader)
    

    #We save the evaluation config in the run directory
    with open(f'{dir}/eval_config.yaml', 'w+') as f:
        yaml.dump(args.__dict__, f, indent=2)

    logger.info(f"Loading config from {dir}")
    logger.info(f"Config: {config}")
    config = Config.from_dict(config)



    #TODO: Do we want to restructure the code to make this config handling centralized? --> Yes probably because this will get very brittle very quickly
    #TODO: Core issue --> The code is dependent on the underlying config structure, which is not good.
    eval_targets = config.eval_config.evaluation_targets
    
    for (cell, pert) in eval_targets:

        try:
        
            logger.info(f"Starting evaluation for {pert} and {cell}")


            pred_fn = f"{prediction_filename(pert, cell)}-preds.npz"
            true_fn = f"{prediction_filename(pert, cell)}-ground_truth.npz"
            control_fn = f"{prediction_filename(pert, cell)}-control.npz"

            r2_and_mse_fn = r2_mse_filename(pert, cell)
            #c_r_fn= c_r_filename(pert, cell)
            DEGs_overlap_fn = DEGs_overlap_filename(pert, cell)
            
            results_exist = ((r2_and_mse_fn in listdir(dir)) & (DEGs_overlap_fn in listdir(dir)))

            logger.debug(f"Checking wheter results already exists for {pert} and {cell} in {dir}: r2_and_mse: {r2_and_mse_fn in listdir(dir)}, DEGs_overlap: {DEGs_overlap_fn in listdir(dir)}")
            preds_exist = pred_fn in listdir(dir) and true_fn in listdir(dir) and control_fn in listdir(dir)

            if not preds_exist:
                logger.warning(f"Predictions for {pert} and {cell} do not exist in {dir}")

                raise FileNotFoundError(f"Predictions for {pert} and {cell} do not exist in {dir}")

            if ((not results_exist) | args.overwrite):

                logger.info(f"Generating evaluations for {pert} and {cell}")

                logger.debug(f"Loading prediction from: {dir}/{pred_fn}")
                logger.debug(f"Loading ground truth from: {dir}/{true_fn}")
                               
                pred_pert = sparse.load_npz(f'{dir}/{pred_fn}')
                true_pert = sparse.load_npz(f'{dir}/{true_fn}')
                control = sparse.load_npz(f'{dir}/{control_fn}')
                
                
                #We need to convert the sparse matrices to dense matrices
                pred_pert = to_dense(pred_pert)
                true_pert = to_dense(true_pert)
                control = to_dense(control)

                logger.debug(f"Data shapes: pred_pert {pred_pert.shape}, true_pert {true_pert.shape}, control {control.shape}")


                pred_pert = sc.AnnData(X=pred_pert.clip(min=0))

                if args.round:
                    pred_pert.X[pred_pert.X <= 0.5] = 0
                #pred_pert.var_names = raw_data.var_names
                
                true_pert = sc.AnnData(X=true_pert)
                #true_pert.var_names = raw_data.var_names
            
                control = sc.AnnData(X=control)
                #control.var_names = raw_data.var_names"""


                logger.debug(f"Getting ground Truth DEGs for {pert} and {cell}")
                true_DEGs_df = get_DEGs(control, true_pert)
                signif_true_DEG = true_DEGs_df[true_DEGs_df['pvals_adj'] < args.pval_threshold]

                logger.debug(f"Number of significant DEGS from ground truth: {signif_true_DEG.shape[0]}")

                logger.debug(f"Getting predicted DEGs for {pert} and {cell}")
                pred_DEGs_df = get_DEGs(control, pred_pert)

        
                logger.debug(f"Getting evaluation metrics for {pert} and {cell}")
                r2_and_mse = get_eval(control, true_pert, pred_pert, true_DEGs_df, [100,50,20], args.pval_threshold, args.log_fold_change_threshold)
                
                """ logger.debug(f"Getting DEG coverage and recall for {pert} and {cell}")
                c_r_results = {p: get_DEG_Coverage_Recall(true_DEGs_df, pred_DEGs_df, p) for p in [x/args.pval_iters for x in range(1,int(args.pval_iters*args.max_p_val))]}"""
                
                
                logger.debug(f"Getting DEG overlaps for {pert} and {cell}")
                DEGs_overlaps = get_DEGs_overlaps(true_DEGs_df, pred_DEGs_df, [100,50,20], args.pval_threshold, args.log_fold_change_threshold)

                with open(f'{dir}/{r2_and_mse_fn}', 'w+') as f:
                    json.dump(r2_and_mse, f, indent=2, cls=NumpyTypeEncoder)

                """with open(f'{dir}/{c_r_fn}', 'w+') as f:
                    json.dump(c_r_results, f, indent=2, cls=NumpyTypeEncoder)"""
                
                with open(f'{dir}/{DEGs_overlap_fn}', 'w+') as f:
                    json.dump(DEGs_overlaps, f, indent=2, cls=NumpyTypeEncoder)

            else:
                logger.info(f"Evaluations for {pert} and {cell} already exist in {dir}, skipping")
    
        except Exception as e:
            logger.error(f"Error processing {pert} and {cell} for dir: {dir} - Error is - {e}")




#Evaluation takes a target model and runs evals on it 

"""
Takes a list of dictionaries and returns a dictionary with the average values for each key, keeping only the keys
which appear (present and not none) at least in occurence_threshold dictionaries or len(dict_list), whichever is smaller.

Average is computed across the occurences of each selected key.
"""
def average_keys(dict_list, occurence_threshold):

    if (len(dict_list) == 0):
        return {}
    

    threshold = min(occurence_threshold, len(dict_list))

    key_occurences = {}

    for d in dict_list:
        for key in d.keys():
            if key in key_occurences and d[key] is not None:
                key_occurences[key] += 1
            elif d[key] is not None:
                key_occurences[key] = 1

    selected_keys = [key for key in key_occurences.keys() if key_occurences[key] >= threshold]



    #Calculate the average of non-None values for each key
    result = {}
    for key in selected_keys:
        for d in dict_list:
            if key in result and d.get(key, None) is not None:
                result[key] += d[key]
            elif key not in result and d.get(key, None) is not None:
                result[key] = d[key]

        result[key] = result[key] / key_occurences[key]

    return result


def average_fold(fold_dir, min_occurences):

    logger.info(f"Averaging evaluations for {fold_dir}")

    with open(f'{fold_dir}/config.yaml') as f:
        config = yaml.load(f, yaml.SafeLoader)
    
    config = Config.from_dict(config)

    eval_targets = config.eval_config.evaluation_targets

    degs_dicts  = []
    r2_mse_dicts = []
    c_r_dicts = []
    random.shuffle(eval_targets)
    for (cell, pert) in eval_targets:

        try :
            with open(f'{fold_dir}/{DEGs_overlap_filename(pert, cell)}', 'rb') as f:
                DEGs_overlaps = json.load(f)
                degs_dicts.append(DEGs_overlaps)
            
            with open(f'{fold_dir}/{r2_mse_filename(pert, cell)}', 'rb') as f:
                r2_mse = json.load(f)
                r2_mse_dicts.append(r2_mse)
            
            """with open(f'{fold_dir}/{c_r_filename(pert, cell)}', 'rb') as f:
                c_r = pickle.load(f)
                c_r_dicts.append(c_r)"""
            
        except Exception as e:
            logger.info(f"Could not load evaluations for {pert} and {cell} in {fold_dir}, will not be included in average stats - {e}")
 

    avg_DEGs_overlaps = average_keys(degs_dicts, min_occurences)
    avg_r2_mse = average_keys(r2_mse_dicts, min_occurences)
    #avg_c_r = average_shared_keys(c_r_dicts)

    with open(f'{fold_dir}/avg_DEGs_overlaps.json', 'w+') as f:
        json.dump(avg_DEGs_overlaps, f, indent=2, cls=NumpyTypeEncoder)
    
    with open(f'{fold_dir}/avg_r2_mse.json', 'w+') as f:
        json.dump(avg_r2_mse, f, indent=2, cls=NumpyTypeEncoder)
    
    """with open(f'{fold_dir}/avg_c_r.pkl', 'wb') as f:
        pickle.dump(avg_c_r, f)"""

"""
Averages all averages contained in subdirectories of the given directory. Assuming average have already been computed in the subdirectories.
"""
def average_directory(dir, min_occurences):
    subdirs = [d for d in dir.iterdir() if d.is_dir()]

    results_DEGs = []
    results_r2_mse = []
    for subdir in subdirs:
        with open(f'{subdir}/avg_DEGs_overlaps.json', 'rb') as f:
            results_DEGs.append(json.load(f))
        
        with open(f'{subdir}/avg_r2_mse.json', 'rb') as f:
            results_r2_mse.append(json.load(f))
   
    avg_DEGs = average_keys(results_DEGs, min_occurences)
    avg_r2_mse = average_keys(results_r2_mse, min_occurences)

    with open(f'{dir}/avg_DEGs_overlaps.json', 'w+') as f:
        json.dump(avg_DEGs, f, indent=2)
    
    with open(f'{dir}/avg_r2_mse.json', 'w+') as f:
        json.dump(avg_r2_mse, f, indent=2)





def is_leaf_dir(path):
    """Check if the given path is a leaf directory."""
    return path.is_dir() and not any(p.is_dir() for p in path.iterdir())

def process_directory(dir_path, args, depth, max_depth, min_compute_average_depth):
    """Process a single directory, either by generating evaluations or recursing further."""
    if is_leaf_dir(dir_path):
        try:
            logger.info(f"Processing leaf directory: {dir_path}")
            generate_evaluation(dir_path, args)
            average_fold(dir_path, args.min_occurence)
        except Exception as e:
            logger.error(f"Error processing directory {dir_path}: {e}")

    elif depth >= max_depth:
        logger.info(f"Reached maximum depth of {max_depth} at {dir_path}")
    else:
        # Convert subdirectories to list and shuffle
        subdirs = [subdir for subdir in dir_path.iterdir() if subdir.is_dir()]
        random.shuffle(subdirs)
        
        # Process in random order
        for subdir in subdirs:
            process_directory(subdir, args, depth + 1, max_depth, min_compute_average_depth)

        # Once all subdirectories have been processed, average the results
        if depth >= min_compute_average_depth:
            try:
                average_directory(dir_path, args.min_occurence)
            except Exception as e:
                logger.error(f"Error computing average for directory {dir_path}: {e}")



def main(*args):

    logger.info("Starting evaluation script")


    parser = argparse.ArgumentParser(description='Analysis settings.')

    parser.add_argument('--root_dir', type=str, default='', help='Top dir where to start comuting evaluations')
    parser.add_argument('--min_occurence', type=int, default=2, help='Minimum number of occurences of a key to be included in the average')
    parser.add_argument('-r', '--round', action='store_true', help='Rounds values <=0.5 to 0 in addition to the clip')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite pre-existing result files')
    parser.add_argument('-pval', '--pval_threshold', type=float, default=0.05, help='Sets maximum adjusted p value for a gene to be called as a DEG')
    parser.add_argument('-lfc', '--log_fold_change_threshold', type=float, default=None, help='Sets minimum absolute log fold change for a gene to be called as a DEG')
    parser.add_argument('--replicates', type=int, default=10, help='Number of replicates to use for p value calculation')
    parser.add_argument('--pval_iters', type=int, default=10000, help='Number of iterations to use for p value calculation')
    parser.add_argument('-l', '--log', dest='loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level (default: %(default)s)", default='INFO')
    parser.add_argument('--min_avg_depth', type=int, default=-1, help='Compute average stats till this depth from root. -1 means no averages beyond leaf directories')
    parser.add_argument('--max_p_val', type=float, default=0.05, help='Maximum p value to use for p value calculation')

    MAX_DEPTH = 15
    args = parser.parse_args()

    root_dir_points = '.'.join(args.root_dir.split('/'))

    root_dir = Path(args.root_dir).resolve()

    logging.basicConfig(filename=f'output_evals_{root_dir_points}.log', filemode='w', level=args.loglevel, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info(f"Starting recursive evaluation from root directory: {root_dir}")

    process_directory(root_dir, args, 0, MAX_DEPTH, args.min_avg_depth)

    logger.info("Recursive evaluation completed.")
    

if __name__ == '__main__':
    main()