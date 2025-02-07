import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import yaml
import pickle
import pandas as pd
from omnicell.evaluation.utils import r2_mse_filename, c_r_filename, DEGs_overlap_filename
from omnicell.data.utils import prediction_filename
import hashlib
import yaml
import json
from pathlib import Path
import argparse

def rearrange_results_df(df, metric = 'all'):
    # Separate and sort lists for each category
    mean_cols_R2 = sorted([name for name in df.index if name.endswith('mean_R2')])
    mean_cols_MSE = sorted([name for name in df.index if name.endswith('mean_MSE')])
    
    var_cols_R2 = sorted([name for name in df.index if name.endswith('var_R2')])
    var_cols_MSE = sorted([name for name in df.index if name.endswith('var_MSE')])
    
    corr_cols_R2 = sorted([name for name in df.index if name.endswith('corr_mtx_R2')])
    corr_cols_MSE = sorted([name for name in df.index if name.endswith('corr_mtx_MSE')])
    
    cov_cols_R2 = sorted([name for name in df.index if name.endswith('cov_mtx_R2')])
    cov_cols_MSE = sorted([name for name in df.index if name.endswith('cov_mtx_MSE')])

    # Combine sorted lists with desired order
    if metric == 'all':
        new_order = mean_cols_R2 + mean_cols_MSE + var_cols_R2 + var_cols_MSE + corr_cols_R2 + corr_cols_MSE + cov_cols_R2 + cov_cols_MSE
    elif metric == 'MSE':
        new_order = mean_cols_MSE + var_cols_MSE + corr_cols_MSE + cov_cols_MSE
    elif metric == 'R2':
        new_order = mean_cols_R2 + var_cols_R2 + corr_cols_R2 + cov_cols_R2
    else:
        print(f'METRIC: {metric} IS NOT AN OPTION')
        
    # Reorder the dataframe based on the new order
    df = df.loc[new_order]

    return(df)

def are_identical(vals):
    return all(i == vals[0] for i in vals)

def df_to_heatmap(df, title, path):

    # Calculate figure size based on dataframe dimensions
    row_count, col_count = df.shape
    base_size = 4  # Base size for a small dataframe
    width = base_size + (col_count * 0.6)  # Increase width for each column
    height = base_size + (row_count * 0.3)  # Increase height for each row
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Adjust layout
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    
    df.columns = [x.replace(' ', '\n') for x in df.columns]
    
    # Calculate font sizes based on dataframe size
    annot_font_size = max(10, min(11, 200 / max(row_count, col_count)))
    tick_font_size = max(8, min(10, 150 / max(row_count, col_count)))
    
    # Adjust heatmap parameters
    sns.heatmap(df, annot=df, linewidths=0.5, ax=ax, cmap='Blues', 
                fmt='.2f', annot_kws={"size": annot_font_size,}, 
                cbar=True, xticklabels=True, yticklabels=True,
                cbar_kws={'shrink': .5})
    
    # Set title and adjust tick labels
    ax.set_title(title, size=min(18, 300 / max(row_count, col_count)), pad=20)
    ax.tick_params(axis='x', labelsize=tick_font_size, rotation=45)
    ax.tick_params(axis='y', labelsize=tick_font_size)

    ax.xaxis.set_ticks_position("top")
    
    # Adjust y-axis label alignment and position
    ax.set_yticklabels(ax.get_yticklabels(), ha='right', va='center')
    ax.yaxis.set_tick_params(pad=5)
    
    # Adjust colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_font_size)
    
    # Tight layout to remove excess white space
    plt.tight_layout()
    
    fig.savefig(f'{path}/{title}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory

def plot_coverage_and_recall(coverage_results, recall_results, dims):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=dims)

    # Plotting Coverage on the first subplot
    ax1.plot(coverage_results, label=coverage_results.columns)
    # Plotting Recall on the second subplot
    ax2.plot(recall_results, label=recall_results.columns)
    
    # Adding titles and labels for the Coverage plot
    ax1.set_title("Coverage by Model")
    ax1.set_xlabel("p-value threshold")
    ax1.set_ylabel("Coverage")
    #ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adding titles and labels for the Recall plot
    ax2.set_title("Recall by Model")
    ax2.set_xlabel("p-value threshold")
    ax2.set_ylabel("Recall")
    ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


"""
Something along those lines will be useful later down the line, i.e. find shared evals across complex runs and compare them
"""
def get_variable_and_fixed_params(all_results, all_results_concat_dict):
    all_config_params = {}
    for key, val in all_results.items():
        for config_key, config_val in val['config'].items():
            if config_key in all_config_params:
                all_config_params[config_key].append(config_val)
            else:
                all_config_params[config_key] = [config_val]
    
    fixed_params = []
    variable_params = []
    for key, val in all_config_params.items():
        if are_identical(val) or len(val) < len(all_results_concat_dict):
            fixed_params.append(key)
        else:
            print(f'Config parameter "{key}" varies with values:\n {val}')
            variable_params.append(key)
    print(f'\nThe following parameters are fixed for all runs:\n {fixed_params}')

    return fixed_params, variable_params


#TODO: No support for folds yet --> Honestly this all might be fore efficient with a notebook
# What are the issues, saving all that shit consistently is a pain, selecting the right stuff etc. but at least we can deploy
# Several folds, several targets in each fold 
def main(*args):


    parser = argparse.ArgumentParser(description='Analysis settings.')

    #Add argument which is a list of paths to the dirs containgin the results
    #result files receives the results files for each

    parser.add_argument('--results_files', type=str, nargs='+', help='List of paths to the files containing the results')
    parser.add_argument('--column_names', type=str, nargs='+', help='Name of the column associated with each file, respectively based on order')

    parser.add_argument('--plot_name', type=str, default='', help='Target perturbation to compare to')
    parser.add_argument('--metrics', choices=['all', 'MSE', 'R2', 'None'], default='None', help='Metrics to plot for the heatmap, set to None if plotting DEG overlaps.')
                        
    parser.add_argument('--save_path', type=str, default='', help='Path to save the results')



    args = parser.parse_args()



    results_files = args.results_files
    column_names = args.column_names

    assert len(results_files) == len(column_names), 'Number of results files and column names must be the same'

    results = {}
    for i, result_file in enumerate(results_files):
        assert Path(result_file).exists(), f'{result_file} does not exist'

        column_name = column_names[i]
    

        with open(result_file, 'rb') as f:
            result_dict = json.load(f)

        results[column_name] = result_dict

    #Find the intersection of the keys in all the results files

    keys = [set(x.keys()) for x in results.values()]
    common_keys = set.intersection(*keys)


    assert len(common_keys) > 0, 'No common keys found in the results files'

    #Subsetting the dicts to only contain the common keys

    for result_name, result_dict in results.items():
        results[result_name] = {key: result_dict[key] for key in common_keys}

        #Create a dataframe with the results

        results_df = pd.DataFrame.from_dict(results[result_name], orient='index')

        results_df.columns = [result_name]

        results[result_name] = results_df


    #Concatenate the dataframes

    results_df = pd.concat(results.values(), axis=1)
        # Sort columns alphanumerically
    results_df = results_df.sort_index(axis=1)

    print(f"Results df before calling heatmap: {results_df}")
    #Create a dataframe with all the results

    if args.metrics == 'None':
        results_df = results_df.sort_index()
        df_to_heatmap(results_df, f'{args.plot_name}', args.save_path)
    else:
        results_df = rearrange_results_df(results_df, metric = args.metrics)
        df_to_heatmap(results_df, f'{args.plot_name}', args.save_path)


if __name__ == '__main__':

    main()