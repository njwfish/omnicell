import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

R2_METRICS = {
    #'all_genes_mean_R2': 'R2 All Genes',
    'all_genes_var_R2': 'R2 Var Genes',
    'all_genes_mean_sub_diff_R2': 'R2 All Genes Sub Diff',
    #'all_genes_mean_fold_diff_R2': 'R2 Fold Diff Genes',
    'all_genes_corr_mtx_R2': 'R2 Corr Matrix',
    'all_genes_cov_mtx_R2': 'R2 Cov Matrix',
    'Top_100_DEGs_sub_diff_R2': 'R2 Top 100 DEGs Sub Diff',
    #'Top_100_DEGs_fold_diff_R2': 'R2 Top 100 DEGs Fold Diff',
    #'Top_100_DEGs_mean_R2': 'R2 Top 100 DEGs Mean',
    'Top_100_DEGs_var_R2': 'R2 Top 100 DEGs Var',
    'Top_100_DEGs_corr_mtx_R2': 'R2 Top 100 DEGs Corr Matrix',
    #'Top_100_DEGs_cov_mtx_R2': 'R2 Top 100 DEGs Cov Matrix',
    'Top_50_DEGs_sub_diff_R2': 'R2 Top 50 DEGs Sub Diff',
    #'Top_50_DEGs_fold_diff_R2': 'R2 Top 50 DEGs Fold Diff',
    'Top_50_DEGs_mean_R2': 'R2 Top 50 DEGs Mean',
    'Top_50_DEGs_var_R2': 'R2 Top 50 DEGs Var',
    'Top_50_DEGs_corr_mtx_R2': 'R2 Top 50 DEGs Corr Matrix',
    'Top_50_DEGs_cov_mtx_R2': 'R2 Top 50 DEGs Cov Matrix',
    'Top_20_DEGs_sub_diff_R2': 'R2 Top 20 DEGs Sub Diff',
    #'Top_20_DEGs_fold_diff_R2': 'R2 Top 20 DEGs Fold Diff',
    'Top_20_DEGs_mean_R2': 'R2 Top 20 DEGs Mean',
    'Top_20_DEGs_var_R2': 'R2 Top 20 DEGs Var',
    'Top_20_DEGs_corr_mtx_R2': 'R2 Top 20 DEGs Corr Matrix',
    #'Top_20_DEGs_cov_mtx_R2': 'R2 Top 20 DEGs Cov Matrix'
}

DEG_METRICS = {
    'Jaccard': 'Jaccard Similarity',
    'Overlap_in_top_100_DEGs': 'Overlap Top 100 DEGs',
    'Overlap_in_top_50_DEGs': 'Overlap Top 50 DEGs',
    'Overlap_in_top_20_DEGs': 'Overlap Top 20 DEGs'
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate model comparison box plots')
    parser.add_argument('--experiments_results', nargs='+', required=True,
                        help='Base directories containing model results')
    parser.add_argument('--model_names', nargs='+', required=True,
                        help='Names for models (must match the number of result paths)')
    parser.add_argument('--plot_name', default='comparison_plot',
                        help='Base name for output plot')
    parser.add_argument('--save_path', default='plots',
                        help='Directory to save generated plots')
    parser.add_argument('--mode', choices=['R2', 'DEG'], required=True,
                        help='Mode: R2 for R2 metrics, DEG for DEG overlaps')
    return parser.parse_args()

def process_directory(root_dir, mode):
    metrics = defaultdict(list)
    file_pattern = 'r2' if mode == 'R2' else 'degs_overlaps'
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Skip non-JSON, skip if file pattern not in filename, skip if 'avg' in name
            if (not filename.endswith('.json')) or (file_pattern not in filename.lower()) or ('avg' in filename.lower()):
                continue
                
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    target_metrics = R2_METRICS if mode == 'R2' else DEG_METRICS
                    for metric in target_metrics:
                        if metric in data:
                            metrics[metric].append(data[metric])
            except Exception as e:
                print(f"[X] Error processing {file_path}: {str(e)}")
    
    return metrics

def generate_plots(all_data, args):
    sns.set_theme(style="whitegrid")
    plot_rows = []
    metric_config = R2_METRICS if args.mode == 'R2' else DEG_METRICS

    # Build a list of dictionaries for plotting
    for model in args.model_names:
        model_metrics = all_data.get(model, {})
        for metric_key in metric_config:
            values = model_metrics.get(metric_key, [])
            display_name = metric_config[metric_key]
            plot_rows.extend(
                {'Metric': display_name, 'Value': v, 'Model': model}
                for v in values
            )
    
    if not plot_rows:
        print("[!] No data available for plotting")
        return
    
    df = pd.DataFrame(plot_rows)
    
    # Determine figure width dynamically based on the number of metrics and models
    num_metrics = df['Metric'].nunique()
    num_models = df['Model'].nunique()

    # You can fine-tune these scaling factors as needed
    # to accommodate how wide or narrow you want the figure.
    if args.mode == 'R2':
        # R2 mode often has many metrics, so we use a larger base width per metric.
        width = max(10, num_metrics * 1.5 + num_models * 1.0)
        plt.figure(figsize=(width, 7))
        sns.boxplot(x='Metric', y='Value', hue='Model', data=df)
        plt.title(f"{args.mode} Metrics Comparison")
        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.xticks(rotation=45, ha='right')
    

    elif args.mode == 'DEG':
        num_metrics = len(DEG_METRICS)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6))
        axes = axes.flatten()
        
        for i, (metric_key, display_name) in enumerate(DEG_METRICS.items()):
            ax = axes[i]
            df_metric = df[df['Metric'] == display_name]
            if df_metric.empty:
                print(f"[!] No data for: {display_name}")
                continue
                
            sns.boxplot(x='Metric', y='Value', hue='Model', data=df_metric, ax=ax)
            ax.set_title(display_name)
            ax.set_xlabel("")
            ax.set_ylabel("Metric Value")
        plt.tight_layout()

    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    os.makedirs(args.save_path, exist_ok=True)
    output_path = os.path.join(args.save_path, f"{args.plot_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Plot saved: {output_path}")
if __name__ == '__main__':
    args = parse_arguments()
    
    if len(args.experiments_results) != len(args.model_names):
        raise ValueError("Mismatch between experiment paths and model names")
    
    all_data = {}
    for model_name, path in zip(args.model_names, args.experiments_results):
        print(f"[*] Processing {model_name} at {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        
        metrics = process_directory(path, args.mode)
        all_data[model_name] = metrics
    
    generate_plots(all_data, args)
