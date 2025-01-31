import os, argparse, json, re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

METRIC_CONFIG = {
    'R2': {
        'R2 All Genes': 'all_genes_mean_R2',
        'R2 Var Genes': 'all_genes_var_R2',
        'R2 Means Genes': 'all_genes_mean_sub_diff_R2'
    },
    'DEG': {
        'DEG Overlap': 'Jaccard'
    }
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect + Display Results')
    parser.add_argument('--experiments_results', default="/home/boucenec/omnicell/results/repogle_k562_essential_raw/norm_log/", help='Base dir with results')
    parser.add_argument('--model_names', nargs='+', default=['nn_oracle'], help='Names of models to compare (subdirectories under base directory)')
    parser.add_argument('--metric', nargs='+', choices=['R2', 'DEG'], default=['R2', 'DEG'], help='Metrics to include in plots')
    parser.add_argument('--plot_name', default='some_plot', help='Title of plots')
    parser.add_argument('--save_path', default='plots', help='Directory to save generated plots')
    return parser.parse_args()

def find_result_dirs(model_dir):
    result_dirs = []
    for root, _, files in os.walk(model_dir):
        has_r2 = any(f.startswith('r2_and_mse_') and f.endswith('.json') for f in files)
        has_deg = any(f.startswith('DEGs_overlaps_') and f.endswith('.json') for f in files)
        if has_r2 or has_deg:
            result_dirs.append(root)
    return result_dirs

def process_gene_files(hash_dir):
    metrics = defaultdict(list)

    for r2_file in os.listdir(hash_dir):
        if r2_file.startswith('r2_and_mse_') and r2_file.endswith('.json'):
            with open(os.path.join(hash_dir, r2_file), 'r') as f:
                data = json.load(f)
                for metric in METRIC_CONFIG['R2'].values():
                    if metric in data:
                        metrics['R2'].append(data[metric])

    for deg_file in os.listdir(hash_dir):
        if deg_file.startswith('DEGs_overlaps_') and deg_file.endswith('.json'):
            with open(os.path.join(hash_dir, deg_file), 'r') as f:
                data = json.load(f)
                if 'Jaccard' in data:
                    metrics['DEG'].append(data['Jaccard'])
    
    return metrics

def generate_plots(data, args):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
  
    df_list = []
    for model_name in args.model_names:
        model_metrics = data.get(model_name, {})
        for metric_group in args.metric:
            for metric_label in METRIC_CONFIG[metric_group]:
                values = model_metrics.get(metric_group, [])
                for value in values:
                    df_list.append({
                        'Model': model_name,
                        'Metric': metric_label,
                        'Value': value
                    })
    
    if not df_list:
        print("[X] No data found.")
        return

    df = pd.DataFrame(df_list)
    ax = sns.boxplot(x='Metric', y='Value', hue='Model', data=df, palette="husl")
    ax.set_title(args.plot_name, fontsize=14)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    
    os.makedirs(args.save_path, exist_ok=True)
    output_path = os.path.join(args.save_path, f"{args.plot_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[O] Plot saved: {output_path}")

if __name__ == '__main__':
    args = parse_arguments()
    print(f'[!] Reading results through: {args.experiments_results}')
    
    all_data = defaultdict(dict)
    
    for model_name in args.model_names:
        model_dir = os.path.join(args.experiments_results, model_name)
        if not os.path.exists(model_dir):
            print(f"[X] Model {model_dir} not found - skipping")
            continue
            
        print(f"[!] Found model to process: {model_name}")
        result_dirs = find_result_dirs(model_dir)
        
        if not result_dirs:
            print(f"[X] Results for {model_name} not found - skipping")
            continue
            
        model_metrics = defaultdict(list)
        for dir_path in result_dirs:
            metrics = process_gene_files(dir_path)
            for key, values in metrics.items():
                model_metrics[key].extend(values)
        
        all_data[model_name] = model_metrics
    
    generate_plots(all_data, args)
