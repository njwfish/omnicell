import os, json, argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

METRIC_CONFIG = {
    'all_genes_mean_R2': 'R2 All Genes',
    'all_genes_var_R2': 'R2 Var Genes',
    'all_genes_mean_sub_diff_R2': 'R2 Means Genes',
    'Jaccard': 'DEG Overlap'
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate model comparison box plots')
    parser.add_argument('--input_paths', nargs='+', required=True,
                        help='Base directories containing model results')
    parser.add_argument('--model_names', nargs='+', required=True,
                        help='Names for models (model name to match the results path)')
    parser.add_argument('--metrics', nargs='+', default=list(METRIC_CONFIG.keys()),
                        help=f'Metrics to include (default: {list(METRIC_CONFIG.keys())})')
    parser.add_argument('--plot_name', default='comparison',
                        help='Base name for output plot')
    parser.add_argument('--save_path', default='plots',
                        help='Directory to save generated plots')
    return parser.parse_args()

def process_directory(root_dir, metrics_to_collect):
    metrics = defaultdict(list)
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.json'):
                continue
                
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for metric in metrics_to_collect:
                        if metric in data:
                            metrics[metric].append(data[metric])
            except Exception as e:
                print(f"[X] Error processing {file_path}: {str(e)}")
    
    return metrics

def generate_plots(all_data, args):
    sns.set_theme(style="whitegrid")

    # 
    plot_rows = []
    for model in args.model_names:
        model_metrics = all_data.get(model, {})
        for metric_key in args.metrics:
            values = model_metrics.get(metric_key, [])
            for v in values:
                plot_rows.append({'Metric': str(metric_key), 'Value': v, 'Model': model})
    
    if not plot_rows:
        print("[!] No data available for plotting")
        return
    
    df = pd.DataFrame(plot_rows)

    plt.figure(figsize=(10, 6))

    # Create boxplots with x-axis as Metric and hue as Model
    sns.boxplot(x='Metric', y='Value', hue='Model', data=df, palette="husl")

    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.xticks(rotation=30)
    plt.legend(title="Model")
    
    os.makedirs(args.save_path, exist_ok=True)
    output_path = os.path.join(args.save_path, f"{args.plot_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[O] Saved plot to: {output_path}")

if __name__ == '__main__':
    args = parse_arguments()
    
    if len(args.input_paths) != len(args.model_names):
        raise ValueError("[X] Number of input paths must match number of model names")

    all_data = {}
    for model_name, path in zip(args.model_names, args.input_paths):
        print(f"[!] Processing {model_name} results from: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[X] Directory not found: {path}")
            
        metrics = process_directory(path, args.metrics)
        all_data[model_name] = metrics
    
    generate_plots(all_data, args)
