import os
import yaml
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def visualize_yaml_points(folder):
    print(f"Processing YAML files in folder: {folder}")
    
    train_weight_dict = defaultdict(list)
    train_group_zero_dict = defaultdict(int)
    
    for file_path in os.listdir(folder):
        if file_path.endswith(".yaml"):
            full_path = os.path.join(folder, file_path)
            print(f"Processing file: {full_path}")
            
            with open(full_path, "r", encoding="utf8") as f:
                config = yaml.safe_load(f)
                train_config = config.get("train", {})
                
                for k, v in train_config.items():
                    train_weight_dict[k].append(float(v))
                
                zero_count = sum(1 for v in train_config.values() if float(v) < 1e-7)
                train_group_zero_dict[zero_count] += 1
    
    print("\n--- Weight Statistics ---\n")
    for k, v in train_weight_dict.items():
        print(f"{k}: Max = {np.max(v):.6f}")
    
    print("\n--- Weight Distributions ---\n")
    for k, v in train_weight_dict.items():
        hist, bins = np.histogram(v, bins=10)
        print(f"{k}:")
        print(f"  Bins: {bins}")
        print(f"  Counts: {hist}")
        print()
    
    print("--- Zero Value Counts ---")
    for k, v in train_weight_dict.items():
        zero_count = sum(1 for x in v if x < 1e-7)
        print(f"{k}: {zero_count}")
    
    print("\n--- Groups Sorted by Zero Value Count ---\n")
    for k, v in sorted(train_group_zero_dict.items()):
        print(f"Groups with {k} zero values: {v}")
    
    # Visualize weight distributions
    plot_weight_distributions(train_weight_dict)

def plot_weight_distributions(train_weight_dict):
    num_keys = len(train_weight_dict)
    fig, axs = plt.subplots(num_keys, 1, figsize=(10, 5 * num_keys), tight_layout=True)
    
    for i, (k, v) in enumerate(train_weight_dict.items()):
        axs[i].hist(v, bins=20, edgecolor='black')
        axs[i].set_title(f'Distribution of weights for {k}')
        axs[i].set_xlabel('Weight value')
        axs[i].set_ylabel('Frequency')
        # log y axis
        axs[i].set_yscale('log')
    
    plt.savefig('weight_distributions.png')
    print("\nWeight distribution plot saved as 'weight_distributions.png'")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, default="config_1m", help="Folder path containing YAML files")
    folder_path = parser.parse_args().folder
    visualize_yaml_points(folder_path)
