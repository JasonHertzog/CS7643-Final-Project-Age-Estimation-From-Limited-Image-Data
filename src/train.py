import argparse
import yaml
import torch
from utils.reproducibility import set_seed

def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # 1. Parse command-line arguments (just the config file path)
    parser = argparse.ArgumentParser(description="Age Estimation Training Baseline")
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', 
                        help='Path to the experiment configuration file')
    args = parser.parse_args()

    # 2. Load hyperparameters
    # config = load_config(args.config)
    # print(f"Loaded config from {args.config}")
    
    # 3. Ensure Reproducibility
    # set_seed(config.get('seed', 42))
    set_seed(42) 

    print("Pipeline Initialized. Ready for dataset loading...")
    

if __name__ == "__main__":
    main()
