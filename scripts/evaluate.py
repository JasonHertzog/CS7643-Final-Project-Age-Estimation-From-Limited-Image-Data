import argparse
import pathlib
import yaml
import torch

from src.utils.reproducibility import set_seed
from src.dataset import get_dataloaders, UTKFaceDataset
from src.utils.engine import evaluate

def load_model(model_name, **kwargs):
    if model_name == "linear":
        from src.models.linear import get_model
    elif model_name == "mlp":
        from src.models.mlp import get_model
    elif model_name == "mlp_dropout":
        from src.models.mlp_dropout import get_model
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return get_model(**kwargs)
     
def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Age Estimation Training Baseline")
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', 
                        help='Path to the experiment configuration file')
    args = parser.parse_args()

    # Load parameters
    config = load_config(args.config)
    set_seed(config.get('seed', 42))

    print("Configuration loaded. Loading data and model...")
    
    # Initialize Data
    data_dir = 'data/processed' # Assumes standard repo structure
    dataloaders = get_dataloaders(data_dir, config['batch_size'], config['image_size'])
    
    # Initialize Model & Optimizer
    if config['task'] == "classification":
        criterion = torch.nn.CrossEntropyLoss()
        target_col = 'age_bin'  # Use age bins for classification
        metric = config.get('metric', 'mae')   # get metric from config, default to accuracy for classification
        out_features = 116 # number of age bins (0-115)
    else:   
        criterion = torch.nn.MSELoss()
        target_col = 'age'  # Use actual age for regression
        metric = config.get('metric', 'mae')        # get metric from config, default to mae for regression
        out_features = 1
    
    if config['save_best']:
        basedir = pathlib.Path(__file__).parent.parent.resolve()
        load_path = basedir / "outputs" / f"checkpoint_{config['experiment_name']}.pth"
        print("Loading model from checkpoint:", load_path)
    else:
        # exit early if we're not saving/loading a model, since this script is meant for evaluation only
        print("No model checkpoint specified for loading. Exiting since this script is meant for evaluation")
        return

    num_classes = len(UTKFaceDataset.AGE_BIN_LABELS)
    model = load_model(
        config['model_name'],
        pretrained=config.get('pretrained', True),
        freeze_backbone=config.get('freeze_backbone', False),
        num_classes=num_classes,
        dropout=config.get('dropout', 0.2),
        out_features=out_features,
    )
    state_dict = torch.load(load_path, weights_only=True)
    model.load_state_dict(state_dict)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    test_stats = evaluate(model, dataloaders['test'], criterion, device=device,task_type=config['task'], target_col=target_col)

    print(f"\nEvaluation Metrics — {config['experiment_name']}")

    #if config['task'] == 'regression':
    rows = [
        ("MAE",    f"{test_stats['mae']:.4f}"),
        ("MSE",    f"{test_stats['mse']:.4f}"),
        ("Acc@3",  f"{test_stats['acc_at_3']:.4f}"),
        ("Acc@5",  f"{test_stats['acc_at_5']:.4f}"), 
    ]
    #else:
    #    rows = [
    #        ("Accuracy", f"{test_stats['accuracy']:.4f}"),
    #    ]

    col_w = max(len("Metric"), max(len(r[0]) for r in rows))  # dynamic column width based on longest metric name
    sep = "+" + "-" * (col_w + 2) + "+" + "-" * 10 + "+"
    print(sep)
    print(f"| {'Metric'.ljust(col_w)} | {'Value':>8} |")
    print(sep)
    for name, val in rows:
        print(f"| {name.ljust(col_w)} | {val:>8} |")
    print(sep)
              
if __name__ == "__main__":
    main()
