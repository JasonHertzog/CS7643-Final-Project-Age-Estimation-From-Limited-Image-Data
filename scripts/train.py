import argparse
import copy
import pathlib
import yaml
import torch
import numpy 

from src.utils.reproducibility import set_seed
from src.dataset import get_dataloaders, UTKFaceDataset
# from src.models.base_model import get_model
from src.utils.engine import train, evaluate
from src.utils.plots import plot_curves

# Model Imports - these are imported dynamically in load_model() based on config
# from src.models.linear import get_model
# from src.models.mlp import get_model
# from src.models.mlp_dropout import get_model      

def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

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

def main():
    parser = argparse.ArgumentParser(description="Age Estimation Training Baseline")
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', 
                        help='Path to the experiment configuration file')
    args = parser.parse_args()

    # Load parameters
    config = load_config(args.config)
    set_seed(config.get('seed', 42))

    print("Pipeline Initialized. Loading data and model...")
    
    # Initialize Data
    data_dir = 'data/processed' # Assumes standard repo structure
    dataloaders = get_dataloaders(data_dir, config['batch_size'], config['image_size'])
    
    # Initialize Model & Optimizer
    if config['task'] == "classification":
        criterion = torch.nn.CrossEntropyLoss()
        target_col = 'age_bin'  # Use age itself as a one year bins for classification
        metric = config.get('metric', 'mae')   # get metric from config, default to accuracy for classification
        out_features = 116 # number of age bins (0-115)
    else:   
        criterion = torch.nn.MSELoss()
        target_col = 'age'  # Use actual age for regression
        metric = config.get('metric', 'mae')        # get metric from config, default to mae for regression
        out_features = 1

    #num_classes = len(UTKFaceDataset.AGE_BIN_LABELS)
    model = load_model(
        config['model_name'],
        pretrained=config.get('pretrained', True),
        freeze_backbone=config.get('freeze_backbone', False),
        dropout=config.get('dropout', 0.2),
        out_features=out_features,
    )
    
    weight_decay=config.get('weight_decay', 0.0)
    if weight_decay > 0:
        # use adamw
        print(f"Using AdamW optimizer with weight decay: {weight_decay}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=weight_decay)
    else:
        print(f"Using Adam optimizer without weight decay")
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)

    best_metric = None
    best_model = None
    best_loss = None

    train_losses, train_metrics = [], []
    val_losses, val_metrics = [], []
    stat_history = {
        'train': [],
        'eval': []
    }
    print("Setup complete! Ready to begin training epochs.")
    
    basedir = pathlib.Path(__file__).parent.parent.resolve() / "outputs"
    save_path = basedir / f"checkpoint_{config['experiment_name']}.pth"
        
    for epoch in range(config['epochs']):
        train_stats = train(model, dataloaders['train'], optimizer, criterion, device=device, task_type=config['task'], target_col=target_col)
        val_stats = evaluate(model, dataloaders['val'], criterion, device=device,task_type=config['task'], target_col=target_col)

        print(
            f"Epoch [{epoch+1}/{config['epochs']}] "
            f"Train Loss: {train_stats['loss']:.4f}, "
            f"Train MAE: {train_stats['mae']:.4f}, "
            f"Train MSE: {train_stats['mse']:.4f}, "
            f"Train Acc@3: {train_stats['acc_at_3']:.4f}, "
            f"Train Acc@5: {train_stats['acc_at_5']:.4f} | "
            f"Val Loss: {val_stats['loss']:.4f}, "
            f"Val MAE: {val_stats['mae']:.4f}, "
            f"Val MSE: {val_stats['mse']:.4f}, "
            f"Val Acc@3: {val_stats['acc_at_3']:.4f}, "
            f"Val Acc@5: {val_stats['acc_at_5']:.4f}, "
            f"Val {metric.upper()}: {val_stats[metric]:.4f}"    
        )
              
        train_losses.append(train_stats['loss'])
        train_metrics.append(train_stats[metric])
        val_losses.append(val_stats['loss'])
        val_metrics.append(val_stats[metric])

        stat_history['train'].append(train_stats)
        stat_history['eval'].append(val_stats)

        if best_metric is None or val_stats[metric] < best_metric:
            best_metric = val_stats[metric]
            best_loss = val_stats['loss']
            best_model = copy.deepcopy(model)

        if config['save_best'] and (epoch+1) % 10 == 0:
            save_path = basedir / f"checkpoint_{config['experiment_name']}_{epoch+1}.pth"
            torch.save(best_model.state_dict(), save_path)
            print(f"Saving Best Val {metric.upper()}: {best_metric:.4f} at Epoch {epoch+1}")

    if config['save_best']:
        save_path = basedir / f"checkpoint_{config['experiment_name']}.pth"
        torch.save(best_model.state_dict(), save_path)
        print(f"Saved model with {metric.upper()} = {best_metric:.4f}")

    # save stats history to a numpy file for later analysis
    stats_save_path = basedir / f"stats_history_{config['experiment_name']}.npy"
    numpy.save(stats_save_path, stat_history)
    print(f"Saved training stats history to {stats_save_path}")

    print(f"Training completely finished! Best Val Loss: {best_loss:.4f}, Best Val {metric.upper()}: {best_metric:.4f}")
    plot_curves(train_losses, train_metrics, val_losses, val_metrics, metric=metric, save_path=f"outputs/experiment_curves_{config['experiment_name']}.png")

if __name__ == "__main__":
    main()
