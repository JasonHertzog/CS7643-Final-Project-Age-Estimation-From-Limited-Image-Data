import argparse
import copy
import pathlib
import yaml
import torch
from src.utils.reproducibility import set_seed
from src.dataset import get_dataloaders
# from src.models.base_model import get_model
from src.utils.regression import train, evaluate
from src.utils.plots import plot_curves

def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(model_name, pretrained=True, **kwargs):
    if model_name == "linear":
        from src.models.linear import get_model
    elif model_name == "mlp":
        from src.models.mlp import get_model
    elif model_name == "mlp_dropout":
        from src.models.mlp_dropout import get_model
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return get_model(pretrained, **kwargs)

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
    model = load_model(config['model_name'], config['pretrained'], dropout=config.get('dropout', 0.2))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)

    best_mae = 1000
    best_model = None
    best_loss = None

    train_losses, train_metrics = [], []
    val_losses, val_metrics = [], []

    print("Setup complete! Ready to begin training epochs.")
    for epoch in range(config['epochs']):
        train_stats = train(model, dataloaders['train'], optimizer, criterion, device=device)
        val_stats = evaluate(model, dataloaders['val'], criterion, device=device)

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
            f"Val Acc@5: {val_stats['acc_at_5']:.4f}"
        )
              

        train_losses.append(train_stats['loss'])
        train_metrics.append(train_stats['mae'])
        val_losses.append(val_stats['loss'])
        val_metrics.append(val_stats['mae'])

        if val_stats['mae'] < best_mae:
            best_mae = val_stats['mae']
            best_loss = val_stats['loss']
            best_model = copy.deepcopy(model)

        if config['save_best']:
            basedir = pathlib.Path(__file__).parent.parent.resolve()
            save_path = basedir / "outputs" / f"checkpoint_{config['model_name']}.pth"
            torch.save(best_model.state_dict(), save_path)
            print(f"Saved model with MAE = {best_mae:.4f}")

    print(f"Training completely finished! Best Val Loss: {best_loss:.4f}, Best Val MAE: {best_mae:.4f}")
    plot_curves(train_losses, train_metrics, val_losses, val_metrics, metric='MAE')

if __name__ == "__main__":
    main()
