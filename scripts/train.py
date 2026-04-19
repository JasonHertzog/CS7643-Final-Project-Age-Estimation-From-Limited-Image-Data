import argparse
import copy
import pathlib
import yaml
import torch
from src.utils.reproducibility import set_seed
from src.dataset import get_dataloaders
from src.models.base_model import get_model
from src.utils.regression import train, evaluate

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

    print("Pipeline Initialized. Loading data and model...")
    
    # Initialize Data
    data_dir = 'data/processed' # Assumes standard repo structure
    dataloaders = get_dataloaders(data_dir, config['batch_size'], config['image_size'])
    
    # Initialize Model & Optimizer
    model = get_model()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)

    best_mae = 1000
    best_model = None
    best_loss = None

    print("Setup complete! Ready to begin training epochs.")
    for epoch in range(config['epochs']):
        _, train_loss, train_mae = train(model, dataloaders['train'], optimizer, criterion, device=device)
        _, val_loss, val_mae = evaluate(model, dataloaders['val'], criterion, device=device)
        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

        
        if val_mae < best_mae:
            best_mae = val_mae
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            
        if config['save_best']:
            basedir = pathlib.Path(__file__).parent.parent.resolve()
            save_path = basedir / "outputs" / f"checkpoint_{config['model_name']}.pth"
            torch.save(best_model.state_dict(), save_path)
            print(f"Saved model with MAE = {best_mae:.4f}")

    print(f"Training completely finished! Best Val Loss: {best_loss:.4f}, Best Val MAE: {best_mae:.4f}")

if __name__ == "__main__":
    main()
