import argparse
import yaml
import torch
from utils.reproducibility import set_seed
from data_loader import get_dataloaders
from model import get_model

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
    data_dir = 'data/raw/utkface' # Assumes standard repo structure
    dataloader = get_dataloaders(data_dir, config['batch_size'], config['image_size'])
    
    # Initialize Model & Optimizer
    model = get_model()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print("Setup complete! Ready to begin training epochs.")
    # Main Training Loop
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, ages) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(images)
            
            # Squeeze outputs to match the shape of the ages tensor
            loss = criterion(outputs.squeeze(), ages)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.4f}")
        
    print("Training completely finished!")

if __name__ == "__main__":
    main()
