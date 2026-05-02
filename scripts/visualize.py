import argparse
import pathlib
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    elif model_name == "mlp_bn":
        from src.models.mlp_bn import get_model # new model with batch norm
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

    # print model parameters
    print(f"\n\033[1mModel Parameters = {sum(p.numel() for p in model.parameters()):,}\033[0m")


    # print model parameters
    print(f"\n\033[1m Generating Cards for {config['experiment_name']} \033[0m")

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)

    samples = []

    # ImageNet mean and std (used in the dataset transforms)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    model.eval()
    with torch.no_grad():
        for batch in dataloaders['test']:
            images, labels = batch          # <-- it's a tuple, not a dict
            images = images.to(device)
            ages   = labels['age'].float()  # labels IS a dict

            outputs = model(images)

            if config['task'] == 'classification':
                age_bins = torch.arange(out_features).float().to(device)
                probs = torch.softmax(outputs, dim=1)
                # add one to age_bins to shift from 0-115 to 1-116, since the dataset ages are 1-indexed (1-116)
                age_bins += 1
                preds = (probs * age_bins).sum(dim=1).cpu()  
            else:
                preds = outputs.squeeze(1).cpu()

            for i in range(len(images)):
                gt   = ages[i].item()
                pred = preds[i].item()
                error = abs(pred - gt)

                # Denormalize using ImageNet stats
                img_tensor = images[i].cpu()
                img_np = (img_tensor * std + mean).permute(1, 2, 0).numpy()
                img_np = img_np.clip(0, 1)

                samples.append({
                    'image': img_np,
                    'gt':    int(round(gt)),
                    'pred':  int(round(pred)),
                    'error': error
                })

    print(f"Total test samples collected: {len(samples)}")      

    # output file
    basedir = pathlib.Path(__file__).parent.parent.resolve() / "outputs"
    save_path_r = basedir / f"cards_random10_{config['experiment_name']}.png"
    save_path_w = basedir / f"cards_worst10_{config['experiment_name']}.png"
    

    # Sort by worst error (biggest mistake first)
    samples_sorted = sorted(samples, key=lambda x: x['error'], reverse=True)
    top10 = samples_sorted[:10]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(
        f"Card Outputs — {config['experiment_name']}\n(10 Worst Predictions)",
        fontsize=14, fontweight='bold', y=1.01
    )

    for idx, ax in enumerate(axes.flat):
        s = top10[idx]
        ax.imshow(s['image'])
        ax.set_title(
            f"GT: {s['gt']}  |  Pred: {s['pred']}\nError: {s['error']:.1f} yrs",
            fontsize=10, color='white',
            bbox=dict(facecolor='black', alpha=0.6, pad=3)
        )
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path_w, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path_w}")

    # random samples
    import random
    random.seed(42)
    fixed10 = random.sample(samples, 10)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(
        f"Card Outputs — {config['experiment_name']}\n(10 Random Predictions)",
        fontsize=14, fontweight='bold', y=1.01
    )

    for idx, ax in enumerate(axes.flat):
        s = fixed10[idx]
        ax.imshow(s['image'])
        ax.set_title(
            f"GT: {s['gt']}  |  Pred: {s['pred']}\nError: {s['error']:.1f} yrs",
            fontsize=10, color='white',
            bbox=dict(facecolor='black', alpha=0.6, pad=3)
        )
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path_r, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path_r}")

if __name__ == "__main__":
    main()
