import math
import matplotlib.pyplot as plt

from scripts.preprocess_data import OUTPUT_DIR

def plot_curves(train_loss_history, train_metric_history, valid_loss_history, valid_metric_history, metric='MAE', save_path=None) -> None:
    """
    Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
    training accuracy and validation accuracy are plot in the same figure too.
    :param train_loss_history: training loss history of epochs
    :param train_metric_history: training metric history of epochs where metric could be any metric
    :param valid_loss_history: validation loss history of epochs
    :param valid_metric_history: validation metric history of epochs where metric could be any metric
    :param metric: metric to be used
    :param save_path: path to save the plot
    :return: None, save two figures in the current directory
    """
    
    # plot the iterative learning curve (accuracy)
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Train", color="cornflowerblue")
    plt.plot(valid_loss_history, label="Valid", color="chartreuse")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(visible=True)
    plt.legend(frameon=True)

    # plot the iterative learning curve (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(train_metric_history, label="Train", color="cornflowerblue")
    plt.plot(valid_metric_history, label="Valid", color="chartreuse")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.title(metric + " Curve")
    plt.grid(visible=True)
    plt.legend(frameon=True)
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

def plot_validation_curves(valid_history, title="None", metric="MAE", label="None", save_path=None) -> None:
    plt.figure(figsize=(9, 5))
    
    valid_metric_history = [epoch_stats[metric.lower()] for epoch_stats in valid_history]

    style_by_epochs = {
        30: {"color": "tab:green", "zorder": 1, "linewidth": 2.0, "alpha": 0.70},
        20: {"color": "tab:orange", "zorder": 3, "linewidth": 2.2, "alpha": 0.95},
        10: {"color": "tab:blue", "zorder": 4, "linewidth": 2.4, "alpha": 1.00},
    }

    num_epochs = min(30, len(valid_metric_history))

    for i in range(math.ceil(num_epochs/10)):
        s = i * 10
        e = s + 10 
        style = style_by_epochs[e] 
        epochs = list(range(1,e+1))
        n = num_epochs if e > num_epochs else e
        
        print(f"Epochs 0-{n}: Best Val {metric} = {min(valid_metric_history[s:e]):.4f}")
        best_val_mae =  min(valid_metric_history[s:e])
        best_epoch = valid_metric_history.index(best_val_mae) + 1
        
        glabel = label + f" ({n} epochs)"
  
        plt.plot(
            epochs,
            valid_metric_history[:n],
            color=style["color"],
            marker="o",
            markersize=4.5,
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=glabel,
        )
            
        plt.scatter(
            best_epoch,
            best_val_mae,
            s=85,
            color=style["color"],
            edgecolor="black",
            zorder=style["zorder"] + 10,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation " + metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
