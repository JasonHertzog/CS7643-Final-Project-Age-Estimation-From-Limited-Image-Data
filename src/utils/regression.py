import torch
import torch.nn as nn

from tqdm import tqdm

def compute_regression_metrics(outputs, targets):
    absolute_errors = torch.abs(outputs - targets)
    squared_errors = (outputs - targets) ** 2

    return {
        "mae": absolute_errors.mean().item(),
        "mse": squared_errors.mean().item(),
        "acc_at_3": (absolute_errors <= 3).float().mean().item(),
        "acc_at_5": (absolute_errors <= 5).float().mean().item(),
    }

def train(model, dataloader, optimizer, criterion, scheduler=None, device='cpu'):
    """Train for one epoch on UTKFace age regression.

    Each batch from the dataloader is expected to be a tuple of
    (images, labels) where images has shape (B, 3, H, W) and labels is a
    dict containing at minimum the key "age" with a float32 tensor of shape (B,).

    Args:
        model      : nn.Module whose forward pass accepts (B, 3, H, W) and
                     returns a tensor of shape (B,) or (B, 1).
        dataloader : DataLoader wrapping a UTKFaceDataset.
        optimizer  : Optimizer (e.g. Adam, SGD).
        criterion  : Loss function (e.g. nn.MSELoss(), nn.L1Loss()).
        scheduler  : Optional LR scheduler stepped once per batch
                     (e.g. OneCycleLR).  Pass None to skip.
        device     : 'cpu', 'cuda', or 'mps'.

    Returns:
        dict containing average metrics as keys: loss, mae, mse, acc_at_3, acc_at_5
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_acc_at_3 = 0.0
    total_acc_at_5 = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, ascii=True)

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        targets = labels['age'].to(device)          # (B,)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)          # (B,) — handles (B,1) output too
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = targets.size(0)
        batch_metrics = compute_regression_metrics(outputs, targets)

        total_loss += loss.item() * batch_size
        total_mae += batch_metrics["mae"] * batch_size
        total_mse += batch_metrics["mse"] * batch_size
        total_acc_at_3 += batch_metrics["acc_at_3"] * batch_size
        total_acc_at_5 += batch_metrics["acc_at_5"] * batch_size
        total_samples += batch_size

        progress_bar.set_description_str(
            "Batch: %d, Loss: %.4f, MAE: %.4f" % ((batch_idx + 1), loss.item(), batch_metrics["mae"]))

    #n = len(dataloader)
    #return total_loss, total_loss / n, total_mae / n
    if total_samples == 0:
        raise ValueError("No samples were processed during training!")
    return {
        "loss": total_loss / total_samples,
        "mae": total_mae / total_samples,
        "mse": total_mse / total_samples,
        "acc_at_3": total_acc_at_3 / total_samples,
        "acc_at_5": total_acc_at_5 / total_samples,
    }


def evaluate(model, dataloader, criterion, device='cpu'):
    """Evaluate on UTKFace age regression (no gradient updates).

    Args:
        model      : Trained nn.Module.
        dataloader : DataLoader wrapping a UTKFaceDataset.
        criterion  : Loss function matching the one used during training.
        device     : 'cpu', 'cuda', or 'mps'.

    Returns:
        total_loss (float), avg_loss (float), mae (float)
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, ascii=True)

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            targets = labels['age'].to(device)      # (B,)

            outputs = model(images).squeeze(1)      # (B,)
            loss = criterion(outputs, targets)

            total_mae += torch.abs(outputs - targets).mean().item()
            total_loss += loss.item()
            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    n = len(dataloader)
    return total_loss, total_loss / n, total_mae / n
