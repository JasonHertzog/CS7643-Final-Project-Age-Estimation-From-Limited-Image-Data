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

def compute_classification_metrics(outputs, targets):
    # for classification, outputs are raw logits - use softmax expectations to get predicted class
    probabilities = torch.softmax(outputs, dim=1)
    # use probabilities to compute expected age for each sample
    expected_ages = (probabilities * torch.arange(probabilities.size(1), device=probabilities.device)).sum(dim=1)
    # usre predicted bins and expected_ages to computed the same metrics as regression for better comparison
    absolute_errors = torch.abs(expected_ages - targets.float())
    squared_errors = (expected_ages - targets.float()) ** 2 
    return {
        "mae": absolute_errors.mean().item(),
        "mse": squared_errors.mean().item(),
        "acc_at_3": (absolute_errors <= 3).float().mean().item(),
        "acc_at_5": (absolute_errors <= 5).float().mean().item()
    }

def train(model, dataloader, optimizer, criterion, task_type='regression', target_col='age', scheduler=None, device='cpu'):
    """Train for one epoch on UTKFace."""
    model.train()
    
    total_loss, total_mae, total_mse, total_acc_at_3, total_acc_at_5, total_acc, total_samples = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    progress_bar = tqdm(dataloader, ascii=True)
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        
        # grab correct target column
        targets = labels[target_col].to(device)
        
        optimizer.zero_grad()
        
        # handle classification vs regression
        if task_type == 'classification':
            comparison_targets = labels['age_bin'].to(device)  
            
            if target_col == 'soft_age':
                # for soft_age target, we need to use KLDivLoss and log_softmax outputs
                criterion = nn.KLDivLoss(reduction='batchmean')
                outputs = torch.log_softmax(model(images), dim=1)
            else:
                targets = targets.long()
                outputs = model(images)
        else:
            targets = targets.float()
            outputs = model(images).squeeze()
            
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
        batch_size = targets.size(0)
        
        if task_type == 'regression':
            batch_metrics = compute_regression_metrics(outputs, targets)
        else:
            batch_metrics = compute_classification_metrics(outputs, comparison_targets)
        
        total_mae += batch_metrics["mae"] * batch_size
        total_mse += batch_metrics["mse"] * batch_size
        total_acc_at_3 += batch_metrics["acc_at_3"] * batch_size
        total_acc_at_5 += batch_metrics["acc_at_5"] * batch_size
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress_bar.set_postfix({'loss': total_loss / total_samples})
        
    return {
        "loss": total_loss / total_samples,
        "mae": total_mae / total_samples,
        "mse": total_mse / total_samples,
        "acc_at_3": total_acc_at_3 / total_samples,
        "acc_at_5": total_acc_at_5 / total_samples,
    }

def evaluate(model, dataloader, criterion, task_type='regression', target_col='age', device='cpu'):
    """Evaluate on UTKFace (no gradient updates)."""
    model.eval()
    
    total_loss, total_mae, total_mse, total_acc_at_3, total_acc_at_5, total_acc, total_samples = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    progress_bar = tqdm(dataloader, ascii=True)
    
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            targets = labels[target_col].to(device)
            
            # handle classification vs regression
            if task_type == 'classification':
                comparison_targets = labels['age_bin'].to(device)  
                
                if target_col == 'soft_age':
                    # for soft_age target, we need to use KLDivLoss and log_softmax outputs
                    criterion = nn.KLDivLoss(reduction='batchmean')
                    outputs = torch.log_softmax(model(images), dim=1)
                else:
                    targets = targets.long()
                    outputs = model(images)
            else:
                targets = targets.float()
                outputs = model(images).squeeze()
                    
            loss = criterion(outputs, targets)
            batch_size = targets.size(0)
            
            if task_type == 'regression':
                batch_metrics = compute_regression_metrics(outputs, targets)
            else:
                batch_metrics = compute_classification_metrics(outputs, comparison_targets)
            
            total_mae += batch_metrics["mae"] * batch_size
            total_mse += batch_metrics["mse"] * batch_size
            total_acc_at_3 += batch_metrics["acc_at_3"] * batch_size
            total_acc_at_5 += batch_metrics["acc_at_5"] * batch_size
        
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            progress_bar.set_postfix({'loss': total_loss / total_samples})
            
    return {
        "loss": total_loss / total_samples,
        "mae": total_mae / total_samples,
        "mse": total_mse / total_samples,
        "acc_at_3": total_acc_at_3 / total_samples,
        "acc_at_5": total_acc_at_5 / total_samples
    }
