# Experiment Configuration & Reproducibility

## Goal
This document records all seeds, hyperparameters, data preprocessing steps, dataset splits, and checkpointing rules used in our experiments. The goal is to ensure that results can be fully reproduced without requiring reverse-engineering of the code.

---

## 1. Random Seeds

```yaml
seed: 42
torch_seed: 42
numpy_seed: 42
```

## 2. Models
### Baseline (Scratch)
```yaml
model: resnet18_scratch
initialization: random
learning_rate: 1e-3
batch_size: 32
epochs: 20
optimizer: Adam
loss_function: MSELoss
```
### Transfer Learning
```yaml
model: resnet18_pretrained
pretrained_on: ImageNet
learning_rate: 1e-4
batch_size: 32
epochs: 20
optimizer: Adam
loss_function: MSELoss
freeze_backbone: true
```
## 3. Data Preprocessing
```yaml
image_size: 224x224
normalization:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
augmentations:
  - horizontal_flip (optional)
```
## 4. Dataset Splits
```yaml
train_split: 80%
validation_split: 10%
test_split: 10%
split_method: random
seed: 42
```
## 5. Checkpointing
```yaml
save_best_model: true
selection_metric: validation_loss
save_frequency: every_epoch
```
## 6. Evaluation Metrics
```yaml
metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
```
