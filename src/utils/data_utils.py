import torch
from torch.utils.data import Subset

def get_dataset_subset(dataset, fraction):
    # utility to grab a fraction of the dataset for limited-data experiments
    num_samples = len(dataset)
    subset_size = int(num_samples * fraction)
    indices = torch.randperm(num_samples).tolist()
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices)
