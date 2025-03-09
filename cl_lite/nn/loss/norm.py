import torch
import torch.nn as nn


class NormLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, centroids, num_old_class):
        centroids_old_norm = torch.linalg.norm(centroids[:num_old_class], dim=1)
        centroids_new_norm = torch.linalg.norm(centroids[num_old_class:], dim=1)
        centroids_old_diff = centroids_old_norm - centroids_new_norm.mean()
        centroids_new_diff = centroids_new_norm - centroids_old_norm.mean()
        return torch.linalg.norm(torch.cat([centroids_old_diff, centroids_new_diff], dim=0), ord=1)
