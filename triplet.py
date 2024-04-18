import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vit_b_16


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
    
class ViT_Triplet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load a pretrained Vision Transformer model
        self.Feature_Extractor = vit_b_16(pretrained=pretrained)
        num_filters = self.Feature_Extractor.heads[0].in_features  # Access the in_features of the ViT head

        # Replace the head of the ViT model with a new sequence of layers
        self.Feature_Extractor.heads = nn.Sequential(
            nn.Linear(num_filters, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 10)
        )

        # Triplet Loss is represented by this simple transformation,
        # This could be your way of reducing the feature dimensions before computing the actual triplet loss
        self.Triplet_Loss = nn.Sequential(
            nn.Linear(10, 2)
        )

    def forward(self, x):
        features = self.Feature_Extractor(x)
        triplets = self.Triplet_Loss(features)
        return triplets