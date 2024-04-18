import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

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
        super(ViT_Triplet, self).__init__()
        if pretrained:
            # Load a pretrained Vision Transformer model from Hugging Face's 'transformers'
            self.Feature_Extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        else:
            # Load a ViT model with default configuration
            config = ViTConfig()
            self.Feature_Extractor = ViTModel(config)

        num_filters = self.Feature_Extractor.config.hidden_size  # Access the hidden_size from the ViT config

        # Replace the head of the ViT model with a new sequence of layers
        self.Feature_Extractor.heads = nn.Sequential(
            nn.Linear(num_filters, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 10)
        )

        # Triplet Loss feature reduction
        self.Triplet_Loss = nn.Sequential(
            nn.Linear(10, 2)
        )

    def forward(self, x):
        # Extract features from the ViT model
        outputs = self.Feature_Extractor(x)
        features = outputs.last_hidden_state[:, 0, :]  # Get the [CLS] token's embeddings
        reduced_features = self.Feature_Extractor.heads(features)
        triplets = self.Triplet_Loss(reduced_features)
        return triplets

# Note that you will need to adjust the input dimensions and data processing as per the requirements of the ViT model.
