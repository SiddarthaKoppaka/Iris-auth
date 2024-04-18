import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch

def triplet_dataset(images, labels):
  """
  Creates triplets for a triplet loss from provided images and labels.

  Args:
      images: A list of images.
      labels: A list of labels corresponding to the images.

  Returns:
      A tuple containing two lists: triplets and labels.
          - triplets: A list of dictionaries with keys 'anchor', 'positive', and 'negative'.
          - labels: The original list of labels.
  """
  triplets = []

  for i, label_anchor in tqdm(enumerate(labels), total=len(labels), desc="Creating triplets"):
    anchor = images[i]
    positive_indices = [j for j in range(len(labels)) if labels[j] == label_anchor and j != i]

    # Check if there's at least one positive image
    if positive_indices:
      positive = images[random.choice(positive_indices)]
      negative = random.choice([images[j] for j in range(len(labels)) if j not in (i, *positive_indices)])
      triplets.append({
          'anchor': anchor,
          'positive': positive,
          'negative': negative
      })

  return triplets, labels


class TripletDataset(Dataset):
    def __init__(self, triplets, labels, transform=None):
        self.triplets = triplets
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        # Retrieve each image data from the triplet
        anchor = self.triplets[idx]['anchor']
        positive = self.triplets[idx]['positive']
        negative = self.triplets[idx]['negative']
        
        # Apply the transformation if any
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        label = self.labels[idx]
        
        return anchor, positive, negative, label
    

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
