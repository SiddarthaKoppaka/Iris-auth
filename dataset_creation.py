import random
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms

def triplet_dataset(images, labels):
    label_to_indices = {label: np.where(labels == label)[0] for label in set(labels)}
    triplets = []

    for i, label_anchor in tqdm(enumerate(labels), total=len(labels), desc="Creating triplets"):
        anchor = images[i]
        positive_indices = label_to_indices[label_anchor]
        positive_indices = positive_indices[positive_indices != i]

        if len(positive_indices) == 0:
            continue  # skip if no valid positive sample found

        negative_indices = np.array([idx for idx in range(len(labels)) if labels[idx] != label_anchor])
        positive = images[random.choice(positive_indices)]
        negative = images[random.choice(negative_indices)]

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
        anchor = self.triplets[idx]['anchor']
        positive = self.triplets[idx]['positive']
        negative = self.triplets[idx]['negative']
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative

