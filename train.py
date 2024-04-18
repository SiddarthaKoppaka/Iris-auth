import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.nn as nn
import torch.nn.functional as F
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import dataset_creation as dc
import load_and_preprocess as p
import triplet

def get_default_device():
  """
  Selects the best available device for PyTorch training.

  Returns:
      A torch.device object representing the chosen device.
  """
  if torch.cuda.is_available():
    # Check for multiple GPUs and choose the first one
    device_count = torch.cuda.device_count()
    if device_count > 1:
      print(f"Using {device_count} GPUs for training.")
      return torch.device('cuda:0')  # Use the first GPU
    else:
      print("Using single GPU for training.")
      return torch.device('cuda')
  else:
    print("Using CPU for training.")
    return torch.device('cpu')

IMAGE_SIZE = 224
BATCH_SIZE = 2
DEVICE = get_default_device()
LEARNING_RATE = 0.005
EPOCHS = 10

def load_dataset(path):
    dataset = p.import_images(path)
    data = dataset.train_test_split(test_size=0.2, shuffle=True)
    data = data['train'].train_test_split(test_size=0.2, shuffle=True)
    train_triplets, train_labels = dc.triplet_dataset(data['train']['image'], data['train']['labels'])
    test_triplets,test_labels = dc.triplet_dataset(data['test']['image'], data['test']['labels'])
    return train_triplets, train_labels, test_triplets, test_labels


def get_train_dataset(triplets, labels, IMAGE_SIZE=224):
    train_dataset = dc.TripletDataset(triplets,labels,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((IMAGE_SIZE,IMAGE_SIZE))]))
    return train_dataset

def main(triplets,images,IMAGE_SIZE = 224,
BATCH_SIZE = 4,
DEVICE = get_default_device(),
LEARNING_RATE = 0.005,
EPOCHS = 10):
    torch.cuda.empty_cache()
    train_dataset = get_train_dataset(train_triplets, train_labels,IMAGE_SIZE = IMAGE_SIZE)
    train_dl = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True)
    vit = triplet.ViT_Triplet()
    vit = vit.to(DEVICE)
    Optimizer = torch.optim.Adam(vit.parameters(),lr = LEARNING_RATE)
    criterion = triplet.TripletLoss()

    for epoch in tqdm(range(EPOCHS), desc='Epochs'):
        running_loss = []
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(tqdm(train_dl,leave=False)):
        
            anchor_img = anchor_img.to(DEVICE)
            positive_img = positive_img.to(DEVICE)
            negative_img = negative_img.to(DEVICE)
            Optimizer.zero_grad()
            anchor_out = vit(anchor_img)
            positive_out = vit(positive_img)
            negative_out = vit(negative_img)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            Optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
            print('Epoch: {}/{} — Loss: {:.4f}'.format(epoch+1, EPOCHS, np.mean(running_loss)))

    return vit

if __name__ == '__main__':
  import argparse


  # Define argument parser
  parser = argparse.ArgumentParser(description='Train ViT model with Triplet Loss')

  # Define arguments with defaults from the function
  parser.add_argument('--image_size', type=int, default=IMAGE_SIZE, help='Size to resize images (default: 224)')
  parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training (default: 32)')
  parser.add_argument('--device', type=str, default=DEVICE, help='Device to use for training (default: cuda if available)')
  parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate for optimizer (default: 0.005)')
  parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs (default: 10)')
  parser.add_argument('--path', type=str, required=True, help='Path to the dataset directory')

  # Parse arguments
  args = parser.parse_args()

  # Load the dataset using the provided path
  train_triplets, train_labels, test_triplets, test_labels = load_dataset(args.path)

  # Train the model with the parsed arguments
  model = main(train_triplets, train_labels, 
               IMAGE_SIZE=args.image_size, 
               BATCH_SIZE=args.batch_size, 
               DEVICE=args.device, 
               LEARNING_RATE=args.learning_rate, 
               EPOCHS=args.epochs)

  # Save the trained model (optional)
  model.save_dict('vit_16_tl.pth')
  # ... (your code to save the model)
