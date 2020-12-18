from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, split_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        # Inserted split path -> is tha path of the file that contains the split of the dataset in train valid and test
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.split_path = split_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.valid_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        elif mode == 'valid':
            self.num_images = len(self.valid_dataset)
        else:
            self.num_images = len(self.test_dataset)


    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        # Creating the dict with the partition in training validation and test
        dict_split = dict()
        lines = [line.rstrip() for line in open(self.split_path, 'r')]
        for line in lines:
          split = line.split()
          dict_split[split[0]]=int(split[1])

        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            
            if (self.mode == 'train'):
              if (dict_split[filename] == 0):
                self.train_dataset.append([filename, label])
            elif (self.mode == 'valid'):
              if (dict_split[filename] == 1):
                self.valid_dataset.append((filename, label))
            else:
              if (dict_split[filename] == 2):
                self.test_dataset.append((filename, label))

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        # Fetching dataset 
        if self.mode == 'train':
          dataset = self.train_dataset
        elif self.mode == 'valid':
          dataset = self.valid_dataset
        else:
          dataset = self.test_dataset

        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.IntTensor(label), filename

    def __len__(self):
        """Return the number of images."""
        return int(np.floor(self.num_images/1))

def get_loader(image_dir, attr_path, split_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, split_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader