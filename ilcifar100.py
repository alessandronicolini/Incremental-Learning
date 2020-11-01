import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR100
from PIL import Image

class ilCIFAR100(CIFAR100):
    """
    Extends CIFAR100 class. Split the dataset into 10 batches, each one containing 10 classes.
    You can retrieve the batches from the attribute "batches", it has different structure according to
    test and train CIFAR100 splits:
        - train -> batches is a dictionary {0:{'train':indexes, 'val':indexes}...} 
        - test -> batches is a dictionary {0:indexes...}
    where the keys are the batch number.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        seed(int): used to ensure reproducibility in shuffling operations.
        val_size(float, optional): between 0 and 1, fraction of data used for validation.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, root, seed, val_size=0.1, train=True, transform=None, target_transform=False, 
    download=True):
        
        super(ilCIFAR100, self).__init__(root=root, train=train, transform=transform, 
        target_transform=target_transform, download=download)
        
        self.targets = np.array(self.targets) # make targets an array to exploit masking
        self._rs = np.random.RandomState(seed) # set random seed
        self._classes_per_batch = self._get_classes_per_batch() 
        if train:
            self.batches = self._make_train_batches(val_size)
        else:
            self.batches = self._make_test_batches()
        

    def _get_classes_per_batch(self):
        """
            Args: 

            Returns:
                2D-array: rows are associated to batch number and columns to batch class labels
        """
        labels = np.arange(0, 100, 1)
        self._rs.shuffle(labels)
        labels = labels.reshape((10, -1)) # each row contains the classes for the corrisponding batch
        return labels

    def _make_test_batches(self):
        """
            Args:

            Returns:
                dictionary {0:indexes...}: key is the batch number and value is a list of the
                associated sample indexes
        """
        batches = {key:[] for key in range(10)}
        for batch in range(10):
            for label in self._classes_per_batch[batch, :]: # select labels of the corrisponding batch
                batches[batch] += list(np.where(self.targets == label)[0]) # np.where() is a tuple, the first element is the indexes array
        return batches

    def _make_train_batches(self, val_size):
        """
            Args: 
                val_size (float): fraction of samples used for validation, assumes values in range [0,1]

            Returns:
                dictionary {0:{'train':indexes, 'val':indexes}...}: key is the batch number and value is
                a dictionary which contains sample indexes both for training and validation.
        """
        batches = {key: {'train': [], 'val': []} for key in range(10)}
        for batch in range(10):
            for label in self.classes_per_batch[batch, :]:
                indexes = list(np.where(self.targets == label)[0])
                val_length = int(len(indexes)*val_size) # to split the data for the current class label
                batches[batch]['val'] += indexes[:val_length]
                batches[batch]['train'] += indexes[val_length:]
            self.__s.shuffle(batches[batch]['val']) # otherwise same class elements are subsequent 
            self._rs.shuffle(batches[batch]['train'])
        return batches