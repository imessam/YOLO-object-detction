import cv2
import torch
import torchvision as tv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class VOCDataset(Dataset):

    def __init__(self, annotations, images_dirs, transform=None):
        """
        Args:
            annotations (Numpy array): Numpy array of annotations.
            images_dirs (list of strings): Directories of the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = annotations
        self.images_dirs = images_dirs
        self.transform = transform

    def __len__(self):
        return len(self.images_dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images_dirs[idx]
        image = plt.imread(img_name)
        annotaion = self.annotations[idx]
        sample = {'image': image, 'annotation': annotaion}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    

    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotation']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_h, new_w))


        return {'image': img, 'annotation': annotations}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotation']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'annotation': torch.from_numpy(annotations)}
    