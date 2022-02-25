import matplotlib.pyplot as plt
import random
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms



class MNISTDatasetWrapper(Dataset):
    """
    A Dataset for learning with subsets of the Fashion MNIST dataset for either the
    original labels or labels that describe how the image has been rotated.
    Rotations will be applied clockwise, with a random choice of one of the
    following degrees: [0, 45, 90, 135, 180, 225, 270, 315]

    - original_dataset - the fashion mnist dataset we got with torchvision
    - pct - percent of data to use
    - for_rotation_classification - True=Use rotation labels.
                                    False=Use original classification labels.
    """

    def __init__(self, original_dataset, pct=1.0, for_rotation_classification=False):
        
        # in case we need to get the subset
        self.dataset_size = int(len(original_dataset)*pct)
        self.dataset = torch.utils.data.Subset(original_dataset, random.choices(range(len(original_dataset)), k=self.dataset_size))
        
        self.for_rotation_classification = for_rotation_classification
        self.label_to_class = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        # tensor image shapte [B,C,H,W]
        self.imageToTensor = transforms.ToTensor()
        # PIL short for Python Image Library
        self.tensorToImage = transforms.ToPILImage()
        # normalize takes mean, std (given for each channel) (works on tensor only)
        # output[channel] = (input[channel] - mean[channel]) / std[channel]
        self.normalize = transforms.Normalize((0.2859,), (0.3530,))
        # this is literally how you get back the original image tensor
        self.denormalize = transforms.Normalize((-0.2859/0.3530,), (1.0/0.3530,))
        # choices of Rotation Degrees
        self.rot_choices = [0, 45, 90, 135, 180, 225, 270, 315]
        # number of roationa degrees to choose from
        self.num_rot = len(self.rot_choices)

    # dumest request I have seen in my entire life candidate here
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Returns a 32x32 MNIST digit and its corresponding clothes
        label, or if self.for_rotation_classification is true returns
        the image rotated by a random rotation amount from
        self.rot_choices. Note: the label should be a number between 0-7,
        not the number of degrees to rotate by.
        """
        img, label = self.dataset[idx]
        if not self.for_rotation_classification:
            return self.normalize(img), label
        else:
            img = self.tensorToImage(img)
            label = random.choice([i for i in range(self.num_rot)])
            rot_degree = self.rot_choices[label]
            img = img.rotate(rot_degree)
            img = self.normalize(self.imageToTensor(img))
            return img, torch.tensor(label).long()

    def show_batch(self, n=3):
        fig, axs = plt.subplots(n, n)
        fig.tight_layout()
        for i in range(n):
            for j in range(n):
                rand_idx = random.randint(0, len(self)-1)
                img, label = self.__getitem__(rand_idx)
                axs[i, j].imshow(self.tensorToImage(self.denormalize(img)), cmap='gray')
                if not self.for_rotation_classification:
                    axs[i, j].set_title('Label: {0} (#{1})'.format(label.item(), self.label_to_class[label.item()]))
                else:
                    axs[i, j].set_title('Label: {0} ({1} Degrees)'.format(label.item(), label.item()*45))
                axs[i, j].axis('off')
