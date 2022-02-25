import torch
import numpy as np
from torch.utils.data import DataLoader


def test(dataset, model, batch_size):
    # TODO initialize a DataLoader on the dataset with the appropriate batch
    # size and shuffling enabled.
    data_loader = None

    correct_count = 0
    for images, labels in data_loader:
        output = model.classify(images.cuda())

        # Calculate accuracy
        _, predictions = torch.max(output, 1)

        # Calculate the number of correctly classified inputs.
        num_correct = sum(predictions==labels)

        correct_count+=num_correct

    # Calculate the float accuracy for the whole dataset.
    accuracy = correct_count / len(dataset)
    print("Testing Accuracy: %.3f"%(accuracy))
