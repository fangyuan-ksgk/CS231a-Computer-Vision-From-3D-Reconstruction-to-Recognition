import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def train(dataset, model, batch_size, epochs):
    """
    A function that trains the given model on the provided dataset.
    1. The momentum method in SGD uses running (exponential) average to denoise against the random noise
       and provide better estimate of the gradient (smaller variance estimator somehow, I am not sure if
       it always work)
    """
    # TODO initialize a DataLoader on the dataset with the appropriate batch
    # size and shuffling enabled.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # TODO initialize this to be a Cross Entropy Classification loss.
    criterion = nn.CrossEntropyLoss()

    # TODO initialize this to be an Stochastic gradient descent optimizer with
    # learning rate set to 0.001 and momentum set to 0.9.
    """
    Are we training on both the embedding and the classifynet or just the classify net?
    """
    params = []
    for param in model.parameters:
        params.append({'params':param})
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)

    losses = []
    accuracies = []
    for e in range(epochs):
        running_loss = 0
        correct_count = 0
        for images, labels in tqdm.tqdm(data_loader,'Training for one epoch'):
            # Training pass
            optimizer.zero_grad()
            #images_cuda = images.cuda()
            #labels_cuda = labels.cuda()

            # Call the model's classify method with images_cuda.
            output = model.classify(images)

            # Call the criterion with the ouput and labels_cuda.
            loss = criterion(output, labels)

            # Calculate accuracy, return the index of maximum values on each batch
            _, predictions = torch.max(output, 1)

            # TODO calculate the number of correctly classified inputs.
            # num_correct = sum(predictions==labels).item()
            num_correct = sum(predictions==labels)

            correct_count += num_correct

            if e > 0: # Don't optimize at epoch 0, to see how model starts
                # Here we use the loss to back propagate errors, and then do an
                # optimization step.

                # Get model weight updates by backpropagating the loss with backward()
                loss.backward()

                # Apply the weight updates with the optimizer with step()
                optimizer.step()

            running_loss += loss.item()

        # Calculate the average loss in this epoch.
        epoch_loss = running_loss / len(data_loader)

        # Calculate the float accuracy of this epoch.
        accuracy = correct_count / len(data_loader)

        losses.append(epoch_loss)
        accuracies.append(accuracy)
        print("Epoch %d - Training loss: %.3f , Training Accuracy: %.3f\n"%(e, 
                                                                          epoch_loss, 
                                                                          accuracy))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses)
    ax2.plot(accuracies)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    plt.show()
