import torch
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

## Define a transform to read the data in as a tensor
data_transform = transforms.Compose([
                 transforms.Resize(28),
                 transforms.CenterCrop(28),
                 transforms.ToTensor()])

# choose the training and test datasets
train_data = FashionMNIST(root='fashionmnist/', train=True,
                                   download=True, transform=data_transform)

test_data = FashionMNIST(root='fashionmnist/', train=False,
                                  download=True, transform=data_transform)


# Print out some stats about the training and test data
print('Train data, number of images: ', len(train_data))
print('Test data, number of images: ', len(test_data))

batch_size = 50

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# specify the image classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

import torch.nn as nn
import torch.nn.functional as F


class ClfNet(nn.Module):

    def __init__(self):
        super(ClfNet, self).__init__()

        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 16, 75)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(75, 10)

    ## TODO: define the feedforward behavior
    def forward(self, x):
        # one activated conv layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        # final output
        return x


# instantiate and print your Net
net = ClfNet()
print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

from torch.autograd import Variable


def train(n_epochs):
    training_loss = []

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            inputs, labels = data

            # wrap them in a torch Variable
            # inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # forward pass to get outputs
            outputs = net(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

            # backward pass to calculate the parameter gradients
            loss.backward()

            # update the parameters
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()

            if batch_i % 1000 == 999:  # print every 1000 mini-batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 1000))
                training_loss.append(running_loss / 1000)
                running_loss = 0.0

    print('Finished Training')
    return training_loss
#
#
# n_epochs = 70
#
# # call train
# training_loss = train(n_epochs)
#
# import matplotlib.pyplot as plt
# plt.plot(training_loss)
# plt.xlabel('k batches')
# plt.ylabel('average loss per batch')
# plt.title('evolution of average training loss per batch')
# plt.show()
#
# # Test accuracy
# import numpy as np
#
# # initialize tensor and lists to monitor test loss and accuracy
# test_loss = torch.zeros(1)
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
#
# # set the module to evaluation mode
# net.eval()
#
# for batch_i, data in enumerate(test_loader):
#
#     # get the input images and their corresponding labels
#     inputs, labels = data
#
#     with torch.no_grad():
#
#         # forward pass to get outputs
#         outputs = net(inputs)
#
#         # calculate the loss
#         loss = criterion(outputs, labels)
#
#         # update average test loss
#         test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))
#
#         # get the predicted class from the maximum value in the output-list of class scores
#         _, predicted = torch.max(outputs.data, 1)
#
#         # compare predictions to true label
#         correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))
#
#         # calculate test accuracy for *each* object class
#         # we get the scalar value of correct items for a class, by calling `correct[i].item()`
#         for i in range(batch_size):
#             label = labels.data[i]
#             class_correct[label] += correct[i].item()
#             class_total[label] += 1
#
# print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))
#
# for i in range(10):
#     if class_total[i] > 0:
#         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
#             classes[i], 100 * class_correct[i] / class_total[i],
#             np.sum(class_correct[i]), np.sum(class_total[i])))
#     else:
#         print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
#
# print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
#     100. * np.sum(class_correct) / np.sum(class_total),
#     np.sum(class_correct), np.sum(class_total)))
#
# # Save model
# model_name = 'FMnist_model_1.pt'
# torch.save(net.state_dict(), model_name)