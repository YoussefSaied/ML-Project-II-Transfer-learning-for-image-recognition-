# %% Global parameters
#Our variables:
YoussefPathModel= '/home/youssef/EPFL/MA1/Machine learning/MLProject2/ML2/youssefserver.modeldict'
Youssefdatapath = '/home/youssef/EPFL/MA1/Machine learning/MLProject2/Data'

#Global variables:
use_saved_model =1
save_trained_model=0
train_or_not =0
epochs =1
PathModel= YoussefPathModel
datapath = Youssefdatapath
proportion_traindata = 0.8 # the proportion of the full dataset used for training



# %% Import Dataset and create trainloader 
import datasetY as dataset
import torch
import importlib
from sampler import BalancedBatchSampler
#from sampler import BalancedBatchSampler2
import itertools
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#importlib.reload(module)


full_dataset = dataset.GG2(datapath)

train_size = int(proportion_traindata * len(full_dataset))
test_size = len(full_dataset) - train_size

# To split the full_dataset
def random_splitY(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths)).tolist()
    return indices, [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in
     zip(itertools.accumulate(lengths), lengths)]

indices, sets = random_splitY(full_dataset, [train_size, test_size])
[trainset, testset]=sets
print(len(trainset))

# Dataloaders
batch_sizev=8 # 32>*>8
test_batch_size = 300

trainset_labels = full_dataset.get_labels()[indices[:train_size]] 
testset_labels= full_dataset.get_labels()[indices[train_size:]] 

samplerv= BalancedBatchSampler(trainset,trainset_labels)
samplertest = BalancedBatchSampler(testset,testset_labels)

trainloader = torch.utils.data.DataLoader(trainset,sampler = samplerv , batch_size= batch_sizev, num_workers=2)
testloader = torch.utils.data.DataLoader(testset,sampler = samplertest , batch_size= test_batch_size, num_workers=2)

# %% Import Neural network

net = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0',
 pretrained=True)

# Change First and Last Layer
net.conv_stem = torch.nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2),
 padding=(1, 1), bias=False)
net.classifier = torch.nn.Linear(1280, 1)
net.to(device)

if not torch.cuda.is_available() : #ie if NOT on the server
    print(net)


# %% Train Neural network

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

momentumv=0.90
lrv=0.01


# To calculate accuracy]
def test_accuracy(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            predicted = net(images)
            #_, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

def train_accuracy(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            predicted = net(images)
            #_, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

criterion = nn.SoftMarginLoss()
optimizer = optim.SGD(net.parameters(), lr=lrv, momentum=momentumv)

net.train()

#Option to use a saved model parameters
if use_saved_model:
    import os
    if os.path.isfile(PathModel):
        if os.stat(PathModel).st_size > 0:
            net.load_state_dict(torch.load(PathModel,map_location=torch.device('cpu')))
            print("Loading model...")
        else: 
            print("Empty file...")

#Training starts
if train_or_not:
    print("Starting training...")
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        test_accuracy = 0.0 
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = torch.unsqueeze(labels, dim =1)
            labels = labels.float()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 5 mini-batches
                print('[%5d, %5d] loss: %.6f, test accuracy: %.3f ' %
                        (epoch+1, i + 1, running_loss/2000 , test_accuracy))
                running_loss = 0.0

    print('Finished Training')
    if save_trained_model:
        if os.path.exists(PathModel):  # checking if there is a file with this name
            os.remove(PathModel)  # deleting the file
        torch.save(net.state_dict(), PathModel)
        print("Saving model...")

if torch.cuda.is_available() : #ie if on the server
    import sys
    sys.exit()

# %% Metrics

# Testing mode for net
net.eval()

#print("Train accuracy: %5d"%train_accuracy(net))

from sklearn import metrics

valdation1_size= test_batch_size

#testset_labels_partial= testset_labels[:valdation1_size] 
testset_partial= iter(testloader).next()
testset_partial_I , testset_partial_labels = testset_partial[0], testset_partial[1] 
# predictions_and_labels = [[net(testset_partial[i][0][None]),testset_partial[i][1]]  in range(valdation1_size)]
predictions = [net(image[None]).item() for image in testset_partial_I ]

# Soft= torch.nn.Softmax()
# predictionsSoft= Soft(predictions,-1*predictions)

fpr, tpr, thresholds = metrics.roc_curve(testset_partial_labels, predictions)

# importing the required module 
import matplotlib.pyplot as plt 
  
# x axis values 
x = fpr
# corresponding y axis values 
y = tpr 
  

# plotting the points  

plt.plot(x, y,marker='x') 
plt.plot(x, x,marker='x')
  
# naming the x axis 
plt.xlabel('False Positive Rate') 
# naming the y axis 
plt.ylabel('True Positive Rate') 
  
# giving a title to my graph 
plt.title('Reciever operating characteristic example') 
  
# function to show the plot 
plt.show()
# %%
# Commands for server: 
# grun *.py
# -t tmux a
# nvidia smi

# %% Optimisation of hyperparameters

import numpy as np

listOfBatchSizes = [8,16,24,32]
listOflr = np.logspace(-6,-1, num=10)

for i,lr, batchSize in enumerate( zip(listOflr,listOfBatchSizes)):
    print('starting test number %5d'%i)
    #pseudo: initialise net? train, save results and net, finaly return best
    

