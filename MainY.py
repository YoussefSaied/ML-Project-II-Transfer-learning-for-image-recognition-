# %% Global parameters
#Our variables:
YoussefPathModel= '/home/youssef/EPFL/MA1/Machine learning/MLProject2/ML2/youssefServer4.modeldict'
Youssefdatapath = '/home/youssef/EPFL/MA1/Machine learning/MLProject2/Data'
YoussefServerPathModel= '/home/saied/ML/ML2/youssefServer5.modeldict'
YoussefServerdatapath = '/data/mgeiger/gg2/data'
YoussefServerPicklingPath = '/home/saied/ML/ML2/'
YoussefPicklingPath = '/home/youssef/EPFL/MA1/Machine learning/MLProject2/ML2/Predictions/'

#Global variables:
use_saved_model =0
save_trained_model=1
train_or_not =1
epochs =2
OnServer =0
if OnServer:
    PicklingPath=YoussefServerPicklingPath
    PathModel= YoussefServerPathModel
    datapath = YoussefServerdatapath
else:
    PicklingPath=YoussefPicklingPath
    PathModel= YoussefPathModel
    datapath = Youssefdatapath
proportion_traindata = 0.8 # the proportion of the full dataset used for training
printevery = 100

# %% Import Dataset and create trainloader 
import datasetY as dataset
import torch
import importlib
from sampler import *
import itertools
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#importlib.reload(module)


full_dataset = dataset.GG2(datapath)

train_size = int(proportion_traindata * len(full_dataset))
test_size = len(full_dataset) - train_size



# To split the full_dataset

indices, sets = random_splitY(full_dataset, [train_size, test_size])
[trainset, testset]=sets
print(len(trainset))

# Dataloaders
batch_sizev=128 # 32>*>8
test_batch_size = 1

# trainset_labels = full_dataset.get_labels()[indices[:train_size]] 
# testset_labels= full_dataset.get_labels()[indices[train_size:]] 

samplerv= BalancedBatchSampler2(trainset)
samplertest = BalancedBatchSampler2(testset)

trainloader = torch.utils.data.DataLoader(trainset , shuffle=True, batch_size= batch_sizev)
testloader = torch.utils.data.DataLoader(testset , shuffle =True, batch_size= test_batch_size)
ROCloader = torch.utils.data.DataLoader(testset,batch_size=1)
# %% Import Neural network

net = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_mobilenetv3_small_minimal_100',
 pretrained=False)

# Change First and Last Layer
net.conv_stem = torch.nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding= (1,1) , bias=False)
net.classifier = torch.nn.Linear(1024, 1)

if torch.cuda.device_count() > 1 and False:
    import torch.nn as nn
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
net.to(device)
if not torch.cuda.is_available() : #ie if NOT on the server
    print(net)

# Replace all batch normalization layers by Instance
def convert_batch_to_instance(model):
    import torch.nn as nn
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features= child.num_features
            setattr(model, child_name, nn.InstanceNorm2d(num_features=num_features))
        else:
            convert_batch_to_instance(child)

#convert_batch_to_instance(net)
if not torch.cuda.is_available() : #ie if NOT on the server
    print(net)
# %% Train Neural network

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

momentumv=0.90
lrv=0.01


# To calculate accuracy
from sampler import accuracy

def train_accuracy(net):
    return accuracy(net, loader= trainloader,device=device)

def test_accuracy(net):
    return accuracy(net, loader= testloader,device=device)

def ROC_accuracy(net):
    return accuracy(net, loader= ROCloader,device=device)


#Option to use a saved model parameters
if use_saved_model:
    import os
    if os.path.isfile(PathModel):
        if os.stat(PathModel).st_size > 0:
            net.load_state_dict(torch.load(PathModel,map_location=torch.device(device   )))
            #convert_batch_to_instance(net)
            print("Loading model...")
        else: 
            print("Empty file...")
        print("Using saved model...")

#Training starts

criterion = nn.SoftMarginLoss()
optimizer = optim.SGD(net.parameters(), lr=lrv, momentum=momentumv)

net.train()

if train_or_not:
    print("Starting training...")
    train_accuracy_list = np.array([0])
    test_accuracy_list = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        print("Starting epoch %d"%(epoch+1))
        running_loss = 0.0
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
            if i % printevery == printevery-1:    # print every n mini-batches
                print('[%5d, %5d] loss: %.6f ' %
                        (epoch+1, i + 1, running_loss/printevery) )
                running_loss = 0.0

        # fixing batch normalization statistics
        #print("Fixing batch normalization statistics...")

        
        # save predictions and labels for ROC curve calculation
        print("Saving predictions and calculating accuracies...")
        if False:
            #net.eval()
            predictions = []
            labels = []
            for k, testset_partial in enumerate(testloader):
                with torch.no_grad():
                    if k <100000:
                        testset_partial_I , testset_partial_labels = testset_partial[0].to(device), testset_partial[1].to(device)
                        predictions += [net(image[None]).item() for image in testset_partial_I ]
                        labels += testset_partial_labels.tolist()
                    else:
                        break

            file_name= PicklingPath+"PredictionsAndLabelsTrial1Epoch"+str(epoch)
            import os
            if os.path.exists(file_name):  # checking if there is a file with this name
                os.remove(file_name)  # deleting the file
            import pickle
            with open(file_name, 'wb') as pickle_file:
                pickle.dump([predictions,labels],pickle_file)
                pickle_file.close()
            print("Pickling done...")

        # calculate and save accuracy and stop if test accuracy increases 
        #net.eval()
        test_accuracyv =  ROC_accuracy(net)
        print("Test accuracy: %5f"%test_accuracyv)
        if test_accuracyv< np.min(train_accuracy_list) and False:
            break
        train_accuracy_list = np.concatenate((train_accuracy_list, np.array([test_accuracyv])))
        net.train()
    import os
    print("Pickling accuracies...")
    file_name= PicklingPath+"accuracies"
    if os.path.exists(file_name):  # checking if there is a file with this name
        os.remove(file_name)  # deleting the file
    import pickle
    with open(file_name, 'wb') as pickle_file:
        pickle.dump(train_accuracy_list,pickle_file)
        pickle_file.close()
    
    print('Finished Training')
    if save_trained_model:
        import os
        if os.path.exists(PathModel):  # checking if there is a file with this name
            os.remove(PathModel)  # deleting the file
        torch.save(net.state_dict(), PathModel)
        print("Saving model...")

if torch.cuda.is_available() : #ie if on the server
    #net.eval()
    test_accuracyv = test_accuracy(net)
    print("Test accuracy: %5f"%test_accuracyv)
    train_accuracyv =  ROC_accuracy(net)
    print("Train accuracy: %5f"%train_accuracyv)
    import sys
    sys.exit()

# %% Metrics

# Testing mode for net
#net.eval()
if True:
    test_accuracyv = test_accuracy(net)
    print("Test accuracy: %5f"%test_accuracyv)

from sklearn import metrics

# ROC curve calculation

# testset_partial= iter(testloader).next()
# testset_partial_I , testset_partial_labels = testset_partial[0], testset_partial[1] 
# predictions = [net(image[None]).item() for image in testset_partial_I ]

predictions = []
labels = []
with torch.no_grad():
    if True:
        for k, testset_partial in enumerate(testloader):
            if k <100000:
                testset_partial_I , testset_partial_labels = testset_partial[0].to(device), testset_partial[1].to(device)
                predictions += [p.item() for p in net(testset_partial_I) ]
                labels += testset_partial_labels.tolist()
            if k%100==0:
                print(k)

        fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)

        # importing the required module 
        import matplotlib.pyplot as plt 
        
        # x axis and y axis values 
        x ,y = fpr, tpr

        # plotting the points  
        plt.plot(x, y,marker='x') 
        plt.plot(x, x,marker='x')
        
        # naming the x axis 
        plt.xlabel('False Positive Rate') 
        # naming the y axis 
        plt.ylabel('True Positive Rate') 
        
        # giving a title to my graph 
        plt.title('Reciever operating characteristic curve') 
        
        # function to show the plot 
        plt.show()

        # plot all ROC curves from pickle 
        print("Pickling accuracies...")

    for epoch in range(epochs): 
        file_name= PicklingPath+"PredictionsAndLabelsTrial1Epoch"+str(epoch)
        import pickle
        with open(file_name, 'rb') as pickle_file:
            [predictions,labels] = pickle.load(pickle_file)
            pickle_file.close()
        
        fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)

        # importing the required module 
        import matplotlib.pyplot as plt 
        # x axis and y axis values 
        x ,y = fpr, tpr

        # plotting the points  
        plt.plot(x, y,marker='x') 
        plt.plot(x, x,marker='x')
        
        # naming the x axis 
        plt.xlabel('False Positive Rate') 
        # naming the y axis 
        plt.ylabel('True Positive Rate') 
        
        # giving a title to my graph 
        plt.title('Reciever operating characteristic curve epoch '+str(epoch)) 
        
        # function to show the plot 
        plt.show()


# %% Optimisation of hyperparameters

import numpy as np

listOfBatchSizes = [8,16,24,32]
listOflr = np.logspace(-6,-1, num=10)

for i,lr, batchSize in enumerate( zip(listOflr,listOfBatchSizes)):
    print('starting test number %5d'%i)
    #pseudo: initialise net? train, save results and net, finaly return best
    

