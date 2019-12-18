# %% Global parameters
#Our variables:
YoussefPathModel= '/home/youssef/EPFL/MA1/Machine learning/MLProject2/ML2/youssefServer4.modeldict' # Path of the weights of the model
Youssefdatapath = '/home/youssef/EPFL/MA1/Machine learning/MLProject2/Data' # Path of data
YoussefServerPathModel= '/home/saied/ML/ML2/youssefServer4.modeldict' # Path of weights of the Model
#Server 5 is init(Batchnorm), not balanced, 128 auc=0.7 after 10 epochs 
#Server 6 is init(Batchnorm), balanced, 128 auc=0.7/0.64 after 2/10 epochs
#Server 7 is init(Batchnorm), balanced, 8 auc=0.74/0.7 after 1/5 epochs
#Server 9 is init(Batchnorm), balanced, 4 auc=0.74 after 1/5 epochs 
#Server 8 is init(Batchnorm), balanced, 128, weightdecay =0.0001 auc =0.64 after 4 epochs 
#Server 10 is SIMPLE is init(Batchnorm), not balanced, 8 auc=0.65/0.7 after 5/15 epochs (redo)
#Server 11 is init(Batchnorm), not balanced, 8 auc= 0.72 after 1 epochs
#Server 12 is Data augmented, init(Batchnorm), balanced, 8,  weightdecay =0.0001 auc=0.825/?? after 10/?? epochs (best) (redo decrease weight decay) (increase lr) (parallelization)
#Server 16 is Data augmented, init(Batchnorm), balanced, 8,  weightdecay =0 auc=??/?? after ??/?? epochs 
#Server 13 is Data augmented, init(Batchnorm), balanced, 128 auc=??/?? after ??/?? epochs (best?)
#Server 14 is Data augmented, SIMPLE, init(Batchnorm), balanced, 128 auc=??/?? after ??/?? epochs
#Server 15 is Data augmented, SIMPLE, init(Batchnorm), balanced, 8 auc=??/?? after ??/?? epochs
#Server 20 is TRANSFERLEARNING, init(Batchnorm), balanced, 128, auc=??/?? after ??/?? epochs
YoussefServerdatapath = '/data/mgeiger/gg2/data' # Path of data
YoussefServerPicklingPath = '/home/saied/ML/ML2/' # Path for pickling 
YoussefPicklingPath = '/home/youssef/EPFL/MA1/Machine learning/MLProject2/ML2/Predictions/' # Path for pickling 
YoussefPathDataset= '/home/youssef/EPFL/MA1/Machine learning/MLProject2/traintestsets.pckl' # Path of training and test dataset
YoussefServerPathDataset= '/home/saied/ML/ML2/traintestsets.pckl' # Path of training and test dataset

#Global variables (booleans):
transfer_learning=0
init_batchnormv =0
use_parallelization=0
simple =0
data_augmentation =1
use_saved_model =1
save_trained_model=1
train_or_not =1
epochs =20
OnServer =1
if OnServer:
    PicklingPath=YoussefServerPicklingPath
    PathModel= YoussefServerPathModel
    PathDataset =YoussefServerPathDataset
    datapath = YoussefServerdatapath
else:
    PicklingPath=YoussefPicklingPath
    PathModel= YoussefPathModel
    PathDataset =YoussefPathDataset
    datapath = Youssefdatapath
proportion_traindata = 0.8 # the proportion of the full dataset used for training
printevery = 1000

print("Server12")

# %% Import Dataset and create trainloader 
import datasetY as dataset
import torch
import importlib
from datasetY import BalancedBatchSampler, BalancedBatchSampler2, random_splitY, accuracy, load_GG2_imagesTransfer, load_GG2_images2
import itertools
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#importlib.reload(module)




# Pickling datasets

if transfer_learning:
    transform=load_GG2_imagesTransfer
else:
    transform=load_GG2_images2

import os
if os.path.isfile(PathDataset):
    if os.stat(PathDataset).st_size > 0:
        import pickle
        with open(PathDataset, 'rb') as pickle_file:
            [full_dataset,trainset,testset] = pickle.load(pickle_file)
        full_dataset.transform=transform
        trainset.transform=transform
        testset.transform=transform
        print("Loading datasets...")
else: 
    full_dataset = dataset.GG2(datapath,data_augmentation=False,transform=transform)

    # To split the full_dataset
    train_size = int(proportion_traindata * len(full_dataset))
    test_size = len(full_dataset) - train_size
    indices, sets = random_splitY(full_dataset, [train_size, test_size])
    [trainset, testset]=sets

    import pickle
    with open(PathDataset, 'wb') as pickle_file:
        pickle.dump([full_dataset,trainset,testset],pickle_file)
    print("Creating and pickling datasets...")

# Data augmentation

if data_augmentation:
    full_dataset.data_augmentation=True
    trainset.data_augmentation=True
    testset.data_augmentation=True

print(len(trainset))

# Dataloaders

batch_sizev=128
test_batch_size = 1

samplerv= BalancedBatchSampler2(trainset)
samplertest = BalancedBatchSampler2(testset)

trainloader = torch.utils.data.DataLoader(trainset, sampler=samplerv, shuffle=False, batch_size= batch_sizev)
testloader = torch.utils.data.DataLoader(testset, sampler=None, shuffle =True, batch_size= test_batch_size)
ROCloader = torch.utils.data.DataLoader(testset,batch_size=1)
# %% Import Neural network

if simple:
    net = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_mobilenetv3_small_minimal_100',
    pretrained=False)

    # Change First and Last Layer
    if not transfer_learning:
        net.conv_stem = torch.nn.Conv2d(4,16,kernel_size=(2,2),bias=False)
    net.classifier = torch.nn.Linear(1024, 1)
else: 
    net = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0',
    pretrained=True)

    # Change First and Last Layer
    if not transfer_learning:
        net.conv_stem = torch.nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    net.classifier = torch.nn.Linear(1280, 1)







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


def init_batchnorm(model): # For initializing the batch normalization layers
    import torch.nn as nn
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features= child.num_features
            setattr(model, child_name, nn.BatchNorm2d(num_features=num_features))
        else:
            convert_batch_to_instance(child)

#convert_batch_to_instance(net)

net.to(device)
if not transfer_learning and init_batchnormv:
        init_batchnorm(net)


#Option to parallelize
print("There are", torch.cuda.device_count(), "GPUs!")
if torch.cuda.device_count() > 1 and use_parallelization:
    import torch.nn as nn
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)
net.to(device)

if not torch.cuda.is_available() : #ie if NOT on the server
    print(net)
# %% Train Neural network

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

momentumv=0.90
lrv=10**-2

print("Learning rate= "+str(lrv))


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

if not transfer_learning:
    optimizer = optim.SGD(net.parameters(), lr=lrv, momentum=momentumv)
else:
    optimizer = optim.SGD(net.classifier.parameters(), lr=lrv, momentum=momentumv)
    for param in net.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = True
    
    
# Decay LR by a factor of 0.1 every 7 epochs
from torch.optim import lr_scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

net.train()

if train_or_not:
    print("Starting training...")
    train_accuracy_list = np.array([0])
    test_accuracy_list = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        exp_lr_scheduler.step()
        print("Starting epoch %d"%(epoch+1))
        print("Learning rate= "+str(lrv))
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
            net.eval()
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
            net.train()
            
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
            if epoch%2 ==0:
                net.eval()
                test_accuracyv =  ROC_accuracy(net)
                print("Test accuracy: %5f"%test_accuracyv)
                if test_accuracyv< np.min(train_accuracy_list) and False:
                    break
                train_accuracy_list = np.concatenate((train_accuracy_list, np.array([test_accuracyv])))
                net.train()


        # AUC for ROC curve
        
        #net.eval()
        net.train()
        from sklearn import metrics
        predictions = []
        labels = []
        with torch.no_grad():
            if True:
                for k, testset_partial in enumerate(testloader):
                    if k <1000:
                        testset_partial_I , testset_partial_labels = testset_partial[0].to(device), testset_partial[1].to(device)
                        predictions += [p.item() for p in net(testset_partial_I) ]
                        labels += testset_partial_labels.tolist()

                auc = metrics.roc_auc_score(labels, predictions)
                print("Test auc: %5f"%auc)
                #train_accuracy_list = np.concatenate((train_accuracy_list, np.array([auc])))
                if False:
                    for k, trainset_partial in enumerate(trainloader):
                        if k <100:
                            trainset_partial_I , trainset_partial_labels = testset_partial[0].to(device), testset_partial[1].to(device)
                            predictions += [p.item() for p in net(testset_partial_I) ]
                            labels += testset_partial_labels.tolist()

                    auc = metrics.roc_auc_score(labels, predictions)
                    print("Train auc: %5f"%auc)
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
    net.train()
    print("Test accuracy: %5f"%test_accuracyv)
    train_accuracyv =  train_accuracy(net)
    print("Train accuracy: %5f"%train_accuracyv)
    import sys
    sys.exit()

# %% Metrics

# Testing mode for net
#net.eval()
if False:
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
        net.train() # remove
        for k, testset_partial in enumerate(testloader):
            if k <10:
                testset_partial_I , testset_partial_labels = testset_partial[0].to(device), testset_partial[1].to(device)
                predictions += [p.item() for p in net(testset_partial_I) ]
                labels += testset_partial_labels.tolist()
            else: break
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

