import torch
is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False
import torch.utils.data
import random
import itertools


def inf_shuffle(xs):
    while xs:
        random.shuffle(xs)
        for x in xs:
            yield x

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)

class BalancedBatchSampler2(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset):
        from collections import defaultdict
        if hasattr(dataset, 'dataset'):
            transform = dataset.dataset.transform
            dataset.dataset.transform = None # trick to avoid useless computations
            indices = defaultdict(list)
            for subset_index, full_data_index in enumerate(dataset.indices):
                _, label = dataset.dataset[full_data_index]
                indices[label].append(subset_index) 
            dataset.dataset.transform = transform
        else:
            transform = dataset.transform
            dataset.transform = None  # trick to avoid useless computations
            indices = defaultdict(list)
            for i in range(0, len(dataset)):
                _, label = dataset[i]
                indices[label].append(i)
            dataset.transform = transform     

        self.indices = list(indices.values())
        self.n = max(len(ids) for ids in self.indices) * len(self.indices)


    def __iter__(self):
        m = 0
        for xs in zip(*(inf_shuffle(xs) for xs in self.indices)):
            for i in xs:  # yield one index of each label
                yield i
                m += 1
                if m >= self.n:
                    return

    def __len__(self):
        return self.n

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


def accuracy(net, loader,device="cpu"):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            predicted = net(images)
            #print(predicted.squeeze())
            predicted = torch.sign(predicted)
            #print(predicted.squeeze())
            #print(labels.squeeze())
            total += labels.size(0)
            correct += (predicted.squeeze() == labels.squeeze()).long().sum().item()
    return correct/total