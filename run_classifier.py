
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import numpy as np

import torch.nn.functional as F

import torch.optim as optim


import argparse
import os

from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch MNIST classifier')

parser.add_argument('-batch_size', type=int, default=100, metavar='N', help='input batch size for training (default: 100)')
parser.add_argument('-num_train_cls',type=int,default=1000,help='number of iterations for training (default:1000)')
parser.add_argument('-eval_size',type=int,default=500, help='Size of the evaluattion (test) dataset (default:100)')

parser.add_argument('-eval_every_n_iter',type=int,default= 100,help='Show Loss values at every N iteration (default:500)')
parser.add_argument('-num_data',type=int,default=200,help='number oflabelled data(default:400)')


parser.add_argument('-dataset',type=str,default='MNIST',help='the name of the dataset to use (MNIST CIFAR10, SVHN) (default=MNIST)')

parser.add_argument('-model_dir',type=str,default='model',help='the directory to store the model)')


args = parser.parse_args()


def get_dataset(is_train, is_labeled, dataset_name, num_data):
    dataset = MNIST(root='/media/iec/Seagate Expansion Drive/mnist_data/data/',
                    train=is_train,
                    transform=transforms.ToTensor(),
                    download=True)
    if is_labeled:
        # num_data = args.num_data
        print("num_data:", num_data)
        
        generator = np.random.default_rng(1001)
        labels, indexs = [], []

        for i in range(len(dataset)):
            _, lab = dataset.__getitem__(i)
            labels.append(lab)
            indexs.append(i)
        labels = np.array(labels)
        indexs = np.array(indexs)
        num_classes = np.max(labels) + 1

        assert num_data % num_classes == 0

        final_indices = []
        for i in range(num_classes):
            tind = list(indexs[labels == i])
            generator.shuffle(tind)
            final_indices.extend(tind[: (num_data // num_classes)])

        dataset = torch.utils.data.Subset(dataset, final_indices)

        assert len(dataset) == num_data
    
    
    return dataset
    

def get_data_iter(batch_size, num_data, dataset_name, is_train=True, infinity=True, is_labeled=False):

    # if batch_size is None:
    #     batch_size = args.batch_size
    
    return inf_data_gen(batch_size, num_data, dataset_name, is_train, infinity, is_labeled)


def inf_data_gen(batch_size, num_data, dataset_name, is_train=True, infinity=True, is_labeled=False):

    # num_data = 0

    loader = DataLoader(
        get_dataset(is_train, is_labeled, dataset_name, num_data),
        batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    if infinity is True:
        while True:
            for img, labels in loader:
                yield img, labels
    else:
        for img, labels in loader:
            yield img, labels



class Classifier_MNIST(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        #output 24 24 6
        self.pool = nn.MaxPool2d(2, 2)
        # kernel= 2 stride=2
        #output 12, 12, 6 
        # output 6, 6 ,16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv2 = nn.Conv2d(24, 16, 5)
        # output 8, 8, 16

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
     

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x



itr_labeled = get_data_iter(batch_size=args.batch_size, num_data=args.num_data, dataset_name=args.dataset, is_labeled=True)


testset = MNIST(root='/media/iec/Seagate Expansion Drive/mnist_data/data/',
                    train=False,
                    transform=transforms.ToTensor(),
                    download=True)

                    
test_loader = DataLoader(dataset=testset,
                         batch_size=args.eval_size,
                         shuffle=False)

classifier = Classifier_MNIST(args)

loss_f_xe = nn.CrossEntropyLoss()
loss_f_bce = nn.BCELoss()


cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('cuda available:', cuda)

if cuda:
    classifer = classifier.cuda()
   
    loss_f_xe = loss_f_xe.cuda()
    loss_f_bce = loss_f_bce.cuda()



optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)


for step in tqdm(range(args.num_train_cls)):
    image_labeled, label = itr_labeled.__next__()

    if cuda:
        image_labeled = image_labeled.cuda()
        label = label.cuda()
    # classifier.zero_grad()

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    logits_c_labeled = classifier(image_labeled)
    if cuda:
        logits_c_labeled = logits_c_labeled.cuda()

    c_crossent_loss = loss_f_xe(logits_c_labeled,label)

    c_crossent_loss.backward()
    optimizer.step()

    if (step + 1) % args.eval_every_n_iter == 0 or step == 0:
        
    
        print(" Step: [%d/%d], CrossEnt_labeled: %.4f" %
                (step + 1, args.num_train_cls, c_crossent_loss.item()))
    

    if (step + 1 ) % args.eval_every_n_iter == 0 or step == 0:
        
        batch_size = args.eval_size #for testing
        test_iter = iter(test_loader)
        test_data = next(test_iter)

        logits_c_test = classifier(Variable(test_data[0]).cuda())

        _, label_predicted_test = torch.max(logits_c_test, 1)
        # onehot_label_predicted_test = nn.functional.one_hot(label_predicted_test,num_classes=10).float()

        c_crossentropy= loss_f_xe(logits_c_test, test_data[1].cuda())

        
        test_accuracy = torch.sum(label_predicted_test == test_data[1].cuda())
    
        final_accuracy = test_accuracy.cpu().numpy() / batch_size


        print("---Eval on test dataset")
        print("---------Prediction Loss: %.4f, Accuracy: %.4f" % (c_crossentropy,final_accuracy))
        print()


print('Finished Training')

print('save the model')


PATH = '/media/iec/Seagate Expansion Drive/mnist_classifier/' + args.model_dir + '/mnist_net.pth'

os.makedirs('/media/iec/Seagate Expansion Drive/mnist_classifier/' + args.model_dir)

torch.save(classifier.state_dict(), PATH)