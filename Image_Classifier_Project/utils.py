# -*- coding: utf-8 -*-

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import math
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
from PIL import Image
from torch.autograd import Variable

parser = argparse.ArgumentParser(description = 'Utility Functions')

#load data
def get_data(path):
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    #transform the data
    data_transforms = {
        "training": transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),

        "validation": transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),

        "test": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
                                                         }

    #Load data with ImageFolder
    image_datasets = {
        "training": datasets.ImageFolder(train_dir, transform=data_transforms["training"]),
        "validation": datasets.ImageFolder(valid_dir, transform=data_transforms["validation"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"])
        }

    #Load data with DataLoader
    dataloaders = {
        "training": torch.utils.data.DataLoader(image_datasets["training"], batch_size=64, shuffle=True),
        "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=32),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=20)
        }

    return data_transforms, image_datasets, dataloaders

#Select a pretrained model
def get_pretrained_model(name):
    pretrained_mdels = set(['vgg16', 'densenet121', 'densenet169', 'densenet161', 'densenet201'])

    return getattr(models, name)

#Define new feed forward network function using ReLU activation and dropout
def train_model(selected_model, hidden_units, lr):

    #selected_model = 'densenet121'
    #init_lr=0.001
    #epochs = 10

    #print(model)

    #Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    #Define features for chosen pretrained model
    if 'vgg' in selected_model:
        feature_input = 25088
    elif 'densenet' in selected_model:
        feature_input = 1024

    classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout()),
            ('fc1', nn.Linear(feature_input, 512)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
            ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return criterion, optimizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#define function to determine loss and accuracy on the validation data
def validation(model, validationloader, criterion):
    valid_loss = 0
    accuracy = 0

    for inputs, labels in validationloader:
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

#Save checkpoint
def save_checkpoint(path='checkpoint.pth',structure ='densenet121', hidden_layer1=120,dropout=0.5,lr=0.001,epochs=12):
    # TODO: Save the checkpoint
    model.class_to_idx = image_datasets['training'].class_to_idx

    filename = selected_model + '.mdl'
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(state, filename)

#Load Checkpoint
def load_checkpoint(im_path):
    state = torch.load(im_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

    class_to_idx = state['class_to_idx']
    class_to_idx = {str(x):str(y) for x,y in class_to_idx.items()}
    idx_to_class = {y:x for x,y in class_to_idx.items()}

    return class_to_idx, idx_to_class

#Process the image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    width, height = im.size
    ratio = height / width
    short_side = 256

    if height > width:
        width = 256
        newsize = (width, height)
    else:
        height = 256
        newsize = (width, height)

    im = im.resize(newsize)

    #use resize method
    new_width, new_height = newsize

    left = (new_width - 224)/2
    top = (new_height - 224)/2
    right = (new_width + 224)/2
    bottom = (new_height + 224)/2

    cropped = im.crop((left, top, right, bottom))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = np.array(cropped) / 255
    image = (image - mean) / std

    #Reorder color channel dimensions
    image = image.transpose((2, 0, 1))

    return  torch.from_numpy(image)


def predict(image_path, model, topk, process_device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    image = process_image(image_path)
    image = torch.unsqueeze(image, 0)
    image = image.to(device).float()
    result = model(Variable(image))
    result = F.softmax(result, dim=1)
    result = result.process_device()

    probs, classes = result.topk(topk)
    probs, classes = probs.detach().numpy()[0], classes.detach().numpy()[0]

    return probs, classes
