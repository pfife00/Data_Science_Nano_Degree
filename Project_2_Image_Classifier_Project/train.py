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

parser = argparse.ArgumentParser(description = 'Train the model')

parser.add_argument('--data_dir', nargs="*", action="store", default="./flowers/")
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument(--'learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=10)
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)

path = parser.data_dir
selected_model = parser.arch
save_path = parser.save_dir
lr = parser.learning_rate
hid_units = parser.hidden_units
num_epochs = parser.epochs
process_device = parser.gpu
drop = parser.dropout

args = parser.parse_args()

#Load the data
data_transforms, image_datasets, dataloaders = get_data(data_dir)

#Choose model
selected_model = get_pretrained_model(selected_model)(pretrained=True)

#train new feedforward network and get returned criterion and optimizer
criterion, optimizer = train_model(selected_model, hid_units, lr)

#Determine whether training on CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

utils.save_checkpoint(path,selected_model,hid_units,dropout,lr)

print("Training Complete")
