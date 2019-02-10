i%matplotlib inline
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
import utils

parser = argparse.ArgumentParser(description = 'Predict the model')

parser.add_argument('im', default='paind-project/flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
parser.add_argument('checkpoint', default='/home/workspace/aipnd-project-master/checkpoint.pth', nargs='*', action="store",type = str)
parser.add_argument('--top_k3', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

im_path = parser.im
topk = parser.top_k3
process_device = parser.gpu
im_input = parser.im
saved_checkpoint = parser.checkpoint

class_to_idx, idx_to_class = utils.load_checkpoint(saved_checkpoint)

image = utils.process_image(im_path)

def get_class_name(idx):
    return cat_to_name[idx_to_class[str(idx)]]

probs, classes = utils.predict(im_path, model, topk, process_device)

class_names = [get_class_name(x) for x in classes]

print('Highest prediction: ' , get_class_name(classes[0]))
