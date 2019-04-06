import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from utils import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix
from dataset import CheXpertDataSet
from models import DenseNet121
from sklearn.metrics import roc_auc_score

cudnn.benchmark = True

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

PATH_DIR = '/ezdh/data'
PATH_TEST = '/ezdh/data/CheXpert-v1.0-small/data_test.csv'
PATH_OUTPUT = "../output/"
os.makedirs(PATH_OUTPUT, exist_ok=True)


NUM_EPOCHS = 6
BATCH_SIZE = 32 # 32 is the max for our memory limitation
USE_CUDA = True  # Set 'True' if you want to use GPU
NUM_WORKERS = 8
num_labels = 14

# Data loading
print('===> Loading entire datasets')
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

transformseq=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize
                                ])

test_dataset = CheXpertDataSet(data_dir=PATH_DIR, image_list_file=PATH_TEST, transform = transformseq)

print(test_dataset)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print('Data Loaded')

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
criterion.to(device)

# load best model
PATH_MODEL = os.path.join(PATH_OUTPUT, "MyCNN.pth")
best_model = torch.load(PATH_MODEL)
test_loss, test_results = evaluate(best_model, device, test_loader, criterion)

# plot confusion matrix 
class_names = ['Positive', 'Negative', 'Uncertain']
label_names = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
for i, label_name in enumerate(label_names): # i th observation
    plot_confusion_matrix(test_results, class_names, i, label_name)

#best_model_prob = torch.nn.Sequential(best_model, nn.Softmax(dim = -1))

