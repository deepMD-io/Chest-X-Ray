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

from utils import train, evaluate#, getprob
from plots import plot_learning_curves, plot_confusion_matrix, plot_roc
from dataset import CheXpertDataSet
from models import DenseNet121
from scipy.special import softmax

cudnn.benchmark = True

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

PATH_DIR = '../data'
PATH_TEST = '../data/CheXpert-v1.0-small/data_test.csv'
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
#test_results = getprob(best_model, device, test_loader)

class_names = ['Negative', 'Positive', 'Uncertain']
label_names = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

#best_model_prob = torch.nn.Sequential(best_model, nn.Softmax(dim = -1))
# convert output to positive probability
def predict_positive(model, device, data_loader):
    model.eval()
    # return a List of probabilities
    #input, target = zip(*data_loader)

    probas = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)
            target = target.detach().to('cpu').numpy()
            targets = np.concatenate((targets, target), axis=0) if len(targets) > 0 else target
            #target = target.to(device)

            output = model(input) # num_batch x 14 x 3
            y_pred = output.detach().to('cpu').numpy()
            y_pred = y_pred[:,:,:2] # drop uncertain
            y_pred = softmax(y_pred, axis = -1)
            y_pred = y_pred[:,:,1] # keep positive only

            probas = np.concatenate((probas, y_pred), axis=0) if len(probas) > 0 else y_pred
    
    return targets, probas

test_targets, test_probs = predict_positive(best_model, device, test_loader)
#print(test_targets)

plot_roc(test_targets, test_probs, label_names)

#best_model_prob = torch.nn.Sequential(best_model, nn.Softmax(dim = -1))

