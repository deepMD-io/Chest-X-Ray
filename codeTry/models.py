import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_labels):
        super(DenseNet121, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7),
            nn.Relu()
            nn.MaxPool2d(kernel_size=2)
            nn.BatchNorm2d(32),
            nn.Dropout(p = 0.15),

            nn.Conv2d(32, 64, kernel_size=5),
            nn.Relu()
            nn.MaxPool2d(kernel_size=2)
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.15),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.Relu()
            nn.MaxPool2d(kernel_size=2)
            nn.BatchNorm2d(128),
            nn.Dropout(p = 0.15),
            
            nn.Conv2d(128, 128, kernel_size=3),
            nn.Relu()
            nn.MaxPool2d(kernel_size=2)
            nn.BatchNorm2d(128),
            nn.Dropout(p = 0.15),

            nn.AdaptiveAvgPool2d(1) # global pooling
        )
          
        self.classifier = nn.Sequential(
            nn.Linear(128, 1000),
            nn.Relu()
            nn.Linear(1000, num_labels),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    pass
