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
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_features = self.densenet121.classifier.in_features # 1024
        self.densenet121.classifier = nn.Linear(num_features, num_labels)#*3)
        #self.num_labels = num_labels
        #self.num_classes = 3 # [p0, p1, p2] for each label

    def forward(self, x):
        x = self.densenet121(x)
        # we don't include sigmoid layer here
        return x#.reshape([len(x), self.num_labels, self.num_classes])

if __name__ == "__main__":
    x = torch.FloatTensor([0,1,2,5,2,3,0,1,2,3,4,5,0,1,2,3,4,5])
    print(x.reshape([6,3]))

    x = x.reshape([2,3,3])
    print(x.detach())
    print(x.detach().max(0))
    print(x.detach().max(1))
    print(x.detach().max(2))
    print(x.detach().max(-1))

    x1 = [1,2,3]
    x2 = [4,3,5]
    data = {'Train Loss':x1,'Valid Loss':x2}
    df = pd.DataFrame(data = data)
    df.index.name = 'epoch'
    print(df)

