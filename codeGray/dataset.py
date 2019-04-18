import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class CheXpertDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        df = pd.read_csv(image_list_file)
        df = df.fillna(0)
        self.transform = transform
        self.imagePaths = []
        self.labels = []
        for i, row in df.iterrows():
            self.imagePaths.append( os.path.join(data_dir,row['Path']) )
            label = list(row[5:].values % 3) # use % 3 to replace -1 with 2, uncertain
            self.labels.append(label)


    def __getitem__(self, index):
        image = Image.open(self.imagePaths[index])#.convert('RGB') # pre-trained models on ImageNet are 'RGB'
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.LongTensor(self.labels[index])

    def __len__(self):
        return len(self.imagePaths)
