import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import Image
import pandas as pd
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, transform=None):

        self.data_flle = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_flle)

    def __getitem__(self, idx):
        
        try:
            print(idx)
            images = Image.open(self.data_flle.iloc[idx, 0])
            labels = self.data_flle.iloc[idx, 1]

            if self.transform:
                images = self.transform(images)
            
            return images, labels            
        except:
            print('=============>', idx)
            pass

def get_all_loaders():

    csv_file = '/data/ronghui_data/data_test1.csv'
    IMAGE_SIZE = 100
    BATCH_SIZE = 128

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # 缩放到 100 * 100 大小
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化
    ])

    train_data = Dataset('train_datas.csv', transform=transform)
    # valid_data = Dataset('valid_datas.csv', transform=transform)
    test_data  = Dataset('test_datas.csv', transform=transform)

    train_loader = DataLoader(dataset=train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

    # valid_loader = DataLoader(dataset=valid_data,
    #                         batch_size=BATCH_SIZE,
    #                         shuffle=True)

    test_loader = DataLoader(dataset=test_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

    return train_loader, test_loader
