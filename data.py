from typing import Any
from skimage import io
from sklearn import model_selection

import numpy as np
import pandas as pd
import torch
import os

import ignite.distributed as idist
import torchvision
import torchvision.transforms as T


class UltraDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, digit_sum, albumentations=None):
     
        self.image_path = image_path
        self.digit_sum = digit_sum
        self.albumentation = albumentations

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):

        image = io.imread(self.image_path[item])
        digit_sum = self.digit_sum[item]
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return torch.tensor(image, dtype=torch.float), torch.tensor(digit_sum, dtype=torch.float)


def setup_data(config: Any):
    
    df = pd.read_csv('../input/ultra-mnist/train.csv')

    df_train, df_valid = model_selection.train_test_split(df, test_size= 0.3) 

    train_images = df_train.Id.values.tolist()
    train_images = [os.path.join(config.data_path,'train',i + '.jpeg') for i in train_images]
    valid_images = df_valid.Id.values.tolist()
    valid_images = [os.path.join(config.data_path,'train',i + '.jpeg') for i in valid_images]

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset_train = UltraDataset(train_images, df_train.digit_sum.values)
    dataset_eval = UltraDataset(valid_images, df_valid.digit_sum.values)

    dataloader_train = idist.auto_dataloader(
        dataset_train,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    dataloader_eval = idist.auto_dataloader(
        dataset_eval,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return dataloader_train, dataloader_eval
