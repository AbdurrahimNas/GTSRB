"""
Module for data preparing and handling functions. If you can't find what you are 
looking for, it means you have to write the function yourself.

Available Functions:
  prepare_data,

"""

import os
import torch
import torchvision

from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader

def prepare_data(train_root:str,
                 test_root:str,
                 train_transforms:torchvision.transforms,
                 test_transforms:torchvision.transforms,
                 batch_size:int=256,
                 num_workers:int=os.cpu_count(),
                 pin_memory:bool=True):
  """
  Creates train and test dataloaders with given transforms.

  Keyword Arguments:
    :arg train_root: Path of the train data
    :type train_root: path 
    :arg test_root: Path of the test data 
    :type test_root: path
    :arg train_transforms: Transform for train data loader to augment the data.
    :type train_transforms: torchvision.transforms
    :arg test_transforms: Transform for test data loader to augment the data.
    :type test_transforms: torchvision.transforms
    :arg batch_size: Batch size of your choice. Default 256.
    :type bacth_size: int 
    :arg num_workers: Number of cpus to use. Default max cpu count of the machine.
    :type num_workers: int 
    :arg pin_memory: Pin memory. Default True
    :type pin_memory: bool

  Example Usage:
    train_dataloader, test_dataloader = prepare_data(train_root:"./data/train_data",
                                                     test_root:"./data/test_data",
                                                     train_transforms=train_transforms,
                                                     test_transforms=test_transforms,
                                                     batch_size=256,
                                                     num_workers=os.cpu_count(),
                                                     pin_memory=True)


  """


  train_data = GTSRB(root="./data/train_data", 
                    split="train",
                    transform=train_transforms,
                    download=True)
  
  test_data = GTSRB(root="./data/test_data",
                    split="test", 
                    transform=test_transforms,
                    download=True)


  train_dataloader = DataLoader(dataset=train_data,
                                shuffle=True,
                                batch_size=256,
                                num_workers=os.cpu_count(),
                                pin_memory=True,
                                generator=torch.Generator(device="cpu"))
  
  test_dataloader = DataLoader(dataset=test_data,
                              shuffle=False,
                              batch_size=256,
                              num_workers=os.cpu_count(),
                              pin_memory=True,
                              generator=torch.Generator(device="cpu"))

  return train_dataloader, test_dataloader
