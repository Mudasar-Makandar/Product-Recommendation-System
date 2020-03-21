import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class NNClassifer(nn.Module):
  def __init__(self, in_features, out_dim, verbose=False):
    super(NNClassifer, self).__init__()
    self.in_features = in_features
    self.out_dim = out_dim
    self.verbose = verbose

    self.fc_model = nn.Sequential(
        nn.Linear(self.in_features, 128),
        nn.Sigmoid(),
        nn.Linear(128, 64),
        nn.Sigmoid(),
        nn.Linear(64, 32),
        nn.Sigmoid(),
        nn.Linear(32, 16),
        nn.Sigmoid(),
        nn.Linear(16, self.out_dim),
        nn.Sigmoid()
    )


  def forward(self, x):
    if self.verbose:
      print(x.size())
    return self.fc_model(x)




