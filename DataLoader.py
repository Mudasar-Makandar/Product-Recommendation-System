import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class DataReader(Dataset):
    def __init__(self, file_path, mapping_dict,cols_to_use,target_cols, verbose=False):
        super(DataReader, self).__init__()
        self.dataframe = pd.read_csv(file_path)
        self.mapping_dict = mapping_dict
        self.target_cols = target_cols
        self.cols_to_use = cols_to_use
        self.verbose = verbose

    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):
        self.unique_dataframe = self.dataframe[self.dataframe['ncodpers']==list(self.dataframe["ncodpers"].unique())[idx]]
        self.unique_dataframe = self.unique_dataframe.reset_index(drop=True)

        for col in self.cols_to_use:
                if col == "indrel_1mes":
                    self.unique_dataframe[col] = self.unique_dataframe[col].astype(str)
                    self.unique_dataframe[col]=  self.unique_dataframe[col].fillna(-99).replace(self.mapping_dict[col])
                if col == "conyuemp":
                    self.unique_dataframe[col] = self.unique_dataframe[col].astype(str)
                    self.unique_dataframe[col]=  self.unique_dataframe[col].replace({"NaN":"NA","nan":"NA"})
                    self.unique_dataframe[col]=  self.unique_dataframe[col].fillna(-99).replace(self.mapping_dict[col])
                if col != "ncodpers" and col != "indrel_1mes" and col != "indrel_1mes":
                    self.unique_dataframe[col]=  self.unique_dataframe[col].fillna(-99).replace(self.mapping_dict[col])
        self.user_profle = list(self.unique_dataframe.loc[0,self.cols_to_use])
        product_list=[]

        for p in range(len(self.unique_dataframe)):
                prod_list = list(self.unique_dataframe.loc[p,self.target_cols].values.astype("int32"))
                product_list.append(prod_list)
        profile_l = []
        profile_l1 = [profile_l.append(i) for i in self.user_profle if type(i)!= np.dtype('str') ]
        user_p = torch.tensor(profile_l).view(1,17)
        product_l = torch.tensor(product_list).view(len(self.unique_dataframe),1,len(self.target_cols))
        return user_p, product_l






if __init__=="__main__":
	data_loader = DataReader("/kaggle/working/train_ver2.csv",mapping_dict,cols_to_use,target_cols)
	from torch.utils.data import DataLoader
	dataloader = DataLoader(data_loader, batch_size=1)
	data_iter = iter(dataloader)
	x,y = data_iter.next()
