import os
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from modules.utils import load_config, load_json
from itertools import product
from itertools import chain
from modules.parser import Parser
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, input_np, output_np, barcode_list, mode):
        self.input_np = input_np
        self.output_np = output_np
        self.barcode_list = barcode_list
        self.mode = mode

    def __len__(self):
        return len(self.input_np)

    def __getitem__(self, index):
        data = self.input_np[index, :]
        label = self.output_np[index, :]
        barcode = self.barcode_list[index]
        return torch.FloatTensor(data), torch.FloatTensor(label), barcode

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_args): # data dir 
        super().__init__()

        self.batch_size = data_args.batch_size   
        self.min_variance_cutoff = data_args.dimension_reduction.min_variance_cutoff
        self.input_dtype = data_args.input 
        self.output_dtype = data_args.output 

        vacuum = pd.read_csv('./data/vacuum.csv')
        charging = pd.read_csv('./data/charging.csv')        

        merged_data = pd.merge(charging.rename(columns = {'set_id':'barcode'}), vacuum, on='barcode')
        parser = Parser(merged_data, align = 'left', expansion=True, samples=-1)

        self.input_df = parser.get_signal(signal_type = self.input_dtype, transpose=False) 
        self.output_df = parser.get_signal(signal_type= self.output_dtype, transpose=False)
        self.barcode_list = self.input_df.index.to_list()  

        # dimension reduction 
        if self.min_variance_cutoff != -1:
            if self.input_dtype == self.output_dtype:
                data_args.dimension_reduction.input_n_components = self.opt_n_component(self.input_df, self.min_variance_cutoff)
                data_args.dimension_reduction.output_n_components = data_args.dimension_reduction.input_n_components
            else : 
                data_args.dimension_reduction.input_n_components = self.opt_n_component(self.input_df, self.min_variance_cutoff)
                data_args.dimension_reduction.output_n_components = self.opt_n_component(self.output_df, self.min_variance_cutoff)

        # fit and transform 
        if self.input_dtype == self.output_dtype:
            self.input_pca = PCA(n_components=data_args.dimension_reduction.input_n_components)
            self.input_pca.fit(self.input_df)
            self.output_pca = self.input_pca
        else : 
            self.input_pca = PCA(n_components=data_args.dimension_reduction.input_n_components)
            self.input_pca.fit(self.input_df)
            self.output_pca = PCA(n_components=data_args.dimension_reduction.output_n_components)
            self.output_pca.fit(self.output_df)
                
        self.input_np = self.input_pca.transform(self.input_df)
        self.output_np = self.output_pca.transform(self.output_df)        

        # 데이터 분할 
        self.n_train = int(len(self.input_np) * 0.6)
        self.n_valid = int(len(self.input_np) * 0.2)
        self.n_test = len(self.input_np) - self.n_train - self.n_valid

        # 데이터셋 생성
        self.train_dataset = CustomDataset(self.input_np[:self.n_train, :], 
                                           self.output_np[:self.n_train, :], 
                                           self.barcode_list[:self.n_train], 
                                           mode = 'train')
        self.val_dataset = CustomDataset(self.input_np[self.n_train:self.n_train+self.n_valid, :], 
                                         self.output_np[self.n_train:self.n_train+self.n_valid, :], 
                                         self.barcode_list[self.n_train:self.n_train+self.n_valid], 
                                         mode = 'val')
        self.test_dataset = CustomDataset(self.input_np[self.n_train+self.n_valid:, :], 
                                          self.output_np[self.n_train+self.n_valid:, :], 
                                          self.barcode_list[self.n_train+self.n_valid:], 
                                          mode = 'test')

    @staticmethod
    def opt_n_component(df:pd.DataFrame, cutoff:float) -> int:
        m = df.shape[1]
        for n_component in range(1,m+1):
            reduce_model = PCA(n_components=n_component)
            reduce_model.fit(df)
            explained_variances_ratio = np.sum(reduce_model.explained_variance_ratio_)
            if np.sum(explained_variances_ratio)>=cutoff:
                return n_component

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size = self.batch_size,
                          shuffle=True) 

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size = self.batch_size,
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size = self.batch_size, 
                          shuffle=False)

