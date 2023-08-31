import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyedflib
import time
import tsfel
from torch.utils import data
from scipy import signal
import datetime
import csv
import h5py
import torch
import re
import pickle

from torch.utils.data import Dataset, DataLoader

 
class ExampleDataset(Dataset):

    def __init__(self, mode='train',n_class = '4c' ):
        self.chn = 'Plethysmogram'
        self.cur_dir = os.getcwd()
        self.n_class = n_class
        self.mode = mode

        #processed PSG data root path
        self.PREFIX_DIR = '/HDD/snubh-psg-processed/' 
        self.LABEL_DIR = 'preprocessing/abcd_2fold_wo1314.pkl'

        if os.path.exists(f'ab_{self.mode}_psg.pkl') == True:
            with open(f'ab_{self.mode}_psg.pkl', "rb") as rp:
                self.pkl_paths = pickle.load(rp)
        else:
            self.pkl_paths = self.__get_pklpaths__()
    
    def __getitem__(self, idx):
        # return data based on index
        # data = self.preprocess(self.data[index]) # if need to process data

        # print(self.pkl_paths.keys())

        # Get file
        data_path = self.pkl_paths[self.mode][idx]
        
        
        label_path = self.pkl_paths[self.n_class][idx]


        # Load data
        data_dict = np.load(data_path, allow_pickle=True)

        label_dict = np.load(label_path, allow_pickle=True).item()
        x = data_dict[self.chn]
        # print(data_path)

        #check if the data is None
        if len(x)==0:           
            print(f"None value data: {data_path}")
        x = torch.tensor(x,dtype=torch.float32)
        y = label_dict[self.n_class]
     
        return x,y
    
    def __len__(self):
        # return the length of data
        return len(self.pkl_paths[self.mode])
    
    def __get_pklpaths__(self):
        #return all group_dir
        data = []
        label = []
        count = 0
        name = self.mode
        pkl_paths = dict()
        if self.mode == "test":
            name = 'val'
        with open(os.path.join(self.cur_dir,self.LABEL_DIR), "rb") as rf:
            clip = pickle.load(rf)
        for k in clip.keys():
            if (k.find(name) != -1):
                for dir_data in clip[k]:
                    filename = os.path.splitext(os.path.basename(dir_data))[0]
                    group = dir_data.split("/")[-3]
                    if os.path.exists(os.path.join(self.PREFIX_DIR,group,self.mode,f"{filename}.pkl")):
                        data.append(os.path.join(self.PREFIX_DIR,group,self.mode,f"{filename}.pkl"))
                        label.append(dir_data)
                    else:
                        count+=1
        pkl_paths[self.n_class] = label
        pkl_paths[self.mode] = data
        print(f"not found {self.mode} files: {count}")
        with open(f'ab_{self.mode}_psg.pkl', "wb") as wf:
            pickle.dump(pkl_paths, wf)


        return pkl_paths

    # read data 
    def __load_data__(self):
        # loop directory of data
        # split and return train_X, valid_X
        X = []
        pkl_list = []
        for group in self.pkl_paths:
            for f in os.listdir(group):
                pkl_list.append(os.path.join(group,f))
                # with open(os.path.join(group,f),'rb') as fp:
                #     pkl = np.load(fp, allow_pickle=True)
                # X += pkl
        return X,pkl_list
    
    def __load_label__(self, label_paths: list):
        # return train_Y, valid_Y
        pass
      
    def preprocess(self, data):
        # do some process on data
        pass




if __name__ == "__main__":
    # Test dataloader
    dataset = ExampleDataset(mode = "test")    
    print(len(dataset))
    dataset = ExampleDataset(mode = "train") 
    print(len(dataset))

    #check sound data files:
    with open('abcd_2fold_wo1314.pkl', 'rb') as f:
            data = pickle.load(f)
    print(data.keys())
   

