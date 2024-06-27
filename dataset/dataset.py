import os
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Subset
from collections import defaultdict


class Cls_dataset(data.Dataset):
    def __init__(self, cls_folder: list) -> None:
        super().__init__()
        # get all the subfolders in the cls_folder
        self.cls_folder = cls_folder
        self.cls_subfolders = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if os.path.isdir(os.path.join(cls_folder, f))]
        self.selected_cls_subfolders = ['12', '13', '14']
        
        self.cls_subfolders = [f for f in self.cls_subfolders if f.split('/')[-1] in self.selected_cls_subfolders]
        
        self.data_paths = [os.path.join(f, d) for f in self.cls_subfolders for d in os.listdir(f) if d.endswith('.mat') and "HVI" in d]
        self.data_paths.sort()
        self.label_paths = [f.replace('HVI', 'Label') for f in self.data_paths]  
        self.rgb_paths = [f.replace('HVI', 'RGB') for f in self.data_paths]
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, index: int) -> tuple:
        data_path = self.data_paths[index]
        label_path = self.label_paths[index]
        rgb_path = self.rgb_paths[index]
        
        data = sio.loadmat(data_path)['filtered_img']
        data = np.transpose(data, (2, 0, 1))
        
        label = sio.loadmat(label_path)['labelnumber']
        
        rgb = sio.loadmat(rgb_path)['img']
        rgb = np.transpose(rgb, (2, 0, 1))
        
        return data, label, rgb
    
    @staticmethod
    def collate_fn(batch: list) -> tuple:
        data, label, rgb = list(zip(*batch))
        data = torch.stack(data, dtype=torch.float32)
        label = torch.stack(label, dtype=torch.int)
        rgb = torch.stack(rgb, dtype=torch.float32)
        return data, label, rgb
    
    
class Recover_rgb_dataset(data.Dataset):
    def __init__(self, recover_folder: str) -> None:
        super().__init__()
        self.recover_folder = recover_folder
        self.data_paths = [os.path.join(recover_folder, f) for f in os.listdir(recover_folder) if f.endswith('.mat') and "HVI" in f]
        self.data_paths.sort()
        self.rgb_paths = [f.replace('HVI', 'RGB') for f in self.data_paths]
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, index: int) -> tuple:
        data_path = self.data_paths[index]
        rgb_path = self.rgb_paths[index]
        
        data = sio.loadmat(data_path)['filtered_img']
        data = np.transpose(data, (2, 0, 1))
        
        rgb = sio.loadmat(rgb_path)['img']
        # rgb = np.transpose(rgb, (2, 0, 1))
        
        return data, rgb
    
    @staticmethod
    def collate_fn(batch: list) -> tuple:
        data, rgb = list(zip(*batch))
        data = torch.stack(data, dtype=torch.float32)
        rgb = torch.stack(rgb, dtype=torch.float32)
        return data, rgb
    