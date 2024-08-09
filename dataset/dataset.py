import os
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Subset
from collections import defaultdict
from scipy.io import loadmat
import random


class Cls_dataset(data.Dataset):
    def __init__(self, cls_folder: list, num_cls: int) -> None:
        super().__init__()
        # get all the subfolders in the cls_folder
        self.cls_folder = cls_folder
        self.num_cls = num_cls
        
        self.cls_subfolders = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if os.path.isdir(os.path.join(cls_folder, f))]
        self.selected_cls_subfolders = ['12', '13', '14', '23', '24', '34']
        
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
        data = torch.from_numpy(np.transpose(data, (2, 0, 1)))
        
        label = sio.loadmat(label_path)['labelnumber'].reshape(-1) # 1 x n vector to (n,)
        #transfer to one-hot
        label = torch.from_numpy(np.sum(np.eye(self.num_cls)[label], axis=0).clip(0, 1).astype(np.int))
        
        rgb = sio.loadmat(rgb_path)['img']
        rgb = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
        
        return data, label, rgb
    
    @staticmethod
    def collate_fn(batch: list) -> tuple:
        data, label, rgb = list(zip(*batch))
        data = torch.stack(data).to(dtype=torch.float32)
        label = torch.stack(label).to(dtype=torch.float32)
        rgb = torch.stack(rgb).to(dtype=torch.float32)
        return data, label, rgb
    
    
class Recover_rgb_dataset(data.Dataset):
    def __init__(self, recover_folder: str, weight_filename) -> None:
        super().__init__()
        # if "image" folder is in the subsubfolder
        #find all the subfolders in the recover_folder
        self.recover_folders = [os.path.join(recover_folder, f) for f in os.listdir(recover_folder) if os.path.isdir(os.path.join(recover_folder, f))]
        self.recover_folders = [os.path.join(f, sub_f) for f in self.recover_folders for sub_f in os.listdir(f) if os.path.isdir(os.path.join(f, sub_f))]

        self.rgb_paths = [os.path.join(f, "image", img) for f in self.recover_folders for img in os.listdir(os.path.join(f, "image")) if img.endswith('.png')]
        self.rgb_paths = random.sample(self.rgb_paths, 20000) if len(self.rgb_paths) > 20000 else self.rgb_paths
        self.rgb_paths.sort()
        
        self.RGB_Wights = loadmat(weight_filename)
        self.RGB_Wights = self.RGB_Wights['w']
        self.RGB_Wights = self.RGB_Wights[:, :, 0]
        
        self.c, self.r = np.shape(self.RGB_Wights)
    
    def __len__(self) -> int:
        return len(self.rgb_paths)
    
    def __getitem__(self, index: int) -> tuple:
        data_path = self.rgb_paths[index]
        
        rgb_image = np.array(Image.open(data_path))
        x, y, z = rgb_image.shape
        
        filtered_img = np.abs(self.RGB_Wights) @ rgb_image.reshape(-1, z).T.astype(np.float16)
        filtered_img = filtered_img.T.reshape(x, y, self.c)
        
        filtered_img = torch.from_numpy(filtered_img).permute(2, 0, 1)
        rgb_image = torch.from_numpy(rgb_image)
        
        return filtered_img, rgb_image
    
    @staticmethod
    def collate_fn(batch: list) -> tuple:
        data, rgb = list(zip(*batch))
        data = torch.stack(data).to(dtype=torch.float32)
        rgb = torch.stack(rgb).long()
        return data, rgb


class Recover_rgb_dataset_visual(data.Dataset):
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
        data = torch.from_numpy(np.transpose(data, (2, 0, 1)))
        
        rgb = torch.from_numpy(sio.loadmat(rgb_path)['imgmat']).round()
        # rgb = np.transpose(rgb, (2, 0, 1))
        
        img_name = data_path.split('/')[-1].replace('.mat', '')
        
        return data, rgb, img_name
    
    @staticmethod
    def collate_fn(batch: list) -> tuple:
        data, rgb, img_name = list(zip(*batch))
        data = torch.stack(data).to(dtype=torch.float32)
        rgb = torch.stack(rgb).long()
        return data, rgb, img_name


if __name__ == "__main__":
    dataset = Recover_rgb_dataset(recover_folder="/data/chaoyi_he/HSI/HSI_RGB/path/NMR_Dataset", 
                                  weight_filename="/data/chaoyi_he/HSI/HSI_RGB/path/NMR_Dataset/filtered_rgbWight.mat")
    