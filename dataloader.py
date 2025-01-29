from PIL import Image
import json 
import numpy as np
import os
import sys
import torch

class SyntheticDataloader(torch.utils.data.Dataset):
    def __init__(self, pth, item, split="train"):
        super().__init__()
        
        data_pth = os.path.join(pth + "/" + item)
        
        with open(data_pth + f"/transforms_{split}.json", "r") as data_file:
            data = json.load(data_file)
            data_file.close()
            
        self.transforms = np.stack([image['transform_matrix'] for image in data['frames']], axis=0)
        
        open_image = lambda img: np.asarray(Image.open(data_pth + img + ".png").convert("RGB")) / 255.
        self.images = np.stack([open_image(img['file_path']) for img in data['frames']], axis=0)
        
        self.total_pxs = self.images[0].shape[0] * self.images[0].shape[1]
        self.img_shape = self.images[0].shape[:2]
        
        f = (self.img_shape[0] / 2) / np.tan( data['camera_angle_x'] / 2 ) 
        self.cam2img = np.eye(3)
        self.cam2img[0, 0] = self.cam2img[1, 1] = f
        self.cam2img[0, 2] = self.cam2img[1, 2] = self.img_shape[0] / 2
        
    def __getitem__(self, idx):
        
        random_px = np.random.randint(0, self.total_pxs)
        px_pos = np.unravel_index(random_px, self.img_shape)
        
        return px_pos
    
    def __len__(self):
        assert len(self.transforms) == len(self.images)
        return len(self.images)