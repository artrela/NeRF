from PIL import Image
from tqdm import tqdm
from typing import Optional, Tuple
import cv2
import json 
import numpy as np
import os
import sys
import torch

class SyntheticDataloader():
    def __init__(self, pth: str, item: str, split: str, 
                item_sampling: bool, resize: int, shuffle: bool, 
                device: torch.device):
        
        self.shuffle = shuffle
        return_alpha = True if item_sampling else False
        self.device = device
        
        data_pth = os.path.join(pth + "/" + item)
        
        with open(data_pth + f"/transforms_{split}.json", "r") as data_file:
            data = json.load(data_file)
            data_file.close()
        
        self.transforms, self.images = [], []
        for frame in tqdm(data['frames'], desc=f"Loading {item} {split} Data: "):
            self.transforms.append(frame['transform_matrix'])
            self.images.append(self.open_image(data_pth + frame['file_path'][1:], return_alpha, resize))
        
        self.transforms = np.stack(self.transforms, axis=0) 
        self.images = np.stack(self.images, axis=0) 
                
        self.H, self.W = self.images[0].shape[:2]
        self.img_shape = self.images[0].shape[:2]

        self.f = (self.W / 2) / np.tan( data['camera_angle_x'] / 2 ) 
        self.cx =  self.H / 2
        self.cy =  self.W / 2
        
        self.cam2img = np.eye(3)
        self.cam2img[0, 0] = self.cam2img[1, 1] = self.f
        self.cam2img[0, 2] = self.cx
        self.cam2img[1, 2] = self.cy
    
    def create_rays(self, R: np.ndarray, t: np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        """ Create rays for a camera at position R, t. Creates HxW rays, and then transforms them 
        to the camera frame. 

        Args:
            R (np.ndarray): Rotation component of a world2cam transform
            t (np.ndarray): Translation component of the world2cam transform

        Returns:
            Tuple[np.ndarray, np.ndarray]: A set of ray origins and directions shape (HxWx3)
        """
        
        u, v = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='xy')
        
        xc =  (u - self.cx) / self.f
        yc = -(v - self.cy) / self.f
        d = np.stack((xc, yc, -np.ones(xc.shape)), -1) @ R.T
        
        o = np.broadcast_to(t, d.shape)
        
        return o, d
    
    
    def open_image(self, pth: str, return_alpha:bool, resize: Optional[int]=0)->np.ndarray:
        """Open a normalized image at the given path, returning either an rgb, rgba image. 
        Optionally resize the image.

        Args:
            pth (str): Path to image
            resize (int): Optional value to resize the image to
            return_alpha (bool): Return a 4 channel image, including the alpha channel 
                if needed. 

        Returns:
            np.ndarray: The post processed image
        """
        img = np.asarray(Image.open(pth + ".png").convert("RGBA"))
        
        if resize != -1:
            img = cv2.resize(img, (resize, resize)) / 255.
        else:
            img /= 255.
        
        return img[..., :3] if not return_alpha else img
    
    
    def predict_color(self, sigmai: torch.Tensor, ci: torch.Tensor, ti: torch.Tensor, d)->torch.Tensor:
        """ Given the density of a ray at a position, its color at the position, and the distance
        between samples, recover the predicted color using the volumetric rendering equatoin

        Args:
            sigmai (torch.Tensor): (rays, N, 1)
            ci (torch.Tensor): (rays, N, 3)
            ti (torch.Tensor): (rays, N)

        Returns:
            torch.Tensor: (rays, 3) the predicted color for each ray
        """
        
        # (rays, N, 1) distance between last ray and next is inf...
        inf = torch.tensor([1e10], device=self.device).expand(ti.shape[0]).unsqueeze(-1)
        deltai = torch.diff(ti * torch.norm(d, dim=-1, keepdim=True), append=inf).unsqueeze(-1) 

        # (rays, N, 1) <- # (rays, N, 1), (rays, N, 1)
        ones = torch.tensor([1.], device=self.device).expand(ti.shape[0]).unsqueeze(-1).unsqueeze(-1)
        Ti = torch.exp( -torch.cumsum(sigmai * deltai, dim=1)) 
        Ti = torch.concat((ones, Ti), dim=1)[..., :-1, :]
        
        # (rays, N, 1) <- # (rays, N. 1), (rays, N, 1)
        alphai = 1. - torch.exp( - sigmai * deltai )
        
        # (rays, 3) <- # (rays, N, 1), (rays, N, 1), (rays, N, 3)
        wi = Ti * alphai
        c_pred = torch.sum(wi * ci, dim=1)
        
        return c_pred 
    

    def sample_rays(self, N: int, tn: float, tf: float, rays: int, stratified: bool)->torch.Tensor:
        """ Sample a time vector 't' within N bins from tn to tf, for 'rays' many rays. Supports
        stratified random samples as well. 

        Args:
            N (int): Number of samples to take
            tn (float): Closest point to sample from 
            tf (float): Fartherst point to sample from 
            rays (int): How many rays to create samples for
            stratified (bool): Randomize the samples or not

        Returns:
            torch.Tensor: A sample of times to complete the equation r = x + o*t
        """
        if stratified:  
            samples = np.random.uniform(low=0, high=(1/N)*(tf - tn), size=(rays, N))
            i = np.expand_dims(np.arange(0, N), 0)
            t = torch.tensor(tn + (i/N)*(tf - tn) + samples, dtype=torch.float32, device="cuda")
        else:
            t = torch.tensor(np.linspace(tn, tf, N), dtype=torch.float32, device='cuda').expand(rays, N)
            
        return t
    
    
    def select_rays(self, o, d, image, rays_per_image):
        
        u, v = np.random.choice(self.H, size=(2, rays_per_image), replace=False)
        
        o = torch.tensor(o[u, v], device=self.device, dtype=torch.float32)
        d = torch.tensor(d[u, v], device=self.device, dtype=torch.float32)
        c_true = torch.tensor(image[u, v], device=self.device, dtype=torch.float32)
        
        return o, d, c_true


    def __iter__(self):
        idx = np.arange(self.__len__())
        
        if self.shuffle:
            np.random.shuffle(idx)
            
        return iter(zip(self.images[idx], self.transforms[idx]))
    
    def __getitem__(self, idx):
        return self.images[idx], self.transforms[idx]

            
    def __len__(self):
        return len(self.images)
    
    
class SyntheticTestDataloader(SyntheticDataloader):
    def __init__(self, pth: str, item: str, 
                img_shape: int, device: torch.device):
        
        self.device = device
        split = 'test'
        
        data_pth = os.path.join(pth + "/" + item)
        
        with open(data_pth + f"/transforms_{split}.json", "r") as data_file:
            data = json.load(data_file)
            data_file.close()
        
        self.transforms = []
        for frame in tqdm(data['frames'], desc=f"Loading {item} {split} Data: "):
            self.transforms.append(frame['transform_matrix'])
        
        self.transforms = np.stack(self.transforms, axis=0) 
                
        self.H = self.W = img_shape
        self.img_shape = (img_shape, img_shape)

        self.f = (self.W / 2) / np.tan( data['camera_angle_x'] / 2 ) 
        self.cx =  self.H / 2
        self.cy =  self.W / 2
        
        self.cam2img = np.eye(3)
        self.cam2img[0, 0] = self.cam2img[1, 1] = self.f
        self.cam2img[0, 2] = self.cx
        self.cam2img[1, 2] = self.cy
        
    def __iter__(self):
        return iter(self.transforms)
    
    def __getitem__(self, idx):
        return self.transforms[idx]

            
    def __len__(self):
        return len(self.transforms)
        
        