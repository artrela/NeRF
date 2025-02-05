from PIL import Image
import json 
import numpy as np
import os
import sys
import torch

class SyntheticDataloader():
    def __init__(self, pth, item, split="train"):
        
        data_pth = os.path.join(pth + "/" + item)
        
        with open(data_pth + f"/transforms_{split}.json", "r") as data_file:
            data = json.load(data_file)
            data_file.close()
        
        transforms = []
        max_t = np.zeros(3)
        for image in data['frames']:
            transforms.append(image['transform_matrix'])
            t = np.array(image['transform_matrix'])[:3, 3]
            if np.linalg.norm(max_t) < np.linalg.norm(t):
                max_t = t
                
        self.transforms = np.stack(transforms, axis=0) / np.linalg.norm(max_t)
        
        open_image = lambda img: np.asarray(Image.open(data_pth + img + ".png").convert("RGB")) / 255.
        self.images = np.stack([open_image(img['file_path']) for img in data['frames']], axis=0)
        
        self.total_pxs = self.images[0].shape[0] * self.images[0].shape[1]
        self.img_shape = self.images[0].shape[:2]
        self.img_shape = (800, 800)
        
        self.f = (self.img_shape[0] / 2) / np.tan( data['camera_angle_x'] / 2 ) 
        self.cx =  self.img_shape[0] / 2
        self.cy =  self.img_shape[0] / 2
        self.cam2img = np.eye(3)
        self.cam2img[0, 0] = self.cam2img[1, 1] = self.f
        self.cam2img[0, 2] = self.cx
        self.cam2img[1, 2] =  self.cy
    
    def create_rays(self, R, t):
        '''
        R.shape = 3x3
        t.shape = 1x3
        '''
        
        # TODO - do this for a SET of Rotation Matric
        u, v = np.mgrid[0:self.img_shape[0], 0:self.img_shape[1]]
        
        xc =  (u - self.cx) / self.f
        yc = -(v - self.cy) / self.f
        
        # TODO change back to 1 afterwards!
        # d = np.dstack((xc, yc, np.ones(xc.shape))) @ R
        d = np.dstack((xc, yc, -np.ones(xc.shape))) @ R
        o = np.ones((*self.img_shape, 1)) @ t
        
        return o, d
    
    def sample_ray(o, d, t):
        pass
        
    
    def __len__(self):
        assert len(self.transforms) == len(self.images)
        return len(self.images)