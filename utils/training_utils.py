from dataset.dataloader import SyntheticDataloader
from model.NeRF import NeRF

import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import tqdm
import yaml

def compute_gamma(lr0: float, lrN: float, N: int)->float:
    """ Compute the gamma factor to obtain lrN after N epochs

    Args:
        lr0 (float): Starting learning rate
        lrN (float): Ending learning rate
        N (int): Epochs

    Returns:
        float: gamma factor
    """
    return (lrN * lr0) ** (1 / N)


def parse_config(path: str)->dict:
    """Given a path to a yaml file, return a dictionary object

    Args:
        path (str): A proposed path to a configuration path

    Raises:
        FileNotFoundError: If the file is not found, prints the path given. 

    Returns:
        dict: A parsed configuration yaml file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration not present at {path}!")
    else:
        with open(path, "r") as config_file:
            config_dict = yaml.safe_load(config_file)
            config_file.close()
    
        return config_dict
    

def render_loop(model: NeRF, dset: SyntheticDataloader, criterion: torch.nn.MSELoss, 
                hparams: dict, train:bool = True):
    
    t_loss = 0
    for idx, (image, pose) in enumerate(dset):
        
        # (H, W, 3), (H, W, 3)
        o, d = dset.create_rays(pose[:3, :3], pose[:3, -1]) 
        
        # (rays, 3), (rays, 3), (rays, 3)
        o, d, c_true = dset.select_rays(o, d, image, hparams['rays_per_image'])
        
        # (rays, Nc)
        t = dset.sample_rays(hparams["Nc"], hparams["tn"], hparams['tf'],
                                    hparams['rays_per_image'], stratified=train)
        
        # (rays, Nc, 3) <- (rays, 1, 3) + (rays, 1, 3)*(rays, Nc, 1)
        x = o.unsqueeze(1) + d.unsqueeze(1)*t.unsqueeze(2)
        d = d.unsqueeze(1).expand(-1, hparams['Nc'], -1)
        
        # (rays, Nc, 1), (rays, Nc, 3)
        sigmas, colors = model(x, d)
        
        # (rays, 3) <- (rays, Nc, 1), (rays, Nc, 3), (rays, Nc)
        c_pred = dset.predict_color(sigmas, colors, t)
        loss = criterion(c_pred, c_true)
        
        if train:
            loss.backward()
            
        t_loss += loss.item()
    
    del o, d, c_true, sigmas, colors, c_pred
    gc.collect()
    torch.cuda.empty_cache
        
    return t_loss

def render_image(model, train_dset, val_dset, pth, batch, hparams):
    
    fig, axs = plt.subplots(2, 2)
    
    for i, (img, pose) in enumerate((train_dset[0], val_dset[0])):
        
        O, D = train_dset.create_rays(pose[:3, :3], pose[:3, -1]) 

        O = O.reshape(-1, 3)
        D = D.reshape(-1, 3)
        
        img_est = np.zeros_like(img)

        name = "train" if i == 0 else 'val'
        for j in tqdm.tqdm(range(0, len(D), batch), desc=f"Rendering {name}"):
            
            o = torch.tensor(O[j:j+batch], device='cuda', dtype=torch.float32)
            d = torch.tensor(D[j:j+batch], device='cuda', dtype=torch.float32)
            
            # (rays, Nc)
            t = train_dset.sample_rays(hparams["Nc"], hparams["tn"], hparams['tf'],
                                    batch, stratified=False)
        
            # (rays, Nc, 3) <- (rays, 1, 3) + (rays, 1, 3)*(rays, Nc, 1)
            x = o.unsqueeze(1) + d.unsqueeze(1)*t.unsqueeze(2)
            d = d.unsqueeze(1).expand(-1, hparams['Nc'], -1)

            # (rays, Nc, 1), (rays, Nc, 3)
            sigmas, colors = model(x, d)

            # (rays, 3) <- (rays, Nc, 1), (rays, Nc, 3), (rays, Nc)
            c_pred = train_dset.predict_color(sigmas, colors, t)
            
            u, v = np.unravel_index(np.arange(j, j+batch), shape=train_dset.img_shape)
            
            img_est[u, v] = c_pred.cpu()
            
        axs[i, 0].imshow(img)
        axs[i, 1].imshow(img_est)
        
    fig.savefig(pth)
    
    del o, d, c_true, sigmas, colors, c_pred
    gc.collect()
    torch.cuda.empty_cache
    
    return
            
    