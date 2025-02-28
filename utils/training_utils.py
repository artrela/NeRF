from dataset.dataloader import SyntheticDataloader
from model.NeRF import NeRF

from PIL import Image
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
    return (lrN / lr0) ** (1 / N)


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
    
    o, d, t, c_true = [], [], [], []
    for idx, (image, pose) in enumerate(dset):
        
        # (H, W, 3), (H, W, 3)
        o_batch, d_batch = dset.create_rays(pose[:3, :3], pose[:3, -1]) 
        
        # (rays, 3), (rays, 3), (rays, 3)
        o_batch, d_batch, c_true_batch = dset.select_rays(o_batch, d_batch, image, hparams['rays_per_image'])
        
        # (rays, Nc)
        t_batch = dset.sample_rays(hparams["Nc"], hparams["tn"], hparams['tf'], hparams['rays_per_image'], stratified=train)
        
        o.append(o_batch)
        d.append(d_batch)
        t.append(t_batch)
        c_true.append(c_true_batch)
    
    o = torch.concat(o, 0)
    d = torch.concat(d, 0)
    t = torch.concat(t, 0)
    c_true = torch.concat(c_true, 0)
    
    # (rays, Nc, 3) <- (rays, 1, 3) + (rays, 1, 3)*(rays, Nc, 1)
    x = o.unsqueeze(1) + d.unsqueeze(1)*t.unsqueeze(2)
    d_norm = d / torch.norm(d, dim=-1, keepdim=True)
    d_norm = d_norm.unsqueeze(1).expand(-1, hparams['Nc'], -1)
    
    # (rays, Nc, 1), (rays, Nc, 3)
    sigmas, colors = model(x, d_norm)
    
    # (rays, 3) <- (rays, Nc, 1), (rays, Nc, 3), (rays, Nc)
    c_pred = dset.predict_color(sigmas, colors, t, d)

    loss = criterion(c_pred, c_true)
        
    if train:
        loss.backward()
    
    del o, d, c_true, sigmas, colors, c_pred
    gc.collect()
    torch.cuda.empty_cache
        
    return loss.item()

def render_image(model, train_dset, val_dset, pth, batch, hparams):
    
    fig, axs = plt.subplots(2, 2)
    
    for i, (img, pose) in enumerate((train_dset[0], val_dset[0])):
        
        O, D = train_dset.create_rays(pose[:3, :3], pose[:3, -1]) 
        O = O.reshape(-1, 3)
        D = D.reshape(-1, 3)
        
        img_est = np.zeros_like(img)
        
        us, vs = np.meshgrid(np.arange(train_dset.H), np.arange(train_dset.W), indexing='xy')
        us = us.flatten()
        vs = vs.flatten()
        
        name = "train" if i == 0 else 'val'
        for j in tqdm.tqdm(range(0, len(us), batch), desc=f"[{name}] Rendering Estimate"):
            
            u = us[j:j+batch]
            v = vs[j:j+batch]

            o = torch.tensor(O[j:j+batch], device='cuda', dtype=torch.float32)
            d = torch.tensor(D[j:j+batch], device='cuda', dtype=torch.float32)
            
            # (rays, Nc)
            t = train_dset.sample_rays(hparams["Nc"], hparams["tn"], hparams['tf'],
                                    batch, stratified=False)
        
            # (rays, Nc, 3) <- (rays, 1, 3) + (rays, 1, 3)*(rays, Nc, 1)
            x = o.unsqueeze(1) + d.unsqueeze(1)*t.unsqueeze(2)
            d_norm = d / torch.norm(d, dim=-1, keepdim=True)
            d_norm = d_norm.unsqueeze(1).expand(-1, hparams['Nc'], -1)
    
            # (rays, Nc, 1), (rays, Nc, 3)
            sigmas, colors = model(x, d_norm)

            # (rays, 3) <- (rays, Nc, 1), (rays, Nc, 3), (rays, Nc)
            c_pred = train_dset.predict_color(sigmas, colors, t, d)
            
            img_est[v, u] = c_pred.cpu()
            
        axs[i, 0].imshow(img)
        axs[i, 1].imshow(img_est)
        axs[i, 0].set_xticks([])
        axs[i, 1].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 1].set_yticks([])
        
    fig.savefig(pth)
    
    del o, d, sigmas, colors, c_pred
    gc.collect()
    torch.cuda.empty_cache
    
    return


def render_video(model, test_dset, pth, batch, hparams):
    
    frames = []
    for i, pose in tqdm.tqdm(enumerate(test_dset), desc=f"[Rendering Video]", total=len(test_dset)):
        
        O, D = test_dset.create_rays(pose[:3, :3], pose[:3, -1]) 
        O = O.reshape(-1, 3)
        D = D.reshape(-1, 3)
        
        img_est = np.zeros((test_dset.H, test_dset.W, 3))
        
        us, vs = np.meshgrid(np.arange(test_dset.H), np.arange(test_dset.W), indexing='xy')
        us = us.flatten()
        vs = vs.flatten()
        
        for j in range(0, len(us), batch):
            
            u = us[j:j+batch]
            v = vs[j:j+batch]

            o = torch.tensor(O[j:j+batch], device='cuda', dtype=torch.float32)
            d = torch.tensor(D[j:j+batch], device='cuda', dtype=torch.float32)
            
            # (rays, Nc)
            t = test_dset.sample_rays(hparams["Nc"], hparams["tn"], hparams['tf'],
                                    batch, stratified=False)
        
            # (rays, Nc, 3) <- (rays, 1, 3) + (rays, 1, 3)*(rays, Nc, 1)
            x = o.unsqueeze(1) + d.unsqueeze(1)*t.unsqueeze(2)
            d_norm = d / torch.norm(d, dim=-1, keepdim=True)
            d_norm = d_norm.unsqueeze(1).expand(-1, hparams['Nc'], -1)
    
            # (rays, Nc, 1), (rays, Nc, 3)
            sigmas, colors = model(x, d_norm)

            # (rays, 3) <- (rays, Nc, 1), (rays, Nc, 3), (rays, Nc)
            c_pred = test_dset.predict_color(sigmas, colors, t, d)
            
            img_est[v, u] = c_pred.cpu()
            
    
        del o, d, sigmas, colors, c_pred
        gc.collect()
        torch.cuda.empty_cache
        
        frames.append(Image.fromarray((img_est * 255.).astype(np.uint8)))
    
    frames[0].save(pth, save_all = True, append_images = frames[1:], optimize = True, duration = 10)
    
    return
            
            
    