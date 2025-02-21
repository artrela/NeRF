from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import glob
import os
import torch
import tqdm

from dataset.dataloader import SyntheticDataloader
from model.NeRF import NeRF
import utils.training_utils as train_utils

def main():
        
    params = train_utils.parse_config("model/config.yaml")
    hparams = params['hparams']
    
    # create and push hparams to summary writer
    logdir = f"runs/{params['name']}/"
    save_model_pth = logdir + "/weights/"
    image_save_pth = logdir + "/images/"
    os.makedirs(save_model_pth, exist_ok=True)
    os.makedirs(image_save_pth, exist_ok=True)
    
    writer = SummaryWriter(logdir)
    
    hparams['ffn_skips'] = str(hparams['ffn_skips'])
    writer.add_hparams(hparams, {}, run_name=params['name'])
    writer.add_text("Additional Notes", params['notes'])
    hparams['ffn_skips'] = list(hparams['ffn_skips'])
    
    # create the nerf
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    epoch = 0
    epochs = hparams['epochs']
    
    coarse_nerf = NeRF(
        x_pe=hparams['x_pe'],
        d_pe=hparams['d_pe'],
        pe_include_inp=hparams['pe_include_input'],
        ffn_layers=hparams['ffn_layers'],
        ffn_skips=hparams['ffn_skips'],
        hid_dim=hparams['hid_dim']
    )
    
    if params['resume_training']:
        latest_model = glob.glob(save_model_pth + "coarse_nerf*.pth").sort()[-1]
        checkpoint = torch.load(latest_model, weights_only=True)
        coarse_nerf.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        
    coarse_nerf = coarse_nerf.to(device)
    model_summary = str(summary(coarse_nerf, depth=4))
    with open(save_model_pth + "/coarse_nerf_arch.txt",  "+w", encoding="utf-8") as arch_file:
        arch_file.write(model_summary)
    
    optimizer = torch.optim.Adam(coarse_nerf.parameters(), hparams['lr'])
    gamma = train_utils.compute_gamma(hparams['lr'], hparams['lr'] * hparams['lr_decay'], epochs)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = torch.nn.MSELoss()
    
    if params['fine_nerf']:
        raise NotImplementedError

    train_dset = SyntheticDataloader("data/nerf_synthetic/", "hotdog", "train",
                                    item_sampling=False, resize=hparams['resize'], shuffle=True, device=device)
    val_dset = SyntheticDataloader("data/nerf_synthetic/", "hotdog", "val",
                                item_sampling=False, resize=hparams['resize'], shuffle=False, device=device)

    if hparams['normalize_scene']:
        train_dset.normalize()
        val_dset.normalize(train_dset.max_t)
    
    for i in tqdm.tqdm(range(epoch, epochs), desc="Training Nerf"):
        
        optimizer.zero_grad()
        train_loss = train_utils.render_loop(coarse_nerf, train_dset, criterion, hparams)
        optimizer.step()
        lr_scheduler.step()
        
        with torch.no_grad():
            val_loss = train_utils.render_loop(coarse_nerf, val_dset, criterion, 
                                                hparams, train=False)
        
        avg_train_loss = train_loss / len(train_dset)
        avg_val_loss = val_loss / len(val_dset)
        writer.add_scalars("loss", {
            'train': avg_train_loss,
            'val': avg_val_loss
        })
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'])
        
        tqdm.tqdm.write(f"== Training Loss: {avg_train_loss:.5f} == Validation Loss: {avg_val_loss:.5f} == ")
            
        if i % params['render_interval'] == 0 and i > 0:
            with torch.no_grad():
                train_utils.render_image(coarse_nerf, train_dset, val_dset, save_model_pth,
                                    params['render_batch'], hparams)
        
        if i % params['save_weights_interval'] == 0 and i > 0:
            torch.save({
                'epoch': i,
                'model_state_dict': coarse_nerf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
                }, save_model_pth + f"/coarse_nerf_{i:0<len(str(epochs))}.pth")

        writer.flush()    
    
    
if __name__ == "__main__":
    main()