import torch
from dataloader import SyntheticDataloader
from render_utils import c_pred, stratified_sampling_rays
import matplotlib.pylab as plt 
import numpy as np
import tqdm
from model import NeRF
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import os

exp_name = "add_relu_slow_val_norm_scene"
writer = SummaryWriter(log_dir=f"runs/{exp_name}/")
os.makedirs(f"data/model_out/{exp_name}", exist_ok=True)
#
epochs = int(10000)
# select a varies amount of images (1/2 the images at a time)
# select a number of rays that goes cleanly into the 
num_images = 100
rays_per_image = 500
Nc = 300
Nf = 128
tn, tf = 2.0, 6.0

device = 'cuda'
coarse_nerf = NeRF()
# fine_nerf = NeRF()
# summary(nerf, input_size=((rays_per_image, Nc, 3), (rays_per_image, Nc, 3)), depth=4, col_names=( "input_size", "output_size", "num_params"))
coarse_nerf.to(device)
# fine_nerf.to(device)

criterion = torch.nn.MSELoss()

lr = 5e-4
optimizer = torch.optim.AdamW(coarse_nerf.parameters(), lr=lr)
# optimizer = torch.optim.AdamW(list(fine_nerf.parameters()) + list(coarse_nerf.parameters()), lr=lr)

gamma = 0.1**(1/epochs)
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=epochs)

train_dset = SyntheticDataloader("data/nerf_synthetic/", "hotdog", "train")
val_dset = SyntheticDataloader("data/nerf_synthetic/", "hotdog", "val")
assert len(train_dset) == len(val_dset)

# max_t = max(np.linalg.norm(train_dset.max_t), np.linalg.norm(val_dset.max_t))
# for i in range(len(train_dset)):
#     train_dset.transforms[i][:3, 3] /= max_t
#     val_dset.transforms[i][:3, 3] /= max_t

fig, ax = plt.subplots(2, 2)
for a in ax.ravel():
    a.set_xticks([])
    a.set_yticks([])

# TODO pass in pixels so u can run test and train
def render(dset, nerf, pose, u, v, rays, w=None, tin=None):
    '''
    u  (rays,)
    v  (rays,)
    '''
    
    o, d = dset.create_rays(pose[:3, :3], pose[None, :3, 3]) # (H, W, 3), (H, W, 3)
    
    '''
    d[u, v] or o[u, v]
    (rays @ pxs (u, v), trans_vec)
    '''
    
    if type(w) == type(None):
        t = stratified_sampling_rays(tn=tn, tf=tf, N=Nc, rays=rays) # (rays, Nc samples along ray)
        
        x = o[u, None, v] + t[..., None]*d[u, None, v] # (rays, Nc samples, xyz)
        d = d[u, None, v].expand(-1, Nc, -1) # arays, Nc samples, xyz)
    else:
        pdf = w / torch.sum(w, axis=1, keepdim=True)
        sample_idx = torch.multinomial(pdf, num_samples=Nf, replacement=True)
        t = torch.gather(tin, -1, sample_idx)
        
        x = o[u, None, v] + t[..., None]*d[u, None, v]
        d = d[u, None, v].expand(-1, Nf, -1)
    
    #TODO check
    sigma, c = nerf(x, d)
    
    c_hat, wi = c_pred(sigma.squeeze(-1), c, t)
    
    return c_hat, wi.squeeze(-1), t

random_images = np.arange(0, len(train_dset), 1)
# random_images = [0, 0] # np.arange(0, len(train_dset), 1)
for i in tqdm.tqdm(range(epochs), desc=f"[Training]", leave=True):
# for i in range(epochs):
    
    # random_images = np.random.randint(0, len(train_dset), num_images)
    np.random.shuffle(random_images)
    
    images, poses = train_dset.images[random_images], train_dset.transforms[random_images]
    
    t_loss = 0
    for pose, image in tqdm.tqdm(zip(poses, images), desc=f"[Train {i}] Rendering", total=100):
    # for idx, (pose, image) in enumerate(zip(poses, images)):
        
        u, v = np.random.randint(0, train_dset.img_shape[0], size=(2, rays_per_image))
        
        optimizer.zero_grad()
        
        # print(20*"=", idx, 20*"=")   
        coarse_c_hat, wi, ti = render(train_dset, coarse_nerf, pose, u, v, rays_per_image)
        # fine_c_hat, _, _ = render(train_dset, fine_nerf, pose, u, v, rays_per_image, wi, ti)
        c_true = torch.tensor(image[u, v], dtype=torch.float32, device=device)
        
        # print("Zero pred c?: ", torch.all(c_hat == 0).item())
        # if torch.all(c_hat == 0).item() == True:
            # tqdm.tqdm.write("[Warning] predicted blank pixels...")
        
        # print(c_hat.detype, c_true.dtype)
        loss = criterion(coarse_c_hat, c_true)# + criterion(fine_c_hat, c_true)
        # loss = criterion(c_true, c_true)
        loss.backward()
        optimizer.step()
        
        t_loss += loss.item()
        
    tqdm.tqdm.write(f"[Train {i}] Loss: {t_loss:.4f}")
    writer.add_scalar("loss/train/", t_loss, i)
    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], i)
    lr_scheduler.step()
    
    if i % 100 == 0 and i > 0:
        
        vpose, vimage = val_dset.transforms[0], val_dset.images[0]
        
        # rn = np.random.randint(0, 100)
        rn = 0 #np.random.randint(0, 100)
        tpose, timage = train_dset.transforms[rn], train_dset.images[rn]
        
        vimage_est = np.zeros_like(image)
        timage_est = np.zeros_like(image)
        
        bundle = 800
        step = bundle // train_dset.img_shape[0]
        for row in tqdm.tqdm(range(0, image.shape[0], step), desc="[Val] Rendering Estimate"):
        # for row in range(image.shape[0]):
            for chunk in range(0, image.shape[1], bundle):
                
                with torch.no_grad():
                    
                    if step <= 1:
                        v = np.arange(chunk, chunk+bundle)
                        u = np.ones_like(v) * row
                    else:
                        v = np.tile(np.arange(train_dset.img_shape[0]), (step, 1))
                        u = np.ones_like(v) * (np.arange(step)+row)[:, None]
                        
                        u = u.flatten()
                        v = v.flatten()
                    
                    vc_hat, wi, ti = render(train_dset, coarse_nerf, vpose, u, v, bundle)
                    # vc_hat, _, _ = render(val_dset, fine_nerf, vpose, u, v, bundle)
                    
                    tc_hat, wi, ti = render(train_dset, coarse_nerf, tpose, u, v, bundle)
                    # tc_hat, _, _ = render(train_dset, fine_nerf, tpose, u, v, bundle)
                    
                    vimage_est[u, v] = vc_hat.cpu().numpy()
                    timage_est[u, v] = tc_hat.cpu().numpy()
                    # print("Zero output?: ", np.all(c_hat.cpu().numpy() == 0))

        ax[0, 0].imshow(vimage)
        ax[0, 1].imshow(vimage_est)
        ax[1, 0].imshow(timage)
        ax[1, 1].imshow(timage_est)
        
        plt.savefig(f"data/model_out/{exp_name}/image_estimate_e{i}.png")

        v_loss = np.mean(np.power(image - vimage_est, 2))
        tqdm.tqdm.write(f"[Val {i}] Loss: {v_loss:0.4f}")
        tqdm.tqdm.write(f"[Val {i}] Estimate Empty: {np.all(vimage_est == 0)}")
        
        writer.add_image("val/image0/", vimage_est, i, dataformats="HWC")
        writer.add_scalar("loss/val/", v_loss, i)

    writer.flush()    
    