import torch
from dataloader import SyntheticDataloader
from render_utils import c_pred, stratified_sampling_rays
import matplotlib.pylab as plt 
import numpy as np
import tqdm
from model import NeRF
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/initial_runs/")
#
epochs = int(100000)
# select a varies amount of images (1/2 the images at a time)
# select a number of rays that goes cleanly into the 
# num_images = 50
rays_per_image = 800
Nc = 64
tn, tf = 0.5, 3

device = 'cuda'
nerf = NeRF()
# summary(nerf, input_size=((rays_per_image, Nc, 3), (rays_per_image, Nc, 3)), depth=4, col_names=( "input_size", "output_size", "num_params"))
nerf.to(device)

criterion = torch.nn.MSELoss()

lr = 1e-4
optimizer = torch.optim.Adam(nerf.parameters(), lr=lr)

gamma = 0.1**(1/epochs)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

train_dset = SyntheticDataloader("data/nerf_synthetic/", "hotdog", "train")
val_dset = SyntheticDataloader("data/nerf_synthetic/", "hotdog", "val")

fig, ax = plt.subplots(1, 2)
ax[0].set_xticks([])
ax[1].set_xticks([])
ax[0].set_yticks([])
ax[1].set_yticks([])

# TODO pass in pixels so u can run test and train
def render(dset, pose, u, v):
    
    o, d = dset.create_rays(pose[:3, :3], pose[:3, 3][None, ...])
    
    t = stratified_sampling_rays(tn=tn, tf=tf, N=Nc, rays=rays_per_image)
    
    x = torch.tensor(o[u, v][:, None, :] + t[..., None]*d[u, v][:, None, :], dtype=torch.float32, device=device)
    d = torch.tensor(d[u, v][:, None, :], dtype=torch.float32, device=device).expand(-1, Nc, -1)
    
    #TODO check
    sigma, c = nerf(x, d)
    c_hat = c_pred(sigma, c, torch.tensor(t, dtype=torch.float32, device=device))
    
    # print("Zero Sigma?: ", torch.all(sigma == 0).item())
    # print("Zero c?: ", torch.all(c == 0).item())
    
    return c_hat

random_images = np.arange(0, len(train_dset), 1)
for i in tqdm.tqdm(range(epochs), desc=f"[Training]", leave=True):
# for i in range(epochs):
    
    # random_images = np.random.randint(0, len(train_dset), num_images)
    np.random.shuffle(random_images)
    
    images, poses = train_dset.images[random_images], train_dset.transforms[random_images]
    
    t_loss = 0
    for pose, image in tqdm.tqdm(zip(poses, images), desc=f"Train {i}] Rendering ", total=100):
    # for idx, (pose, image) in enumerate(zip(poses, images)):
        
        u, v = np.random.randint(0, train_dset.img_shape[0], size=(2, rays_per_image))
        
        optimizer.zero_grad()
        
        # print(20*"=", idx, 20*"=")   
        c_hat = render(train_dset, pose, u, v)
        c_true = torch.tensor(image[u, v], dtype=torch.float32, device=device)
        
        # print("Zero pred c?: ", torch.all(c_hat == 0).item())
        if torch.all(c_hat == 0).item() == True:
            print("[Warning] predicted blank pixels...")
        
        # print(c_hat.detype, c_true.dtype)
        loss = criterion(c_hat, c_true)
        loss.backward()
        optimizer.step()
        
        t_loss += loss.item()
        
    tqdm.tqdm.write(f" [Train {i}] Loss: {t_loss:.3f}")
    writer.add_scalar("loss/train/", t_loss, i)
    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], i)
    lr_scheduler.step()
    
    if i % 100 == 0:
        pose, image = val_dset.transforms[0], val_dset.images[1]
        
        image_est = np.zeros_like(image)
        for row in tqdm.tqdm(range(image.shape[0]), desc="[Val] Rendering Estimate"):
        # for row in range(image.shape[0]):
            for chunk in range(0, image.shape[1], rays_per_image):
                
                with torch.no_grad():
                    u = np.arange(chunk, chunk+rays_per_image)
                    v = np.ones_like(u) * row
                    c_hat = render(val_dset, pose, u, v)
                    
                    image_est[u, v] = c_hat.cpu().numpy()
                    # print("Zero output?: ", np.all(c_hat.cpu().numpy() == 0))

        ax[0].imshow(image)
        ax[1].imshow(image_est)
        
        plt.savefig(f"data/model_out/image_estimate_e{i}.png")

        tqdm.tqdm.write(f"[Val {i}] Rendered Estimate")
        tqdm.tqdm.write(f"[Val {i}] Estimate Empty: {np.all(image_est == 0)}")
        
        writer.add_image("val/image0/", image_est, i, dataformats="HWC")

    writer.flush()    
    