import torch
from typing import Optional
import numpy as np
import torch.nn.functional as F

#TODO Add more flexibility, better documentation to these classes
class PositionalEncoding(torch.nn.Module):
    def __init__(self, L: int):
        super().__init__()
        
        self.L = L
        _map = np.array([[(2**i) * np.pi, (2**i) * np.pi] for i in range(L)]).flatten()
        self.register_buffer('map', torch.tensor(_map, dtype=torch.float32))
                
    def forward(self, x: torch.Tensor)->torch.Tensor:
        """Generate positional encoding for tensor x.
        Args:
            x (torch.Tensor): (BxNx3) The input vector x (camera position) or d (camera direction)

        Returns:
            torch.Tensor: (BxNx2L) A higher dimensional mapping of the input vector
        """
        
        b, r, xyz = x.shape
        gamma_x = torch.reshape(x[..., None] * self.map, shape=(b, r, self.L*xyz*2))
        gamma_x[...,  ::2] = torch.sin(gamma_x[...,  ::2])
        gamma_x[..., 1::2] = torch.cos(gamma_x[..., 1::2])
        
        return gamma_x
        

class FFN(torch.nn.Module):
    def __init__(self, in_feats: int, num_layers: int, skips: list, out_feats: Optional[int]=None):
        super().__init__()
        
        layers = []
        for layer  in range(num_layers):
            if layer == 0:
                layers.append(torch.nn.Linear(in_feats, 256))
            elif layer in skips:
                layers.append(torch.nn.Linear(256+in_feats, 256))
            elif out_feats and layer == num_layers-1:
                layers.append(torch.nn.Linear(256, out_feats))
            else:
                layers.append(torch.nn.Linear(256, 256))
            
        self.network = torch.nn.ModuleList(layers)
        self.skips = skips
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        
        inp = x.clone()
        for idx, layer in enumerate(self.network):
            
            if idx not in self.skips:
                x = layer(x)
            else:
                x = layer(torch.concat((x, inp), dim=-1))
                
            if idx != len(self.network)-1:
                x = F.relu(x)
                
        return x
                
        

class NeRF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = torch.nn.ModuleDict({
            "x_pe": PositionalEncoding(10),
            "d_pe": PositionalEncoding(4),
            "ffn": FFN(in_feats=60, num_layers=8, skips=[4]),
            "sigma_layer": torch.nn.Linear(in_features=256, out_features=1),
            "proj": torch.nn.Linear(in_features=256+24, out_features=128),
            "out": torch.nn.Linear(in_features=128, out_features=3)
        })
        
    def forward(self, x: torch.Tensor, d: torch.Tensor):
        '''
        x.shape = rays, samples, xyz
        d.shape = rays, samples, dir
        '''
        
        # rays, samples, embedding_size (60)
        gamma_x = self.model["x_pe"](x) 

        # rays, samples, 256
        hidden_feats = self.model["ffn"](gamma_x) 
        
        # rays, samples
        sigma = F.relu(self.model["sigma_layer"](hidden_feats).squeeze(-1))
        
        # # rays, samples, embedding_size (24)
        gamma_d = self.model["d_pe"](d) 
        
        # rays, samples, 128
        rgb = F.relu(self.model["proj"](torch.cat((hidden_feats, gamma_d), dim=-1))) 
        
        # rays, samples, rgb
        rgb = F.sigmoid(self.model["out"](rgb))
        
        return sigma, rgb