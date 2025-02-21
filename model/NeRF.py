from typing import Tuple
import torch
import torch.nn.functional as F

from model.FFN import FFN
from model.PositionalEncoding import PositionalEncoding

class NeRF(torch.nn.Module):
    def __init__(self, 
            x_pe: int,
            d_pe: int,
            pe_include_inp: bool,
            ffn_layers: int,
            ffn_skips: list,
            hid_dim: int,
        ):
        super().__init__()
        
        x_pe_out = x_pe * 2 * 3 # pe_dim * len([sin, cos]) * xyz 
        d_pe_out = d_pe * 2 * 3 # pe_dim * len([sin, cos]) * xyz
        
        if pe_include_inp: # prepend original vals to list
            x_pe_out += 3
            d_pe_out += 3
        
        self.model = torch.nn.ModuleDict({
            "x_pe": PositionalEncoding(x_pe, pe_include_inp),
            "d_pe": PositionalEncoding(d_pe, pe_include_inp),
            "ffn": FFN(x_pe_out, hid_dim, num_layers=ffn_layers, skips=ffn_skips,),
            "sigma_layer": torch.nn.Linear(hid_dim, 1),
            "feature_layer": torch.nn.Linear(hid_dim + d_pe_out, hid_dim // 2),
            "rgb": torch.nn.Linear(hid_dim // 2, 3)
        })
        
    def forward(self, x: torch.Tensor, d: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        """ Pass the positon and direction vectors through the MLP, creating an estimated
        density and color at each position.

        Args:
            x (torch.Tensor): (B, N, 3) A set of N samples along B rays, for the dim(3) pos vector
            d (torch.Tensor): (B, N, 3) A set of N samples along B rays, for the dim(3) dir vector

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        gamma_x = self.model["x_pe"](x) # (rays, samples, 3) -> (rays, samples, x_pe_out)
        hidden_feats = self.model["ffn"](gamma_x) # (rays, samples, x_pe_out) -> (rays, samples, hid_dim)
        
        # (rays, samples, hid_dim) -> (rays, samples, 1)
        sigma = F.relu(self.model["sigma_layer"](hidden_feats))
        
        gamma_d = self.model["d_pe"](d) # (rays, samples, 3) -> (rays, samples, d_pe_out)
        
        # (rays, samples, hid_dim + d_pe_out) -> (rays, samples, hid_dim // 2)
        rgb = F.relu(self.model["feature_layer"](torch.cat((hidden_feats, gamma_d), dim=-1))) 
        
        # (rays, samples, hid_dim // 2) -> (rays, samples, 3)
        rgb = F.sigmoid(self.model["rgb"](rgb))
        
        return sigma, rgb