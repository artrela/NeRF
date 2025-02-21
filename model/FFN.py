import torch
import torch.nn.functional as F

class FFN(torch.nn.Module):
    def __init__(self, in_feats: int, hid: int, num_layers: int, skips: list):
        """ A FFN to project the position into hidden features, with some skip layers.

        Args:
            in_feats (int): The amount of input features, taken from the positional encoded positions
            hid (int): Dimensions of hidden features to project to
            num_layers (int): Number of total layers
            skips (list): Layers to include a skip at
        """
        super().__init__()
        layers = []
        for layer  in range(num_layers):
            if layer == 0:
                layers.append(torch.nn.Linear(in_feats, hid))
            elif layer in skips:
                layers.append(torch.nn.Linear(hid+in_feats, hid))
            else:
                layers.append(torch.nn.Linear(hid, hid))
            
        self.network = torch.nn.ModuleList(layers)
        self.skips = set(skips)
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        """ Pass the input position x through a ffn, with skips as needed.
        All layers include a rectified nonlinear layer. 

        Args:
            x (torch.Tensor): A tensor of (B, N, PE) a batch of rays with N samples along the 
            ray, the original xyz positions have been encoded to PE dimensions.

        Returns:
            torch.Tensor: A set of hidden features for the positions.
        """
        x0 = x
        for idx, layer in enumerate(self.network):
            if idx in self.skips:
                x = layer(torch.concat((x0, x), dim=-1))
            else:
                x = layer(x)
            
            x = F.relu(x)            

        return x
                