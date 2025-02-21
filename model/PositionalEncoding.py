import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, L: int, include_input: bool):
        super().__init__()
        
        self.include_input = include_input
        _map = torch.tensor([[(2**i) * torch.pi, (2**i) * torch.pi] for i in range(L)], dtype=torch.float32).flatten()
        self.register_buffer('map', _map)
                
    def forward(self, x: torch.Tensor)->torch.Tensor:
        """Generate positional encoding for tensor x.
        Args:
            x (torch.Tensor): (BxNx3) The input vector x (camera position) or d (camera direction)

        Returns:
            torch.Tensor: (BxNx2L or 2L+3) A higher dimensional mapping of the input vector
        """
        
        b, r, _ = x.shape
        gamma_x = torch.reshape(x[..., None] * self.map, shape=(b, r, -1))
        gamma_x[...,  ::2] = torch.sin(gamma_x[...,  ::2])
        gamma_x[..., 1::2] = torch.cos(gamma_x[..., 1::2])
        
        if self.include_input:
            gamma_x = torch.concat((x, gamma_x), dim=-1)
        
        return gamma_x