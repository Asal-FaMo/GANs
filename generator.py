import torch
import torch.nn as nn

class View(nn.Module):
    """reshape utility for nn.Sequential"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

class Generator(nn.Module):
    """
    MNIST DCGAN Generator 

    Layer 1:  Linear(z_dim -> 7*7*128) + BN1d + ReLU
    Layer 2:  ConvTranspose2d(128 -> 64,  k=4,s=2,p=1)  -> 14x14x64 + BN2d + ReLU
    Layer 3:  ConvTranspose2d(64  -> 32,  k=4,s=2,p=1)  -> 28x28x32 + BN2d + ReLU
    Layer 4:  ConvTranspose2d(32  -> 1,   k=4,s=1,p=1)  -> 28x28x1  + Tanh    (بدون BN)
    """
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            # Dense -> (7,7,128)
            nn.Linear(z_dim, 7 * 7 * 128),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.ReLU(inplace=True),

            View((-1, 128, 7, 7)),

            # 7 -> 14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 14 -> 28
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 28 -> 28 (same size), بدون BatchNorm
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

def init_dcgan_weights(m):
    """DCGAN init: Normal(0,0.02) """
    classname = m.__class__.__name__
    if 'Conv' in classname:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif 'BatchNorm' in classname:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif 'Linear' in classname:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

if __name__ == "__main__":
    
    G = Generator(100)
    G.apply(init_dcgan_weights)
    z = torch.randn(4, 100)
    out = G(z)
    print("Generator out:", out.shape, "min/max:", float(out.min()), float(out.max()))
