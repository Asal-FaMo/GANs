import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    MNIST DCGAN Discriminator (مطابق سند پروژه)

    Conv1: (1 -> 64),  kernel=4, stride=2, padding=1   -> (14x14x64),  LeakyReLU(0.2), بدون BN
    Conv2: (64 -> 128), kernel=4, stride=2, padding=1   -> (7x7x128),  BN + LeakyReLU(0.2)
    Conv3: (128 -> 256), kernel=4, stride=2, padding=1  -> (3x3x256),  BN + LeakyReLU(0.2)
    Flatten -> Dense(2304 -> 1024) + BN + LeakyReLU
             -> Dense(1024 -> 1)   (logit)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Conv1: بدون BN
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),   # 28->14
            nn.LeakyReLU(0.2, inplace=True),

            # Conv2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 14->7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 7->3
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # 256*3*3 = 2304
            nn.Linear(256 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),                           # logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

def init_dcgan_weights(m):
    """DCGAN init: Normal(0,0.02) برای Conv/Linear، (1,0.02) برای BN-weights."""
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
    # Self-test
    D = Discriminator()
    D.apply(init_dcgan_weights)
    x = torch.randn(4, 1, 28, 28)   # batch of fake MNIST
    out = D(x)
    print("Discriminator out:", out.shape)  # باید [4,1] باشه
