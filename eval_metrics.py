import torch
from torchvision import transforms
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

@torch.no_grad()
def compute_is_fid(G, real_loader, z_dim=100, n_fake=10000, device="cpu"):
    # آماده‌سازی مترک‌ها
    is_metric  = InceptionScore(normalize=True).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    # تبدیل: [-1,1] -> [0,1], تکرار کانال، تغییر اندازه به 299
    to_metric = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1) / 2),       # [-1,1] -> [0,1]
        transforms.Lambda(lambda x: x.repeat(1,3,1,1)), # 1ch -> 3ch
        transforms.Resize((299,299)),
    ])

    # 1) عبور داده‌های واقعی برای FID
    for real, _ in real_loader:
        real = real.to(device)
        real = to_metric(real)
        fid_metric.update(real, real=True)

    # 2) تولید نمونه‌های جعلی و آپدیت هر دو مترک
    bs = 128
    done = 0
    while done < n_fake:
        b = min(bs, n_fake - done)
        z = torch.randn(b, z_dim, device=device)
        fake = G(z).to(device)
        fake = to_metric(fake)
        is_metric.update(fake)
        fid_metric.update(fake, real=False)
        done += b

    is_mean, is_std = is_metric.compute()   # IS
    fid = fid_metric.compute()              # FID
    return float(is_mean), float(is_std), float(fid)
