# eval_metrics.py
import torch
from torchvision import transforms

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

@torch.inference_mode()
def compute_is_fid(G, real_loader, z_dim=100, n_fake=3000, device="cpu", fake_bs=128):
    """
    IS: expects float in [0,1]
    FID (torch-fidelity backend): expects uint8 in [0,255]
    MNIST is 1ch 28x28 -> convert to 3ch 299x299 for metrics
    """

    # متریک‌ها روی device
    is_metric  = InceptionScore(splits=10, normalize=True).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    # ---- Transforms ----
    # از [-1,1] به [0,1] + 3 کاناله + 299x299 (برای IS)
    to_float_for_is = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1) / 2),             # [-1,1] -> [0,1]
        transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),    # 1ch -> 3ch
        transforms.Resize((299, 299)),
    ])

    # از [-1,1] به uint8 در [0,255] + 3 کاناله + 299x299 (برای FID)
    to_uint8_for_fid = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1) / 2),             # [-1,1] -> [0,1]
        transforms.Lambda(lambda x: (x * 255.0).clamp(0, 255).to(torch.uint8)),
        transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),    # 1ch -> 3ch
        transforms.Resize((299, 299)),
    ])

    # 1) عبور real برای FID (uint8)
    for real, _ in real_loader:
        real = real.to(device)                # real: [-1,1] float
        real_u8 = to_uint8_for_fid(real)      # uint8 [0,255], 3ch, 299
        fid_metric.update(real_u8, real=True)

    # 2) تولید fake و آپدیت هر دو مترک
    done = 0
    while done < n_fake:
        b = min(fake_bs, n_fake - done)
        z = torch.randn(b, z_dim, device=device)
        fake = G(z)                               # [-1,1] float

        # IS: float [0,1]
        fake_f = to_float_for_is(fake)
        is_metric.update(fake_f)

        # FID: uint8 [0,255]
        fake_u8 = to_uint8_for_fid(fake)
        fid_metric.update(fake_u8, real=False)

        done += b

    is_mean, is_std = is_metric.compute()
    fid = fid_metric.compute()

    # پاکسازی
    is_metric.reset(); fid_metric.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(is_mean), float(is_std), float(fid)
