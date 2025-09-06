import os, math, torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

class GANTrainer:
    """
    آموزش DCGAN برای MNIST با Loss های گفته‌شده در پروژه (BCE):
      - Discriminator:  L_D = -1/2 [ log D(x) + log (1 - D(G(z))) ]
      - Generator:      L_G = - log D(G(z))
    به‌صورت عملی از BCEWithLogitsLoss استفاده می‌کنیم (خروجی D لاجیت است).
    """

    def __init__(
        self,
        G, D,
        z_dim=100,
        lr=2e-4, betas=(0.5, 0.999),
        device=None,
        out_dir="./outputs",
        label_smoothing=0.9,   # real=0.9 (one-sided smoothing)
        flip_labels_p=0.0,     # احتمال کوچک برعکس‌کردن برچسب‌ها (به‌طور پیش‌فرض خاموش)
        d_steps=1,             # در صورت قوی‌بودن G می‌توان D را بیشتر آپدیت کرد و بلعکس
        save_every=1,          # ذخیره‌ی تصاویر هر چند epoch
        ckpt_every=5,          # ذخیره‌ی چک‌پوینت
    ):
        self.G, self.D = G, D
        self.z_dim = z_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.G.to(self.device); self.D.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.g_opt = optim.Adam(self.G.parameters(), lr=lr, betas=betas)
        self.d_opt = optim.Adam(self.D.parameters(), lr=lr, betas=betas)

        self.fixed_noise = torch.randn(64, z_dim, device=self.device)  # برای لاگ ثابت
        self.out_dir = out_dir
        self.img_dir = os.path.join(out_dir, "samples"); os.makedirs(self.img_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(out_dir, "checkpoints"); os.makedirs(self.ckpt_dir, exist_ok=True)

        # ترفندهای پایداری
        self.label_smoothing = label_smoothing
        self.flip_labels_p = flip_labels_p
        self.d_steps = d_steps
        self.save_every = save_every
        self.ckpt_every = ckpt_every

    @torch.no_grad()
    def _save_samples(self, epoch):
        self.G.eval()
        fake = self.G(self.fixed_noise).detach().cpu()
        # نرمال‌سازی به [-1,1] مطابق خروجی Tanh
        save_image(fake, os.path.join(self.img_dir, f"epoch_{epoch:03d}.png"),
                   nrow=8, normalize=True, range=(-1, 1))
        self.G.train()

    def _maybe_flip(self, y):
        """برعکس‌کردن تصادفی برچسب‌ها برای کمی نویز (اختیاری)."""
        if self.flip_labels_p <= 0: return y
        flip_mask = (torch.rand_like(y) < self.flip_labels_p).float()
        return (1.0 - y) * flip_mask + y * (1.0 - flip_mask)

    def train(self, loader, epochs=50):
        for epoch in range(1, epochs + 1):
            d_loss_epoch, g_loss_epoch = 0.0, 0.0

            for real, _ in loader:
                real = real.to(self.device)
                bsz = real.size(0)

                # ---------- Train Discriminator ----------
                for _ in range(self.d_steps):
                    self.D.zero_grad(set_to_none=True)

                    # real labels (با smoothing)
                    y_real = torch.full((bsz, 1), self.label_smoothing, device=self.device)
                    y_real = self._maybe_flip(y_real)  # اختیاری

                    # fake labels
                    y_fake = torch.zeros((bsz, 1), device=self.device)

                    # D(real)
                    logits_real = self.D(real)
                    loss_real = self.criterion(logits_real, y_real)

                    # D(fake) با G(z) جدا شده از گرادیان
                    z = torch.randn(bsz, self.z_dim, device=self.device)
                    fake = self.G(z).detach()
                    logits_fake = self.D(fake)
                    loss_fake = self.criterion(logits_fake, y_fake)

                    d_loss = 0.5 * (loss_real + loss_fake)  # مطابق فرمول سند (میانگین‌گیری با 1/2)
                    d_loss.backward()
                    self.d_opt.step()

                d_loss_epoch += d_loss.item()

                # ---------- Train Generator ----------
                self.G.zero_grad(set_to_none=True)
                z = torch.randn(bsz, self.z_dim, device=self.device)
                fake = self.G(z)
                logits_fake_for_g = self.D(fake)

                # هدف G: D(G(z))≈1  -->  y=1
                y_true = torch.ones((bsz, 1), device=self.device)
                g_loss = self.criterion(logits_fake_for_g, y_true)
                g_loss.backward()
                self.g_opt.step()

                g_loss_epoch += g_loss.item()

            d_loss_epoch /= len(loader)
            g_loss_epoch /= len(loader)
            print(f"[{epoch:03d}/{epochs}] D: {d_loss_epoch:.4f} | G: {g_loss_epoch:.4f}")

            if epoch % self.save_every == 0:
                self._save_samples(epoch)

            if epoch % self.ckpt_every == 0:
                torch.save({
                    "epoch": epoch,
                    "G": self.G.state_dict(),
                    "D": self.D.state_dict(),
                    "g_opt": self.g_opt.state_dict(),
                    "d_opt": self.d_opt.state_dict(),
                }, os.path.join(self.ckpt_dir, f"gan_{epoch:03d}.pt"))
