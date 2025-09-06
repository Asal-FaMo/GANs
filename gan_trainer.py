import os, math, torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt


class GANTrainer:

    def __init__(
        self,
        G, D,
        z_dim=100,
        g_lr=2e-4, d_lr=2.5e-5, betas=(0.5, 0.999),  # << TTUR      d_lr=1e-4
        device=None,
        out_dir="./outputs",
        label_smoothing=0.9,
        fake_label_soft=0.05,        # << soft fake          fake_label_soft=0.1, 
        flip_labels_p=0.02,         # << روشن                 flip_labels_p=0.05
        d_steps=1,
        g_steps=2,                  # << قابل افزایش به 2*********one to two*******************
        save_every=1,
        ckpt_every=5,
        inst_noise_sigma=0.02,      # << instance noise شروع               inst_noise_sigma=0.05,
        inst_noise_anneal=0.90,     # << هر ایپاک ضربدر این                    inst_noise_anneal=0.98,
        save_epochs=(1,25,50,100), 
        log_filename="training.log",

    ):
        self.G, self.D = G, D
        self.z_dim = z_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.G.to(self.device); self.D.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.g_opt = optim.Adam(self.G.parameters(), lr=g_lr, betas=betas)
        self.d_opt = optim.Adam(self.D.parameters(), lr=d_lr, betas=betas)

        self.fixed_noise = torch.randn(64, z_dim, device=self.device)  # برای لاگ ثابت
        self.out_dir = out_dir
        self.img_dir = os.path.join(out_dir, "samples"); os.makedirs(self.img_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(out_dir, "checkpoints"); os.makedirs(self.ckpt_dir, exist_ok=True)

        # ترفندهای پایداری
        self.fake_label_soft = fake_label_soft
        self.flip_labels_p = flip_labels_p
        self.d_steps = d_steps
        self.g_steps = g_steps
        self.inst_noise_sigma = inst_noise_sigma
        self.inst_noise_anneal = inst_noise_anneal
        self.save_every = save_every
        self.ckpt_every = ckpt_every
        self.label_smoothing = label_smoothing
#_____________3rd
        # self.save_epochs = set(save_epochs)
        # self.history = {"d": [], "g": []}
        # self.log_path = os.path.join(self.out_dir, log_filename)
        # os.makedirs(self.out_dir, exist_ok=True)
        self.save_epochs = set(save_epochs)
        self.history = {"d": [], "g": [], "g_lr": [], "d_lr": []}
        self.log_path = os.path.join(self.out_dir, log_filename)
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[Logger] Writing log to: {os.path.abspath(self.log_path)}")  # اضافه کن__________________________________________
    # هدر لاگ
        with open(self.log_path, "w") as f:
            f.write("epoch | d_loss | g_loss | g_lr | d_lr \n")
            f.write("---------------------------------------------------------------\n")


    @torch.no_grad()
    def _save_samples(self, epoch):
        self.G.eval()
        fake = self.G(self.fixed_noise).detach().cpu()
        # امن‌تر: مطمئن شو داخل بازه‌ی [-1,1] است
        fake = fake.clamp_(-1, 1)
        save_image(
            fake,
            os.path.join(self.img_dir, f"epoch_{epoch:03d}.png"),
            nrow=8,
            normalize=True,
            value_range=(-1, 1)  # <<— به جای range
        )
        self.G.train()


    def _maybe_flip(self, y):
        if self.flip_labels_p <= 0: return y
        flip_mask = (torch.rand_like(y) < self.flip_labels_p).float()
        return (1.0 - y) * flip_mask + y * (1.0 - flip_mask)

    def _add_instance_noise(self, x, sigma):
        if sigma <= 0: return x
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp_(-1, 1)  # چون ورودی‌ها [-1,1] نرمال شده‌اند
    #__________________________________________________________new one
    def _current_lrs(self):
        g_lr = self.g_opt.param_groups[0]["lr"]
        d_lr = self.d_opt.param_groups[0]["lr"]
        return g_lr, d_lr

    def _log_epoch(self, epoch, epochs, d_loss_epoch, g_loss_epoch):
        g_lr, d_lr = self._current_lrs()
        line = f"[{epoch:03d}/{epochs}] D: {d_loss_epoch:.4f} | G: {g_loss_epoch:.4f} | g_lr={g_lr:.6f} | d_lr={d_lr:.6f} "
        # فایل .log
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
    # ترمینال
        print(line)
    #________________________________________________________________
    def train(self, loader, epochs=100):
        for epoch in range(1, epochs + 1):
            self.G.train(); self.D.train() #_____________________________2nd chang: it wasnt
            d_loss_epoch, g_loss_epoch = 0.0, 0.0

            for real, _ in loader:
                real = real.to(self.device)
                bsz = real.size(0)

                # ---------- Train Discriminator ----------
                d_loss_sum = 0.0
                for _ in range(self.d_steps):
                    self.D.zero_grad(set_to_none=True)

                    # real labels (با smoothing)
                    y_real = torch.full((bsz, 1), self.label_smoothing, device=self.device)
                    y_real = self._maybe_flip(y_real)  # اختیاری

                    # fake labels
                    y_fake = torch.full((bsz,1), self.fake_label_soft, device=self.device)  # 0.1

                    # D(real)
                    real_noisy = self._add_instance_noise(real, self.inst_noise_sigma)
                    logits_real = self.D(real_noisy)
                    loss_real = self.criterion(logits_real, y_real)

                    # D(fake) با G(z) جدا شده از گرادیان
                    z = torch.randn(bsz, self.z_dim, device=self.device)
                    fake = self.G(z).detach()
                    fake_noisy = self._add_instance_noise(fake, self.inst_noise_sigma)
                    logits_fake = self.D(fake_noisy)
                    loss_fake = self.criterion(logits_fake, y_fake)


                    d_loss = 0.5 * (loss_real + loss_fake)  # مطابق فرمول سند (میانگین‌گیری با 1/2)
                    d_loss.backward()
                    self.d_opt.step()
                    d_loss_sum += d_loss.item() #_____________________2nd chng: wasnt 

                d_loss_epoch += d_loss_sum / self.d_steps   #______________________2nd chng: d_loss_epoch += d_loss.item()

                # ---------- Train Generator ----------
                g_loss_sum = 0.0
                for _ in range(self.g_steps):
                  self.G.zero_grad(set_to_none=True)
                  z = torch.randn(bsz, self.z_dim, device=self.device)
                  fake = self.G(z)
                  fake_noisy = self._add_instance_noise(fake, self.inst_noise_sigma)
                  logits_fake_for_g = self.D(fake_noisy)
                  y_true = torch.ones((bsz, 1), device=self.device)
                  g_loss = self.criterion(logits_fake_for_g, y_true)
                  g_loss.backward()
                  self.g_opt.step()
                  g_loss_sum += g_loss.item()   #______________________2nd: same as up

                g_loss_epoch += g_loss_sum / self.g_steps       #__________2nd: g_loss_epoch += g_loss.item()
                
            # log_line = f"[{epoch:03d}/{epochs}] D: {d_loss_epoch:.4f} | G: {g_loss_epoch:.4f}\n"
            # with open(self.log_path, "a") as f:
            #     f.write(log_line)

            #print(log_line.strip())  # همان خروجی روی صفحه

            #print(f"[{epoch:03d}/{epochs}] D: {d_loss_epoch:.4f} | G: {g_loss_epoch:.4f}")

            

            d_loss_epoch /= len(loader)
            g_loss_epoch /= len(loader)
            g_lr, d_lr = self._current_lrs()

            
# تاریخچه برای نمودارها
            self.history["d"].append(d_loss_epoch)
            self.history["g"].append(g_loss_epoch)
            self.history["g_lr"].append(g_lr)
            self.history["d_lr"].append(d_lr)

            self.inst_noise_sigma *= self.inst_noise_anneal

            self._log_epoch(epoch, epochs, d_loss_epoch, g_loss_epoch)

            if epoch % self.save_every == 0 or getattr(self, "save_epochs", set()):
                if (epoch % self.save_every == 0) or (epoch in getattr(self, "save_epochs", set())):
                    self._save_samples(epoch)

            if epoch % self.ckpt_every == 0:
                torch.save({
                    "epoch": epoch,
                    "G": self.G.state_dict(),
                    "D": self.D.state_dict(),
                    "g_opt": self.g_opt.state_dict(),
                    "d_opt": self.d_opt.state_dict(),
                }, os.path.join(self.ckpt_dir, f"gan_{epoch:03d}.pt"))
        # --- خارج از حلقهٔ for epoch ---
        plt.figure()
        plt.plot(self.history["d"], label="D loss")
        plt.plot(self.history["g"], label="G loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.savefig(os.path.join(self.out_dir, "loss_curves.png"), bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(self.history["g_lr"], label="G LR")
        plt.plot(self.history["d_lr"], label="D LR")
        plt.xlabel("Epoch"); plt.ylabel("LR"); plt.legend()
        plt.savefig(os.path.join(self.out_dir, "lr_curves.png"), bbox_inches="tight")
        plt.close()

