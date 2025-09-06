import torch
from dataset_loader import DatasetLoader
from generator import Generator, init_dcgan_weights as initG
from discriminator import Discriminator, init_dcgan_weights as initD
from gan_trainer import GANTrainer

def main():
    # 1) Data
    dl = DatasetLoader(root="./data", batch_size=128, num_workers=2)
    train_loader = dl.get_train_loader(allow_download=True)

    # 2) Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(z_dim=100).to(device); G.apply(initG)
    D = Discriminator().to(device);      D.apply(initD)

    # 3) Trainer (طبق سند: lr=2e-4, betas=(0.5, 0.999))
    trainer = GANTrainer(
        G, D,
        z_dim=100,
        g_lr=2e-4, d_lr=5e-5, betas=(0.5, 0.999),    #g_lr=2e-4, d_lr=1e-4, betas=(0.5, 0.999),
        device=device,
        out_dir="./outputs",
        label_smoothing=0.9,
        fake_label_soft=0.05,  #0.1
        flip_labels_p=0.02,   #0.05
        d_steps=1,
        g_steps=2,               # اگر لازم شد 2 کن*****yk bod krdm 2
        save_every=1,
        ckpt_every=5,
        inst_noise_sigma=0.02,   #0.05
        inst_noise_anneal=0.90,    #0.98
    )

    # 4) Train
    trainer.train(train_loader, epochs=10)  # برای تست سریع: 5-10 ایپاک

if __name__ == "__main__":
    main()
