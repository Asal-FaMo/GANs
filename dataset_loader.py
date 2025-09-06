from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def _build_transform(image_size: int, normalize_to: Tuple[float, float]):
    lo, hi = normalize_to
    if (lo, hi) == (-1.0, 1.0):
        norm = transforms.Normalize((0.5,), (0.5,))  # [0,1] -> [-1,1]
    else:
        raise ValueError("normalize_to باید (-1.0, 1.0) باشد برای سازگاری با Tanh.")
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        norm,
    ])

def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    import random, numpy as np
    random.seed(worker_seed); np.random.seed(worker_seed)

class DatasetLoader:
    """
    استفاده:
        dl = DatasetLoader(root="./data", batch_size=128)
        train_loader = dl.get_train_loader(allow_download=False)  # روی سیستم: دانلود ممنوع
        # در Colab می‌توانی:
        # train_loader = dl.get_train_loader(allow_download=True)
    """
    def __init__(
        self,
        root: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        image_size: int = 28,
        normalize_to: Tuple[float, float] = (-1.0, 1.0),
        pin_memory: Optional[bool] = None, 
        seed: int = 42,
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.normalize_to = normalize_to
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self.seed = seed

        self._g = torch.Generator()
        self._g.manual_seed(self.seed)

        self.transform = _build_transform(self.image_size, self.normalize_to)

    def _build_loader(self, train: bool, allow_download: bool) -> DataLoader:
        try:
            ds = datasets.MNIST(
                root=self.root,
                train=train,
                download=allow_download,  
                transform=self.transform
            )
        except RuntimeError as e:
            raise FileNotFoundError(
                f"MNIST در مسیر '{self.root}' یافت نشد و دانلود غیرفعال است.\n"
                f"راه‌حل‌ها:\n"
                f"  1) دیتاست را از قبل در این مسیر قرار بده (ساختار torchvision).\n"
                f"  2) در Colab یا جایی که اجازه داری، با allow_download=True صدا بزن.\n"
                f"  3) مسیر root را به جایی که MNIST موجود است تغییر بده."
            ) from e

        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True if train else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=train,              
            worker_init_fn=_seed_worker,
            generator=self._g,
        )
        return loader

    def get_train_loader(self, allow_download: bool = True) -> DataLoader:
        return self._build_loader(train=True, allow_download=allow_download)

    def get_test_loader(self, allow_download: bool = True) -> DataLoader:
        return self._build_loader(train=False, allow_download=allow_download)
