# Fast MNIST

The [PyTorch MNIST dataset](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist) is **SLOW** by default, because it wants to conform to the usual interface of returning a PIL image. This is unnecessary if you just want a normalized MNIST and are not interested in image transforms (such as rotation, cropping). By folding the normalization into the dataset initialization you can **save your CPU and speed up training by 2-3x**.

The bottleneck when training on MNIST with a GPU and a small-ish model is **the CPU**. In fact, even with six dataloader workers on a six core i7, the GPU utilization is only ~5-10%. Using FastMNIST increases GPU utilization to ~20-25% and reduces CPU utilization to near zero. On my particular model the steps per second with batch size 64 went from ~150 to ~500.

Instead of the default MNIST dataset, use this:

```
import torch
from torchvision.datasets import MNIST

device = torch.device('cuda')

class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target
```

And call the dataloader like this:

```
from torch.utils.data import DataLoader

train_dataset = FastMNIST('data/MNIST', train=True, download=True)
test_dataset = FastMNIST('data/MNIST', train=False, download=True)

# num_workers=0 is very important!
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=0)
```

Results in 2-3x speedup (500it/s on a 1080Ti and a smallish MLP), uses near zero CPU (compared to full CPU usage normally).
