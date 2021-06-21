This is a collection of PyTorch snippets that have helped me (and others) significantly with doing ML research.

* [Minimal CIFAR-10](minimal_cifar) A very simple training script to get 94% accuracy with just ~150 lines of code, for an easy but strong baseline.
* [FastMNIST](fast_mnist): a drop-in replacement for the standard MNIST dataset which speeds up training on the **GPU** (by 2x !) by avoiding unnecessary preprocessing that pegs the cpu at 100% for small models.
* [Subset of ImageNet](subset_of_imagenet): it's remarkably difficult to train on a subset of the ImageNet classes with the default TorchVision datasets. This snippet makes the minimal changes to them to make it very easy.
* [ImageNet dogs vs not dogs](imagenet_dogs_vs_notdogs): a standardized setup for the ImageNet Dogs vs Not Dogs out-of-distribution detection task.

In a separate repository, [Slurm for ML](https://github.com/y0ast/slurm-for-ml), I explain how I use `slurm` job arrays without pain using a simple 1 file shell script.
