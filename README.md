This is a collection of PyTorch snippets that have helped me (and others) significantly with doing ML research.

* A minimal [CIFAR-10 training script](minimal_cifar) to get 94% in ~150 lines, for a simple but strong baseline.
* [FastMNIST](fast_mnist): a drop-in replacement for the standard MNIST dataset which speeds up training on the **GPU** (2x !) by avoiding unnecessary preprocessing.
* [Subset of ImageNet](subset_of_imagenet): it's remarkably difficult to train on a subset of the ImageNet classes with the default TorchVision datasets. This snippet makes the minimal changes to them to make it very easy.

In a [separate repository](https://github.com/y0ast/slurm-for-ml) I explain how I use `slurm` job arrays without pain using a simple 1 file shell script.
