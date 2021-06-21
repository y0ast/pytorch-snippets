from os.path import join
from torchvision import datasets, transforms


def get_imagenet(root, dogs):
    input_size = 224
    num_classes = 118

    if dogs:
        root = join(root, "imagenet_dogs/")
    else:
        root = join(root, "imagenet_notdogs/")

    traindir = join(root, "train/")
    valdir = join(root, "val/")

    normalize = transforms.Normalize(
        # Standard ImageNet preprocessing, not specialized for Dogs
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    return input_size, num_classes, train_dataset, val_dataset


def get_imagenet_dogs(root):
    return get_imagenet(root, True)


def get_imagenet_notdogs(root):
    return get_imagenet(root, False)
