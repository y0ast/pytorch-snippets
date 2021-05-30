import os

from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

def get_imagenet(root, num_classes=100):
    class SubDatasetFolder(datasets.DatasetFolder):
        def _find_classes(self, dir):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            classes.sort()
            classes = classes[:num_classes]  # overwritten from original to take subset
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            return classes, class_to_idx

    input_size = 224

    traindir = os.path.join(root, "train/")
    valdir = os.path.join(root, "val/")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Reimplements ImageFolder arguments
    train_dataset = SubDatasetFolder(
        traindir,
        default_loader,
        IMG_EXTENSIONS,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    test_dataset = SubDatasetFolder(
        valdir,
        default_loader,
        IMG_EXTENSIONS,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    return train_dataset, test_dataset