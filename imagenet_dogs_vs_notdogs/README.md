## ImageNet Dogs vs Not Dogs

- [imagenet.txt](imagenet.txt) -> contains all 1000 classes from ImageNet
- [imagenet_dogs.txt](imagenet_dogs.txt) -> contains 118 dog classes in ImageNet (obtained from https://arxiv.org/abs/1606.04080)
- [imagenet_notdogs.txt](imagenet_notdogs.txt) -> contains 882 not dog classes in ImageNet
- [imagenet_notdogs_subset.txt](imagenet_notdogs_subset.txt) -> contains 118 not dog classes in ImageNet (subset of above)

[imagenet.txt](imagenet.txt) created using:

```
ls train > imagenet.txt
```
(train is the folder inside `ILSVRC2017_CLS-LOC.tar.gz`)

[imagenet_notdogs.txt](imagenet_notdogs.txt) created using:

```
comm -3 <(sort imagenet.txt) <(sort imagenet_dogs.txt) > imagenet_notdogs.txt
```

[imagenet_notdogs_subset.txt](imagenet_notdogs_subset.txt) created using:

```
shuf -n 118 imagenet_notdogs.txt > imagenet_notdogs_subset.txt
```

In a folder with the `train/` and `val/` ImageNet sub-folders:

```
mkdir -p imagenet_notdogs/train
mkdir -p imagenet_notdogs/val
cat imagenet_notdogs.txt | xargs -I{} cp -R train/{} imagenet_notdogs/train/
cat imagenet_notdogs.txt | xargs -I{} cp -R val/{} imagenet_notdogs/val/

mkdir -p imagenet_dogs/train
mkdir -p imagenet_dogs/val
cat imagenet_dogs.txt | xargs -I{} cp -R train/{} imagenet_dogs/train/
cat imagenet_dogs.txt | xargs -I{} cp -R val/{} imagenet_dogs/val/
```

(`val/` can be obtained using [Soumith's valprep.sh script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh))

An example way to use these folders in PyTorch is provided in [dataset.py](dataset.py).
