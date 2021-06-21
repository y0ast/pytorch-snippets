## Imagenet Dogs vs Not Dogs

- [imagenet.txt](imagenet.txt) -> contains all 1000 classes from imagenet
- [imagenet_dogs.txt](imagenet_dogs.txt) -> contains 118 dog classes in imagenet (obtained from https://arxiv.org/abs/1606.04080)
- [imagenet_notdogs.txt](imagenet_notdogs.txt) -> contains 882 not dog classes in imagenet
- [imagenet_notdogs_subset.txt](imagenet_notdogs_subset.txt) -> contains 118 not dog classes in imagenet (subset of above)

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
shuf -n 118 imagenet.txt > imagenet_notdogs_subset.txt
```

To copy all folders in one of the files (with val preprocessed using [Soumith's script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)):

```
mkdir -p imagenet_notdogs/train
mkdir -p imagenet_notdogs/val
cat <filename.txt> | xargs -I{} cp -R train/{} imagenet_notdogs/train/
cat <filename.txt> | xargs -I{} cp -R val/{} imagenet_notdogs/val/
```
