## ImageNot

From: ["What classifiers know what they don't?"](https://arxiv.org/abs/2107.06217)

### Create dataset on disk

In a folder with the `train/` and `val/` ImageNet sub-folders:

```
mkdir -p imagenot_in-domain/train
mkdir -p imagenot_in-domain/val
cat imagenot_in-domain.txt | xargs -I{} cp -R train/{} imagenot_in-domain/train/
cat imagenot_in-domain.txt | xargs -I{} cp -R val/{} imagenot_in-domain/val/

mkdir -p imagenot_out-domain/train
mkdir -p imagenot_out-domain/val
cat imagenot_out-domain.txt | xargs -I{} cp -R train/{} imagenot_out-domain/train/
cat imagenot_out-domain.txt | xargs -I{} cp -R val/{} imagenot_out-domain/val/
```

(`val/` can be obtained using [Soumith's valprep.sh script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh))

An example way to use these folders in PyTorch is provided in [dataset.py](dataset.py).
