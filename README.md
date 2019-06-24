# chainer-fm-gan

This is a re-implimentation of [FM-GAN](https://github.com/LiqunChen0606/FM-GAN)
using chainer.

## Requirements:

* Python 3.5+
* Chainer 5.2+
* wget, nkf, mecab (for preprocessing dataset)

## How to run

### Dataset

data/ directory contains scripts to download the Livedoor news corpus
and preprocess them.

Just type the following:

```
cd data && make all
```

### Train

```
export GPU=0 # if you want to use GPU
./train.sh # pretrain LSTM-VAE
./gantrain.sh # GAN training
```

### License

3-clause BSD
