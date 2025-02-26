# <span style="color:rgb(135, 206, 235)">C</span>ontrastive <span style="color:rgb(135, 206, 235)">S</span>parse <span style="color:rgb(135, 206, 235)">R</span>epresentation Learning
Official Code for Paper: **Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation**

## Set Up
Pip install the requirements file in this directory. Note that a python3 distribution is required:
```
pip3 install -r requirements.txt
```

### Preparing the Dataset
Following the ImageNet training pipeline of [FFCV](https://github.com/libffcv/ffcv-imagenet) for ResNet50, generate the dataset with the following command (`IMAGENET_DIR` should point to a PyTorch style [ImageNet dataset](https://github.com/MadryLab/pytorch-imagenet-dataset)):

```bash
# Required environmental variables for the script:
cd train/
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/

# Serialize images with:
# - 500px side length maximum
# - 50% JPEG encoded, 90% raw pixel values
# - quality=90 JPEGs
./write_imagenet.sh 500 0.50 90
```

### Get Pretrained ImageNet1k Embeddings
For training and evaluation simplicity, we first extract image embeddings using Models from [Timm](https://github.com/huggingface/pytorch-image-models).

In our paper, we select [resnet50d.ra4_e3600_r224_in1k](https://huggingface.co/timm/resnet50d.ra4_e3600_r224_in1k) as our pre-trained visual backbone.
To extract embeddings, run the following command:
```bash
python inference/pretrained_embeddings.py \
      --train_data_ffcv  /path/to/train.ffcv \
      --eval_data_ffcv    /path/to/val.ffcv \
      --model_name "pre-trained visual backbone" \
```
