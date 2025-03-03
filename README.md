# <span style="color:rgb(135, 206, 235)">C</span>ontrastive <span style="color:rgb(135, 206, 235)">S</span>parse <span style="color:rgb(135, 206, 235)">R</span>epresentation Learning
Official Code for Paper: **Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation**


In this repo, we will release (**updating**):

- Environment Dependencies &#x2705;
- Checkpoints &#x1F4CC;
- Reproducing Experiments &#x2705;
  - Dataset preparations &#x2705;
  - Configurations 
  - Training &#x2705;
  - Evaluation &#x2705;

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

### Get Pre-trained ImageNet1k Embeddings
For training and evaluation simplicity, we first extract image embeddings using Models from [Timm](https://github.com/huggingface/pytorch-image-models).

In our paper, we select [resnet50d.ra4_e3600_r224_in1k](https://huggingface.co/timm/resnet50d.ra4_e3600_r224_in1k) as our pre-trained visual backbone.
To extract embeddings, run the following command:
```bash
python inference/pretrained_embeddings.py \
      --train_data_ffcv  /path/to/train.ffcv \
      --eval_data_ffcv    /path/to/val.ffcv \
      --model_name "pre-trained visual backbone" \
```

### Train Contrastvie Sparse Representation on Imagenet1K
```bash
python main.py \
      --pretrained_emb /path/to/pretrained_emb \
      --model_name "pre-trained visual backbone" \
      --use_ddp False \     #set True if you want to use multi-GPU
      --gpu 1\              # GPU ID, set None if you use multi-GPU
      --batch-size 1024 * 4 \
      --lr 4e-4 \
      --use_CL True \       #whether to use contrastive learning
      --topk 8 \            #topk for CSR
      --auxk 512 \          #auxiliary sparse code size
      --hidden-size 8192 \  #"By default, 4 * visual backbone embedding size"   
```
### Get CSR Embeddings for 1-NN Evaluation
```bash
python inference/csr_inference.py \
      --train_emb_path  /path/to/train_emb \
      --eval_emb_path    /path/to/val_emb \
      --model_name "pre-trained visual backbone" \
      --topk 8\
      --hidden-size 8192  #"By default, 4 * visual backbone embedding size"
      --cse_ckpt "CSR ckpt path"\
```

### Get Evaluation Results
We use [FAISS](https://github.com/facebookresearch/faiss) for KNN evaluation and calculate Top1 Accuracy under different sparsity conditions.
Note that we follow the pipeline of [MRL](https://github.com/RAIVNLab/MRL/tree/main/retrieval) for fair comparision.
```bash
cd retrieval
# Get FAISS index
python faiss_nn.py --topk 8
# Evaluate Top1 accuracy
python compute_metrics.py --topk 8
```
