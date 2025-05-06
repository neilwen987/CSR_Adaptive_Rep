#  [Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation](http://arxiv.org/abs/2503.01776) (ICML 2025)
By [Tiansheng Wen\*](https://neilwen987.github.io/), [Yifei Wang\*](https://yifeiwang77.com/), [Zequn Zeng](https://joeyz0z.github.io/), Zhong Peng, Yudi Su, [Xinyang Liu](https://xinyangatk.github.io/),
[Bo Chen](https://web.xidian.edu.cn/bchen/en/index.html), Hongwei Liu, [Stefanie Jegelka](https://people.csail.mit.edu/stefje/) and [Chenyu You](https://chenyuyou.me/)

<a href='https://arxiv.org/abs/2503.01776'><img alt="Static Badge" src="https://img.shields.io/badge/Paper-arXiv-red"></a>

![Overview](./assets/overview.jpg)

In this paper, we show that *sparse coding* offers a compelling alternative for achieving adaptive representation with minimal overhead and higher fidelity. We propose **C**ontrastive **S**parse **R**epresentation, a method that sparsifies pre-trained embeddings into a high-dimensional but *selectively activated* feature space. By leveraging lightweight autoencoding and task-aware contrastive objectives, CSR preserves semantic quality while allowing flexible, cost-effective inference at different sparsity levels. Extensive experiments on image, text, and multimodal benchmarks demonstrate that CSR consistently outperforms MRL in terms of both accuracy and retrieval speed-often by large margins-while also cutting training time to a fraction of that required by MRL. Our results establish sparse coding as a powerful paradigm for adaptive representation learning in real-world applications where efficiency and fidelity are both paramount. 

## &#x1F680; &#x1F680; News
- 2025.04.25 Training & Evaluation Pipeline for Multimodal Retrieval is Now Live! We further provide pre-computed ImageNet1k embeddings at [Dataset Link](https://huggingface.co/datasets/W1nd-navigator/CSR-precompute-embeds) for easy follow up! 🙌🙌
- 2025.03.25 Evaluation framework for multimodal retrieval tasks is now online!! &#x1FABC;&#x1FABC;
- 2025.03.07  [Weights](https://drive.google.com/drive/folders/1fI4ip-tcjSrmXtFANmIDTh1wERPlfySO?usp=sharing)
 for visual embeds(k=8 & 32), multimodal embeds(k=64) are now online!! &#x1F601;&#x1F601;
- 2025.03.05  Code released! Let's embrace sparsity!! &#x1F389;&#x1F389;

In this repo, we will release (**updating**):

- Environment Dependencies &#x2705;
- Checkpoints &#x1F4CC;
  - Visual ckpt (on ImageNet)&#x2705;
  - Text ckpt (on partly MTEB datasets) &#x1F4CC;
  - MutilModal ckpt (on MS COCO)&#x2705;
- Reproducing Experiments &#x2705;
  - Visual Exp&#x2705;
    - Dataset preparations &#x2705;
    - Training &#x2705;
    - Evaluation &#x2705;
  - Text Exp &#x1F4CC;
  - MultiModal Exp &#x2705;
    - Dataset preparations &#x2705;
    - Training &#x2705;
    - Evaluation &#x2705;
  - Retrieval Time Evaluation &#x1F4CC;
- Upload CSR on HugginFace &#x1F4CC;

## Set Up
Pip install the requirements file in this directory. Note that a python3 distribution is required:
```
pip3 install -r requirements.txt
```

## Reproduce Visual Exp on Imagenet1k
### Get pre-compute embeds from Huggingface🤗!
We provide embeds extracted by FF2048 backbones (same backbone weights with MRL), and embeds by SoTA backbones at
[Dataset Link](https://huggingface.co/datasets/W1nd-navigator/CSR-precompute-embeds) 

To train CSR with different visual backbones, please follow the preparation steps below.
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
For training and evaluation simplicity, we precompute image embeddings using models from [Timm](https://github.com/huggingface/pytorch-image-models).

In our paper, we select [resnet50d.ra4_e3600_r224_in1k](https://huggingface.co/timm/resnet50d.ra4_e3600_r224_in1k) as our pre-trained visual backbone.
To extract embeddings, run the following command:
```bash
python inference/pretrained_embeddings.py \
      --train_data_ffcv  /path/to/train.ffcv \
      --eval_data_ffcv    /path/to/val.ffcv \
      --model_name "pre-trained visual backbone" \
```
Then stack embeds together:
```bash
python stack_emb.py
```
FYI : I did this only because memory constrain on my computer, otherwise you can directly infer the entire training embeds without stack operation.

### Train Contrastvie Sparse Representation on Imagenet1K
```bash
python main_visual.py \
      --pretrained_emb /path/to/pretrained_emb \
      --model_name "pre-trained visual backbone" \
      --use_ddp False \     # set True if you want to use multi-GPU
      --gpu 1\              # GPU ID, set None if you use multi-GPU
      --batch-size 1024 * 4 \
      --lr 4e-4 \
      --use_CL True \       # whether to use contrastive learning
      --topk 8 \            # topk for CSR
      --auxk 512 \          # auxiliary sparse code size
      --hidden-size 8192 \  # By default, 4 * visual backbone embedding size
```
### Get CSR Embeddings for Evaluation
```bash
python inference/csr_inference.py \
      --train_emb_path  /path/to/train_emb \
      --eval_emb_path    /path/to/val_emb \
      --model_name "pre-trained visual backbone" \
      --topk 8\
      --hidden-size 8192  # By default, 4 * visual backbone embedding size
      --cse_ckpt "CSR ckpt path"\
```

### Get Evaluation Results
We use [FAISS](https://github.com/facebookresearch/faiss) for KNN evaluation and calculate Top1 Accuracy under different sparsity conditions.
We follow the pipeline of [MRL](https://github.com/RAIVNLab/MRL/tree/main/retrieval) for a fair comparison.
It should be noted that this evaluation process runs on a 128-CPU server, requires approximately 150GB of memory, and takes about 20 minutes to complete. 
Further optimization is needed, and we welcome collaboration.  

```bash
cd retrieval
# Get FAISS index
python faiss_nn.py --topk 8
# Evaluate Top1 accuracy
python compute_metrics.py --topk 8
```

## Reproduce MultiModal Exp

### Train CSR on CC3M and eval on MS COCO and Flickr30K (Zero-Shot)

```bash
torchrun --nproc_per_node=1 ./main_multimodal.py\
         --train-data "./cc3m_wds/cc3m-train-{0000..575}.tar" # path to training datsets
         --train-num-samples 2905954 \
         --dataset-type webdataset \
         --precision amp \
         --workers 16 \
         --model "ViT-B-16"\
         --pretrained "dfn2b" \
         --epochs 5 \
         --save-frequency 1 \
         --report-to tensorboard \
         --csr-topk 64 \
         --csr-auxk 1024 \
         --csr-cl_coef 1 \
         --csr-hidden-size 2048 \
         --csr-input-dim 512 \
         --batch-size 1024 \
         --lr 4e-4 \
         --wd 1e-4 \
         --precision amp \
         --grad-clip-norm 1.0 \
         --save-frequency 1 \
```

### Evaluate CSR performance
```bash
python main_multimodal.py eval
    --dataset=mscoco_captions \
    --dataset_root="DATASET_ROOT" \
    --task=zeroshot_retrieval \
    --model="ViT-B-16" \
    --pretrained="dnf2b" \
    --output="./result.json" \
    --batch_size=64 \
    --recall_k=5 \
    --csr=True\
    --csr_ckpt=$CSR_CKPT\
    --topk=$Topk\
    --hidden-size=2048\
    --rep_dim=$Topk
```

And the results can be fine in ./result.json.

### Citing this paper
If you find this work useful, please cite the accompanying paper:
```
@misc{wen2025matryoshkarevisitingsparsecoding,
      title={Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation}, 
      author={Tiansheng Wen and Yifei Wang and Zequn Zeng and Zhong Peng and Yudi Su and Xinyang Liu and Bo Chen and Hongwei Liu and Stefanie Jegelka and Chenyu You},
      year={2025},
      eprint={2503.01776},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.01776}, 
}
```
### Acknowledgements
This repository was built off of [Sparse_AutoEncoder](https://github.com/openai/sparse_autoencoder), [Torchvision](https://github.com/pytorch/vision).


