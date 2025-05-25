#  [Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation](http://arxiv.org/abs/2503.01776) (ICML 2025)
By [Tiansheng Wen\*](https://neilwen987.github.io/), [Yifei Wang\*](https://yifeiwang77.com/), [Zequn Zeng](https://joeyz0z.github.io/), Zhong Peng, Yudi Su, [Xinyang Liu](https://xinyangatk.github.io/),
[Bo Chen](https://web.xidian.edu.cn/bchen/en/index.html), Hongwei Liu, [Stefanie Jegelka](https://people.csail.mit.edu/stefje/) and [Chenyu You](https://chenyuyou.me/)

<a href='https://arxiv.org/abs/2503.01776'><img alt="Static Badge" src="https://img.shields.io/badge/Paper-arXiv-red"></a>

![Overview](./assets/overview.jpg)

In this paper, we show that *sparse coding* offers a compelling alternative for achieving adaptive representation with minimal overhead and higher fidelity. We propose **C**ontrastive **S**parse **R**epresentation, a method that sparsifies pre-trained embeddings into a high-dimensional but *selectively activated* feature space. By leveraging lightweight autoencoding and task-aware contrastive objectives, CSR preserves semantic quality while allowing flexible, cost-effective inference at different sparsity levels. Extensive experiments on image, text, and multimodal benchmarks demonstrate that CSR consistently outperforms MRL in terms of both accuracy and retrieval speed-often by large margins-while also cutting training time to a fraction of that required by MRL. Our results establish sparse coding as a powerful paradigm for adaptive representation learning in real-world applications where efficiency and fidelity are both paramount. 

## &#x1F680; &#x1F680; News
- 2025.05.25 **Major Update**. We have thoroughly reorganized our repository with the following changes: 🎉🎉
  - Minor code changes on Visual Experiments, especially dataset preparation.
  - Training & Evaluation Pipeline for Text Experiments, including text classification, text clustering and text retrieval.
  - Minor changes in codes for Multimodal Experiments.
  - More detailed instructions for Data Preparation & Training & Evaluation.
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
  - Text Exp;
    - Dataset preparations &#x2705;
    - Training &#x2705;
    - Evaluation &#x1F4CC;
  - MultiModal Exp &#x2705;
    - Dataset preparations &#x2705;
    - Training &#x2705;
    - Evaluation &#x2705;
  - Retrieval Time Evaluation &#x1F4CC;
- Upload CSR on HugginFace &#x1F4CC;

## Set up
You only need to prepare an empty conda environment with Python 3 (reference version: Python 3.8.20) and `pip install` the `requirements.txt` file in this directory.
```sh
conda create --name csr python=3.8.20
pip install -r requirements.txt
```

## Vision Representation
First, please move to our `vision_representation` codebase directory.
```shell
cd ./vision_representation/
```
### Get Imagenet1k embeds
#### Way I (Recommended): Get pre-compute embeds from Huggingface🤗
We provide embeds extracted by FF2048 backbones (same backbone weights with MRL), and embeds by [SoTA backbone](https://huggingface.co/timm/resnet50d.ra4_e3600_r224_in1k) at [Dataset Link](https://huggingface.co/datasets/W1nd-navigator/CSR-precompute-embeds) .
#### Way II: Prepare embeds from start
To train CSR with different visual backbones, please follow the preparation steps below.
##### Prepare Imagenet1k dataset
**Step I**: Download Imagenet1k dataset and bounding box annotations from [Imagenet1k Official Website](https://www.image-net.org/).
**Step II**: Convert the original dataset to [Pytorch style](https://github.com/MadryLab/pytorch-imagenet-dataset).
```shell
# Prepare the annotations.txt file for both training and validation set
python ./dataset_preparation annotations.py --xml_dir "/path/to/train/annotation/directory" --output_file "/path/to/annotation.txt/directory"
# Convert original dataset to Pytorch style
python ./dataset_preparation to_pytorch_style.py --split_path "/path/to/pytorch/style/dataset"
```
**Step III**: Follow the pipeline of [FFCV](https://github.com/libffcv/ffcv-imagenet) for ResNet50 to generate the dataset with the following command (`IMAGENET_DIR` should point to a Pytorch style [ImageNet dataset](https://github.com/williamFalcon/pytorch-imagenet-dataset)).
```shell
cd dataset_preparation
# Required environmental variables for the script:
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/

# Serialize images with:
# - dataset type: train/val
# - 500px side length maximum
# - 50% JPEG encoded, 90% raw pixel values
# - quality=90 JPEGs
./write_imagenet.sh "train" 500 0.50 90
./write_imagenet.sh "val" 500 0.50 90
```
##### Get Pre-trained ImageNet1k Embeddings
For training and evaluation simplicity, we precompute image embeddings using models from [Timm](https://github.com/huggingface/pytorch-image-models).
In our paper, we select [resnet50d.ra4_e3600_r224_in1k](https://huggingface.co/timm/resnet50d.ra4_e3600_r224_in1k) as our pre-trained visual backbone. To extract embeddings, run the following command: 
```shell
python pretrained_embeddings.py \
	--train_data_ffcv  /path/to/train.ffcv \
	--eval_data_ffcv    /path/to/val.ffcv \
	--model_name "pre-trained visual backbone" \
```

Then stack embeds together:
```shell
python stack_emb.py
```
> Note:  We did this only in consideration of memory constrain on our computer, otherwise you can directly infer the entire training embeds without stack operation.
### Train Contrastive Sparse Representation on Imagenet1K
After getting embeds, you can train CSR with `main_visual.py`. You must customize the `pretrained_emb` (Path to the embeds) and `model_name` (`timm` model's name). For other parameters, you can both follow the default settings and customize them. The trained models will be saved to `./ckpt/CSR_topk_{args.topk}/`.
```sh
python main_visual.py \
      --pretrained_emb /path/to/pretrained_emb \
      --model_name "pre-trained visual backbone" \
      --use_ddp False \     # set True if you want to use multi-GPU
      --gpu 1 \              # GPU ID, set None if you use multi-GPU
      --batch-size 1024 * 4 \
      --lr 4e-4 \
      --use_CL True \       # whether to use contrastive learning
      --topk 8 \            # topk for CSR
      --auxk 512 \          # auxiliary sparse code size
      --hidden-size 8192 \  # By default, 4 * visual backbone embedding size
```
### Get CSR embeddings for Evaluation
You can get CSR embeddings with `csr_inference.py`. You must customize the `train_embed_path`, `eval_emb_path` and `csr_ckpt`. The embeds will be saved in `./retrieval/` by default. **Note:** Considering that the CSR embeddings are too large, we split them into chunks stored in the same directory with `chunk_original_npz_file.py`. Minor code change is needed in method `generate_retrieval_data` in `utils.py` is needed if you prefer inference with single `.npz` file.
```sh
# Chunk original embeddings
python chunk_npz_file.py \
	--input_path "Path/to/original/embeddings" \
	--output_path "Path/to/chunk/directory" \
	--chunk_size "Number of samples per chunk"
# Inference
python csr_inference.py \
      --train_emb_path  /path/to/train_emb \
      --eval_emb_path    /path/to/val_emb \
      --model_name "pre-trained visual backbone" \
      --topk 8\
      --hidden-size 8192  # By default, 4 * visual backbone embedding size
      --csr_ckpt "CSR ckpt path"\
```
### Get Evaluation Results
We use [FAISS](https://github.com/facebookresearch/faiss) for KNN evaluation and calculate Top1 Accuracy under different sparsity conditions. It should be noted that this evaluation process runs on a 128-CPU server, requires approximately 150GB of memory, and takes about 20 minutes to complete. Further optimization is needed, and we welcome collaboration. You only need to set `--topk` if all generated files are not moved.
```shell
# Get FAISS index
python ./retrieval/faiss_nn.py --topk 8
# Evaluate Top1 accuracy
python ./retrieval/compute_metrics.py --topk 8
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


