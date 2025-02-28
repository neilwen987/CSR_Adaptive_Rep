'''
Code to evaluate MRL models on different validation benchmarks. 
'''
import sys 
sys.path.append("../") # adding root folder to the path

import torch 
import torchvision
from torchvision import transforms
from torchvision.models import *
from torchvision import datasets
from tqdm import tqdm

from argparse import ArgumentParser
from utils import *
import timm
from collections import OrderedDict
from typing import List
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder


parser=ArgumentParser()

# model args
parser.add_argument('--train_data_ffcv', default='/mnt/b6358dbf-93d5-42d7-adee-9793f027e744/WTS/Matryoshka_NCL/examples-main/'
                         'imagenet/ffcv_imagenet/train_500_0.50_90.ffcv',
                    help='path to training dataset (default: imagenet)')
parser.add_argument('--eval_data_ffcv', default='/mnt/b6358dbf-93d5-42d7-adee-9793f027e744/WTS/Matryoshka_NCL/examples-main/'
                         'imagenet/ffcv_imagenet/val_500_0.50_90.ffcv',
                    help='path to evaluation dataset (default: imagenet)')
parser.add_argument('--workers', type=int, default=16, help='num workers for dataloader')
parser.add_argument('--batch_size', default=256,type=int,help='batch size')
parser.add_argument('--embed_save_path', default='../retrieval/pretrained_emb', help='path to save database and query arrays for retrieval', type=str)
parser.add_argument('--model_name', default='resnet50d.ra4_e3600_r224_in1k',help='timm model name')

args = parser.parse_args()

model = timm.create_model(args.model_name, pretrained=True, num_classes=1000,)
state_dict = torch.load(args.backbone_ckpt, map_location='cpu')
model.load_state_dict(state_dict)
model.cuda()
model.eval()


# We follow data processing pipe line from FFCV
IMG_SIZE = 256
CENTER_CROP_SIZE = 224
IMAGENET_MEAN = np.array([0.5, 0.5, 0.5]) * 255
IMAGENET_STD = np.array([0.5, 0.5, 0.5]) * 255
DEFAULT_CROP_RATIO = 224 / 256
decoder = RandomResizedCropRGBImageDecoder((224, 224))
res_tuple = (256, 256)
device = torch.device("cuda")

image_pipeline: List[Operation] = [
    decoder,
    RandomHorizontalFlip(),
    ToTensor(),
    ToDevice(device, non_blocking=True),
    ToTorchImage(),
    NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
]

label_pipeline: List[Operation] = [
    IntDecoder(),
    ToTensor(),
    Squeeze(),
    ToDevice(device, non_blocking=True)
]
order = OrderOption.RANDOM

database_loader = Loader(args.train_data_ffcv,
                      batch_size=args.batch_size,
                      num_workers=args.workers,
                      order=order,
                      os_cache=1,
                      drop_last=True,
                      pipelines={
                          'image': image_pipeline,
                          'label': label_pipeline
                      },
                      )

cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
image_pipeline = [
    cropper,
    ToTensor(),
    ToDevice(device, non_blocking=True),
    ToTorchImage(),
    NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
]

label_pipeline = [
    IntDecoder(),
    ToTensor(),
    Squeeze(),
    ToDevice(device,
             non_blocking=True)
]

queryset_loader = Loader(args.eval_data_ffcv,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    }, )

if not os.path.exists(args.embed_save_path):
    os.makedirs(args.embed_save_path+'/train_emb', exist_ok=True)
    os.makedirs(args.embed_save_path + '/val_emb', exist_ok=True)


print("Inferencing Training Dataset")
generate_pretrained_embed(model, database_loader,args.embed_save_path+'/train_emb')

print("Inferencing Evaluation Dataset")
generate_pretrained_embed(model, queryset_loader,args.embed_save_path + '/val_emb')

    