import os
import math
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from datetime import datetime
import torch.nn as nn
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip_train.scheduler import cosine_lr
from open_clip_train.distributed import is_master, init_distributed_device
from open_clip_train.logger import setup_logging
from open_clip_train.precision import get_autocast
from open_clip_train.data import get_data
from open_clip_train.params import ParseKwargs
from model_zoo import CSR
from torch.optim.lr_scheduler import StepLR

class CSRLoss(torch.nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        auxk_coef=1/32,
        cl_coef =1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.auxk_coef = auxk_coef
        self.cl_coef = cl_coef
         
        
        self.labels = {}

    def normalized_mse(self, recon, xs, criterion):
        
        xs_mu = xs.mean(dim=0)
        loss = criterion(recon, xs) / criterion(
            xs_mu[None, :].broadcast_to(xs.shape), xs
        )
        return loss
            
    def get_ground_truth(self, device, dtype, batch_size):
        
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        if self.world_size > 1 and self.local_loss:
            labels = labels + batch_size * self.rank
        return labels
    
    def get_logits(self, image_features, text_features, logit_scale):
        
        
        
        image_features_dim = F.normalize(image_features, dim=-1)
        text_features_dim = F.normalize(text_features, dim=-1)
        
        
        logits_per_image = logit_scale * image_features_dim @ text_features_dim.t()
        logits_per_text = logits_per_image.t()
        
        
        return logits_per_image, logits_per_text
    
    def forward(self, image_features, text_features, img_recon,text_recon,
                img_recon_4k,text_recon4k,img_auxk,text_auxk,logit_scale, prebias,criterion,output_dict=False):
        device = image_features.device
        batch_size = image_features.shape[0]
        
        
        labels = self.get_ground_truth(device, image_features.dtype, batch_size)
        
        # Calculate Recon Loss
        recon_loss = criterion(image_features, img_recon) + criterion(text_features, text_recon)
        recon_loss_4k = criterion(image_features, img_recon_4k) + criterion(text_features,text_recon4k)

        loss_auxk_1 = self.normalized_mse(img_auxk, image_features - img_recon.detach() + prebias.detach(), criterion,).nan_to_num(0)
        loss_auxk_2 = self.normalized_mse(text_auxk, text_features - text_recon.detach() + prebias.detach(), criterion,).nan_to_num(0)
        loss_auxk = (loss_auxk_1 + loss_auxk_2) / 2

        # Calculate CL-CL Loss
        logits_per_image, logits_per_text = self.get_logits(img_recon, text_recon, logit_scale)
        
        
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        cl_cl_loss = (loss_img + loss_txt) / 2
        
        
        total_loss = recon_loss + 0.125 * recon_loss_4k + self.auxk_coef * loss_auxk + self.cl_coef* cl_cl_loss
        
        return {
            'loss': total_loss,
            'loss_k': recon_loss,
            'loss_4k': recon_loss_4k,
            'loss_auxk': loss_auxk,
            'loss_cl': cl_cl_loss,
        }


class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(model, csr_layer, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = torch.float32
    if args.precision == "fp16":
        input_dtype = torch.float16
    elif args.precision == "bf16":
        input_dtype = torch.bfloat16

    model.eval()  
    csr_layer.train()  
    
    
    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    
    end = torch.cuda.Event(enable_timing=True)
    start = torch.cuda.Event(enable_timing=True)
    
    end.record()
    
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        
        start.record()
        torch.cuda.synchronize()
        data_time = end.elapsed_time(start) / 1000
        data_time_m.update(data_time)
        
        
        optimizer.zero_grad()
        
        
        with autocast():
            
            with torch.no_grad():
                model_out = model(images, texts)
                image_features = model_out["image_features"]
                text_features = model_out["text_features"]
                logit_scale = model_out["logit_scale"]
            
            
            _, _, latents_k_1, recons_1, recons_aux_1 = csr_layer(image_features)
            _, _, latents_k_2, recons_2, recons_aux_2 = csr_layer(text_features)
            
            
            _, _, _, recons_3, _ = csr_layer(image_features, topk=4 * args.csr_topk)
            _, _, _, recons_4, _ = csr_layer(text_features, topk=4 * args.csr_topk)
            
            losses = loss(
                image_features,
                text_features,
                recons_1,
                recons_2,
                recons_3,
                recons_4,
                recons_aux_1,
                recons_aux_2,
                logit_scale,
                csr_layer.module.pre_bias,
                criterion = nn.MSELoss().to(device)
                
            )
            
            total_loss = losses['loss']
        
       
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(csr_layer.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(csr_layer.parameters(), args.grad_clip_norm)
            optimizer.step()
        
        
        end.record()
        torch.cuda.synchronize()
        batch_time = end.elapsed_time(start) / 1000
        batch_time_m.update(batch_time)
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or i == num_batches_per_epoch - 1):
            batch_size = images.shape[0]
            for loss_name, loss_val in losses.items():
                if loss_name not in losses_m:
                    losses_m[loss_name] = AverageMeter()
                losses_m[loss_name].update(loss_val.item(), batch_size)
            
            loss_log = " ".join([
                f"{loss_name.capitalize()}: {loss_m.val:.5g} ({loss_m.avg:.5g})"
                for loss_name, loss_m in losses_m.items()
            ])
            
            samples_per_second = args.batch_size * args.world_size / batch_time_m.val
            
            logging.info(
                f"Train Epoch: {epoch} [{i}/{num_batches_per_epoch}] "
                f"Data: {data_time_m.avg:.3f} "
                f"Batch: {batch_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} " + loss_log
            )
            
            
            if tb_writer is not None:
                tb_writer.add_scalar("train/batch_time", batch_time_m.val, step)
                tb_writer.add_scalar("train/data_time", data_time_m.val, step)
                tb_writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
                for loss_name, loss_m in losses_m.items():
                    tb_writer.add_scalar(f"train/{loss_name}", loss_m.val, step)
            
        end.record()
    
    
    return {loss_name: loss_m.avg for loss_name, loss_m in losses_m.items()}

def main():
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Training CSR')
    
    
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="train data path",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="val dataset path",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="number of training samples",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="number of validation samples",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="auto",
        help="dataset type",
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",

    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",

    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",

    )
    
    
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-16",
        help="CLIP backbone",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="pre-trained CLIP",
    )
    parser.add_argument(
        "--csr-topk",
        type=int,
        default=64,
        help="topk",
    )
    parser.add_argument(
        "--csr-cl_coef",
        type=float,
        default=1.0,
        help="cl coef",
    )
    parser.add_argument(
        "--csr-auxk",
        type=int,
        default=512,
        help="auxk",
    )
    parser.add_argument(
        "--csr-dead-threshold",
        type=int,
        default=30,
        help="dead_threshold",
    )
    parser.add_argument(
        "--csr-hidden-size",
        type=int,
        default=2048,
        help="hidden_size",
    )
    parser.add_argument(
        "--csr-input-dim",
        type=int,
        default=512,
        help="input_dim",
    )
    
    # 训练参数
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="bsz",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-4,
        help="lr",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10000,

    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,

    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",

    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,

    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=1,

    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="./logs/",

    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,

    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,

    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,

    )
    

    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,

    )
    parser.add_argument(
        "--dist-backend",
        default=None,
        type=str,

    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,

    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,

    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    
    args = parser.parse_args()
    
    
    if args.name is None:
        args.name = f"csr_training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    
    args.distributed = torch.cuda.is_available() and torch.distributed.is_available()
    if args.distributed:
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
        args.local_rank = torch.distributed.get_rank()
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)
        args.device = device
    else:
        args.local_rank = 0
        args.rank = 0
        args.world_size = 1
        device = torch.device(args.device)
        args.device = device
    
    
    log_base_path = os.path.join(args.logs_dir, args.name)
    os.makedirs(log_base_path, exist_ok=True)
    log_file = os.path.join(log_base_path, "train.log")
    setup_logging(log_file=log_file, level=logging.INFO)
    
    if is_master(args):
        logging.info(f"Training Parameters: {args}")
    
    
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        output_dict=True,
    )
    
    # 用于生成embedding的CSR层
    csr_layer = CSR(
        n_latents=args.csr_hidden_size,
        topk=args.csr_topk,
        auxk=args.csr_auxk,
        normalize=False,
        n_inputs=args.csr_input_dim,
        dead_threshold=args.csr_dead_threshold
    ).to(device)
    
    tokenizer = get_tokenizer(args.model)
    
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=0,
        tokenizer=tokenizer,
    )
    
    
    if args.distributed:
        csr_layer = DistributedDataParallel(csr_layer, device_ids=[device])
    
    
    loss = CSRLoss(
        local_loss=False,
        gather_with_grad=False,
        rank=args.local_rank,
        world_size=args.world_size,
        cl_coef=args.csr_cl_coef
    )
    
    
    optimizer = torch.optim.AdamW(
        csr_layer.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=6.25 * 1e-10,
        weight_decay=args.wd,
    )
    
    
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1) 
    
    scaler = None
    if args.precision == "amp":
        scaler = torch.cuda.amp.GradScaler()
    
    # TensorBoard writer
    tb_writer = None
    if is_master(args) and args.report_to == 'tensorboard':
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=os.path.join(log_base_path, "tensorboard"))
        except ImportError:
            logging.warning('tensorboard not available')
    
    
    for epoch in range(args.epochs):
        if is_master(args):
            logging.info(f"start training {epoch}")
        
        train_metrics = train_one_epoch(
            model=model,
            csr_layer=csr_layer,
            data=data,
            loss=loss,
            epoch=epoch,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            args=args,
            tb_writer=tb_writer,
        )
        
      
        if not args.skip_scheduler:
            scheduler.step()
        
      
        if is_master(args) and ((epoch + 1) % args.save_frequency == 0 or epoch == args.epochs - 1):
            checkpoint_path = os.path.join(log_base_path, f'checkpoint_{epoch}.pt')
            if args.distributed:
                state_dict = csr_layer.module.state_dict()
            else:
                state_dict = csr_layer.state_dict()
            torch.save({
                'epoch': epoch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)
            logging.info(f"saving ckpt to {checkpoint_path}")
    
   
    if tb_writer is not None:
        tb_writer.close()

    if is_master(args):
        logging.info("Done!")

if __name__ == "__main__":
    main() 
