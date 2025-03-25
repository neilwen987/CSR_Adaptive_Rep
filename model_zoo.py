import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional

class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


class CSR(nn.Module):
    def __init__(
            self, n_latents: int, topk :int ,auxk :int ,normalize :bool, n_inputs: int ,dead_threshold :int,
            pre_trained_backbone: Optional[nn.Module] = None,
    ) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
        :param activation: activation function
        :param tied: whether to tie the encoder and decoder weights
        """
        super().__init__()

        self.pre_trained_backbone = pre_trained_backbone
        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder: nn.Module = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.decoder: TiedTranspose = TiedTranspose(self.encoder)
        self.topk = topk
        self.auxk = auxk
        self.normalize = normalize

        self.stats_last_nonzero: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_latents, dtype=torch.long))

        def auxk_mask_fn(x):
            dead_mask = self.stats_last_nonzero > dead_threshold
            x.data *= dead_mask  # inplace to save memoryr
            return x

        self.auxk_mask_fn = auxk_mask_fn

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(
            x, self.encoder.weight, self.latent_bias
        )
        return latents_pre_act


    def LN(self ,x: torch.Tensor, eps: float = 1e-5) :
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def preprocess(self, x: torch.Tensor):
        if not self.normalize:
            return x, dict()
        x, mu, std = self.LN(x)
        return x, dict(mu=mu, std=std)

    def top_k(self, x: torch.Tensor, topk = None) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        if topk is None:
            topk = self.topk
        topk = torch.topk(x, k=topk, dim=-1)
        z_topk = torch.zeros_like(x)
        z_topk.scatter_(-1, topk.indices, topk.values)
        latents_k = F.relu(z_topk)
        ## set num nonzero stat ##
        tmp = torch.zeros_like(self.stats_last_nonzero)
        tmp.scatter_add_(
            0,
            topk.indices.reshape(-1),
            (topk.values > 1e-5).to(tmp.dtype).reshape(-1),
        )
        self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
        self.stats_last_nonzero += 1
        ## end stats ##

        if self.auxk:
            aux_topk = torch.topk(
                input= self.auxk_mask_fn(x),
                k=self.auxk,
            )
            z_auxk = torch.zeros_like(x)
            z_auxk.scatter_(-1, aux_topk.indices, aux_topk.values)
            latents_auxk = F.relu(z_auxk)
        return latents_k , latents_auxk

    def decode(self, latents: torch.Tensor ,info=None) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """

        ret = self.decoder(latents) + self.pre_bias

        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def get_pretrain_embedding(self, x:torch.Tensor)-> torch.Tensor:
        with torch.no_grad():
            x = self.pre_trained_backbone.forward_features(x)
            x = self.pre_trained_backbone.forward_head(x, pre_logits=True)
        return x


    def csr_forward(self, x: torch.Tensor,topk) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """

        x , info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents_k, latents_auxk = self.top_k(latents_pre_act,topk)
        recons = self.decode(latents_k ,info)
        recons_aux = self.decode(latents_auxk ,info)
        return latents_pre_act, latents_k, recons ,recons_aux

    def forward(self, x: torch.Tensor,topk=None) -> torch.Tensor:
        # pretrained_embed = self.get_pretrain_embedding(x)
        # latents_pre_act, latents_k, recons, recons_aux = self.sae_forward(pretrained_embed)
        # return pretrained_embed,latents_pre_act, latents_k, recons, recons_aux
        latents_pre_act, latents_k, recons, recons_aux = self.csr_forward(x,topk=topk)
        return x, latents_pre_act, latents_k, recons, recons_aux

class CustomDataset(Dataset):
    def __init__(self,train_data):
        self.train_data = train_data
        # self.labels = train_data['labels']
        self.N = self.train_data.shape[0]

    def __getitem__(self, index):
        data = self.train_data[index,:]
        labels = 0
        return (np.squeeze(data), labels)

    def __len__(self):
        return self.N


