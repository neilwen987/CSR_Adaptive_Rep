import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
import pandas as pd
# from clip_benchmark.datasets.builder import build_dataset
from tqdm import tqdm
from torch.cuda.amp import autocast
from pathlib import Path
import glob

'''
Retrieval utility methods.
'''
activation = {}
fwd_pass_x_list = []
fwd_pass_y_list = []

def get_activation(name):
	"""
	Get the activation from an intermediate point in the network.
	:param name: layer whose activation is to be returned
	:return: activation of layer
	"""
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook


def append_feature_vector_to_list(activation, label):
	"""
	Append the feature vector to a list to later write to disk.
	:param activation: image feature vector from network
	:param label: ground truth label
	"""
	for i in range (activation.shape[0]):
		x = activation[i].cpu().detach().numpy()
		y = label[i].cpu().detach().numpy()
		fwd_pass_y_list.append(y)
		fwd_pass_x_list.append(x)


def dump_feature_vector_array_lists(config_name, output_path):
	"""
	Save the database and query vector array lists to disk.
	:param config_name: config to specify during file write
	:param output_path: path to dump database and query arrays after inference
	"""

	# save X (n x 2048), y (n x 1) to disk, where n = num_samples
	X_fwd_pass = np.asarray(fwd_pass_x_list, dtype=np.float32)
	y_fwd_pass = np.asarray(fwd_pass_y_list, dtype=np.float16).reshape(-1,1)

	np.save(output_path+'/'+str(config_name)+'-X.npy', X_fwd_pass)
	np.save(output_path+'/'+str(config_name)+'-y.npy', y_fwd_pass)


def generate_retrieval_data(model, emb_path, output_path, args, mode):
    model.eval()
    
    dump_path = os.path.join(output_path, f'SOTA_CSR_' + 'topk_' + str(args.topk))
    
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
        
    global fwd_pass_x_list
    global fwd_pass_y_list
    fwd_pass_x_list = []
    fwd_pass_y_list = []
    
    chunk_files = sorted(glob.glob(os.path.join(emb_path, '*.npz')))
    if not chunk_files:
        raise RuntimeError(f"No .npz files found in {emb_path}")
    
    with torch.no_grad():
        with autocast():
            all_data = []
            all_labels = []
            for chunk_path in tqdm(chunk_files):
                img_emb = torch.from_numpy(np.load(chunk_path)['data'])
                target = torch.from_numpy(np.load(chunk_path)['label'])
                _, _, feature, _, _ = model(img_emb.cuda())
                # append_feature_vector_to_list(feature, target.cuda())
                all_data.append(feature.cpu())
                all_labels.append(target)
    
    combined_data = np.concatenate(all_data, axis=0)
    combined_label = np.concatenate(all_labels, axis=0)
    
    np.savez(dump_path + f"_{mode}.npz", data=combined_data, label=combined_label)
    # np.savez(dump_path + f"_{mode}.npz", data=combined_data)

def generate_pretrained_embed(model, data_loader, emb_path,):
	"""
	Iterate over data in dataloader, get feature vector from model inference, and save to array to dump to disk.
	:param model: ResNet50 model loaded from disk
	:param data_loader: loader for database or query set
	:param config: name of configuration for writing arrays to disk
	:param rep_size: representation size for fixed feature model
	:param output_path: path to dump database and query arrays after inference
	"""
	img_path = os.path.join(emb_path,'img')
	label_path = os.path.join(emb_path,'label')
	if not os.path.exists(img_path):
		os.makedirs(img_path)
	if not os.path.exists(label_path):
		os.makedirs(label_path)
	with torch.no_grad():
		with autocast():
			for i_batch, (images, target) in enumerate(tqdm(data_loader)):
				feature = model.forward_features(images.cuda())
				feature = model.forward_head(feature, pre_logits=True)
				# append_feature_vector_to_list(feature, target.cuda(), rep_size)
				img = feature.detach().cpu().numpy()
				'''save pretrained embedding '''
				np.save(f'{img_path}/emb_{i_batch}.npy', img)
				np.save(f'{label_path}/emb_{i_batch}.npy',target.detach().cpu().numpy())


	# re-initialize empty lists
	global fwd_pass_x_list
	global fwd_pass_y_list
	fwd_pass_x_list = []
	fwd_pass_y_list = []


def stack_emb(root_path='./retrieval/pretrained_emb/train_emb'):
	data_list =[]
	img_path = os.path.join(root_path,'img')
	label_path = os.path.join(root_path,'label')
	for file in tqdm(os.listdir(img_path)):
		data = np.load(os.path.join(img_path,file))
		data_list.append(data)
	data_stack = np.concatenate(data_list,axis=0)
	np.save('imgenet1k_train_emb.npy',data_stack)


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out

def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)

def get_mm_dataset(root_path, dataset='mscoco_captions'):
	for split in ['train','val']:
		ds = build_dataset(dataset, root=root_path, split=split,
						   task="captioning")  # this downloads the dataset if it is not there already
		coco = ds.coco
		imgs = coco.loadImgs(coco.getImgIds())
		future_df = {"filepath": [], "title": []}
		abs_path =  Path(root_path).resolve()
		for img in imgs:
			caps = coco.imgToAnns[img["id"]]
			for cap in caps:
				future_df["filepath"].append(abs_path/f'{split}2014'/img["file_name"])
				future_df["title"].append(cap["caption"])
		pd.DataFrame.from_dict(future_df).to_csv(
			os.path.join(root_path, f"{split}2014.csv"), index=False, sep="\t"
		)
	return print(f'Successfully create {dataset} dataste')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--root_path', type=str,required=True,help='root path for mm dataset')
	parser.add_argument('--dataset_name', type=str, default='mscoco_captions',help='dataset name')
	args = parser.parse_args()
	get_mm_dataset(args.root_path,args.dataset_name)
