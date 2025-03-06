import os
from tqdm import tqdm
import numpy as np
data_list =[]

embed_path = './retrieval/pretrained_emb/train_emb'
for file in tqdm(os.listdir(embed_path)):
    data = np.load(os.path.join(embed_path,file))
    data_list.append(data)
data_stack = np.concatenate(data_list,axis=0)
print(data_stack.shape)
np.save('imgenet1k_train_emb.npy',data_stack)