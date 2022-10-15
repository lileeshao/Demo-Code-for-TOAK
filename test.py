import numpy as np
import os
import json

def load_id2idx(base_dir):
    id2idx_file = os.path.join(base_dir, 'id2idx.json')
    id2idx = {}
    id2idx = json.load(open(id2idx_file)) 
    for k, v in id2idx.items():
        id2idx[str(k)] = v
    return id2idx

'''tid2idx = load_id2idx('./dataset/douban/target')
sid2idx = load_id2idx('./dataset/douban/source')
h_file = './dataset/douban/train'
h = np.zeros((3906,1118))
with open(h_file) as f:
    for line in f:
        s,anchor = line.strip().split()
        h[sid2idx[str(s)], tid2idx[str(anchor)]] = 1

np.save('./dataset/douban/h.npy',h)'''

h = np.load('./dataset/douban/h.npy', allow_pickle=True)
h = h.T
node_num = h.shape[0]
h_indicator = np.zeros(node_num)
for i in range(node_num):
    if np.sum(h[i,:])>=1:
        h_indicator[i] = 1

