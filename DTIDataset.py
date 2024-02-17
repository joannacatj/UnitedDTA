import pandas as pd
import torch
import numpy as np
import json

from PIL import Image
from dgl import load_graphs
from torch.utils.data import DataLoader,Dataset
import dgl
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
if torch.cuda.is_available():
    device = torch.device('cuda')
img_transformer = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
class DTADataset(Dataset):
    def __init__(self,dataset='Davis',idx=None,label=None):
        self.dataset=dataset
        #with open(compound_embedding, 'r') as file:
        #    self.compound_emd = json.load(file)
        #with open(protein_embedding, 'r') as file:
        #    self.protein_emd= json.load(file)
        self.idlist=pd.read_csv(idx)
        #self.compound_emd
        with open(label, 'r') as file:
            self.label_value = json.load(file)
    def __len__(self):
        return len(self.idlist)

    def __getitem__(self,idx):
        compound_id=self.idlist['COMPOUND_ID'].iloc[idx]
        protein_id=self.idlist['PROTEIN_ID'].iloc[idx]
        dir='data/'+self.dataset+'/predata/compound_embedding/'+str(compound_id)+'.npy'
        compound_onehot=np.load(dir)
        filename='data/'+self.dataset+'/predata/compound_image/'+str(compound_id)+'.png'
        compound_img = Image.open(filename).convert('RGB')
        compound_img=img_transformer(compound_img)
        compound_img=normalize(compound_img)
        #compound_onehot=self.compound_emd[str(compound_id)]['onehot']
        #compound_adj=self.compound_emd[str(compound_id)]['adj']
        #compound_fea=self.compound_emd[str(compound_id)]['fea']
        dir='data/'+self.dataset+'/predata/compound_graph/'+str(compound_id)+'.bin'
        compound_graph,_=load_graphs(dir)
        compound_graph=compound_graph[0]
        dir='data/'+self.dataset+'/predata/compound_3d_embedding/'+str(compound_id)+'.npy'
        compound_3d=np.load(dir)
        dir='data/'+self.dataset+'/predata/protein_embedding/'+protein_id+'.npz'
        #protein_onehot=self.protein_emd[protein_id]['onehot']
        #protein_word2vec=self.protein_emd[protein_id]['word2vec']
        #protein_bio=self.protein_emd[protein_id]['bio']
        #protein_fea=self.protein_emd[protein_id]['fea']
        protein_train = np.load(dir)
        protein_onehot=protein_train['arr_0']
        #print(protein_onehot)
        protein_word2vec=protein_train['arr_1']
        dir='data/'+self.dataset+'/predata/protein_graph/'+str(protein_id)+'.bin'
        protein_graph,_=load_graphs(dir)
        protein_graph=protein_graph[0]
        dir='data/'+self.dataset+'/predata/ESM_embedding_pocket/'+str(protein_id)+'.npy'
        protein_3d=np.load(dir)
        protein_len=protein_3d.shape[0]
        label = self.label_value.get(str(compound_id), {}).get(protein_id, [0.0])
        return compound_onehot,compound_graph,compound_img,compound_3d,protein_onehot,protein_word2vec,protein_graph,protein_3d,label,protein_len

    def collate_fn(self,data):
        batch_size=len(data)
        compound_onehot, compound_graph, compound_img,compound_3d,protein_onehot, protein_word2vec, protein_graph,protein_3d,label,protein_len = map(list,zip(
            *data))
        #print(protein_len)
        for i in range(batch_size):
            compound_graph[i].ndata['graph_id'] = torch.tensor([i] * compound_graph[i].number_of_nodes())
            protein_graph[i].ndata['graph_id'] = torch.tensor([i] * protein_graph[i].number_of_nodes())
        # 将 compound_graph 和 protein_graph 进行 batch 操作
        compound_graph = dgl.batch(compound_graph)
        protein_graph = dgl.batch(protein_graph)
        max_protein_len = max(protein_len)
        #print(max_protein_len)

        for i in range(batch_size):
            if protein_3d[i].shape[0] < max_protein_len:
                protein_3d[i] = np.pad(protein_3d[i],
                                              ((0, max_protein_len - protein_3d[i].shape[0]), (0, 0)),
                                              mode='constant', constant_values=(0, 0))
        # 转换为 Tensor
        compound_onehot = torch.FloatTensor(compound_onehot)
        compound_img = torch.stack(compound_img)
        compound_3d = torch.FloatTensor(compound_3d)
        protein_onehot = torch.FloatTensor(protein_onehot)
        protein_word2vec = torch.FloatTensor(protein_word2vec)
        protein_3d=torch.FloatTensor(protein_3d)
        label = torch.FloatTensor(label)

        return compound_onehot, compound_graph, compound_img, compound_3d,protein_onehot, protein_word2vec, protein_graph, protein_3d,label

# 创建一个 Dataset 实例
#dataset = DTADataset(dataset='Davis', idx='/mnt/sdb/home/hjy/Summary-DTA/data/Davis/processed/test/fold1/idx_id.csv', label='/mnt/sdb/home/hjy/Summary-DTA/data/Davis/label.json')
'''

# 创建一个 DataLoader 实例
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)
# 遍历 DataLoader
for batch in dataloader:
    compound_onehot, compound_graph, compound_img, protein_onehot, protein_word2vec, protein_graph, label = batch

    # 在这里进行你的验证操作，例如打印批次数据的大小或其他操作
    print('Compound One-Hot Shape:', compound_onehot.shape)
    print('Compound Graph:', compound_graph.edata['feats'].shape)
    print('Compound Image Shape:', compound_img.shape)
    print('Protein One-Hot Shape:', protein_onehot.shape)
    print('Protein Word2Vec Shape:', protein_word2vec.shape)
    print('Protein Graph:', protein_graph)
    print('Label Shape:', label.shape)
'''

