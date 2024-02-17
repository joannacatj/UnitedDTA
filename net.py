import torch
import torch.nn as nn
import torch.nn.functional as F
from model import sequence_model,graph_model,image_model,graph_model_new
from model.multimodal_fusion import Protein_Fusion, EncoderLayer_Self, EncoderLayer_Cross, MultiHeadAttention, \
    CrossMultiAttention, PositionwiseFeedforward
from model.nt_xent import NTXentLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SDTAnet(nn.Module):
    def __init__(self,protein_len,compound_len,protein_emb_dim,compound_emb_dim,compound_fea_dim,protein_fea_dim,out_dim,dropout,model_name=None):
        super(SDTAnet,self).__init__()
        self.protein_len=protein_len
        self.compound_len=compound_len
        self.protein_emb_dim=protein_emb_dim
        self.compound_emb_dim=compound_emb_dim
        #一维方式：TextCNN
        self.protein_cnn=sequence_model.TextCNN(protein_len,protein_emb_dim,num_filters=100,filter_sizes = [4,8,12])
        self.compound_cnn=sequence_model.TextCNN(compound_len,compound_emb_dim,num_filters=100,filter_sizes = [4,6,8])
        #一维方式：RNN系列
        self.compound_lstm=sequence_model.BILSTM(compound_emb_dim,64,3,128)
        self.protein_lstm = sequence_model.BILSTM(protein_emb_dim,64,3,128)
        #一维方式：Transformer
        self.protein_encoder=sequence_model.Encoder(protein_emb_dim,256,3,5,0.1,device)
        self.compound_encoder=sequence_model.Encoder(compound_emb_dim,256,3,5,0.1,device)
        self.decoder=sequence_model.Decoder(256,256,3,8,256,sequence_model.DecoderLayer,sequence_model.MultiHeadAttention,sequence_model.PositionwiseFeedforward,0.1,device)
        #二维方式：GCN
        self.protein_gcn=graph_model.GCN(protein_fea_dim,256,128)
        self.compound_gcn=graph_model.GCN(compound_fea_dim,256,128)
        #二维方式：GAT
        self.protein_gat=graph_model.GAT(protein_fea_dim,64,0.2,0.1,2)
        self.compound_gat=graph_model.GAT(compound_fea_dim,64,0.2,0.1,2)
        #二维方式：RESNET
        self.compound_image=image_model.load_model('ResNet18',128)
        self.out_dim=out_dim
        self.model_name=model_name
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.dropout = nn.Dropout(dropout)  # 表示正则化Dropout，防止过拟合
        self.fc1=nn.Linear(256,1024)
        self.fc2=nn.Linear(1024,512)
        self.out=nn.Linear(512,self.out_dim)
    def forward(self,compound_onehot,protein_onehot,compound_adj,compound_fea,protein_adj,protein_fea,compound_image):
        #一维预处理方式（one-hot)
        #CNN部分
        #compound_emd=self.compound_cnn(compound_onehot)
        #protein_emd=self.protein_cnn(protein_onehot)
        #RNN系列部分
        compound_emd,compound_global_emd=self.compound_lstm(compound_onehot)
        protein_emd,protein_global_emd=self.protein_lstm(protein_onehot)
        #Transformer部分
        #使用的时候要把fc的128*2改为128
        #compound_emd=self.compound_encoder(compound_onehot)
        #print(compound_emd.shape)
        #protein_emd=self.protein_encoder(protein_onehot)
        #print(protein_emd.shape)
        #xc=self.decoder(compound_emd,protein_emd)
        #GCN部分
        compound_emd_global,compound_emd_local=self.compound_gcn(compound_adj,compound_fea)
        #print('compound_emd',compound_emd)
        protein_emd_global,protein_emd_local=self.protein_gcn(protein_adj,protein_fea)
        #GAT部分
        #compound_emd_global,compound_emd_local=self.compound_gat(compound_adj,compound_fea)
        #print('compound_emd',compound_emd)
        #protein_emd_global,protein_emd_local=self.protein_gat(protein_adj,protein_fea)
        #RESNET部分
        #compound_emd=self.compound_image(compound_image)
        #结合方式：concat
        xc=torch.cat((compound_emd_global,protein_emd_global),1).float()
        #xc=torch.cat((compound_emd,protein_emd),1).float()
        #print(xc.dtype)
        #最后获得维度【batch_size,2*hid_dim】
        #结合方式：attention
        # 计算相关性分数
        #score_matrix = torch.matmul(protein_emd, compound_emd.t())
        # 归一化得到注意力权重
        #attention_weights = F.softmax(score_matrix, dim=1)
        # 加权组合得到最终特征向量
        #combined_emd = torch.matmul(attention_weights, compound_emd)
        #最终获得维度【batch_size,hid_dim】
        #结合方式：decoder
        #结合方式：matmul
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        #print(out)
        return out

class SDTNET_NEW(nn.Module):
    def __init__(self, protein_len, compound_len, protein_emb_dim, compound_emb_dim, compound_fea_dim, protein_fea_dim,
                 out_dim, dropout, model_name=None):
        super(SDTNET_NEW, self).__init__()
        self.protein_len = protein_len
        self.compound_len = compound_len
        self.protein_emb_dim = protein_emb_dim
        self.compound_emb_dim = compound_emb_dim
        # 一维方式：TextCNN
        self.protein_cnn = sequence_model.TextCNN(protein_len, protein_emb_dim, num_filters=100,
                                                  filter_sizes=[4, 8, 12])
        self.compound_cnn = sequence_model.TextCNN(compound_len, compound_emb_dim, num_filters=100,
                                                   filter_sizes=[4, 6, 8])
        # 一维方式：RNN系列
        self.compound_lstm = sequence_model.BILSTM(compound_emb_dim, 64, 3, 128)
        self.protein_lstm = sequence_model.BILSTM(protein_emb_dim, 64, 3, 128)
        # 一维方式：Transformer
        self.protein_encoder = sequence_model.Encoder(protein_emb_dim, 256, 3, 5, 0.1, device)
        self.compound_encoder = sequence_model.Encoder(compound_emb_dim, 256, 3, 5, 0.1, device)
        self.decoder = sequence_model.Decoder(256, 256, 3, 8, 256, sequence_model.DecoderLayer,
                                              sequence_model.MultiHeadAttention, sequence_model.PositionwiseFeedforward, 0.1,
                                              device)
        #二维方式：GCN
        self.compound_gcn=graph_model_new.GCNNet(compound_fea_dim)
        self.protein_gcn=graph_model_new.GCNNet(protein_fea_dim)
        #二维方式：GAT
        self.compound_gat=graph_model_new.GATNet(compound_fea_dim)
        self.protein_gat=graph_model_new.GATNet(protein_fea_dim)
        #二维方式：GIN
        self.compound_gin=graph_model_new.GINConvNet(compound_fea_dim)
        self.protein_gin=graph_model_new.GINConvNet(protein_fea_dim)
        #三维方式：Graph Transformer
        self.compound_graph_transformer=graph_model_new.GraphTransformer(device, n_layers=10, node_dim=44, edge_dim=10, hidden_dim=128,
                                                        out_dim=128, n_heads=8, in_feat_dropout=0.0, dropout=0.2, pos_enc_dim=8)
        self.protein_graph_transformer=graph_model_new.GraphTransformer(device,n_layers=10,node_dim=41,edge_dim=5,hidden_dim=128,out_dim=128,n_heads=8,in_feat_dropout=0.2,dropout=0.2,pos_enc_dim=8)
        # 二维方式：RESNET
        self.compound_image = image_model.load_model('ResNet18', 128)
        # 三维方式：unimol和esm2
        self.compound_3D=nn.Linear(512,128)
        self.protein_3d=nn.Linear(320,128)
        #fusion方式
        self.protein_fusion=Protein_Fusion(emd_dim=128, n_layers=4, n_heads=8, pf_dim=256, dropout=0.1, fusion_layer=2, use_bottleneck=True, test_with_bottlenecks=True,encoderlayer_self=EncoderLayer_Self,encoderlayer_cross=EncoderLayer_Cross,self_attention=MultiHeadAttention,cross_attention=CrossMultiAttention,feedfowardposition=PositionwiseFeedforward)
        self.out_dim = out_dim
        self.model_name = model_name
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)  # 表示正则化Dropout，防止过拟合
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.out_dim)
        self.nt_xent_criterion = NTXentLoss(device,batch_size=64,temperature=0.1,use_cosine_similarity=False)

    def forward(self, compound_onehot, protein_onehot, compound_graph,protein_graph,
                compound_image,compound_3d,protein_3d,train=True):
        loss=0
        # 一维预处理方式（one-hot)
        # CNN部分
        if self.model_name=='CNN':
            compound_emd=self.compound_cnn(compound_onehot)
            protein_emd=self.protein_cnn(protein_onehot)
        # RNN系列部分
        if self.model_name=='RNN':
            compound_emd, compound_global_emd = self.compound_lstm(compound_onehot)
            protein_emd, protein_global_emd = self.protein_lstm(protein_onehot)
            xc = torch.cat((compound_global_emd, protein_global_emd), 1).float()
        # Transformer部分
        # 使用的时候要把fc的128*2改为128
        if self.model_name=='Transformer':
            compound_emd=self.compound_encoder(compound_onehot)
            protein_emd=self.protein_encoder(protein_onehot)
            xc=self.decoder(compound_emd,protein_emd)
        #GCN部分
        if self.model_name=='GCN':
            compound_emd,compound_global_emd = self.compound_gcn(compound_graph)
            protein_emd,protein_global_emd = self.protein_gcn(protein_graph)
            xc = torch.cat((compound_global_emd, protein_global_emd), 1).float()
        #GIN部分
        if self.model_name=='GIN':
            compound_emd = self.compound_gin(compound_graph)
            protein_emd = self.protein_gin(protein_graph)
        #GAT部分
        if self.model_name=='GAT':
            compound_emd, compound_global_emd = self.compound_gat(compound_graph)
            protein_emd, protein_global_emd = self.protein_gat(protein_graph)
            xc = torch.cat((compound_global_emd, protein_global_emd), 1).float()
        #image对比学习部分
        if self.model_name=='Constractive Learning':
            compound_i = self.compound_image(compound_image)
            compound_emd,compound_global_emd =self.compound_gat(compound_graph)
            protein_emd=self.protein_cnn(protein_onehot)
            loss=self.nt_xent_criterion(compound_global_emd,compound_i)
            xc = torch.cat((compound_global_emd, protein_emd), 1).float()
        if self.model_name=='Graph Transformer':
            compound_emd,compound_global_emd=self.compound_graph_transformer(compound_graph)
            protein_emd,protein_global_emd=self.protein_graph_transformer(protein_graph)
            print('compound_emd:', compound_emd.shape)
            print('protein_emd:', protein_emd.shape)
            xc = torch.cat((compound_global_emd, protein_global_emd), 1).float()
        if self.model_name=='Pretrained 3D':
            compound_global_emd=self.compound_3D(compound_3d)
            protein_emd=self.protein_3d(protein_3d)
            protein_global_emd=torch.mean(protein_emd,dim=1)
            xc = torch.cat((compound_global_emd, protein_global_emd), 1).float()
        if self.model_name=='multimodel fusion':
            protein_emd_s, protein_global_emd_s = self.protein_lstm(protein_onehot)
            protein_emd_g, protein_global_emd_g = self.protein_gat(protein_graph)
            protein_emd_e = self.protein_3d(protein_3d)
            bottleneck = torch.randn(2, 5, 128).to(device)
            protein_emd=self.protein_fusion(protein_emd_s,protein_emd_g,protein_emd_e,bottleneck,train)
            compound_emd = self.compound_cnn(compound_onehot)
            xc=torch.cat((compound_emd,protein_emd),1).float()

        #compound_emd=self.compound_gin(compound_graph)
        #protein_emd=self.protein_gin(protein_graph)
        # RESNET部分
        # compound_emd=self.compound_image(compound_image)
        # 结合方式：concat

        # xc=torch.cat((compound_emd,protein_emd),1).float()
        # print(xc.dtype)
        # 最后获得维度【batch_size,2*hid_dim】
        # 结合方式：attention
        # 计算相关性分数
        # score_matrix = torch.matmul(protein_emd, compound_emd.t())
        # 归一化得到注意力权重
        # attention_weights = F.softmax(score_matrix, dim=1)
        # 加权组合得到最终特征向量
        # combined_emd = torch.matmul(attention_weights, compound_emd)
        # 最终获得维度【batch_size,hid_dim】
        # 结合方式：decoder
        # 结合方式：matmul
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        #print(out)
        return out,loss