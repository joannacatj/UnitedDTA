import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.nn import GATConv,GINConv,global_add_pool
from torch.nn import Sequential, Linear, ReLU

from model.graph_transformer_edge_layer import GraphTransformerLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dgl_split(bg, feats):
    max_num_nodes = int(bg.batch_num_nodes().max())  # 计算批处理中最大的节点数量，以便为重排后的特征数据创建正确的大小。
    batch = torch.cat(
        [torch.full((1, x.type(torch.int)), y) for x, y in zip(bg.batch_num_nodes(), range(bg.batch_size))],
        dim=1).reshape(-1).type(torch.long).to(bg.device)
    cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(
        dim=0)])  # 根据每个批次中的节点数量，为每个节点分配一个批次索引。这将创建一个大小为(总节点数量,)的张量，其中每个元素对应一个节点的批次索引。
    # print(cum_nodes)
    idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)  # 创建一个从0到总节点数量的索引张量。
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)  # 根据节点的批次索引和累积节点数，计算节点在重排后特征数据中的索引。
    size = [bg.batch_size * max_num_nodes] + list(feats.size())[
                                             1:]  # 计算重排后特征数据的大小，其中批次维度为bg.batch_size * max_num_nodes，其余维度与输入特征数据相同。
    out = feats.new_full(size, fill_value=0)
    out[idx] = feats  # 将输入特征数据按照计算得到的索引重排到out张量中。
    out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[
                                                    1:])  # 按照批次大小和最大节点数对重排后的特征数据进行形状变换，使其成为大小为[bg.batch_size, max_num_nodes] + list(feats.size())[1:]的张量。
    return out
class GCNNet(torch.nn.Module):
    def __init__(self,num_features,output_dim=128,
                 dropout=0.2):
        '''
        定义一个名为GCNNet的神经网络模型，它继承自torch.nn.Module类。
        在__init__函数中定义了模型的各种参数，包括输出层的数量（n_output）、过滤器数量（n_filters）、输入特征的维度（embed_dim）、药物特征的数量（num_features_xd）、蛋白质特征的数量（num_features_xt）、输出特征的维度（output_dim）和dropout率（dropout）等。
        '''
        super(GCNNet, self).__init__()  # 这段代码是调用父类的初始化函数，以便在新的类GCNNet中继承所有父类的属性和方法，方便调用

        # 处理SMILES的神经网络模型
        #self.n_output = n_output  # 表示输出的维度即模型最后一层的输出维度
        self.conv1 = GCNConv(num_features,
                             num_features)  # 这行代码创建了一个实例，并将其赋值给属性变量self.conv1，该层是基于图形数据的卷积神经网络，适用于图形数据的特征提取，两个参数分别代表输入输出维度，在forward函数中通过调用self.conv1(x,edge_index)来进行图卷积操作，这个也就是第一层的图卷积神经网络
        self.conv2 = GCNConv(num_features, num_features * 2)  # 创建第二层图卷积神经网络，同上只不过这里输入输出维度变了注意一下
        self.conv3 = GCNConv(num_features * 2, num_features * 4)  # 创建第三层图卷积神经网络，这里输入输出维度也改变了
        self.fc_g1 = torch.nn.Linear(num_features * 4, 1024)  # 表示全连接层，输入维度为78*4，输出维度为1024
        self.fc_g2 = torch.nn.Linear(1024, output_dim)  # 表示全连接层，输入维度为1024，输出维度为128（潜在变量）
        self.relu = nn.ReLU()  # 表示激活函数relu
        self.dropout = nn.Dropout(dropout)  # 表示正则化Dropout，防止过拟合
    def forward(self,g):
        x=g.ndata['feats'].to(device)#格式为[num_nodes,num_features]
        # 使用 edges() 方法获取图中所有边的源节点和目标节点的索引
        x = x.float()
        src, dst = g.edges()
        # 将 (src, dst) 张量元组转换为维度为 [edge_nums, 2] 的 torch.tensor
        edge_index = torch.stack([src, dst], dim=1).T
        #print(edge_index.dtype)
        batch=g.ndata['graph_id'].to(device)
        # 图模型前向传播
        x = self.conv1(x, edge_index)  # 第一层GCN
        x = self.relu(x)  # 非线性激活函数

        x = self.conv2(x, edge_index)  # 第二层GCN
        x = self.relu(x)

        x = self.conv3(x, edge_index)  # 第三层GCN
        x = self.relu(x)
        out = dgl_split(g, x)
        out = self.relu(self.fc_g1(out))  # 先进行全连接层，再用激活函数激活
        out = self.dropout(out)  # 正则化
        out = self.fc_g2(out)
        out = self.dropout(out)
        x = gmp(x, batch)  # 全局最大池化,batch
        x = self.relu(self.fc_g1(x))  # 先进行全连接层，再用激活函数激活
        x = self.dropout(x)  # 正则化
        x = self.fc_g2(x)
        x = self.dropout(x)
        return out,x

class GATNet(torch.nn.Module):
    def __init__(self,num_features,output_dim=128,dropout=0.2):
        super(GATNet,self).__init__()
        #图卷积模块
        self.gcn1 = GATConv(num_features,num_features,heads=10,dropout=dropout)       #加入了dropout层防止过拟合
        self.gcn2 = GATConv(num_features*10,output_dim,dropout=dropout)                  #加入了多头注意力10输入特征维度也要乘10
        self.fc_g1 = nn.Linear(output_dim,output_dim)
        #激活和正则化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, g):
        x = g.ndata['feats'].to(device)  # 格式为[num_nodes,num_features]
        # 使用 edges() 方法获取图中所有边的源节点和目标节点的索引
        x = x.float()
        src, dst = g.edges()
        # 将 (src, dst) 张量元组转换为维度为 [edge_nums, 2] 的 torch.tensor
        edge_index = torch.stack([src, dst], dim=1).T
        batch = g.ndata['graph_id'].to(device)
        # 药物模型
        x = F.dropout(x, p=0.2, training=self.training)  # 这里的training=self.training是保证训练集采用正则化dropout而测试集不用
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        out = dgl_split(g, x)
        out=self.fc_g1(out)
        out = self.relu(out)
        x = gmp(x, batch)
        x = self.fc_g1(x)
        x = self.relu(x)
        return out,x

class GINConvNet(torch.nn.Module):
    def __init__(self, num_features, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        #图卷积模型
        nn1 = Sequential(Linear(num_features,dim),ReLU(),Linear(dim,dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)                                 #批量归一化用来提高模型泛化能力

        nn2 = Sequential(Linear(dim,dim),ReLU(),Linear(dim,dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim,dim),ReLU(),Linear(dim,dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        self.fc1_xd = nn.Linear(dim,output_dim)

    def forward(self, g):
        x = g.ndata['feats'].to(device)  # 格式为[num_nodes,num_features]
        # 使用 edges() 方法获取图中所有边的源节点和目标节点的索引
        x = x.float()
        src, dst = g.edges()
        # 将 (src, dst) 张量元组转换为维度为 [edge_nums, 2] 的 torch.tensor
        edge_index = torch.stack([src, dst], dim=1).T
        batch = g.ndata['graph_id'].to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return x

class GraphTransformer(nn.Module):
    def __init__(self, device, n_layers, node_dim, edge_dim, hidden_dim, out_dim, n_heads, in_feat_dropout, dropout,
                 pos_enc_dim):
        super(GraphTransformer, self).__init__()

        self.device = device
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.linear_h = nn.Linear(node_dim, hidden_dim)
        self.linear_e = nn.Linear(edge_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout, self.layer_norm,
                                                           self.batch_norm, self.residual)
                                     for _ in range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, n_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))

    def dgl_split(self, bg, feats):
        max_num_nodes = int(bg.batch_num_nodes().max())#计算批处理中最大的节点数量，以便为重排后的特征数据创建正确的大小。
        batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in zip(bg.batch_num_nodes(), range(bg.batch_size))],
                       dim=1).reshape(-1).type(torch.long).to(bg.device)
        cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])#根据每个批次中的节点数量，为每个节点分配一个批次索引。这将创建一个大小为(总节点数量,)的张量，其中每个元素对应一个节点的批次索引。
        #print(cum_nodes)
        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)#创建一个从0到总节点数量的索引张量。
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)#根据节点的批次索引和累积节点数，计算节点在重排后特征数据中的索引。
        size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]#计算重排后特征数据的大小，其中批次维度为bg.batch_size * max_num_nodes，其余维度与输入特征数据相同。
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats#将输入特征数据按照计算得到的索引重排到out张量中。
        out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])#按照批次大小和最大节点数对重排后的特征数据进行形状变换，使其成为大小为[bg.batch_size, max_num_nodes] + list(feats.size())[1:]的张量。
        return out

    def forward(self, g):
        # input embedding
        batch = g.ndata['graph_id'].to(device)
        g = g.to(self.device)
        h = g.ndata['feats'].float().to(self.device)
        h_lap_pos_enc = g.ndata['lap_pos_enc'].to(self.device)
        e = g.edata['feats'].float().to(self.device)

        # sign_flip = torch.rand(h_lap_pos_enc.size(1), device=h.device)
        # sign_flip[sign_flip >= 0.5] = 1.0
        # sign_flip[sign_flip < 0.5] = -1.0
        # h_lap_pos_enc = h_lap_pos_enc * sign_flip.unsqueeze(0)

        h = self.linear_h(h)
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        h = h + h_lap_pos_enc
        h = self.in_feat_dropout(h)

        e = self.linear_e(e)

        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)

        g.ndata['h'] = h
        output=self.dgl_split(g,h)
        x = gmp(h, batch)
        # h = dgl.mean_nodes(g, 'h')

        return output,x