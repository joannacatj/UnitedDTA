import math
import torch.nn.functional as F
import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, att_dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.att_dropout = att_dropout

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        self.Wq = nn.Linear(emb_dim, emb_dim, bias=False).to(device)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=False).to(device)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=False).to(device)

        self.fc = nn.Linear(emb_dim, emb_dim).to(device)

    def forward(self, x, pad_mask=None):
        # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        batch_size = x.size(0)

        # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # 分头 [batch_szie, num_heads, seq_len, depth] = [3, 8, 5, 512/8=64]
        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_szie, num_heads, seq_len, seq_len] = [3, 8, 5, 5]
        att_weights = torch.matmul(Q, K.transpose(-2, -1))
        att_weights = att_weights / math.sqrt(self.depth)

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, seq_len, seq_len] -> [batch_size, nums_head, seq_len, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)

        # 自己的多头注意力效果没有torch的好，我猜是因为它的dropout给了att权重，而不是fc
        if self.att_dropout > 0.0:
            att_weights = F.dropout(att_weights, p=self.att_dropout)

        # [batch_szie, num_heads, seq_len, depth] = [3, 8, 5, 64]
        output = torch.matmul(att_weights, V)

        # 不同头的结果拼接 [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)

        output = self.fc(output)
        return output

class CrossMultiAttention(nn.Module):
    def __init__(self,emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(CrossMultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads


        #self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim).to(device)
        self.Wk = nn.Linear(emb_dim, emb_dim).to(device)
        self.Wv = nn.Linear(emb_dim, emb_dim).to(device)

        #self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x, context, pad_mask=None):
        '''

        :param x: [batch_size,seq_len,emb_dim]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        batch_size= x.shape[0]
        Q = self.Wq(x)  # [batch_size, seq_len, emb_dim] = [3, 262144, 512]
        K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(context)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, h*w, depth]
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)   # [batch_size, seq_len, emb_dim]
        #输出的结果是x的seq_len!
        #print(out.shape)
        return out
class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1).to(device)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1).to(device)  # convolution neural units
        self.do = nn.Dropout(dropout).to(device)
    def forward(self, x):
        # x = [batch size, sent len, hid dim]
        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]
        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch size, pf dim, sent len]
        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]
        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]
        return x
class EncoderLayer_Self(nn.Module):
    def __init__(self,emd_dim,n_heads,pf_dim,dropout,multiheadattention,positionwisefeedforward):
        super().__init__()
        self.ln = nn.LayerNorm(emd_dim).to(device)
        self.sa = multiheadattention(emd_dim, n_heads, dropout)
        self.ea = multiheadattention(emd_dim, n_heads)
        self.pf = positionwisefeedforward(emd_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
    def forward(self, trg,trg_mask=None):
        trg= self.sa(trg,trg_mask)
        trg = self.ln(trg + self.do(trg))
        trg= self.ea(trg, trg_mask)
        trg = self.ln(trg + self.do(trg))
        trg = self.ln(trg + self.do(self.pf(trg)))
        #输出维度为[batch_size,seq_len,hid_dim]
        return trg
class EncoderLayer_Cross(nn.Module):
    def __init__(self,emd_dim,n_heads,pf_dim,dropout,crossmultiattention,positionwisefeedforward):
        super().__init__()
        self.ln = nn.LayerNorm(emd_dim)
        self.sa = crossmultiattention(emd_dim, n_heads, dropout)
        self.ea = crossmultiattention(emd_dim, n_heads)
        self.pf = positionwisefeedforward(emd_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
    def forward(self,trg,src,trg_mask=None,src_mask=None):
        trg = self.sa(trg,trg,trg_mask)

        trg = self.ln(trg + self.do(trg))

        trg = self.ea(trg, src, src_mask)

        trg = self.ln(trg + self.do(trg))

        trg = self.ln(trg + self.do(self.pf(trg)))
        #输出维度为[batch_size,seq_len,hid_dim]
        return trg

class Protein_Fusion(nn.Module):
    def __init__(self,emd_dim,n_layers,n_heads,pf_dim,dropout,fusion_layer,use_bottleneck,test_with_bottlenecks,encoderlayer_self,encoderlayer_cross,self_attention,cross_attention,feedfowardposition):
        super(Protein_Fusion, self).__init__()
        self.emd_dim=emd_dim
        self.n_layers=n_layers
        self.n_heads=n_heads
        self.pf_dim=pf_dim
        self.dropout=dropout
        self.fusion_layer=fusion_layer
        self.use_bottleneck=use_bottleneck
        self.test_with_bottlenecks=test_with_bottlenecks
        self.encoderlayer_self=encoderlayer_self
        self.encoderlayer_cross=encoderlayer_cross
        self.self_attention=self_attention
        self.cross_attention=cross_attention
        self.feedforwardposition=feedfowardposition

    def forward(self,seq_emd,graph_emd,esm_emd,bottleneck,train=True):
        #print('before seq',seq_emd.shape)
        #print('before graph',graph_emd.shape)
        #print('before esm',esm_emd.shape)
        x_combined=None
        use_bottlenecks = train or self.test_with_bottlenecks
        for lyr in range(self.n_layers):
            encoders={}
            encoders['seq_self']=self.encoderlayer_self(self.emd_dim,self.n_heads,self.pf_dim,self.dropout,self.self_attention,self.feedforwardposition)
            encoders['graph_self'] = self.encoderlayer_self(self.emd_dim, self.n_heads, self.pf_dim, self.dropout,self.self_attention,self.feedforwardposition)
            encoders['esm_self'] = self.encoderlayer_self(self.emd_dim, self.n_heads, self.pf_dim, self.dropout,self.self_attention,self.feedforwardposition)
            encoders['seq_cross']=self.encoderlayer_cross(self.emd_dim,self.n_heads,self.pf_dim,self.dropout,self.cross_attention,self.feedforwardposition)
            encoders['graph_cross'] = self.encoderlayer_cross(self.emd_dim, self.n_heads, self.pf_dim, self.dropout,self.cross_attention,self.feedforwardposition)
            encoders['esm_cross'] = self.encoderlayer_cross(self.emd_dim, self.n_heads, self.pf_dim, self.dropout,self.cross_attention,self.feedforwardposition)
            if (lyr < self.fusion_layer or (self.use_bottleneck and not use_bottlenecks)):
                seq_emd=encoders['seq_self'](seq_emd)
                graph_emd=encoders['graph_self'](graph_emd)
                esm_emd=encoders['esm_self'](esm_emd)
            else:
                if self.use_bottleneck:
                    bottle=[]
                    t_mod=seq_emd.shape[1]
                    in_mod=torch.cat((seq_emd,bottleneck),1)
                    out_mod=encoders['seq_self'](in_mod)
                    seq_emd= out_mod[:, :t_mod]  # 将编码器的输出切片，获取与原始输入模态相同时间步数的部分作为新的输入数据
                    #print(out_mod[:, t_mod:].shape)
                    bottle.append(out_mod[:, t_mod:])
                    t_mod = graph_emd.shape[1]
                    in_mod = torch.cat((graph_emd, bottleneck), 1)
                    out_mod = encoders['graph_self'](in_mod)
                    #print(out_mod[:, t_mod:].shape)
                    graph_emd = out_mod[:, :t_mod]  # 将编码器的输出切片，获取与原始输入模态相同时间步数的部分作为新的输入数据
                    bottle.append(out_mod[:, t_mod:])
                    t_mod = esm_emd.shape[1]
                    in_mod = torch.cat((esm_emd, bottleneck), 1)
                    out_mod = encoders['seq_self'](in_mod)
                    esm_emd = out_mod[:, :t_mod]  # 将编码器的输出切片，获取与原始输入模态相同时间步数的部分作为新的输入数据
                    #print(out_mod[:, t_mod:].shape)
                    #print('after seq', seq_emd.shape)
                    #print('after graph', graph_emd.shape)
                    #print('after esm', esm_emd.shape)
                    bottle.append(out_mod[:, t_mod:])
                    # 在axis=-1维度上堆叠张量
                    stacked = torch.stack(bottle, dim=-1)
                    # 在axis=-1维度上取平均值
                    bottleneck = torch.mean(stacked, dim=-1)
                else:
                    other_modal=torch.cat((graph_emd,esm_emd),1)
                    seq_emd_new=encoders['seq_cross'](seq_emd,other_modal)
                    other_modal = torch.cat((seq_emd, esm_emd), 1)
                    graph_emd_new = encoders['graph_cross'](graph_emd, other_modal)
                    other_modal = torch.cat((graph_emd, seq_emd), 1)
                    esm_emd_new = encoders['seq_cross'](esm_emd, other_modal)
                    x_combined=torch.cat([seq_emd_new, graph_emd_new, esm_emd_new], dim=1)
        if x_combined is not None:
            x_out=x_combined
        else:
            x_out=torch.cat([seq_emd, graph_emd, esm_emd], dim=1)
        encoded=torch.norm(x_out,dim=2)
        layer=nn.Linear(encoded.shape[1],128).to(device)
        encoded=layer(encoded)

        return encoded

