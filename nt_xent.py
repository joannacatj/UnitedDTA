import torch
import numpy as np
'''
在这段代码中，输入参数zis和zjs表示正样本和负样本的表示。它们的维度应该是 (batch_size, representation_dim)。

具体而言，zis和zjs都是表示样本的张量，其中 batch_size 是一个批次中样本的数量，representation_dim 是每个样本的表示维度。这段代码中的批次大小 batch_size 在初始化函数中传递给了 NTXentLoss 类。

假设 batch_size 是 16，representation_dim 是 128，那么 zis 和 zjs 的维度将是 (16, 128)。这意味着在每个批次中，正样本和负样本的表示都是一个维度为 128 的向量。
'''

class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size  # 批次大小
        self.temperature = temperature  # 温度参数
        self.device = device  # 设备（CPU或GPU）
        self.softmax = torch.nn.Softmax(dim=-1)  # softmax函数
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)  # 用于屏蔽相同表示的样本的掩码
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)  # 相似度计算函数
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")  # 交叉熵损失函数

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)  # 余弦相似度计算函数
            return self._cosine_similarity
        else:
            return self._dot_similarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)  # 对角矩阵
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)  # k=-batch_size的下对角矩阵
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)  # k=batch_size的上对角矩阵
        mask = torch.from_numpy((diag + l1 + l2))  # 相关掩码矩阵
        mask = (1 - mask).type(torch.bool)  # 将掩码矩阵转换为布尔类型
        return mask.to(self.device)  # 将掩码矩阵移动到指定设备上

    @staticmethod
    def _dot_similarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)  # 点积相似度计算函数
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_similarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))  # 余弦相似度计算函数
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)  # 将正样本和负样本的表示拼接在一起

        similarity_matrix = self.similarity_function(representations, representations)  # 计算相似度矩阵

        # 过滤出正样本的得分
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        #print(positives.shape)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature  # 对logits进行温度调节

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()  # 创建标签（全零）
        loss = self.criterion(logits, labels)  # 计算交叉熵损失

        return loss / (2 * self.batch_size)  # 返回归一化后的损失