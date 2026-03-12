import torch.nn as nn
import mmd
import backbone
import torch
import cmmd
import cmmd_2
import cmmd_3
import cmmd_4
import cmmd_5
import numpy as np
from guessmatch import MatchWeighting
# from softmatch import SoftMatchWeighting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
# from softmatch_2 import SoftMatchWeighting
GAMMA = 10 ** 3

class SelfAttention(nn.Module):
    def __init__(self, dropout):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(d, dtype=torch.float))
        self.attention_weights = torch.softmax(scores, dim=2)
        return torch.bmm(self.dropout(self.attention_weights), values)

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_hiddens, num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = SelfAttention(dropout)
        self.W_q = nn.Linear(feature_dim, num_hiddens, bias=bias)
        self.W_k = nn.Linear(feature_dim, num_hiddens, bias=bias)
        self.W_v = nn.Linear(feature_dim, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class Transfer_Net(nn.Module):
    def __init__(self, num_class, base_net='CFE', base_net_eye='CFE_eye', transfer_loss='mmd',
                 use_bottleneck=False, width=32, confidence_threshold=0, num_hiddens=128, num_heads=4):
        super(Transfer_Net, self).__init__()
        self.base_network = backbone.network_dict[base_net]()
        self.base_network_eye = backbone.network_dict[base_net_eye]()

        self.eeg_dim = 64
        self.eye_dim = 64

        self.self_attention_eeg = MultiHeadAttention(self.eeg_dim, num_hiddens, num_heads, 0.5)
        self.self_attention_eye = MultiHeadAttention(self.eye_dim, num_hiddens, num_heads, 0.5)
        self.cross_attention = MultiHeadAttention(self.eeg_dim, num_hiddens, num_heads, 0.5)

        self.P = torch.randn(num_class, 64)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        self.confidence_threshold = confidence_threshold
        self.num_class = num_class
        # 分类器输入维度改为 num_hiddens * 3
        classifier_layer_list = [nn.Linear(num_hiddens * 3, width), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(width, num_class)]
        self.classifier = nn.Sequential(*classifier_layer_list)
        self.softmax = nn.Softmax(dim=1)

    def fuse_features(self, eeg_features, eye_features):
        # 脑电自注意力
        eeg_self = self.self_attention_eeg(eeg_features.unsqueeze(1), eeg_features.unsqueeze(1),
                                           eeg_features.unsqueeze(1))
        # 眼动自注意力
        eye_self = self.self_attention_eye(eye_features.unsqueeze(1), eye_features.unsqueeze(1),
                                           eye_features.unsqueeze(1))
        # 脑电查询眼动的交叉注意力
        cross = self.cross_attention(eeg_features.unsqueeze(1), eye_features.unsqueeze(1), eye_features.unsqueeze(1))
        # 融合所有结果
        fused = torch.cat([eeg_self.squeeze(1), eye_self.squeeze(1), cross.squeeze(1)],
                          dim=1)  # (batch_size, num_hiddens * 3)
        return fused

    def forward(self, e, source, target, s_label):

        # 提取特征
        source_data = self.base_network(source[:, :310])
        target_data = self.base_network(target[:, :310])
        source_data_eye = self.base_network_eye(source[:, 310:])
        target_data_eye = self.base_network_eye(target[:, 310:])

        # 融合特征
        fused_source = self.fuse_features(source_data, source_data_eye)
        fused_target = self.fuse_features(target_data, target_data_eye)

        # 分类器输出
        source_clf = self.classifier(fused_source)
        target_clf = self.classifier(fused_target)

        # 目标域预测概率
        t_label_ = target_clf.data.max(1)[1]  # 可保留用于其他用途，但 CMMD 不需要
        target_clf = self.softmax(target_clf)
        target_probabilities = torch.max(target_clf, dim=1)[0]

        # 初始化 SoftMatch 加权模块
        match = MatchWeighting(self.num_class, momentum=0.999, lambda_max=1.0)
        weights = match.compute_weights(target_clf, e)

        # 置信度筛选
        confidence_mask = target_probabilities > self.confidence_threshold
        confident_target_data = fused_target[confidence_mask]

        # 确保 s_label 是 one-hot 编码
        if s_label.dim() == 1:  # 如果输入是类别索引
            s_label = nn.functional.one_hot(s_label, num_classes=self.num_class).float()

        # 转移损失
        transfer_loss = self.adapt_loss(fused_source, fused_target, self.transfer_loss)

        # CMMD 损失（使用软标签）
        cmmd_loss = cmmd_5.cmmd(
            fused_source,
            confident_target_data,
            s_label,  # 源域 one-hot 标签
            target_clf[confidence_mask],  # 目标域软标签
            weights[confidence_mask]  # 权重
        )
        return source_clf, transfer_loss, cmmd_loss

    def predict(self, x):
        features = self.base_network(x[:, :310])
        features_eye = self.base_network_eye(x[:, 310:])
        fused = self.fuse_features(features, features_eye)
        clf = self.classifier(fused)
        return clf

    def adapt_loss(self, X, Y, adapt_loss):
        loss = mmd.mmd_rbf_noaccelerate(X, Y)
        return loss

    def visualization(self, source, source_labels, target, target_labels, tsne=1):
        # 提取源域和目标域的 EEG 和眼动特征
        feature_source_eeg = self.base_network(source[:, :310])
        feature_source_eye = self.base_network_eye(source[:, 310:])
        feature_target_eeg = self.base_network(target[:, :310])
        feature_target_eye = self.base_network_eye(target[:, 310:])

        # 融合特征
        fused_source = self.fuse_features(feature_source_eeg, feature_source_eye)
        fused_target = self.fuse_features(feature_target_eeg, feature_target_eye)

        # 提取分类器输出
        source_feature = self.classifier(fused_source)
        target_feature = self.classifier(fused_target)

        # 将结果转为 numpy 数组
        source_feature = source_feature.cpu().detach().numpy()
        fused_source = fused_source.cpu().detach().numpy()
        source_labels = np.argmax(source_labels.cpu().detach().numpy(), axis=1)

        target_feature = target_feature.cpu().detach().numpy()
        fused_target = fused_target.cpu().detach().numpy()
        target_labels = np.argmax(target_labels.cpu().detach().numpy(), axis=1)

        colors1 = '#00CED1'  # 点的颜色蓝绿色
        colors2 = '#DC143C'  # 深红色
        colors3 = '#008000'  # 绿色
        area = 0.5 ** 2  # 点面积

        if tsne == 0:
            # 绘制源域的散点图
            x0_source = fused_source[np.where(source_labels == 0)[0]]
            x1_source = fused_source[np.where(source_labels == 1)[0]]
            x2_source = fused_source[np.where(source_labels == 2)[0]]

            # 绘制目标域的散点图
            x0_target = fused_target[np.where(target_labels == 0)[0]]
            x1_target = fused_target[np.where(target_labels == 1)[0]]
            x2_target = fused_target[np.where(target_labels == 2)[0]]

            # 画散点图
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(x0_source[:, 0], x0_source[:, 1], x0_source[:, 2], s=area, c=colors1, alpha=0.1,
                       label='Source Class 0')
            ax.scatter(x1_source[:, 0], x1_source[:, 1], x1_source[:, 2], s=area, c=colors2, alpha=0.1,
                       label='Source Class 1')
            ax.scatter(x2_source[:, 0], x2_source[:, 1], x2_source[:, 2], s=area, c=colors3, alpha=0.4,
                       label='Source Class 2')

            ax.scatter(x0_target[:, 0], x0_target[:, 1], x0_target[:, 2], s=area, c=colors1, alpha=0.4, marker='^',
                       label='Target Class 0')
            ax.scatter(x1_target[:, 0], x1_target[:, 1], x1_target[:, 2], s=area, c=colors2, alpha=0.4, marker='^',
                       label='Target Class 1')
            ax.scatter(x2_target[:, 0], x2_target[:, 1], x2_target[:, 2], s=area, c=colors3, alpha=0.4, marker='^',
                       label='Target Class 2')

            ax.legend()
            plt.show()
            # plt.savefig('tsne_3d.png')

        else:
            # 使用 t-SNE 进行降维
            source_feature_tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=3000).fit_transform(
                fused_source.astype('float32'))
            target_feature_tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=3000).fit_transform(
                fused_target.astype('float32'))

            # 绘制源域的散点图
            x0_source = source_feature_tsne[np.where(source_labels == 0)[0]]
            x1_source = source_feature_tsne[np.where(source_labels == 1)[0]]
            x2_source = source_feature_tsne[np.where(source_labels == 2)[0]]

            # 绘制目标域的散点图
            x0_target = target_feature_tsne[np.where(target_labels == 0)[0]]
            x1_target = target_feature_tsne[np.where(target_labels == 1)[0]]
            x2_target = target_feature_tsne[np.where(target_labels == 2)[0]]

            plt.scatter(x0_source[:, 0], x0_source[:, 1], c='none', marker='o', edgecolors=colors1, alpha=0.5,
                        label='Source Class 0')
            plt.scatter(x1_source[:, 0], x1_source[:, 1], c='none', marker='o', edgecolors=colors2, alpha=0.5,
                        label='Source Class 1')
            plt.scatter(x2_source[:, 0], x2_source[:, 1], c='none', marker='o', edgecolors=colors3, alpha=0.5,
                        label='Source Class 2')

            plt.scatter(x0_target[:, 0], x0_target[:, 1], c='none', marker='*', edgecolors=colors1, alpha=0.5,
                        label='Target Class 0')
            plt.scatter(x1_target[:, 0], x1_target[:, 1], c='none', marker='*', edgecolors=colors2, alpha=0.5,
                        label='Target Class 1')
            plt.scatter(x2_target[:, 0], x2_target[:, 1], c='none', marker='*', edgecolors=colors3, alpha=0.5,
                        label='Target Class 2')

            plt.legend().set_visible(False)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            # plt.savefig('tsne_2d.png')