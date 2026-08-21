import torch
import torch.nn.functional as F

# 最小方差估计，用于数值稳定性
min_var_est = 1e-8

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, epsilon=1e-5):
    """
    计算高斯核矩阵，用于 CMMD 和 MMD 计算。
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2) + epsilon

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth = max(bandwidth, min_var_est)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def cmmd(source, target, s_label, t_probs, weights=None, kernel_mul=2.0, kernel_num=5, fix_sigma=None, cmmd_weight=0.1, reg_lambda=0.0, beta=0.5, epsilon=1e-8):
    """
    计算加权条件最大均值差异 (CMMD) 损失，支持软标签。
    """
    # 获取批次大小和类别数
    batch_size_s = source.size(0)
    batch_size_t = target.size(0)
    num_classes = s_label.size(1)

    # 确保 s_label 和 t_probs 是浮点型
    s_label = s_label.float()
    t_probs = t_probs.float()

    # 如果目标域为空，返回零损失
    if batch_size_t == 0:
        return torch.tensor(0.0, device=source.device)

    # 计算高斯核矩阵
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    # 提取子矩阵
    XX = kernels[:batch_size_s, :batch_size_s]
    YY = kernels[batch_size_s:, batch_size_s:]
    XY = kernels[:batch_size_s, batch_size_s:]
    YX = kernels[batch_size_s:, :batch_size_s]

    # 处理权重
    if weights is None:
        weights = torch.ones(batch_size_t, device=target.device)
    else:
        lambda_max = 1.0
        weights = torch.clamp(weights, min=0.0, max=lambda_max)
        weights = torch.pow(weights, beta)
        weights_t = weights.view(-1, 1)  # 不归一化总和

    # 初始化损失
    loss = 0.0

    # 对每个类别计算 CMMD 损失
    for c in range(num_classes):
        s_mask_c = s_label[:, c].view(-1, 1)  # (batch_size_s, 1)
        t_probs_c = t_probs[:, c].view(-1, 1)  # (batch_size_t, 1)
        weighted_t_probs_c = t_probs_c * weights_t

        # 源域内部的核均值
        sum_s_mask_c = torch.sum(s_mask_c)
        loss_XX_c = torch.sum(s_mask_c.t() @ XX @ s_mask_c) / (sum_s_mask_c ** 2 + epsilon)

        # 目标域内部的加权核均值
        sum_weighted_t_probs_c = torch.sum(weighted_t_probs_c)
        loss_YY_c = torch.sum(weighted_t_probs_c.t() @ YY @ weighted_t_probs_c) / (sum_weighted_t_probs_c ** 2 + epsilon)

        # 源域与目标域的加权核均值
        loss_XY_c = torch.sum(s_mask_c.t() @ XY @ weighted_t_probs_c) / (sum_s_mask_c * sum_weighted_t_probs_c + epsilon)

        # 目标域与源域的加权核均值
        loss_YX_c = torch.sum(weighted_t_probs_c.t() @ YX @ s_mask_c) / (sum_weighted_t_probs_c * sum_s_mask_c + epsilon)

        # 类别 c 的 CMMD 损失
        loss_c = loss_XX_c + loss_YY_c - loss_XY_c - loss_YX_c
        loss += loss_c

    # 归一化损失
    loss /= num_classes

    # 移除正则化项（或设为 0）
    # loss = loss + reg_lambda * loss ** 2

    # 应用缩放因子
    loss = cmmd_weight * loss
    loss = torch.relu(loss)  # 确保损失非负

    return loss