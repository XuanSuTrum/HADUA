import torch
import torch.nn.functional as F
import numpy as np

class MatchWeighting:
    def __init__(self, num_classes, momentum=0.9, lambda_max=1.0, temperature=1.0):
        self.num_classes = num_classes
        self.momentum = momentum  # EMA 的动量参数
        self.lambda_max = lambda_max
        self.mu = 0.5
        self.sigma = 1.0
        # 初始化类别分布估计
        self.class_dist = (torch.ones(num_classes) / num_classes).cuda()  # 初始均匀分布
        self.alpha = 0.3  # 初始对齐强度
        self.temperature = temperature  # 调整因子的温度参数

    def update_gaussian_params(self, probabilities):
        max_probs = torch.max(probabilities, dim=1)[0]
        batch_mu = torch.mean(max_probs).item()
        batch_sigma = torch.std(max_probs, unbiased=True).item()
        self.mu = self.momentum * self.mu + (1 - self.momentum) * batch_mu
        self.sigma = self.momentum * self.sigma + (1 - self.momentum) * max(batch_sigma, 1e-5)

    def update_class_dist(self, probabilities):
        """动态更新类别分布"""
        batch_class_dist = torch.mean(probabilities, dim=0)  # 当前批次的类别分布
        self.class_dist = self.momentum * self.class_dist + (1 - self.momentum) * batch_class_dist

    def uniform_alignment(self, probabilities, current_epoch):
        """优化后的 Uniform Alignment"""
        # 使用 sigmoid 衰减调整对齐强度
        effective_alpha = self.alpha / (1 + np.exp((current_epoch - 20) / 6))
        uniform_dist = torch.ones(self.num_classes, device=probabilities.device) / self.num_classes
        adjusted_dist = effective_alpha * uniform_dist + (1 - effective_alpha) * self.class_dist
        # 增强调整因子，引入温度参数
        adjust_factor = (adjusted_dist / (self.class_dist + 1e-8)) ** self.temperature
        adjusted_probs = probabilities * adjust_factor
        adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)
        return adjusted_probs

    def compute_weights(self, probabilities, current_epoch):
        self.update_class_dist(probabilities)  # 更新类别分布
        adjusted_probs = self.uniform_alignment(probabilities, current_epoch)
        max_probs = torch.max(adjusted_probs, dim=1)[0]
        self.update_gaussian_params(probabilities)
        weights = torch.ones_like(max_probs) * self.lambda_max
        mask = max_probs < self.mu
        weights[mask] = self.lambda_max * torch.exp(
            -((max_probs[mask] - self.mu) ** 2) / (2 * self.sigma ** 2)
        )
        return weights