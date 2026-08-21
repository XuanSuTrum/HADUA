import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
class feature_extractor(nn.Module):
    def __init__(self,hidden_1 = 64,hidden_2 =64):
         super(feature_extractor,self).__init__()
         self.fc1=nn.Linear(310,hidden_1)
         self.fc2=nn.Linear(hidden_1,hidden_2)
         self.dropout1 = nn.Dropout(p=0.25)
         self.dropout2 = nn.Dropout(p=0.25)

    def forward(self,x):
         x=self.fc1(x)
         x=F.relu(x)
         x=self.fc2(x)
         x=F.relu(x)
         return x
    def get_parameters(self) -> List[Dict]:
         params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
                  ]
         return params
class CFE(nn.Module):
    def __init__(self):
        super(CFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(310, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.module[0].parameters(), "lr_mult": 1},  # nn.Linear(310, 256)
            {"params": self.module[3].parameters(), "lr_mult": 1},  # nn.Linear(256, 128)
            {"params": self.module[6].parameters(), "lr_mult": 1},  # nn.Linear(128, 64)
        ]
        return params

class CFE_eye(nn.Module):
    def __init__(self):
        super(CFE_eye, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(33, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
    def forward(self, x):
        x = self.module(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.module[0].parameters(), "lr_mult": 1},  # nn.Linear(310, 256)
            {"params": self.module[3].parameters(), "lr_mult": 1},  # nn.Linear(256, 128)
            {"params": self.module[6].parameters(), "lr_mult": 1},  # nn.Linear(128, 64)
        ]
        return params

network_dict = {"simple_net": feature_extractor,
                'CFE': CFE,
                'CFE_eye': CFE_eye}