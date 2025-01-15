import torch.nn.functional as F
import torch
import torch.nn as nn

# Proposed Encoder
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(ConvBlock, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv(x)

class BranchBlock(nn.Module):
    def __init__(self, kernel_size, in_channels=1, out_channels=64, stride=1):
        super(BranchBlock, self).__init__()
        self.block1 = ConvBlock(in_channels, out_channels, kernel_size, stride)
        self.block2 = ConvBlock(out_channels, out_channels, kernel_size, stride)
        
    def forward(self, x):
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        out1 = self.block1(x)
        out2 = self.block2(out1)
        return out2 + out1

class MultiStageEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, num_branches=3, final_channels=3):
        super(MultiStageEncoder, self).__init__()
        self.branches = nn.ModuleList([
            BranchBlock(kernel_size=3+2*i, in_channels=in_channels, out_channels=out_channels)
            for i in range(num_branches)
        ])
        
        # Add projection layer to reduce channels to final_channels (3)
        self.projection = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, final_channels, 1)
        )
        
    def forward(self, x):
        if x.size(-1) != 64 or x.size(-2) != 64:
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Get multi-scale features
        branch_outputs = [branch(x) for branch in self.branches]
        combined = torch.cat(branch_outputs, dim=1)
        
        # Project to final number of channels
        return self.projection(combined)
