
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
        out1 = self.block1(x)
        out2 = self.block2(out1)
        return out2 + out1

class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(192, 128, 3, padding=1)  # Input: 192 channels (64*3 from encoder branches)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)  # Changed output to 1 channel for grayscale
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # or you could use sigmoid if you prefer

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.tanh(self.conv3(x))  # Output range: [-1, 1]
        return x
        
class MultiStageEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, num_branches=3):
        super(MultiStageEncoder, self).__init__()
        self.branches = nn.ModuleList([
            BranchBlock(kernel_size=3+2*i, in_channels=in_channels, out_channels=out_channels)
            for i in range(num_branches)
        ])
        self.out_channels = out_channels * num_branches
        
    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        return torch.cat(branch_outputs, dim=1)