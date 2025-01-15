import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
import torch.nn.functional as F

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.MSEncoder.MSEncoder import MultiStageEncoder

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class DecoderBlock(nn.Module):
    def __init__(self, in_channels=3):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        if x.size(-1) != 64 or x.size(-2) != 64:
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.tanh(self.conv3(x))
        return x

class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        
        # Initialize encoder with 3 output channels to match UNet expectations
        self.vqgan = MultiStageEncoder(in_channels=1, out_channels=64, num_branches=3, final_channels=3)
        self.decoder = DecoderBlock(in_channels=3)
        
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
            
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError

    def forward(self, x, x_cond, context=None):
        with torch.no_grad():
            # Ensure grayscale input
            if x.shape[1] != 1:
                x = x.mean(dim=1, keepdim=True)
            if x_cond.shape[1] != 1:
                x_cond = x_cond.mean(dim=1, keepdim=True)
            
            # Ensure correct spatial dimensions
            if x.size(-1) != 64 or x.size(-2) != 64:
                x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
            if x_cond.size(-1) != 64 or x_cond.size(-2) != 64:
                x_cond = F.interpolate(x_cond, size=(64, 64), mode='bilinear', align_corners=False)
                
            x_latent = self.encode(x, cond=False)
            x_cond_latent = self.encode(x_cond, cond=True)
            
        context = self.get_cond_stage_context(x_cond)
        return super().forward(x_latent.detach(), x_cond_latent.detach(), context)

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        x_latent = self.vqgan(x)
        
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        
        out = self.decoder(x_latent)
        return out

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        #return context
        return None
        
    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet, Decoder")
            params = itertools.chain(self.denoise_fn.parameters(), 
                                   self.cond_stage_model.parameters(),
                                   self.decoder.parameters())
        else:
            print("get parameters to optimize: UNet, Decoder")
            params = itertools.chain(self.denoise_fn.parameters(), 
                                   self.decoder.parameters())
        return params

    @torch.no_grad()
    def sample(self, x_cond, clip_denoised=False, sample_mid_step=False):
        if x_cond.shape[1] != 1:
            x_cond = x_cond.mean(dim=1, keepdim=True)
        if x_cond.size(-1) != 64 or x_cond.size(-2) != 64:
            x_cond = F.interpolate(x_cond, size=(64, 64), mode='bilinear', align_corners=False)
            
        x_cond_latent = self.encode(x_cond, cond=True)
        
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(
                y=x_cond_latent,
                context=self.get_cond_stage_context(x_cond),
                clip_denoised=clip_denoised,
                sample_mid_step=sample_mid_step
            )
            
            out_samples = []
            for i in tqdm(range(len(temp)), desc="save output sample mid steps"):
                out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), desc="save one step sample mid steps"):
                out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
                
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop(
                y=x_cond_latent,
                context=self.get_cond_stage_context(x_cond),
                clip_denoised=clip_denoised,
                sample_mid_step=sample_mid_step
            )
            out = self.decode(temp, cond=False)
            return out
