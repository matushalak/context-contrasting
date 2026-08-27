import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import torchvision
import math


class TransformerEncoder_(nn.Module):
    def __init__(self, emb_dim, num_heads, num_enc_layers, mlp_ratio=4, post_norm=True, return_attention_weights=False):
        super(TransformerEncoder_, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * mlp_ratio,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        self.post_norm = post_norm
        self.return_attention_weights = return_attention_weights
        if self.post_norm:
            self.norm = nn.LayerNorm(emb_dim)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attention_weights = None
        for i, layer in enumerate(self.transformer_encoder.layers):
            if i == len(self.transformer_encoder.layers) - 1 and self.return_attention_weights:
                src, attention_weights = self._extract_attention_weights(layer, src, src_mask, src_key_padding_mask)
            else:
                src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        if self.post_norm:
            src = self.norm(src)
        return src, attention_weights
    
    def _extract_attention_weights(self, layer, src, src_mask, src_key_padding_mask):
        attn_output, attn_output_weights = layer.self_attn(src, src, src, 
                                                           attn_mask=src_mask, 
                                                           key_padding_mask=src_key_padding_mask,
                                                           need_weights=True,
                                                           average_attn_weights=False)
        
        x = src
        if layer.norm_first:
            x = x + layer.dropout1(attn_output)
            x = x + layer._ff_block(layer.norm2(x))
        else:
            x = layer.norm1(x + layer.dropout1(attn_output))
            x = layer.norm2(x + layer._ff_block(x))
        return x, attn_output_weights



###### SeqJEPA_PLS ######
class SeqJEPA_PLS(nn.Module):
    def __init__(self, fovea_size, img_size, ema, ema_decay=0.996, **kwargs):
        super().__init__()
        self.fovea_size = fovea_size
        self.img_size = img_size
        self.ema = ema
        self.ema_decay = ema_decay
        self.n_channels = kwargs["n_channels"]
        self.num_classes = kwargs["num_classes"]
        self.num_heads = kwargs["num_heads"]
        self.num_enc_layers = kwargs["num_enc_layers"]
        self.action_projdim = kwargs["act_projdim"]
        self.action_latentdim = kwargs["act_latentdim"]
        self.act_cond = kwargs["act_cond"]
        self.learn_act_emb = kwargs["learn_act_emb"]
        self.cifar_resnet = kwargs["cifar_resnet"]
        self.criterion = nn.CosineSimilarity(dim=1)

        self.encoder = torchvision.models.resnet18(pretrained=False, zero_init_residual=True)
        self.res_out_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        
        if self.cifar_resnet:
            self.encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = torch.nn.Identity()
        if self.ema:
            self.target_encoder = copy.deepcopy(self.encoder)
            for param in self.target_encoder.parameters():
                param.requires_grad = False
        else:
            print("Not using EMA")
        
        if self.learn_act_emb:
            self.action_proj = nn.Sequential(
                nn.Linear(self.action_latentdim, self.action_projdim, bias=False), torch.nn.BatchNorm1d(self.action_projdim, affine=False))
            self.emb_dim = self.res_out_dim + self.action_projdim
        else:
            print("Not learning action embeddings")
            self.emb_dim = self.res_out_dim
        
        self.pred_hidden = kwargs["pred_hidden"]
        if self.learn_act_emb and self.act_cond:
            if self.pred_hidden <= 0:
                self.pred_hidden = self.res_out_dim
            self.predictor = nn.Sequential(
                nn.Linear(self.emb_dim+self.action_projdim, self.pred_hidden,bias=False),
                nn.BatchNorm1d(self.pred_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.pred_hidden, self.res_out_dim),)
        else:
            self.emb_dim = self.res_out_dim
            if self.pred_hidden <= 0:
                self.pred_hidden = self.res_out_dim
            self.predictor = nn.Sequential(
                nn.Linear(self.emb_dim, self.pred_hidden, bias=False),
                nn.BatchNorm1d(self.pred_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.pred_hidden, self.res_out_dim),)
                
        ### Transformer Encoder
        self.agg_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        self.transformer_encoder = TransformerEncoder_(self.emb_dim, self.num_heads,
                                                       self.num_enc_layers, mlp_ratio=4, post_norm=True)
        

    def _update_target_network(self):
        for online_params, target_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.ema_decay * target_params.data + (1 - self.ema_decay) * online_params.data

    def update_moving_average(self):
        """Update the target network using EMA."""
        self._update_target_network()
    
    def add_probes(self):
        for param in self.parameters():
            param.requires_grad = False
        self.pos_regressor = nn.Linear(self.res_out_dim*2, 2)
        self.agg_classifier = nn.Linear(self.emb_dim, self.num_classes)
        self.res_classifier = nn.Linear(self.res_out_dim, self.num_classes)
    
    def forward(self, fov_x_obs, fov_x_last, action_latents):
        ### pred encodings
        num_saccades = fov_x_obs.shape[1] + 1
        fov_x_obs = fov_x_obs.reshape(-1, self.n_channels, self.fovea_size, self.fovea_size)
        fov_x_last = fov_x_last.reshape(-1, self.n_channels, self.fovea_size, self.fovea_size)
        if self.ema:
            fov_encs_last = self.target_encoder(fov_x_last)
            fov_encs_last_detached = fov_encs_last.detach()
        else:
            fov_encs_last = self.encoder(fov_x_last)
            fov_encs_last_detached = fov_encs_last.detach() 
        ### obs encodings
        fov_encs_obs = self.encoder(fov_x_obs)
        fov_encs_obs = fov_encs_obs.reshape((-1, num_saccades-1, self.res_out_dim))
        ### action conditioning
        if self.act_cond:
            act_enc_obs = action_latents[:,:-1,:].reshape(-1, num_saccades-1, self.action_latentdim)
            act_enc_last = action_latents[:,-1,:].reshape(-1, self.action_latentdim)
            relative_act_enc_obs = torch.zeros_like(act_enc_obs)
            relative_act_enc_obs[:,:-1] = act_enc_obs[:,1:] - act_enc_obs[:,:-1]
            act_enc_last = act_enc_last - act_enc_obs[:,-1]
            relative_act_enc_obs = relative_act_enc_obs.reshape(-1, self.action_latentdim)
            relative_act_enc_obs = self.action_proj(relative_act_enc_obs)
            relative_act_enc_obs = relative_act_enc_obs.reshape(-1, num_saccades-1, self.action_projdim)
            relative_act_enc_obs[:,-1,...] = 0.
            fov_encs_obs = torch.cat((fov_encs_obs, relative_act_enc_obs), dim=-1)
            act_enc_last = self.action_proj(act_enc_last)

        fov_encs_reshape = fov_encs_obs.reshape((-1, num_saccades-1, self.emb_dim))
        B, N, _ = fov_encs_reshape.shape
        agg_tokens = self.agg_token.expand(B, -1, -1)

        x = torch.cat((agg_tokens, fov_encs_reshape), dim=1)
        
        x, _ = self.transformer_encoder(x)
            
        agg_out = x[:, 0]
        if self.act_cond:
            agg_out_conditioned = torch.cat((agg_out, act_enc_last), dim=-1)
        else:
            agg_out_conditioned = agg_out
        pred_out = self.predictor(agg_out_conditioned)
        
        loss = 1-self.criterion(pred_out, fov_encs_last_detached).mean()
        z1 = fov_encs_obs[:,0,:self.res_out_dim]
        if num_saccades == 2:
            z2 = fov_encs_last_detached
        else:
            z2 = fov_encs_obs[:,1,:self.res_out_dim]

        return loss, agg_out, z1, z2
