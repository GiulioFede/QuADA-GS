import torch
import torch.nn as nn
import math
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from models.attention.attention import WindowCrossAttnBlock, GSSelfAttnBlock
from models.quad_refiner.adaptive_sparse_quad_splatting import AdaptiveSparseQuadtreeSplatting

'''
     The transformer that extracts and processes features.
     # From: https://github.com/ChrisDud0257/GSASR/blob/master/utils/fea2gs.py

     The AdaptiveSparseQuadtreeSplatting (Neural Routing Architecture and the Hierarchical Pointer Convolution) is our core contribution
'''

class QuADA_GS(nn.Module):
    def __init__(self, 
                    inchannel=64, 
                    channel=180, 
                    num_heads=6, 
                    num_crossattn_blocks=1, 
                    num_crossattn_layers=2, 
                    num_selfattn_blocks = 6, 
                    num_selfattn_layers = 6,
                    num_gs_seed=144, 
                    gs_up_factor=1.0, 
                    window_size=12, 
                    img_range=1.0, 
                    shuffle_scale1 = 2, 
                    shuffle_scale2 = 2, 
                    use_checkpoint = False,
                    type_of_image_encoder = 'rdn',
                    
                    use_learnable_structure_map = False,
                    default_step_size = 1.4):

        super(QuADA_GS, self).__init__()
        self.channel = channel
        self.nhead = num_heads
        self.gs_up_factor = gs_up_factor
        self.num_gs_seed = num_gs_seed
        self.window_size = window_size
        self.img_range = img_range
        self.use_checkpoint = use_checkpoint
        self.type_of_image_encoder = type_of_image_encoder

        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))
        self.gs_up_factor_sqrt = int(math.sqrt(gs_up_factor))

        self.shuffle_scale1 = shuffle_scale1
        self.shuffle_scale2 = shuffle_scale2

        # shared gaussian embedding and its pos embedding
        self.gs_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel), requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel), requires_grad=True)

        if type_of_image_encoder == 'rdn':
            from models.image_encoder.rdn.rdn import RDNNOUP
            self.image_encoder = RDNNOUP()
        elif type_of_image_encoder == 'edsr':
            from models.image_encoder.edsr.edsr import EDSRNOUP
            self.image_encoder = EDSRNOUP()
        else:
            raise Exception(f"The type of image encoder [{self.image_encoder}] is not evailable.")

        self.img_feat_proj = nn.Sequential(
            nn.Conv2d(inchannel, channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )

        self.window_crossattn_blocks = nn.ModuleList([
            WindowCrossAttnBlock(dim=channel,
                                 window_size=window_size,
                                 num_heads=num_heads,
                                 num_layers=num_crossattn_layers,
                                 num_gs_seed=num_gs_seed) for i in range(num_crossattn_blocks)
        ])

        self.gs_selfattn_blocks = nn.ModuleList([
            GSSelfAttnBlock(dim=channel,
                            num_heads=num_heads,
                            num_selfattn_layers=num_selfattn_layers,
                            num_gs_seed_sqrt=self.num_gs_seed_sqrt
                            ) for i in range(num_selfattn_blocks)
        ])

        # GS sigma_x, sigma_y
        self.mlp_block_sigma = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(2 * gs_up_factor))
        )

        # GS rho
        self.mlp_block_rho = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(1 * gs_up_factor))
        )

        # GS alpha
        self.mlp_block_alpha = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(1 * gs_up_factor))
        )

        # GS RGB values
        self.mlp_block_rgb = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(3 * gs_up_factor))
        )

        # GS mean_x, mean_y
        self.mlp_block_mean = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(2 * gs_up_factor))
        )

        self.scale_mlp = nn.Sequential(
            nn.Linear(1, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, channel)
        )

        self.quad_refiner = AdaptiveSparseQuadtreeSplatting(channel, channel, use_learnable_structure_map=use_learnable_structure_map, default_step_size=default_step_size)



    @staticmethod
    def get_N_reference_points(h, w, device='cuda'):
        step_y = 1 / h
        step_x = 1 / w
        ref_y, ref_x = torch.meshgrid(torch.linspace(step_y / 2, 1 - step_y / 2, h, dtype=torch.float32, device=device),
                                      torch.linspace(step_x / 2, 1 - step_x / 2, w, dtype=torch.float32, device=device))
        reference_points = torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), -1)
        reference_points = reference_points[None, :, None]
        return reference_points

    def forward(self, lr_image, scale, inference=False, grt_image=None, tau=None):
        device = lr_image.device
        srcs = self.image_encoder(lr_image)  # b,c,h,w

        b, c, h, w = srcs.shape  ###srcs is pad to the size that could be divided by window_size
        scale = torch.full((b, 1), float(scale), device=device)
        query = self.gs_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (w // self.window_size),
                                                                   1, 1)  # b, h_count*w_count, num_gs_seed, channel
        query = query.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                              self.channel)  # b*h_count*w_count, num_gs_seed, channel

        scale = 1 / scale
        # scale = scale.unsqueeze(1)  # b*1
        scale_embedding = self.scale_mlp(scale)  # b*channel
        scale_embedding = scale_embedding.unsqueeze(1).unsqueeze(2).repeat(1, (h // self.window_size) * (
                    w // self.window_size), self.num_gs_seed, 1)  # b, h_count*w_count, num_gs_seed, channel
        
        
        scale_embedding = scale_embedding.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                                      self.channel) # b*h_count*w_count, num_gs_seed, channel

        query_pos = self.pos_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (
                    w // self.window_size), 1, 1)  # b, h_count*w_count, num_gs_seed, channel

        feat = self.img_feat_proj(srcs)  # b*channel*h*w

        query_pos = query_pos.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                                      self.channel)  # b*h_count*w_count, num_gs_seed, channel

        for block in self.window_crossattn_blocks:
            if self.use_checkpoint:
                query = checkpoint(block, query, query_pos, feat, scale_embedding)
            else:
                query = block(query, query_pos, feat, scale_embedding)  # b*h_count*w_count, num_gs_seed, channel

        resi = query
        for block in self.gs_selfattn_blocks:
            if self.use_checkpoint:
                query = checkpoint(block, query, query_pos, h // self.window_size, w // self.window_size, scale_embedding)
            else:
                query = block(query, query_pos, h // self.window_size, w // self.window_size, scale_embedding)
        query = query + resi

        query = rearrange(query, '(b m n) (h w) c -> b c (m h) (n w)', m=h // self.window_size, n=w // self.window_size,
                          h=self.num_gs_seed_sqrt)
        
        out_feats_list, out_coords_list, out_scales_list, predictor_loss, out_sigma_scales_list, H_up, W_up, weight_map_hr, level_probs = self.quad_refiner(query, grt_image=grt_image, training=not(inference), tau=tau)
        return out_feats_list, out_coords_list, out_scales_list, predictor_loss, out_sigma_scales_list, H_up, W_up, weight_map_hr, level_probs



    def predict_gaussians(self, query, b, w, h, start_position=None, predict_offset=True, gs_scales=None, chunk_size=None):


        if chunk_size is not None and chunk_size > 0:
            sigma_chunks, rho_chunks, alpha_chunks, rgb_chunks = [], [], [], []
            if predict_offset:
                mean_chunks = []
            
            # Processiamo in blocchi lungo N (dim=1)
            for q_chunk in torch.split(query, chunk_size, dim=1):
                sigma_chunks.append(self.mlp_block_sigma(q_chunk))
                rho_chunks.append(self.mlp_block_rho(q_chunk))
                alpha_chunks.append(self.mlp_block_alpha(q_chunk))
                rgb_chunks.append(self.mlp_block_rgb(q_chunk))
                
                if predict_offset:
                    mean_chunks.append(self.mlp_block_mean(q_chunk))
            
            # Concateniamo i risultati parziali e facciamo il reshape
            query_sigma = torch.cat(sigma_chunks, dim=1).reshape(b, -1, 2)
            query_rho   = torch.cat(rho_chunks, dim=1).reshape(b, -1, 1)
            query_alpha = torch.cat(alpha_chunks, dim=1).reshape(b, -1, 1)
            query_rgb   = torch.cat(rgb_chunks, dim=1).reshape(b, -1, 3)
            
            if predict_offset:
                raw_offset = torch.cat(mean_chunks, dim=1).reshape(b, -1, 2)
        else:
            query_sigma = self.mlp_block_sigma(query).reshape(b, -1, 2)
            query_rho = self.mlp_block_rho(query).reshape(b, -1, 1)
            query_alpha = self.mlp_block_alpha(query).reshape(b, -1, 1)
            query_rgb = self.mlp_block_rgb(query).reshape(b, -1, 3)

            if predict_offset:
                # Output grezzo della rete
                raw_offset = self.mlp_block_mean(query).reshape(b, -1, 2)
        
        if predict_offset:
            # Normalizziamo per [w, h] e sommiamo alla posizione iniziale
            scale_tensor = torch.tensor([[[w, h]]], device=query.device)
            query_mean = (raw_offset / scale_tensor) + start_position.reshape(1, -1, 2)
        else:
            query_mean = start_position

        query = torch.cat([query_sigma, query_rho, query_alpha, query_rgb, query_mean], dim=-1)
        
        return query