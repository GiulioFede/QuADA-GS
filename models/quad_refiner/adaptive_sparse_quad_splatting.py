
import torch.nn as nn
import torch
import torch.nn.functional as F
from .neural_routing_architecture import StructurePrimedRouter
from .hpc import HierarchicalPointerConv


class AdaptiveSparseQuadtreeSplatting(nn.Module):
    def __init__(self, in_dim, feature_dim, default_step_size=1.4, use_dense_rendering = False, use_learnable_structure_map=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.default_step_size = default_step_size
        self.use_dense_rendering = use_dense_rendering
        self.use_learnable_structure_map = use_learnable_structure_map
        

        # 1. Base Upsampling 
        self.uniform_upsample = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.GroupNorm(num_groups=6, num_channels=feature_dim)
        )

        
        if use_learnable_structure_map:
            self.level_router = StructurePrimedRouter(
                feature_dim=feature_dim,
                num_levels=3
            )

            #The epoch marking the transition from a static structure map to a learnable structure map.
            self.start_learning_structure_map_epoch = 19400

            #If set to True, the ground-truth structure map is utilized for both training and inference.
            self.use_structure_map_from_ground_truth = False

        else:
            # 2. Density Predictor
            self.density_predictor = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // 2, 1, 1) 
            )


        # 3. Sparse Expanders 
        # Expand channels C -> C*4 per generate 4 children
        self.expand_L1 = nn.Linear(feature_dim, feature_dim * 4)
        self.expand_L2 = nn.Linear(feature_dim, feature_dim * 4)
        
        # 4. Spatially-Aware Pointer Convolution
        # 4. Spatially-Aware Pointer Convolution (Cascade Interleaved)
        self.pointer_conv_L1 = HierarchicalPointerConv(feature_dim)
        
        # Replace the 5x5: two 3x3 in cascade 
        self.pointer_conv_L2_a = HierarchicalPointerConv(feature_dim)
        self.pointer_conv_L2_b = HierarchicalPointerConv(feature_dim)

        self.scales = torch.tensor([4.0, 2.0, 1.0])

    def quantize_map(self, x, thresholds):
        x = x.squeeze(1)
        t = torch.tensor(thresholds, device=x.device, dtype=x.dtype)
        return torch.bucketize(x, t, right=False).long()

    def get_target_level_map(self, image, max_levels=4.0):
        # 1. Get Structure Map (0.0 - 1.0)
        struct_map = self.get_structural_complexity_map(image, border_ignore=0) 
        
        struct_map[struct_map < 0.01] = 0.0

        target_levels = torch.sqrt(struct_map) * max_levels
        
        return target_levels # (B, 1, H, W) with float values [0, 4.0]


    def get_structural_complexity_map(self, image, kernel_size=3, border_ignore=4):
        """
        Compute Structure Map.
        """
        # --- ANTI-NAN FIX: Force float32 precision ---
        # Prevents the squares (Ix**2, trace**2) from overflowing (inf) in FP16
        image = image.to(torch.float32)

        # 1. Grayscale
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image

        # 2. Calculating Gradients with REFLECT PADDING
        # Instead of conv2d(padding=1), we pad first
        padding = 1
        gray_padded = F.pad(gray, (padding, padding, padding, padding), mode='reflect')
        
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        
        # Conv2d
        Ix = F.conv2d(gray_padded, kernel_x, padding=0)
        Iy = F.conv2d(gray_padded, kernel_y, padding=0)

        #3. Gradient products
        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        Ixy = Ix * Iy

        # 4. Local Sum (Window Tensor Structure)
        k_window = 5 
        pad_pool = k_window // 2
        Ix2_pad = F.pad(Ix2, (pad_pool, pad_pool, pad_pool, pad_pool), mode='reflect')
        Iy2_pad = F.pad(Iy2, (pad_pool, pad_pool, pad_pool, pad_pool), mode='reflect')
        Ixy_pad = F.pad(Ixy, (pad_pool, pad_pool, pad_pool, pad_pool), mode='reflect')
        
        avg_pool = nn.AvgPool2d(kernel_size=k_window, stride=1, padding=0)
        
        sum_Ix2 = avg_pool(Ix2_pad)
        sum_Iy2 = avg_pool(Iy2_pad)
        sum_Ixy = avg_pool(Ixy_pad)

        # 5. Eigenvalues
        det = (sum_Ix2 * sum_Iy2) - (sum_Ixy ** 2)
        trace = sum_Ix2 + sum_Iy2
        delta = torch.sqrt(torch.clamp(trace**2 - 4*det, min=1e-6))
        
        lambda_1 = (trace + delta) / 2.0
        lambda_2 = (trace - delta) / 2.0 

        # 6. Scores
        texture_score = lambda_2 
        edge_score = lambda_1 - lambda_2 

        combined_score = (3.0 * texture_score) + (1.0 * edge_score)
        combined_score = torch.clamp(combined_score, min=0.0) 
        
        # Normalization
        robust_max = torch.quantile(combined_score, 0.98)
        final_map = torch.clamp(combined_score / (robust_max + 1e-6), 0.0, 1.0)


        # ---  BORDER MASKING ---
        # Reset the outer borders to avoid artifacts
        if border_ignore > 0:
            final_map[..., :border_ignore, :] = 0.0 # Top
            final_map[..., -border_ignore:, :] = 0.0 # Bottom
            final_map[..., :, :border_ignore] = 0.0 # Left
            final_map[..., :, -border_ignore:] = 0.0 # Right
        
        return final_map


    def _split_sparse_points(self, feats, y, x, b, target_lvl, current_lvl, offset_step, expand_layer):
        " Divides a point into 4 children using a Linear Layer "
        N, C = feats.shape
        device = feats.device
        
        # Expansion feature: (N, C) -> (N, C*4) -> (N, 4, C)
        new_feats = expand_layer(feats).view(N, 4, C)
        
        # Coordinates of the four child nodes (top-left, top-right, bottom-left, bottom-right)
        offsets = torch.tensor([[0,0], [0,1], [1,0], [1,1]], device=device) * offset_step
        
        y_new = y.unsqueeze(1) + offsets[:, 0].unsqueeze(0) # (N, 4)
        x_new = x.unsqueeze(1) + offsets[:, 1].unsqueeze(0) # (N, 4)
        b_new = b.unsqueeze(1).repeat(1, 4)
        tgt_new = target_lvl.unsqueeze(1).repeat(1, 4)
        
        # Flat
        return (
            new_feats.reshape(-1, C), 
            y_new.flatten(), 
            x_new.flatten(), 
            b_new.flatten(), 
            tgt_new.flatten(),
            torch.full((N * 4,), current_lvl + 1, dtype=torch.long, device=device)
        )

    def forward(self, feature_map, training=True, grt_image=None, chunk_size=None, tau=None):
        B, C_in, H_in, W_in = feature_map.shape
        device = feature_map.device

        # --- 1. BASE GRID & DENSITY ---
        base_feats = self.uniform_upsample(feature_map) 

        B, C, H_base, W_base = base_feats.shape
        H_fine, W_fine = H_base * 4, W_base * 4

        if self.use_learnable_structure_map == False:
            density_map = self.density_predictor(base_feats.detach())


        predictor_loss = 0.0
        weight_map_hr = None
        
        if self.use_learnable_structure_map == False:
            if grt_image is not None and training:
                gt_levels = self.get_target_level_map(grt_image, 2).detach()
                if gt_levels.shape[-2:] != density_map.shape[-2:]:
                    gt_levels = F.interpolate(gt_levels, size=density_map.shape[-2:], mode='bilinear')
                predictor_loss = F.mse_loss(density_map, gt_levels)
                weight_map_hr = 1 + F.interpolate(gt_levels.float(), size=grt_image.shape[-2:], mode='nearest')

            # --- 2. TRINITY THEOREM ---
            if training and grt_image is not None:
                density_for_quant = density_map.detach()
                rand_val = torch.rand(1).item()
                P_NOISE = 0.1

                # Let's generate random thresholds within a reasonable range 
                t1 = torch.empty(1).uniform_(-0.45, 0.4).item()
                t2 = torch.empty(1).uniform_(0.7, 1.8).item()
                
                thresholds = [min(t1, t2), max(t1, t2)] 
                
                if rand_val < P_NOISE:
                    noisy_map = density_for_quant + torch.randn_like(density_for_quant) * 0.5
                    levels = self.quantize_map(noisy_map, thresholds)
                else:
                    levels = self.quantize_map(density_for_quant, thresholds)
            else:
                levels = self.quantize_map(density_map, [0.1, 1.4])

            tgt_curr = levels.flatten()
            level_probs = [1.0] * B

        else:

            with torch.no_grad():
                if training or self.use_structure_map_from_ground_truth:
                    structure_map = self.get_target_level_map(grt_image, max_levels=2.0)
                    if structure_map.shape[-2:] != (H_base, W_base):
                        structure_map = F.interpolate(structure_map, size=(H_base, W_base),
                                                    mode='bilinear', align_corners=False)
                else:
                    structure_map = None

            if self.use_structure_map_from_ground_truth == False :
                levels_map, predictor_loss, level_one_hot, level_probs = self.level_router(
                    base_feats,
                    structure_map=structure_map,
                    tau=tau,
                    training=training,
                )

                tgt_curr   = levels_map.flatten()
                weight_map_hr = [1.0] * B
            else:

                if training and grt_image is not None:
                    density_for_quant = structure_map.detach()
                    rand_val = torch.rand(1).item()
                    P_NOISE = 0.1

                    # Let's generate random thresholds within a reasonable range 
                    t1 = torch.empty(1).uniform_(-0.45, 0.4).item()
                    t2 = torch.empty(1).uniform_(0.7, 1.8).item()
                    
                    thresholds = [min(t1, t2), max(t1, t2)] 
                    
                    if rand_val < P_NOISE:
                        noisy_map = density_for_quant + torch.randn_like(density_for_quant) * 0.5
                        levels = self.quantize_map(noisy_map, thresholds)
                    else:
                        levels = self.quantize_map(density_for_quant, thresholds)
                else:
                    levels = self.quantize_map(structure_map, [0.1, 1.4])         

                   
                tgt_curr = levels.flatten()
                level_probs = [1.0] * B
                weight_map_hr = [1.0] * B

            
            
        
        # --- 3. INITIALIZING POINTS (L0) ---
        # Convert everything to linear format (N, C)
        b_idx, y_base, x_base = torch.meshgrid(
            torch.arange(B, device=device),
            torch.arange(H_base, device=device),
            torch.arange(W_base, device=device),
            indexing='ij'
        )
        
        feats_curr = base_feats.permute(0, 2, 3, 1).reshape(-1, C)


        b_curr = b_idx.flatten()
        
        # Base coordinates projected onto the fine grid (Top-Left)
        y_curr = y_base.flatten() * 4
        x_curr = x_base.flatten() * 4
        
        # The target level for each point (which determines whether it should split or stop)
        lvl_curr = torch.zeros_like(tgt_curr) # Everyone starts at L0

        # --- STEP 4: REFINEMENT CASCADE WITH INTERLEAVED MIXING ---
        mask_split_L1 = (tgt_curr >= 1)
        
        # Array for those who stop at L0
        f0_stop = feats_curr[~mask_split_L1]
        y0_stop = y_curr[~mask_split_L1]
        x0_stop = x_curr[~mask_split_L1]
        b0_stop = b_curr[~mask_split_L1]
        l0_stop = lvl_curr[~mask_split_L1]
        
        f1_stop, y1_stop, x1_stop, b1_stop, l1_stop = [], [], [], [], []
        f2, y2, x2, b2, l2 = [], [], [], [], []

        if mask_split_L1.any():
            # 4.1 Generate the L1 children
            f1_active, y1_active, x1_active, b1_active, t1_active, l1_active = self._split_sparse_points(
                feats_curr[mask_split_L1], y_curr[mask_split_L1], x_curr[mask_split_L1], 
                b_curr[mask_split_L1], tgt_curr[mask_split_L1], 0, offset_step=2, expand_layer=self.expand_L1
            )
            
            # 4.2 INTERMEDIATE MIXING L1 (Blending the midrange frequencies)
            # Temporarily combine those who have stopped (L0) with the new children (L1)
            tmp_f = torch.cat([f0_stop, f1_active], dim=0)
            tmp_y = torch.cat([y0_stop, y1_active], dim=0)
            tmp_x = torch.cat([x0_stop, x1_active], dim=0)
            tmp_b = torch.cat([b0_stop, b1_active], dim=0)
            tmp_l = torch.cat([l0_stop, l1_active], dim=0)
            
            # Let's apply L1 convolution
            tmp_f = self.pointer_conv_L1(tmp_f, tmp_y, tmp_x, tmp_l, tmp_b, H_base, W_base, chunk_size, training)
            
            # Let's retrieve the updated tensors
            num_L0 = f0_stop.shape[0]
            f0_stop = tmp_f[:num_L0]      
            f1_active = tmp_f[num_L0:]    
            
            # 4.3 Split L1 -> L2
            mask_split_L2 = (t1_active == 2)
            
            # Array for those stopping at L1
            f1_stop = f1_active[~mask_split_L2]
            y1_stop = y1_active[~mask_split_L2]
            x1_stop = x1_active[~mask_split_L2]
            b1_stop = b1_active[~mask_split_L2]
            l1_stop = l1_active[~mask_split_L2]
            
            if mask_split_L2.any():
                # Generating L2 micro-details
                f2, y2, x2, b2, t2, l2 = self._split_sparse_points(
                    f1_active[mask_split_L2], y1_active[mask_split_L2], x1_active[mask_split_L2], 
                    b1_active[mask_split_L2], t1_active[mask_split_L2], 1, offset_step=1, expand_layer=self.expand_L2
                )

        # Let's group everything together (skipping empty lists)
        all_feats = torch.cat([f for f in (f0_stop, f1_stop, f2) if len(f) > 0], dim=0)
        all_y = torch.cat([y for y in (y0_stop, y1_stop, y2) if len(y) > 0], dim=0)
        all_x = torch.cat([x for x in (x0_stop, x1_stop, x2) if len(x) > 0], dim=0)
        all_b = torch.cat([b for b in (b0_stop, b1_stop, b2) if len(b) > 0], dim=0)
        all_l = torch.cat([l for l in (l0_stop, l1_stop, l2) if len(l) > 0], dim=0)

        # --- STEP 5: FINAL CASCADE L2 (5x5 Receptive Field) ---
        all_feats = self.pointer_conv_L2_a(all_feats, all_y, all_x, all_l, all_b, H_base, W_base, chunk_size, training)
        all_feats = self.pointer_conv_L2_b(all_feats, all_y, all_x, all_l, all_b, H_base, W_base, chunk_size, training)



        offsets_center = torch.tensor([2.0, 1.0, 0.5], device=device)
        
        # FIX
        all_l = all_l.long()
        
        y_center = (all_y.float() + offsets_center[all_l]) / H_fine
        x_center = (all_x.float() + offsets_center[all_l]) / W_fine
        
        all_coords = torch.stack([x_center, y_center], dim=-1) # (N, 2)
        scales = self.scales.to(device)
        all_scales = scales[all_l].unsqueeze(-1)               # (N, 1)

        # Let's group by batch and sort
        sort_idx = torch.argsort(all_b)
        all_feats, all_coords = all_feats[sort_idx], all_coords[sort_idx]
        all_scales, all_b = all_scales[sort_idx], all_b[sort_idx]
        
        counts = torch.bincount(all_b, minlength=B)
        sections = counts.tolist()

        if self.use_dense_rendering:
            # Let's prepare the instructions: for each soarse point, specify its location (y, x, lvl)
            mapping_info = torch.stack([all_y, all_x, all_l], dim=-1)
            mapping_list = torch.split(mapping_info, sections)
            h_out, w_out = H_fine, W_fine 
        else:
            mapping_list = [None] * B
            h_out, w_out = H_base, W_base


        return (
            torch.split(all_feats, sections), 
            torch.split(all_coords, sections), 
            mapping_list, 
            predictor_loss, 
            torch.split(all_scales, sections), 
            h_out, w_out, weight_map_hr, level_probs
        )

    def get_dynamic_hyperparams(self, current_epoch):
        # 1. Virtual epoch (for a possible restart)
        restart_epoch = self.start_learning_structure_map_epoch
        virtual_epoch = current_epoch - restart_epoch if current_epoch >= restart_epoch else 0
        
        # --- 2. TAU SCHEDULING (Exploration) ---
        tau_start = 1.0
        tau_end = 0.5
        decay_rate = 0.995 # Gradual decrease (approximately 45 steps to reach 0.1)
        
        current_tau = max(tau_end, tau_start * (decay_rate ** virtual_epoch))

        # --- 3. PREDICTOR LOSS WEIGHT (The Concrete Wall) ---
        target_lambda = 0.1
        
        # A 3-epoch mini-warmup to avoid initial shock
        warmup_epochs = 3
        progress = min(1.0, virtual_epoch / warmup_epochs)
        current_lambda_predictor = target_lambda * progress


        # --- 4. STRUCTURE SIMILARITY DECAY ---
        structure_start = 1.0
        structure_end = 0.0

        hold_epochs = 800   # 🔴 
        decay_rate_structure = 0.98  # decrease speed

        if current_epoch < restart_epoch:
            structure_similarity_decay = -1
        elif virtual_epoch < hold_epochs:
            structure_similarity_decay = structure_start
        else:
            decay_epoch = virtual_epoch - hold_epochs
            structure_similarity_decay = max(
                structure_end,
                structure_start * (decay_rate_structure ** decay_epoch)
            )

        return current_tau, current_lambda_predictor, structure_similarity_decay