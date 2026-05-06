
import torch.nn as nn
import torch
import torch.nn.functional as F

class StructurePrimedRouter(nn.Module):
    """
        The structure map defines an expected cost budget per pixel.
        The router can deviate by ±band_tolerance from that budget—in either direction.
        Within the band: zero penalty; L1 loss has free rein.
        Outside the band: smooth penalty.
        
        During inference: structure_map=None; the router works solely on the features.
    """
    def __init__(self, feature_dim: int, num_levels: int = 3,
                 band_tolerance_low: float = 0.05, #indicates the percentage, relative to the cost of the structure map, below which to go if necessary
                 band_tolerance_up: float = 0.2, #indicates the percentage, relative to the cost of the structure map, that should be exceeded if necessary
                 band_weight: float = 1.0,
                 entropy_weight: float = 0.05,
                 quadtree_penalty_weight: float = 1.5,
                 checkboard_weight: float = 0.025,
                 structure_imitation_weight: float = 0.3,
                 struct_sharpness: float = 3.0,
                 min_gaussians_to_use = 30000):
        super().__init__()
        self.band_tolerance_low   = band_tolerance_low
        self.band_tolerance_up   = band_tolerance_up
        self.band_weight      = band_weight
        self.entropy_weight   = entropy_weight
        self.quadtree_penalty_weight = quadtree_penalty_weight
        self.struct_sharpness = struct_sharpness
        self.min_gaussians_to_use = min_gaussians_to_use
        self.structure_imitation_weight = structure_imitation_weight
        self.checkboard_weight = checkboard_weight

        # Normalized Costs L0=0.0625, L1=0.25, L2=1.0
        self.register_buffer("level_costs",
            # torch.tensor([0.2, 0.25, 1.0]).view(1, 3, 1, 1))
            torch.tensor([0.2, 0.25, 1.0]).view(1, 3, 1, 1))
        # Level centers for the structure_map → prob conversion
        self.register_buffer("level_centers",
            torch.tensor([0.0, 1.0, 2.0]).view(1, 3, 1, 1))

        self.router = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, 1, 1),
            nn.GroupNorm(6, feature_dim // 2),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, feature_dim // 4, 3, 1, 2, dilation=2),
            nn.GELU(),
            nn.Conv2d(feature_dim // 4, num_levels, 1),
        )

        nn.init.normal_(self.router[-1].weight, std=1e-4)
        nn.init.zeros_(self.router[-1].bias)


    #TODO: sostituisci i costi con self.level_costs
    def structure_map_to_cost(self, structure_map):
        """
        Calculate the EXACT budget for Gaussians.
        structure_map is ALREADY processed by get_target_level_map (with sqrt and *2.0 applied).
        """
        cost = torch.full_like(structure_map, 0.0625)
        cost[structure_map >= 0.1] = 0.25
        cost[structure_map >= 1.4] = 1.0
        
        return cost

    def get_min_budget(self, H, W, global_struct_budget):
        H_fine, W_fine = H * 4, W * 4
        max_gaussians = H_fine * W_fine
        
        min_budget = self.min_gaussians_to_use / max_gaussians

        noise = (torch.rand_like(global_struct_budget) * 0.06) - 0.03
        dynamic_floor = min_budget * (1.0 + noise)

        global_struct_budget = torch.maximum(global_struct_budget, dynamic_floor)

        return global_struct_budget


    def forward(self, base_feats, structure_map=None, tau=1.0, training=True):
        B, C, H, W = base_feats.shape
        device = base_feats.device

        logits      = self.router(base_feats.detach())
        logits = 4.0 * torch.tanh(logits / 4.0)
        level_probs = F.softmax(logits, dim=1)

        if training:
            level_one_hot = F.gumbel_softmax(logits, tau=tau, hard=True, dim=1)
            levels        = level_one_hot.argmax(dim=1).long()

            # -----------------------------------------------------------------
            # --- FIX: BUDGET CALCULATION BASED ON ACTUAL DECISIONS (HARD) ---
            # -------------------------------------------------------------- ---
            # Let's use level_one_hot instead of level_probs! 
            # The gradients will pass through perfectly thanks to Gumbel-Softmax (STE).
            router_cost = (level_one_hot * self.level_costs).sum(dim=1, keepdim=True)
            # -----------------------------------------------------------------


            router_loss = torch.tensor(0.0, device=device)

            if structure_map is not None:
                if structure_map.shape[-2:] != (H, W):
                    structure_map = F.interpolate(structure_map, (H, W),
                                                mode='bilinear', align_corners=False)

                struct_cost = self.structure_map_to_cost(structure_map)

                global_struct_budget = struct_cost.mean(dim=[1, 2, 3]) * 1.4

                if self.min_gaussians_to_use is not None:
                    global_struct_budget = self.get_min_budget(H, W, global_struct_budget)


                global_router_budget = router_cost.mean(dim=[1, 2, 3])

                lo = global_struct_budget * (1.0 - self.band_tolerance_low)
                hi = global_struct_budget * (1.0 + self.band_tolerance_up)

                over_budget  = F.relu(global_router_budget - hi)
                under_budget = F.relu(lo - global_router_budget)

                # -----------------------------------------------------------------
                # NEW: LOCAL DITHERING LOSS (Checkerboard Effect)
                # ----- ------------------------------------------------------------
                # 1. router_cost contains the ACTUAL costs selected by Gumbel (0.0625, 0.25, 1.0)
                # 2. struct_cost contains the IDEAL costs calculated by the structure map
                
                # We calculate the average cost over micro-windows (e.g., 2x2 or 4x4)
                # Kernel 2 creates very dense dithering. Kernel 4 creates more visible blocks.
                patch_size = 2 
                local_router_budget = F.avg_pool2d(router_cost, kernel_size=patch_size, stride=patch_size)
                local_struct_budget = F.avg_pool2d(struct_cost, kernel_size=patch_size, stride=patch_size)
                
                # Force the network to map L0, L1, and L2 locally to match the target
                structure_imitation_loss = F.mse_loss(local_router_budget, local_struct_budget)
                # -----------------------------------------------------------------

                # Re-enabling entropy regularization to encourage decisive routing
                entropy = - (level_probs * torch.log(level_probs + 1e-8)).sum(dim=1).mean()

                # -----------------------------------------------------------------
                # --- QUADTREE PENALTY (Prohibition of Extreme Jumps) ---
                # -----------------------------------------------------------------
                p_L0 = level_probs[:, 0, :, :] # (B, H, W)
                p_L1 = level_probs[:, 1, :, :] # (B, H, W)
                p_L2 = level_probs[:, 2, :, :] # (B, H, W)

                # Prevent L0 and L2 from touching each other 
                clash_02_x = p_L0[:, :, :-1] * p_L2[:, :, 1:] + p_L2[:, :, :-1] * p_L0[:, :, 1:]
                clash_02_y = p_L0[:, :-1, :] * p_L2[:, 1:, :] + p_L2[:, :-1, :] * p_L0[:, 1:, :]
                quadtree_penalty = clash_02_x.mean() + clash_02_y.mean()


                # -----------------------------------------------------------------
                # --- NEW: CHECKERBOARD PENALTY  ---
                # We prevent amorphous areas by penalizing adjacent identical pixels.
                # -----------------------------------------------------------------
                # 1. How often does L0 touch L0?
                clash_00_x = p_L0[:, :, :-1] * p_L0[:, :, 1:]
                clash_00_y = p_L0[:, :-1, :] * p_L0[:, 1:, :]
                
                # 2. How often does L1 touch L1?
                clash_11_x = p_L1[:, :, :-1] * p_L1[:, :, 1:]
                clash_11_y = p_L1[:, :-1, :] * p_L1[:, 1:, :]
                
                # 3. How often does L2 interact with L2?
                clash_22_x = p_L2[:, :, :-1] * p_L2[:, :, 1:]
                clash_22_y = p_L2[:, :-1, :] * p_L2[:, 1:, :]

                w0, w1, w2 = 1.0, 0.1, 1.2

                checkerboard_penalty = w0 * (clash_00_x.mean() + clash_00_y.mean()) + \
                                       w1 * (clash_11_x.mean() + clash_11_y.mean()) + \
                                       w2 * (clash_22_x.mean() + clash_22_y.mean())


                router_loss = (over_budget + under_budget).mean() + \
                              (self.entropy_weight * entropy) + \
                              (self.quadtree_penalty_weight * quadtree_penalty) + \
                              (self.checkboard_weight * checkerboard_penalty) + \
                              (self.structure_imitation_weight * structure_imitation_loss)



        else:
            levels      = logits.argmax(dim=1).long()
            router_loss = torch.tensor(0.0, device=device)
            level_one_hot = None

        return levels, router_loss, level_one_hot, level_probs