
import torch.nn as nn
import torch

class HierarchicalPointerConv(nn.Module):
    """
    Performs spatially exact 3x3 convolution using a Pointer Grid (Index Map) to avoid massive dense allocations.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.spatial_conv = nn.Linear(channels * 9, channels)
        self.act = nn.GELU()
        self.register_buffer("zero_pad_feat", torch.zeros(1, channels))

    def forward(self, all_feats, y_fine, x_fine, levels, batch_idx, H_base, W_base, chunk_size=None, training=True):
        N, C = all_feats.shape
        device = all_feats.device
        
        H_fine, W_fine = H_base * 4, W_base * 4
        B = batch_idx.max().long().item() + 1
        
        # Padding trick: the Nth index will always point to zeros
        feats_padded = torch.cat([all_feats, self.zero_pad_feat], dim=0)
        
        ptr_grid = torch.full(
            (B, H_fine + 2, W_fine + 2), 
            fill_value=N, dtype=torch.long, device=device
        ) #Create a grid with height H_fine and width W_fine, filled with the total number of features (by summing those in each batch)
        
        point_indices = torch.arange(N, dtype=torch.long, device=device)

        def paint_level(lvl, block_size):
            mask = (levels == lvl)
            if not mask.any(): return
            
            b_m, y_m, x_m, idx_m = batch_idx[mask], y_fine[mask], x_fine[mask], point_indices[mask]
            
            gy, gx = torch.meshgrid(torch.arange(block_size, device=device), 
                                    torch.arange(block_size, device=device), indexing='ij')
            gy, gx = gy.flatten().unsqueeze(0), gx.flatten().unsqueeze(0)
            
            Y_dest = (y_m.unsqueeze(1) + gy).flatten() + 1
            X_dest = (x_m.unsqueeze(1) + gx).flatten() + 1
            B_dest = b_m.unsqueeze(1).repeat(1, block_size**2).flatten()
            I_dest = idx_m.unsqueeze(1).repeat(1, block_size**2).flatten()
            
            ptr_grid[B_dest.long(), Y_dest.long(), X_dest.long()] = I_dest

        # Let's paint the canvas: from the lowest (L0) to the finest (L2)
        paint_level(0, block_size=4)
        paint_level(1, block_size=2)
        paint_level(2, block_size=1)

        # -----------------------------------------------------------------
        # -- - ASYMMETRICAL 3x3 “BOUNDARY-HUGGING” LOOKUP ---
        # -----------------------------------------------------------------
        
        # 1. Map the level to the block size (L0=4, L1=2, L2=1)
        block_sizes = torch.tensor([4, 2, 1], device=device, dtype=torch.long)
        bs = block_sizes[levels] # (N,) 
        
        # 2. Let's create the exact offsets to read the edges and center of the block
        # The three reading points along the axis: [previous pixel, center, next pixel]
        offsets = torch.stack([
            torch.full_like(bs, -1),  # Left
            bs // 2,                  # Center 
            bs                        # Right
        ], dim=1) # Shape: (N, 3)
        
        #3. Combinatorial grid for the 9 neighbors
        k_range = torch.tensor([0, 1, 2], device=device)
        idx_y, idx_x = torch.meshgrid(k_range, k_range, indexing='ij')
        idx_y = idx_y.flatten() # (9,) [0, 0, 0, 1, 1, 1, 2, 2, 2]
        idx_x = idx_x.flatten() # (9,) [0, 1, 2, 0, 1, 2, 0, 1, 2]

        if training == False and chunk_size is not None and N > chunk_size:
            # Empty final container
            out_conv = torch.empty((N, C), device=device, dtype=all_feats.dtype)
            
            # --- SUPER CHUNKING ---
            # We ALSO calculate the block coordinates to clear the RAM
            for i in range(0, N, chunk_size):
                # Let's extract the sub-blocks (no heavy allocation)
                bs_c = bs[i : i + chunk_size]
                y_c = y_fine[i : i + chunk_size].unsqueeze(1)
                x_c = x_fine[i : i + chunk_size].unsqueeze(1)
                b_c = batch_idx[i : i + chunk_size].unsqueeze(1).repeat(1, 9)

                offsets_c = torch.stack([
                    torch.full_like(bs_c, -1),  
                    bs_c // 2,                  
                    bs_c                        
                ], dim=1) 
                
                # Coordinate calculation and clamping ONLY for this chunk (RAM usage: negligible)
                g_Y = (y_c + offsets_c[:, idx_y] + 1).clamp(0, H_fine + 1)
                g_X = (x_c + offsets_c[:, idx_x] + 1).clamp(0, W_fine + 1)
                
                idx_chunk = ptr_grid[b_c.long(), g_Y.long(), g_X.long()]
                
                neigh_chunk = feats_padded[idx_chunk].view(-1, 9 * C)
                out_conv[i : i + chunk_size] = self.act(self.spatial_conv(neigh_chunk))
                
            return all_feats + out_conv

        else:
            # --- IN TRAINING ---
            offsets = torch.stack([
                torch.full_like(bs, -1), bs // 2, bs
            ], dim=1) 
            
            gather_Y = (y_fine.unsqueeze(1) + offsets[:, idx_y] + 1).clamp(0, H_fine + 1)
            gather_X = (x_fine.unsqueeze(1) + offsets[:, idx_x] + 1).clamp(0, W_fine + 1)
            gather_B = batch_idx.unsqueeze(1).repeat(1, 9)
            
            idx_3x3 = ptr_grid[gather_B.long(), gather_Y.long(), gather_X.long()]
            neighbor_feats = feats_padded[idx_3x3] 
            
            x = neighbor_feats.view(N, 9 * C)
            out = self.act(self.spatial_conv(x))
            return all_feats + out