import torch




def rendering_cuda_dmax_adaptive(sigma_x, sigma_y, rho, coords, colours_with_alpha, sr_size, step_size,  device, dmax=1):
    from utils.gs_utils.utils.gs_cuda_dmax_adaptive.gswrapper_adaptve import GSCUDA
    sigmas = torch.cat([sigma_y/step_size*2/(sr_size[1] - 1), sigma_x/step_size*2/(sr_size[0] - 1),  rho], dim=-1).contiguous()  # (gs num, 3)
    coords[:, 0] = (coords[:, 0] + 1 - 1/sr_size[1]) * sr_size[1] / (sr_size[1] - 1) - 1.0
    coords[:, 1] = (coords[:, 1] + 1 - 1/sr_size[0]) * sr_size[0] / (sr_size[0] - 1) - 1.0
    colours_with_alpha = colours_with_alpha.contiguous()  # (gs num, 3)
    rendered_img = torch.zeros(sr_size[0], sr_size[1], 3).to(device).type(torch.float32).contiguous()
    final_image = GSCUDA.apply(sigmas, coords, colours_with_alpha, rendered_img, dmax)
    final_image = final_image.permute(2, 0, 1).contiguous()
    return final_image




def rasterize_image(gs_parameters, #gaussian parameters (b, 9) --> (Sx, Sy, Theta, Opacity, R, G, B, X, Y)
                    resolution, #resolution of the final image
                    scale, #scale es. x2, x5,...
                    default_step_size = 1.2,
                    cell_scale = None,
                    dmax_levels = None):



    gs_parameters = gs_parameters.float() 
    b = gs_parameters.shape[0]

    if isinstance(resolution, (int, float)):
        grt_resolution = torch.tensor([int(resolution), int(resolution)], device=gs_parameters.device)
    elif isinstance(resolution, (list, tuple)):
        grt_resolution = torch.tensor(resolution, device=gs_parameters.device)
    else:
        grt_resolution = resolution.to(gs_parameters.device)

    final_scale = scale


    if dmax_levels is None:
        current_dmax = torch.where(cell_scale > 3.0, 0.1, 
                torch.where(cell_scale > 1.5, 0.1, 0.1)).flatten().contiguous()
    else:
        v1, v2, v3 = dmax_levels
        current_dmax = torch.where(cell_scale > 3.0, v1, 
                torch.where(cell_scale > 1.5, v2, v3)).flatten().contiguous()


    step_size = default_step_size/ final_scale
    sigma_scaling_factor = 1.0
    all_images = []
    for b_i in range(b):
        if cell_scale is None:
            cell_scale = 1


        # prepare gaussian properties
        sigma_x = 0.99999 *(torch.sigmoid(gs_parameters[b_i, :, 0:1]) * cell_scale * sigma_scaling_factor) + 1e-9
        sigma_y = 0.99999 *(torch.sigmoid(gs_parameters[b_i, :, 1:2]) * cell_scale * sigma_scaling_factor) + 1e-9
        
        rho = 0.999999 * torch.tanh(gs_parameters[b_i, :, 2:3])
        alpha = torch.sigmoid(gs_parameters[b_i, :, 3:4])
        colours = torch.sigmoid(gs_parameters[b_i, :, 4:7])
        coords = (gs_parameters[b_i, :, 7:9] * 2 - 1)
        colours_with_alpha = colours * alpha

        final_image = rendering_cuda_dmax_adaptive(sigma_x, sigma_y, rho, coords, colours_with_alpha, grt_resolution, step_size, dmax=current_dmax, device=sigma_x.device)
        all_images.append(final_image.unsqueeze(0))
    
    all_images = torch.cat(all_images, dim=0)  

    return all_images
