import os
import torch
from torch.utils.cpp_extension import load
from torch.autograd import Function
from torch.autograd.function import once_differentiable

build_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'build')
os.makedirs(build_path, exist_ok=True)
file_path = os.path.split(os.path.abspath(__file__))[0]


import gscuda_adaptive
GSWrapper = gscuda_adaptive

class GSCUDA(Function):
    @staticmethod
    def forward(ctx, sigmas, coords, colors, rendered_img, dmax_tensor):
        # Salviamo dmax_tensor perché ora è un Tensor PyTorch
        ctx.save_for_backward(sigmas, coords, colors, dmax_tensor)
        h, w, c = rendered_img.shape
        s = sigmas.shape[0]
        GSWrapper.gs_render(sigmas, coords, colors, rendered_img, s, h, w, c, dmax_tensor)
        return rendered_img

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        sigmas, coords, colors, dmax_tensor = ctx.saved_tensors
        h, w, c = grad_output.shape
        s = sigmas.shape[0]
        grads_sigmas = torch.zeros_like(sigmas)
        grads_coords = torch.zeros_like(coords)
        grads_colors = torch.zeros_like(colors)
        GSWrapper.gs_render_backward(sigmas, coords, colors, grad_output.contiguous(), grads_sigmas, grads_coords, grads_colors, s, h, w, c, dmax_tensor)
        # 5 input nel forward -> 5 output nel backward (dmax non ha gradiente)
        return (grads_sigmas, grads_coords, grads_colors, None, None)

def gaussiansplatting_render(sigmas, coords, colors, image_size, dmax=0.1):
    sigmas = sigmas.contiguous() 
    coords = coords.contiguous() 
    colors = colours_with_alpha = colors.contiguous()
    
    # Se dmax è un numero, lo espandiamo in un Tensor lungo quanto le gaussiane
    if isinstance(dmax, (int, float)):
        dmax_tensor = torch.full((sigmas.shape[0],), float(dmax), device=sigmas.device, dtype=torch.float32)
    else:
        dmax_tensor = dmax.to(sigmas.device).to(torch.float32).contiguous()

    h, w = image_size[:2]
    c = colors.shape[-1]
    rendered_img = torch.zeros(h, w, c, device=sigmas.device, dtype=torch.float32)
    return GSCUDA.apply(sigmas, coords, colors, rendered_img, dmax_tensor)

if __name__ == "__main__":
    sigmas = torch.randn(10, 3).cuda()
    coords = torch.randn(10, 2).cuda()
    colors = torch.randn(10, 3).cuda()
    image_size = (100, 100)
    # Test con tensor variabile
    dmax_adaptive = torch.linspace(0.1, 0.5, 10).cuda()
    rendered_img = gaussiansplatting_render(sigmas, coords, colors, image_size, dmax_adaptive)
    print("Successo! Shape output:", rendered_img.shape)