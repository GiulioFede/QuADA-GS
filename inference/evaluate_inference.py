
import argparse
from models.main_model import model_versions
import torch
import os
import glob
import cv2
import numpy as np
import torch.nn.functional as F
from utils.raster import rasterize_image
import time
import torch
import torch.nn.functional as F
import numpy as np
import math
import pytorch_lightning as pl



def preprocess(x, denominator):
    # pad input image to be a multiple of denominator
    _,c,h,w = x.shape
    if h % denominator > 0:
        pad_h = denominator - h % denominator
    else:
        pad_h = 0
    if w % denominator > 0:
        pad_w = denominator - w % denominator
    else:
        pad_w = 0
    x_new = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
    return x_new

def postprocess(x, gt_size_h, gt_size_w):
    x_new = x[:, :, :gt_size_h, :gt_size_w]
    return x_new


class Model_Lighting(pl.LightningModule):

    def __init__(
        self,
        model_name = None,
    ):
        super().__init__()

        self.model_name = model_name

        self.__init__model_version__()

    
    def __init__model_version__(self):
        
        self.model = model_versions[self.model_name]()



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model_Lighting.load_from_checkpoint(
                                strict=False,
                                checkpoint_path=args.load_checkpoint,
                                model_name=args.model,
                                map_location=device)
    
    model = model.model
    model.to(device) 
    model.eval()

    type_of_inference = args.type_of_inference
    dmax = torch.tensor(args.dmax, device=device, dtype=torch.float32)

    # LOAD all LR image path
    lr_image_paths = sorted(glob.glob(os.path.join(args.path_to_image_dataset, "x"+str(args.scale_to_evaluate), "LR", "*.png")))

    dataset_name = args.path_to_image_dataset.split("/")[-1]
    result_dir = os.path.join(args.results_dir, dataset_name, "x"+str(args.scale_to_evaluate))
    os.makedirs(result_dir, exist_ok=True)

    time_cost_list = []
    used_memory_list = []

    print(f"Start inference on {dataset_name} with scale {args.scale_to_evaluate}")

    img_i = 0
    for image_path in lr_image_paths:
        image_name = os.path.basename(image_path)
        img_i+=1

        img_cv = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img_cv[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        gt_size = [math.floor(args.scale_to_evaluate * img.shape[2]), math.floor(args.scale_to_evaluate * img.shape[3])]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()            
            torch.cuda.reset_peak_memory_stats() 
            torch.cuda.synchronize()

        start = time.time()

        with torch.no_grad():
            if type_of_inference == "tiling":
                raise ValueError("Tiling process not yet implemented. Please use parallel instead.")
            else:
                lq_pad = preprocess(img, 12)
                gt_size_pad = torch.tensor([math.floor(args.scale_to_evaluate* lq_pad.shape[2]), math.floor(args.scale_to_evaluate * lq_pad.shape[3])])
                gt_size_pad = gt_size_pad.unsqueeze(0)

                out_feats_list, out_coords_list, mapping_list, _, out_sigma_scales_list, H_up, W_up, _, _ = model(lq_pad, scale=args.scale_to_evaluate, inference=True)
                gs_scales = out_sigma_scales_list[0]

                gaussians = model.predict_gaussians(out_feats_list[0].unsqueeze(0),
                                                    1,
                                                    W_up,
                                                    H_up, 
                                                    out_coords_list[0].unsqueeze(0),
                                                    gs_scales=gs_scales).squeeze(0)
                
                #TODO: USA dmax 0.1, 0.25, 0.5 per scale >= 4
                sr_results = rasterize_image(gs_parameters=gaussians.unsqueeze(0), 
                                           resolution=gt_size_pad[0], 
                                           scale=args.scale_to_evaluate,
                                           default_step_size=model.quad_refiner.default_step_size,
                                           cell_scale=gs_scales,
                                           dmax_levels = dmax)


        torch.cuda.synchronize()
        end = time.time()
        time_cost = end - start
        

        if torch.cuda.is_available():
            used_memory = torch.cuda.max_memory_allocated()
            print(f"{img_i} Image: {image_name} | Time: {time_cost*1000:.4f} ms | GPU Peak: {used_memory /1024 /1024:.2f} MB")
        else:
            used_memory = 0

        time_cost_list.append(time_cost)
        used_memory_list.append(used_memory) 

        output = postprocess(sr_results, gt_size[0], gt_size[1])

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        
        cv2.imwrite(os.path.join(result_dir, image_name), output)

        del sr_results
        del img
    
    # Final statistics
    if len(time_cost_list) > 2:
        # Time: -> ms
        # Memory: MB
        times_ms = np.array(time_cost_list) * 1000
        mems_mb = np.array(used_memory_list) / (1024 * 1024)

        # --- Time (Skip first two) ---
        valid_times = times_ms[2:] 
        avg_time = np.mean(valid_times)
        std_time = np.std(valid_times)
        
        # --- Memory (Skip first two) ---
        valid_mems = mems_mb[2:]
        avg_mem = np.mean(valid_mems)
        std_mem = np.std(valid_mems)
        min_mem = np.min(valid_mems)
        max_mem = np.max(valid_mems)

        print("\n" + "="*40)
        print(f"FINAL STATISTICS (Skipped first 2 samples)")
        print("="*40)
        
        # Format: Mean ± Std
        print(f"TIME   : {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"MEMORY : {avg_mem:.2f} ± {std_mem:.2f} MB")
        
        # Range: Min and Max
        print(f"MEM Range: [{min_mem:.2f} - {max_mem:.2f}] MB")
        print("="*40)
    else:
        print("Not enough samples to calculate statistics (need > 2).")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_image_dataset", type=str, default="/raid/homes/giulio.federico/DiGT/gaussian_vae/data/benchmarks/AnyScaleTestBicubic/DIV2K100")
    parser.add_argument("--scale_to_evaluate", type=int, choices=[2,3,4,6,8,12,16,24,30], default=12.7) 
    parser.add_argument("--results-dir", type=str, default="/raid/homes/giulio.federico/QuADA-GS/da_eliminare")
    parser.add_argument("--load_checkpoint", type=str, default="/raid/homes/giulio.federico/DiGT/gaussian_vae/results/063-QuASAR_v2/checkpoints/last-x12_L1_L2_last.ckpt")
    parser.add_argument("--model", type=str, choices=list(model_versions), default="QuADA_GS_v2")
    parser.add_argument("--type_of_inference", type=str, choices=["parallel"], default="parallel") #TODO "tiling"
    parser.add_argument(
    "--dmax", 
    type=float, 
    nargs="+", 
    default=[0.1, 0.1, 0.1],
    help="Determines the rasterization speed for each level. Lower values mean higher speed. "
        "If the input LR is very small and/or the scale is very high, "
        "increase these values of L1 and L2 (dmaxL1>dmaxL2) to avoid holes or artifacts."
)

    args = parser.parse_args()
    main(args)