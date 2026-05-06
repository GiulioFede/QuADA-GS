<div align="center">

# 🔬 QuADA-GS
### Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution

<br>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

<img src="assets/teaser.png" width="100%" alt="QuADA-GS Teaser"/>

</div>

---

## 📋 Overview

**QuADA-GS** introduces a novel framework for arbitrary-scale image super-resolution by learning to **adaptively allocate Gaussians** across the image, enabling high-quality upscaling at any scale factor without retraining.

---

## ✅ Release Checklist

- [x] **Inference code** — including runtime and memory profiling
- [x] **Evaluation code** — PSNR, SSIM, LPIPS, DISTS metrics
- [ ] **Training code**

---

## ⚙️ Requirements

Before running inference, make sure your environment is properly set up:

```bash
# Clone the repository
git clone https://github.com/your-username/QuADA-GS.git
cd QuADA-GS

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Inference

### 1 — Prepare your dataset

Place your low-resolution benchmark images in a directory of your choice. The pipeline expects the following structure:

```
your_dataset/
├── image_001.png
├── image_002.png
└── ...
```

> **Tip:** The example below uses the **DIV2K 100-image validation set** with bicubic downsampling at arbitrary scale (`AnyScaleTestBicubic`).

---

### 2 — Run evaluation

Use the following command to run inference with a pretrained model:

```bash
python inference/evaluate_inference.py \
    --path_to_image_dataset "/path/to/your/dataset" \
    --results-dir /path/to/output/directory \
    --pretrained_model RDN_best
```

#### Full working example (DIV2K benchmark):

```bash
python inference/evaluate_inference.py \
    --path_to_image_dataset "/raid/homes/giulio.federico/DiGT/gaussian_vae/data/benchmarks/AnyScaleTestBicubic/DIV2K100" \
    --results-dir /raid/homes/giulio.federico/da_eliminare/QuADA-GS/da_el \
    --pretrained_model RDN_best
```

---

### 3 — Arguments reference

| Argument | Type | Description |
|---|---|---|
| `--path_to_image_dataset` | `str` | Path to the folder containing the input low-resolution images |
| `--results-dir` | `str` | Directory where super-resolved outputs will be saved |
| `--pretrained_model` | `str` | Name of the pretrained checkpoint to load (e.g. `RDN_best`) |

---

### 4 — Output

After inference completes, the super-resolved images will be saved in the directory specified by `--results-dir`:

```
results-dir/
├── image_001_SR.png
├── image_002_SR.png
└── metrics.json       ← PSNR / SSIM summary
```

---

## 📊 Metrics Evaluation

Once inference is complete and the super-resolved images are saved, you can compute the full suite of image quality metrics (PSNR, SSIM, LPIPS, DISTS) with:

```bash
python inference/compute_metrics.py \
    --sr-dir /path/to/super_resolved_images \
    --gt-dir /path/to/ground_truth_images
```

#### Full working example (DIV2K benchmark):

```bash
python inference/compute_metrics.py \
    --sr-dir /raid/homes/giulio.federico/da_eliminare/QuADA-GS/da_el \
    --gt-dir "/raid/homes/giulio.federico/DiGT/gaussian_vae/data/benchmarks/AnyScaleTestBicubic/DIV2K100"
```

#### Arguments reference

| Argument | Description |
|---|---|
| `--sr-dir` | Path to the folder containing the super-resolved output images |
| `--gt-dir` | Path to the folder containing the ground-truth high-resolution images |

#### Computed metrics

| Metric | Description |
|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio — measures pixel-level fidelity |
| **SSIM** | Structural Similarity Index — evaluates perceptual structure |
| **LPIPS** | Learned Perceptual Image Patch Similarity — deep feature-based perceptual quality |
| **DISTS** | Deep Image Structure and Texture Similarity — texture-aware perceptual metric |

Results are printed to console and saved as `metrics_summary.json` inside `--sr-dir`.

---

## 📁 Project Structure

```
QuADA-GS/
├── inference/
│   └── evaluate_inference.py   # Main evaluation script
├── models/                     # Model architectures
├── pretrained/                 # Pretrained checkpoints
├── data/                       # Dataset utilities
└── README.md
```

---

## 📌 Notes

- The model supports **arbitrary scale factors** at inference time — no retraining needed.
- Results are evaluated using standard **PSNR** and **SSIM** metrics.
- The default pretrained checkpoint `RDN_best` was trained on the DIV2K training set.

---

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{quada_gs,
  title     = {QuADA-GS: Learning to Adaptively Allocate Gaussians for Arbitrary-Scale Image Super-Resolution},
  author    = {Your Name et al.},
  year      = {2025}
}
```

---

<div align="center">
Made with ❤️ — QuADA-GS Team
</div>