# CIFAR-100 Transformer Image Classification

## Overview

This project trains and compares two transformer-based image classifiers on the CIFAR-100 dataset:

1. **Primary Model: Swin Transformer**
   - Hierarchical Vision Transformer
   - Window-based multi-head self-attention
   - Shifted-window attention
   - Hierarchical feature representation

2. **Baseline Model: Vision Transformer (ViT)**
   - Standard single-scale Transformer
   - Global self-attention over image patches

The main research question is:

> Does adding locality and hierarchical feature processing improve Vision Transformer learning when training data are limited?

Both models are trained from randomly initialized weights and use the same:

- Dataset splits
- Data augmentations
- Optimizer
- Learning-rate schedule
- Training duration
- Batch size
- Evaluation procedure

The ViT baseline is parameter-matched with the Swin Transformer to ensure performance differences are caused by architecture differences rather than model capacity.

---

# Dataset

## CIFAR-100

CIFAR-100 contains:

- 60,000 RGB images
- Image resolution: 32 × 32
- 100 object classes

Official dataset split:

| Split | Number of Images |
|---|---:|
| Training | 50,000 |
| Testing | 10,000 |

The official training set is divided into:

| Split | Number of Images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Testing | 10,000 |

The training/validation split uses:

- Stratified sampling
- Fixed random seed (42)
- No overlap between splits

---

# Repository Structure
```
vision_transformer/

├── README.md
├── requirements.txt
├── experiments.pdf

├── configs/
│ ├── primary.yaml
│ └── vit_baseline.yaml

├── src/
│ ├── init.py
│ ├── data.py
│ ├── models.py
│ ├── train.py
│ ├── evaluate.py
│ ├── metrics.py
│ └── utils.py

├── tests/
│ ├── test_data.py
│ ├── test_models.py
│ └── test_parameters.py

├── logs/
│ ├── primary_training.csv
│ └── vit_training.csv

├── checkpoints/

└── figures/
├── training_curves.png
├── confusion_matrix.png
└── error_examples.png
```

---

# Installation

Create the environment:

```bash
conda create -n cifar-transformers python=3.11

conda activate cifar-transformers

Install dependencies:

pip install -r requirements.txt

```

# Dataset Processing
### Input Resolution

Both models use the original CIFAR-100 image resolution:
32 × 32 × 3

No resizing is applied.

The original resolution is used because CIFAR-100 images are already small and resizing to ImageNet resolution would introduce unnecessary interpolation artifacts.

### Patch Embedding

Both models use:

Patch size: 4 × 4

The input image is divided into:

(32 / 4) × (32 / 4)

= 8 × 8

= 64 patches

### Normalization

Images are normalized using CIFAR-100 dataset statistics:

Mean:
[0.5071, 0.4867, 0.4408]

Standard deviation:
[0.2675, 0.2565, 0.2761]


### Data Augmentation

The training pipeline applies:

Tensor conversion
Normalization
Random horizontal flipping
Random cropping with padding
Color jitter
Random erasing

The validation and testing pipelines use:

Tensor conversion
Normalization

Random augmentation is not applied during validation/testing because evaluation should be deterministic and provide consistent measurements.

# Model Architectures
### Primary Model: Swin Transformer

The primary model uses a hierarchical Vision Transformer architecture.

Components:

Patch embedding
Window-based multi-head self-attention
Shifted-window attention
Hierarchical transformer stages
Patch merging
Classification head with 100 outputs

The Swin Transformer introduces locality through:

Local attention windows
Shifted windows
Multi-scale hierarchical feature processing

### Baseline Model: Vision Transformer

The baseline model uses standard global self-attention.

Components:

Patch embedding
Learnable positional embeddings
Class token
Transformer encoder blocks
Global multi-head self-attention
Feed-forward MLP layers
Classification head with 100 outputs

The ViT baseline is configured so that its trainable parameter count is within 10% of the Swin Transformer.

This ensures a fair comparison between:

Hierarchical local attention
Global attention

Training Configuration

Both models use identical training settings.

Optimizer
AdamW
Weight Decay
0.05
Initial Learning Rate
5e-4
Learning Rate Schedule
Cosine Annealing
Warmup
10 epochs
Loss Function
Cross Entropy Loss

Label smoothing:
0.1
Batch Size
128
Training Duration
200 epochs
Mixed Precision

Enabled using automatic mixed precision (AMP).

Gradient Clipping
Maximum gradient norm:
1.0
Random Seed
42
Training
Train Swin Transformer
python src/train.py \
    --config configs/primary.yaml


Train Vision Transformer
python src/train.py \
    --config configs/vit_baseline.yaml
Training Logs

Training logs include:

Epoch number
Training loss
Training accuracy
Validation loss
Validation accuracy
Learning rate
Epoch duration

Logs are exported as CSV files:

logs/

├── primary_training.csv
└── vit_training.csv
Evaluation
Evaluate Swin Transformer
python src/evaluate.py \
    --config configs/primary.yaml \
    --checkpoint checkpoints/swin/best_model.pt
Evaluate Vision Transformer
python src/evaluate.py \
    --config configs/vit_baseline.yaml \
    --checkpoint checkpoints/vit/best_model.pt
Evaluation Metrics

The evaluation pipeline reports:

Top-1 accuracy
Top-5 accuracy
Macro F1 score
Per-class precision
Per-class recall
Per-class F1 score
Five highest F1 classes
Five lowest F1 classes
Common confusion patterns
Error Analysis
Confusion Matrix

Generated file:

figures/confusion_matrix.png

The confusion matrix is used to identify classes that are frequently confused.

Misclassified Examples

Generated file:

figures/error_examples.png

The visualization contains 12 misclassified test examples.

Selection method:

Most confident incorrect predictions

Each example includes:

Input image
True label
Predicted label
Prediction probability
Automated Tests

Run all tests:

pytest tests/ -v
Dataset Tests

File:

tests/test_data.py

Tests:

Dataset split sizes
Train/validation/test disjointness
Stratified split verification
Transform output shape
Model Tests

File:

tests/test_models.py

Tests:

Model forward pass
Output dimensions
CIFAR-100 input compatibility
Patch-size validation
Parameter Tests

File:

tests/test_parameters.py

Tests:

Parameter count calculation
Trainable parameter validation
ViT/Swin parameter matching requirement
Reproducibility

All experiments use fixed random seeds:

Python random seed
NumPy seed
PyTorch seed
CUDA seed

The YAML configuration files define:

Dataset settings
Model architecture
Optimization parameters
Evaluation settings
Results

Final results will be reported in the experimental report.

Model	Top-1 Accuracy	Top-5 Accuracy	Parameters	Macro F1
Swin Transformer	TBD	TBD	TBD	TBD
Vision Transformer	TBD	TBD	TBD	TBD