# Edge-Optimized SRCNN Super-Resolution Model

Production-ready 2× super-resolution model optimized for mobile CPU inference with <30ms target latency.

## Project Structure

```
srcnn/
├── src/                          # Core source code
│   ├── model.py                  # Model architecture
│   ├── data.py                   # Dataset loader
│   ├── train.py                  # Training script
│   ├── eval.py                   # Evaluation script
│   ├── export.py                 # Export to TorchScript/ONNX
│   ├── quantize_ptq.py           # Post-training quantization
│   ├── quantize_qat.py           # Quantization-aware training
│   ├── benchmark.py              # Benchmarking
│   └── utils.py                  # Utilities
├── tests/                        # Unit tests
│   └── test_model_forward.py
├── data/                         # Data directory
│   ├── DIV2K_train_HR/           # Training images
│   └── processed/                # Preprocessed data
├── checkpoints/                  # Model checkpoints
│   ├── fp32/                     # FP32 checkpoints
│   └── int8/                     # INT8 checkpoints
├── artifacts/                    # Exported models and logs
│   ├── models/                   # Exported models
│   └── logs/                     # Training logs
├── docs/                         # Documentation
├── main.py                       # Entry point
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Quick Start

### Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
python main.py train --epochs 150 --batch_size 16
```

### Evaluation
```bash
python main.py eval
```

### Export
```bash
python main.py export
```

### Quantization
```bash
python main.py quantize-ptq
python main.py quantize-qat --epochs 20
```

### Benchmarking
```bash
python main.py benchmark --num_threads 1
```

### Testing
```bash
python main.py test
```

## Direct Script Usage

You can also run scripts directly from src/:

```bash
python src/train.py --train_dir dataset/DIV2K_train_HR/DIV2K_train_HR --epochs 150
python src/eval.py --checkpoint checkpoints/fp32/best.pth
python src/export.py --checkpoint checkpoints/fp32/best.pth
python src/benchmark.py --torchscript artifacts/models/model_fp32.pt
```

## Model Specifications

| Metric | Value |
|--------|-------|
| Parameters | ~250k |
| FLOPs (256×256) | ~2.1 GFLOPs |
| Model Size (FP32) | ~1.0 MB |
| Model Size (INT8) | ~0.3 MB |
| Target Latency | <30ms (256×256, CPU) |
| PSNR (DIV2K val) | 32-34 dB |
| SSIM (DIV2K val) | 0.88-0.92 |

## Training Configuration

- **Optimizer**: Adam (lr=1e-3, weight_decay=0)
- **Scheduler**: CosineAnnealingLR
- **Loss**: L1 (MAE)
- **Batch Size**: 16
- **Patch Size**: 96×96
- **Epochs**: 150
- **Mixed Precision**: AMP enabled

## Artifacts

After training and export:

- `checkpoints/fp32/best.pth` - Best FP32 model
- `checkpoints/fp32/last.pth` - Last epoch checkpoint
- `artifacts/models/model_fp32.pt` - TorchScript export
- `artifacts/models/model_fp32.onnx` - ONNX export
- `artifacts/logs/training_log.csv` - Training metrics

## References

- SRCNN: https://arxiv.org/abs/1501.04112
- DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
