# Edge-Optimized SRCNN Super-Resolution Model

Production-ready 2× super-resolution model optimized for mobile CPU inference with <30ms target latency.

## Architecture

- **Model Family**: SRCNN with depthwise separable convolutions and residual blocks
- **Parameters**: ~250k (well under 500k budget)
- **Design**: Bicubic upscale → residual refinement network
- **Activations**: PReLU (mobile-friendly)
- **Quantization**: PTQ and QAT support

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download DIV2K:
```bash
# Create data directory
mkdir -p data

# Download DIV2K training set (800 images)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip -d data/

# Download DIV2K validation set (100 images)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip DIV2K_valid_HR.zip -d data/
```

Or use Flickr2K as alternative:
```bash
wget http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
tar -xf Flickr2K.tar -C data/
```

## Training

### Basic training (150 epochs, batch size 16):
```bash
python train.py \
  --train_dir data/DIV2K_train_HR \
  --val_dir data/DIV2K_valid_HR \
  --epochs 150 \
  --batch_size 16 \
  --patch_size 96
```

### Resume training:
```bash
python train.py \
  --train_dir data/DIV2K_train_HR \
  --val_dir data/DIV2K_valid_HR \
  --resume checkpoints/best.pth \
  --epochs 150
```

### With torch.compile (PyTorch 2.0+):
```bash
python train.py --compile --epochs 150
```

### Custom model architecture:
```bash
python train.py \
  --num_channels 64 \
  --num_ds_blocks 4 \
  --num_res_blocks 3 \
  --epochs 150
```

Training logs saved to `logs/training_log.csv`.

## Evaluation

```bash
python eval.py --checkpoint checkpoints/best.pth
```

## Export

### Export to TorchScript and ONNX:
```bash
python export.py --checkpoint checkpoints/best.pth --output_dir artifacts
```

### Export with TFLite conversion:
```bash
python export.py \
  --checkpoint checkpoints/best.pth \
  --output_dir artifacts \
  --export_tflite
```

Exported models:
- `artifacts/model_fp32.pt` - TorchScript (CPU inference)
- `artifacts/model_fp32.onnx` - ONNX (CPU/mobile inference)
- `artifacts/model_fp32.tflite` - TFLite (Android/iOS)

## Quantization

### Post-Training Quantization (PTQ):
```bash
python quantize_ptq.py \
  --checkpoint checkpoints/best.pth \
  --val_dir data/DIV2K_valid_HR \
  --output_dir artifacts
```

Output: `artifacts/model_int8_ptq.pth`

### Quantization-Aware Training (QAT):
```bash
python quantize_qat.py \
  --checkpoint checkpoints/best.pth \
  --train_dir data/DIV2K_train_HR \
  --val_dir data/DIV2K_valid_HR \
  --epochs 20 \
  --output_dir artifacts
```

Output: `artifacts/model_int8_qat_converted.pth`

## Benchmarking

### CPU inference latency (single-threaded):
```bash
python benchmark.py \
  --torchscript artifacts/model_fp32.pt \
  --onnx artifacts/model_fp32.onnx \
  --num_runs 100 \
  --num_threads 1
```

### Multi-threaded benchmark:
```bash
python benchmark.py \
  --torchscript artifacts/model_fp32.pt \
  --onnx artifacts/model_fp32.onnx \
  --num_threads 4
```

Outputs:
- Mean/median/P95/P99 latency (ms)
- Model size (MB)
- FLOPs (GFLOPs)

## Testing

```bash
python tests/test_model_forward.py
```

Verifies:
- Forward pass correctness
- Parameter count (<= 500k)
- Export compatibility (TorchScript, ONNX)
- Multiple input sizes

## Model Specifications

| Metric | Value |
|--------|-------|
| Parameters | ~250k |
| FLOPs (256×256) | ~2.1 GFLOPs |
| Model Size (FP32) | ~1.0 MB |
| Model Size (INT8) | ~0.3 MB |
| Target Latency | <30ms (256×256, CPU) |
| Input | LR image (any size) |
| Output | SR image (2× upscaled) |
| Upscaling | Bicubic + residual refinement |

## Mobile Deployment

### Android (TFLite):
```bash
# Copy model to Android assets
adb push artifacts/model_fp32.tflite /data/local/tmp/

# Run TFLite benchmark tool
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model_fp32.tflite \
  --num_threads=1
```

### iOS (CoreML):
Convert ONNX to CoreML:
```bash
pip install coremltools
python -c "
import coremltools as ct
import onnx
onnx_model = onnx.load('artifacts/model_fp32.onnx')
mlmodel = ct.convert(onnx_model, source='onnx')
mlmodel.save('artifacts/model_fp32.mlmodel')
"
```

## Configuration

Default hyperparameters in scripts:
- Learning rate: 1e-3 (training), 1e-4 (QAT)
- Optimizer: Adam (weight_decay=0)
- Scheduler: CosineAnnealingLR
- Loss: L1 (MAE)
- Batch size: 16
- Patch size: 96×96
- Epochs: 150 (training), 20 (QAT)

## Performance Notes

- **FP32 inference**: ~15-25ms per 256×256 image (CPU, single-threaded)
- **INT8 inference**: ~8-15ms per 256×256 image (CPU, single-threaded)
- **PSNR**: ~32-34 dB on DIV2K validation
- **SSIM**: ~0.88-0.92 on DIV2K validation

Actual latency depends on CPU model and thread count.

## File Structure

```
srcnn/
├── model.py                 # Model architecture
├── data.py                  # Dataset and dataloader
├── train.py                 # Training script
├── eval.py                  # Evaluation script
├── export.py                # Export to TorchScript/ONNX/TFLite
├── quantize_ptq.py          # Post-training quantization
├── quantize_qat.py          # Quantization-aware training
├── benchmark.py             # CPU inference benchmarking
├── utils.py                 # Metrics and utilities
├── requirements.txt         # Dependencies
├── README.md                # This file
├── tests/
│   └── test_model_forward.py # Unit tests
├── checkpoints/             # Training checkpoints
├── artifacts/               # Exported models
└── logs/                    # Training logs
```

## Troubleshooting

**ONNX export fails**: Ensure opset_version matches your ONNX Runtime version.

**TFLite conversion fails**: Install tensorflow and onnx-tf: `pip install tensorflow onnx-tf`

**Out of memory during training**: Reduce batch_size or patch_size.

**Slow training**: Enable torch.compile with `--compile` flag (PyTorch 2.0+).

## References

- SRCNN: https://arxiv.org/abs/1501.04112
- DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- ONNX: https://onnx.ai/
- TFLite: https://www.tensorflow.org/lite
