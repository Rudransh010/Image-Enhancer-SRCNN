# benchmark.py - CPU inference latency and model metrics benchmarking
import argparse
import torch
import numpy as np
import os
import time
from pathlib import Path

from src.model import create_model, count_parameters
from src.utils import count_flops, get_model_size_mb


def benchmark_torchscript(model_path, input_size=(1, 3, 256, 256), num_runs=100, num_threads=1):
    print(f"\nBenchmarking TorchScript (num_threads={num_threads})...")
    torch.set_num_threads(num_threads)
    
    model = torch.jit.load(model_path)
    model.eval()
    
    dummy_input = torch.randn(input_size)
    
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    latencies = np.array(latencies)
    
    return {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }


def benchmark_onnx(model_path, input_size=(1, 3, 256, 256), num_runs=100, num_threads=1):
    print(f"\nBenchmarking ONNX Runtime (num_threads={num_threads})...")
    try:
        import onnxruntime as ort
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        
        session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        
        dummy_input = np.random.randn(*input_size).astype(np.float32)
        input_name = session.get_inputs()[0].name
        
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        latencies = np.array(latencies)
        
        return {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'min': np.min(latencies),
            'max': np.max(latencies)
        }
    except ImportError:
        print("onnxruntime not installed")
        return None


def main(args):
    print("=" * 60)
    print("SRCNN Edge Model Benchmark")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_model(args.num_channels, args.num_ds_blocks, args.num_res_blocks)
    model = model.to(device)
    
    params = count_parameters(model)
    print(f"\nModel Parameters: {params:,}")
    assert params <= 500_000, f"Model exceeds 500k parameter limit!"
    
    input_size = (1, 3, args.input_height, args.input_width)
    flops = count_flops(model, input_size)
    print(f"FLOPs (GFLOPs): {flops:.3f}")
    
    print(f"\nInput size: {input_size}")
    print(f"Target inference budget: <30ms (design goal)")
    
    results = {}
    
    if args.torchscript and os.path.exists(args.torchscript):
        ts_results = benchmark_torchscript(args.torchscript, input_size, args.num_runs, args.num_threads)
        results['TorchScript'] = ts_results
        print(f"  Mean: {ts_results['mean']:.2f}ms")
        print(f"  Median: {ts_results['median']:.2f}ms")
        print(f"  P95: {ts_results['p95']:.2f}ms")
        print(f"  P99: {ts_results['p99']:.2f}ms")
        
        ts_size = get_model_size_mb(args.torchscript)
        print(f"  Model size: {ts_size:.2f}MB")
    
    if args.onnx and os.path.exists(args.onnx):
        onnx_results = benchmark_onnx(args.onnx, input_size, args.num_runs, args.num_threads)
        if onnx_results:
            results['ONNX'] = onnx_results
            print(f"  Mean: {onnx_results['mean']:.2f}ms")
            print(f"  Median: {onnx_results['median']:.2f}ms")
            print(f"  P95: {onnx_results['p95']:.2f}ms")
            print(f"  P99: {onnx_results['p99']:.2f}ms")
            
            onnx_size = get_model_size_mb(args.onnx)
            print(f"  Model size: {onnx_size:.2f}MB")
    
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    print(f"Parameters: {params:,}")
    print(f"FLOPs: {flops:.3f} GFLOPs")
    print(f"Input: {input_size}")
    
    for backend, metrics in results.items():
        print(f"\n{backend}:")
        print(f"  Mean latency: {metrics['mean']:.2f}ms")
        print(f"  Median latency: {metrics['median']:.2f}ms")
        print(f"  P95 latency: {metrics['p95']:.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torchscript', type=str, default='artifacts/models/model_fp32.pt')
    parser.add_argument('--onnx', type=str, default='artifacts/models/model_fp32.onnx')
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--input_height', type=int, default=256)
    parser.add_argument('--input_width', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=48)
    parser.add_argument('--num_ds_blocks', type=int, default=3)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    
    args = parser.parse_args()
    main(args)
