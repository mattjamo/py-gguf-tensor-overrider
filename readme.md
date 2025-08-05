# GGUF Tensor Overrider - Python Version

A Python port of the GGUF Tensor Overrider tool for optimizing GGUF files for NVIDIA GPUs on Windows.

Adapted from https://github.com/k-koehler/gguf-tensor-overrider

Adapterer: Matthew O'Brien

## Prerequisites

- Python 3.7 or higher
- NVIDIA GPU with drivers installed
- `nvidia-smi` command available in PATH

## Installation

### Automatic Installation (Windows)

1. Download `install.bat`
2. Run as administrator (right-click → "Run as administrator")
3. Follow the prompts

### Manual Installation

1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make the script executable and add to PATH (optional)

## Usage

```bash
python gguf_tensor_overrider.py -g "https://example.com/model.gguf" -c 4096
```

### Command Line Arguments

- `-g, --gguf-url`: URL of the GGUF file to optimize (required)
- `-c, --context-length`: Context length for optimization (required)
- `--context-quantization-size`: Context quantization size (4, 8, or 16, default: 16)
- `--no-check`: Skip system resource limits check (useful when using swap)
- `--gpu-percentage`: Percentage of GPU memory to use (default: 0.9)
- `--granular-gpu-percentage`: Set percentage for each GPU (format: "0.9,0.8,0.7")
- `--verbose`: Enable verbose logging

### Examples

Basic usage:
```bash
python gguf_tensor_overrider.py -g "https://huggingface.co/model/model.gguf" -c 4096
```

With custom GPU memory usage:
```bash
python gguf_tensor_overrider.py -g "https://huggingface.co/model/model.gguf" -c 4096 --gpu-percentage 0.8
```

With granular GPU control:
```bash
python gguf_tensor_overrider.py -g "https://huggingface.co/model/model.gguf" -c 4096 --granular-gpu-percentage "0.9,0.8,0.7"
```

Verbose output:
```bash
python gguf_tensor_overrider.py -g "https://huggingface.co/model/model.gguf" -c 4096 --verbose
```

## Supported Architectures

- Llama (including Llama4)
- Qwen3 and Qwen3MoE
- DeepSeek2
- Dots1
- Hunyuan-MoE
- Generic architectures (with fallback extraction)

## Multi-part GGUF Files

The tool automatically detects and handles multi-part GGUF files with the naming pattern:
`basename-00001-of-00003.gguf`

## Output

The tool generates a command string that can be used with llama.cpp to optimize tensor placement across your GPUs and CPU. The output format is:

```
-ngl 0 -ot "tensor_name=CUDA0" -ot "another_tensor=CPU" ...
```

## Memory Allocation Strategy

The tool uses a multi-pass allocation strategy:

1. **Embedding tensors**: Allocated to CPU (compatibility)
2. **Attention tensors + KV cache**: Allocated by block to GPUs
3. **FFN tensors**: Allocated to available devices
4. **Gate tensors**: For MoE models, allocated to available devices
5. **Norm tensors**: Allocated to available devices
6. **Remaining tensors**: Allocated to available devices

## Troubleshooting

### "nvidia-smi not found"
- Install NVIDIA GPU drivers
- Ensure nvidia-smi is in your system PATH

### "Model does not fit in memory"
- Reduce context length (`-c` parameter)
- Use lower context quantization size (`--context-quantization-size 8` or `4`)
- Use `--no-check` if you have swap memory available

### Download failures
- Check internet connection
- Verify the GGUF URL is accessible
- Some models may require authentication tokens

## Differences from Original

This Python version maintains feature parity with the original Node.js version but includes:

- Native Python implementation (no Node.js dependency)
- Windows-optimized installation script
- Simplified GGUF parsing (header-only for metadata extraction)
- Better error handling and logging

## License

This project has a slightly modified license from the original repository. This repo adopts Apache 2.0 with k-koehler license appended which is as follows:
Go wild. The code in this repository is free, open source, modifiable, distributable, whatever-the-fuck-you-wantable.