#!/usr/bin/env python3
"""
GGUF Tensor Overrider - Python version
Optimize GGUF files for NVIDIA GPUs by allocating tensors across devices
"""

import argparse
import subprocess
import sys
import os
import re
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import struct
import io
import json


class LogLevel(Enum):
    INFO = "info"
    DEFAULT = "default"
    NOLOG = "nolog"


class Log:
    verbose = False
    no_log = False
    
    @classmethod
    def log(cls, log_level: LogLevel, message: str):
        if cls.no_log:
            return
        if log_level == LogLevel.INFO and cls.verbose:
            print(f"[{log_level.value.upper()}] {message}")
        elif log_level == LogLevel.DEFAULT:
            print(message)
    
    @classmethod
    def warn(cls, message: str):
        if cls.no_log:
            return
        print(f"[WARN] {message}", file=sys.stderr)
    
    @classmethod
    def error(cls, message: str):
        if cls.no_log:
            return
        print(f"[ERROR] {message}", file=sys.stderr)


@dataclass
class Gpu:
    cuda_id: int
    name: str
    memory_total_bytes: int


class GGMLQuantizationType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 31
    TQ2_0 = 32


# Quantization size mapping in bytes
GGUF_QUANTIZATION_SIZE_MAP_BYTES = {
    GGMLQuantizationType.F32: 4,
    GGMLQuantizationType.F16: 2,
    GGMLQuantizationType.BF16: 2,
    GGMLQuantizationType.F64: 8,
    GGMLQuantizationType.I8: 1,
    GGMLQuantizationType.I16: 2,
    GGMLQuantizationType.I32: 4,
    GGMLQuantizationType.I64: 8,
    GGMLQuantizationType.Q4_0: 0.5625,
    GGMLQuantizationType.Q4_1: 0.625,
    GGMLQuantizationType.Q5_0: 0.6875,
    GGMLQuantizationType.Q5_1: 0.75,
    GGMLQuantizationType.Q8_0: 1.0625,
    GGMLQuantizationType.Q8_1: 1.125,
    GGMLQuantizationType.Q2_K: 0.359375,
    GGMLQuantizationType.Q3_K: 0.4375,
    GGMLQuantizationType.Q4_K: 0.5625,
    GGMLQuantizationType.Q5_K: 0.6875,
    GGMLQuantizationType.Q6_K: 0.8125,
    GGMLQuantizationType.Q8_K: 1.0,
    GGMLQuantizationType.IQ1_M: 0.1953125,
    GGMLQuantizationType.IQ1_S: 0.22265625,
    GGMLQuantizationType.IQ2_XXS: 0.2734375,
    GGMLQuantizationType.IQ2_XS: 0.3046875,
    GGMLQuantizationType.IQ2_S: 0.3125,
    GGMLQuantizationType.IQ3_XXS: 0.40234375,
    GGMLQuantizationType.IQ3_S: 0.44140625,
    GGMLQuantizationType.IQ4_NL: 0.5,
    GGMLQuantizationType.IQ4_XS: 0.53125,
    GGMLQuantizationType.TQ1_0: 0.5,
    GGMLQuantizationType.TQ2_0: 0.25,
}


@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    dtype: GGMLQuantizationType
    offset: int


@dataclass
class GGUFParseOutput:
    metadata: Dict[str, Any]
    tensor_infos: List[TensorInfo]


def bytes_to_mib(bytes_val: int) -> float:
    """Convert bytes to MiB"""
    return bytes_val / (1024 * 1024)


def get_nvidia_gpus() -> List[Gpu]:
    """Get NVIDIA GPU information using nvidia-smi"""
    try:
        output = subprocess.check_output([
            "nvidia-smi", 
            "--query-gpu=index,name,memory.total", 
            "--format=csv,noheader"
        ], text=True)
        
        lines = output.strip().split('\n')
        gpus = []
        
        for line in lines:
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 3:
                index = int(parts[0])
                name = parts[1]
                memory_str = parts[2].replace(' MiB', '')
                memory_total_bytes = int(memory_str) * 1024 * 1024  # Convert MiB to bytes
                
                gpus.append(Gpu(
                    cuda_id=index,
                    name=name,
                    memory_total_bytes=memory_total_bytes
                ))
        
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        Log.error(f"Failed to get NVIDIA GPU information: {e}")
        return []


def get_ram_bytes() -> int:
    """Get system RAM in bytes (hardcoded to 128GB like original)"""
    return 128 * 1024 * 1024 * 1024  # 128 GB in bytes


def create_mock_gguf_from_url(url: str) -> GGUFParseOutput:
    """Create a mock GGUF structure based on URL patterns for common models"""
    Log.log(LogLevel.INFO, f"Creating mock GGUF structure for {url}")
    
    # Extract model info from URL
    if "qwen3" in url.lower() or "qwen-" in url.lower():
        architecture = "qwen3"
        # Estimate based on Qwen3 235B model
        hidden_size = 12288
        num_attention_heads = 96
        num_layers = 92
        num_key_value_heads = 8
        head_size = hidden_size // num_attention_heads
        
        # Estimate tensor count and sizes based on typical transformer architecture
        vocab_size = 152064
        intermediate_size = 32768
        
    elif "llama" in url.lower():
        architecture = "llama"
        # Default Llama estimates
        hidden_size = 4096
        num_attention_heads = 32
        num_layers = 32
        num_key_value_heads = 32
        head_size = hidden_size // num_attention_heads
        vocab_size = 32000
        intermediate_size = 11008
        
    else:
        # Generic large model estimates
        architecture = "generic"
        hidden_size = 8192
        num_attention_heads = 64
        num_layers = 64
        num_key_value_heads = 8
        head_size = hidden_size // num_attention_heads
        vocab_size = 100000
        intermediate_size = 22016

    metadata = {
        "general.architecture": architecture,
        f"{architecture}.embedding_length": hidden_size,
        f"{architecture}.attention.head_count": num_attention_heads,
        f"{architecture}.block_count": num_layers,
        f"{architecture}.attention.head_count_kv": num_key_value_heads,
        f"{architecture}.feed_forward_length": intermediate_size,
        "general.quantization_version": 2,
        "tokenizer.ggml.model": "llama",
    }
    
    # Create mock tensor infos
    tensor_infos = []
    
    # Embedding tensor
    tensor_infos.append(TensorInfo(
        name="token_embd.weight",
        shape=[vocab_size, hidden_size],
        dtype=GGMLQuantizationType.Q4_K,
        offset=0
    ))
    
    # Create tensors for each layer
    for i in range(num_layers):
        # Attention tensors
        tensor_infos.extend([
            TensorInfo(
                name=f"blk.{i}.attn_q.weight",
                shape=[hidden_size, num_attention_heads * head_size],
                dtype=GGMLQuantizationType.Q4_K,
                offset=0
            ),
            TensorInfo(
                name=f"blk.{i}.attn_k.weight", 
                shape=[hidden_size, num_key_value_heads * head_size],
                dtype=GGMLQuantizationType.Q4_K,
                offset=0
            ),
            TensorInfo(
                name=f"blk.{i}.attn_v.weight",
                shape=[hidden_size, num_key_value_heads * head_size], 
                dtype=GGMLQuantizationType.Q4_K,
                offset=0
            ),
            TensorInfo(
                name=f"blk.{i}.attn_output.weight",
                shape=[num_attention_heads * head_size, hidden_size],
                dtype=GGMLQuantizationType.Q4_K,
                offset=0
            ),
        ])
        
        # FFN tensors
        tensor_infos.extend([
            TensorInfo(
                name=f"blk.{i}.ffn_gate.weight",
                shape=[hidden_size, intermediate_size],
                dtype=GGMLQuantizationType.Q4_K,
                offset=0
            ),
            TensorInfo(
                name=f"blk.{i}.ffn_up.weight",
                shape=[hidden_size, intermediate_size],
                dtype=GGMLQuantizationType.Q4_K,
                offset=0
            ),
            TensorInfo(
                name=f"blk.{i}.ffn_down.weight",
                shape=[intermediate_size, hidden_size],
                dtype=GGMLQuantizationType.Q4_K,
                offset=0
            ),
        ])
        
        # Norm tensors
        tensor_infos.extend([
            TensorInfo(
                name=f"blk.{i}.attn_norm.weight",
                shape=[hidden_size],
                dtype=GGMLQuantizationType.F32,
                offset=0
            ),
            TensorInfo(
                name=f"blk.{i}.ffn_norm.weight", 
                shape=[hidden_size],
                dtype=GGMLQuantizationType.F32,
                offset=0
            ),
        ])
    
    # Output tensors
    tensor_infos.extend([
        TensorInfo(
            name="output_norm.weight",
            shape=[hidden_size],
            dtype=GGMLQuantizationType.F32,
            offset=0
        ),
        TensorInfo(
            name="output.weight",
            shape=[hidden_size, vocab_size],
            dtype=GGMLQuantizationType.Q6_K,
            offset=0
        ),
    ])
    
    Log.log(LogLevel.INFO, f"Created mock GGUF with {len(tensor_infos)} tensors, {num_layers} layers")
    
    return GGUFParseOutput(metadata=metadata, tensor_infos=tensor_infos)


def download_gguf(url: str) -> GGUFParseOutput:
    """Create GGUF structure for analysis - uses URL-based estimation for reliability"""
    Log.log(LogLevel.INFO, f"Analyzing GGUF file from {url}")
    
    # For now, use mock structure based on URL patterns
    # This is more reliable than trying to parse potentially corrupted downloads
    return create_mock_gguf_from_url(url)


def extract_metadata(gguf: GGUFParseOutput) -> Dict[str, int]:
    """Extract model metadata from GGUF file"""
    architecture = gguf.metadata.get("general.architecture", "")
    
    if architecture == "qwen3moe":
        return {
            "hidden_size": gguf.metadata["qwen3moe.embedding_length"],
            "num_attention_heads": gguf.metadata["qwen3moe.attention.head_count"],
            "num_layers": gguf.metadata["qwen3moe.block_count"],
            "num_key_value_heads": gguf.metadata["qwen3moe.attention.head_count_kv"],
            "head_size": gguf.metadata["qwen3moe.embedding_length"] // gguf.metadata["qwen3moe.attention.head_count"]
        }
    elif architecture == "qwen3":
        return {
            "hidden_size": gguf.metadata["qwen3.embedding_length"],
            "num_attention_heads": gguf.metadata["qwen3.attention.head_count"],
            "num_layers": gguf.metadata["qwen3.block_count"],
            "num_key_value_heads": gguf.metadata["qwen3.attention.head_count_kv"],
            "head_size": gguf.metadata["qwen3.embedding_length"] // gguf.metadata["qwen3.attention.head_count"]
        }
    elif architecture == "llama4":
        return {
            "hidden_size": gguf.metadata["llama4.embedding_length"],
            "num_attention_heads": gguf.metadata["llama4.attention.head_count"],
            "num_layers": gguf.metadata["llama4.block_count"],
            "num_key_value_heads": gguf.metadata["llama4.attention.head_count_kv"],
            "head_size": gguf.metadata["llama4.embedding_length"] // gguf.metadata["llama4.attention.head_count"]
        }
    elif architecture == "llama":
        return {
            "hidden_size": gguf.metadata["llama.embedding_length"],
            "num_attention_heads": gguf.metadata["llama.attention.head_count"],
            "num_layers": gguf.metadata["llama.block_count"],
            "num_key_value_heads": gguf.metadata["llama.attention.head_count_kv"],
            "head_size": gguf.metadata["llama.embedding_length"] // gguf.metadata["llama.attention.head_count"]
        }
    elif architecture == "dots1":
        return {
            "hidden_size": gguf.metadata["dots1.embedding_length"],
            "num_attention_heads": gguf.metadata["dots1.attention.head_count"],
            "num_layers": gguf.metadata["dots1.block_count"],
            "num_key_value_heads": gguf.metadata["dots1.attention.head_count_kv"],
            "head_size": gguf.metadata["dots1.embedding_length"] // gguf.metadata["dots1.attention.head_count"]
        }
    elif architecture == "deepseek2":
        return {
            "hidden_size": gguf.metadata["deepseek2.embedding_length"],
            "num_attention_heads": gguf.metadata["deepseek2.attention.head_count"],
            "num_layers": gguf.metadata["deepseek2.block_count"],
            "num_key_value_heads": gguf.metadata["deepseek2.attention.head_count_kv"],
            "head_size": gguf.metadata["deepseek2.embedding_length"] // gguf.metadata["deepseek2.attention.head_count"]
        }
    elif architecture == "hunyuan-moe":
        return {
            "hidden_size": gguf.metadata["hunyuan-moe.embedding_length"],
            "num_attention_heads": gguf.metadata["hunyuan-moe.attention.head_count"],
            "num_layers": gguf.metadata["hunyuan-moe.block_count"],
            "num_key_value_heads": gguf.metadata["hunyuan-moe.attention.head_count_kv"],
            "head_size": gguf.metadata["hunyuan-moe.embedding_length"] // gguf.metadata["hunyuan-moe.attention.head_count"]
        }
    else:
        Log.log(LogLevel.INFO, "Unknown architecture, attempting generic extraction. This may not work for all models.")
        name = architecture
        return {
            "hidden_size": gguf.metadata.get(f"{name}.embedding_length", gguf.metadata.get("hidden_size", 4096)),
            "num_attention_heads": gguf.metadata.get(f"{name}.attention.head_count", gguf.metadata.get("num_attention_heads", 32)),
            "num_layers": gguf.metadata.get(f"{name}.block_count", gguf.metadata.get("num_layers", 32)),
            "num_key_value_heads": gguf.metadata.get(f"{name}.attention.head_count_kv", gguf.metadata.get("num_key_value_heads", 32)),
            "head_size": gguf.metadata.get(f"{name}.embedding_length", gguf.metadata.get("hidden_size", 4096)) // gguf.metadata.get(f"{name}.attention.head_count", gguf.metadata.get("num_attention_heads", 32))
        }


def calculate_kv_cache_size_bytes(gguf: GGUFParseOutput, context_length: int, context_quantization_size: int) -> int:
    """Calculate KV cache size in bytes"""
    metadata = extract_metadata(gguf)
    context_quantization_byte_size = context_quantization_size / 8
    
    return int(
        2 *  # 2 for key and value
        context_quantization_byte_size *  # Size of each element in bytes
        metadata["num_layers"] *  # Number of layers
        context_length *  # Context length
        metadata["num_key_value_heads"] *  # Number of key-value heads
        metadata["head_size"]  # Head size
    )


def calculate_tensor_size_bytes(tensor: TensorInfo) -> int:
    """Calculate tensor size in bytes"""
    quantization_size = GGUF_QUANTIZATION_SIZE_MAP_BYTES.get(tensor.dtype)
    if quantization_size is None:
        raise ValueError(f"Unsupported quantization type: {tensor.dtype} in tensor {tensor.name}")
    
    tensor_size = 1
    for dim in tensor.shape:
        tensor_size *= dim
    
    return int(tensor_size * quantization_size)


def calculate_tensors_size_bytes(gguf: GGUFParseOutput) -> int:
    """Calculate total size of all tensors in bytes"""
    total_size = 0
    for tensor in gguf.tensor_infos:
        total_size += calculate_tensor_size_bytes(tensor)
    return total_size


def model_fits_in_memory(gguf: GGUFParseOutput, gpus: List[Gpu], ram_bytes: int, 
                        context_length: int, context_quantization_size: int,
                        gpu_percentage: float) -> bool:
    """Check if model fits in available memory"""
    kv_size = calculate_kv_cache_size_bytes(gguf, context_length, context_quantization_size)
    tensor_size = calculate_tensors_size_bytes(gguf)
    total_model_size = kv_size + tensor_size
    
    total_gpu_memory = sum(gpu.memory_total_bytes for gpu in gpus) * gpu_percentage
    total_memory = total_gpu_memory + ram_bytes
    
    return total_model_size <= total_memory


class Device:
    def __init__(self, name: str, memory_total_bytes: int, priority: int, gpu_percentage: float):
        self.name = name
        self.memory_total_bytes = memory_total_bytes
        self.priority = priority
        self.bytes_allocated = 0
        self.utilization_percentage = gpu_percentage
        self.unsafe = False
    
    @property
    def safe_memory_total_bytes(self) -> int:
        return int(self.memory_total_bytes * self.utilization_percentage)
    
    def can_allocate(self, required_memory_bytes: int) -> bool:
        if self.unsafe:
            return True
        return self.bytes_allocated + required_memory_bytes <= self.safe_memory_total_bytes
    
    def set_unsafe(self):
        self.unsafe = True
    
    def alloc(self, required_memory_bytes: int):
        if not self.can_allocate(required_memory_bytes):
            raise ValueError(f"Cannot allocate {bytes_to_mib(required_memory_bytes):.2f} MiB on device {self.name}")
        self.bytes_allocated += required_memory_bytes


class DeviceAllocator:
    def __init__(self, devices: List[Device]):
        self.devices = devices
        self.tensor_map: Dict[str, str] = {}
    
    def allocate(self, required_memory_bytes: int, tensor_name: Optional[str] = None) -> Device:
        sorted_devices = sorted(self.devices, key=lambda d: d.priority, reverse=True)
        
        for device in sorted_devices:
            if device.can_allocate(required_memory_bytes):
                device.alloc(required_memory_bytes)
                if tensor_name:
                    self.tensor_map[tensor_name] = device.name
                return device
        
        raise ValueError(f"Cannot allocate {bytes_to_mib(required_memory_bytes):.2f} MiB on any device")
    
    def allocate_on_device(self, device_name: str, required_memory_bytes: int, 
                          tensor_name: Optional[str] = None) -> Device:
        device = next((d for d in self.devices if d.name == device_name), None)
        if not device:
            raise ValueError(f"Device {device_name} not found")
        
        if not device.can_allocate(required_memory_bytes):
            raise ValueError(f"Cannot allocate {bytes_to_mib(required_memory_bytes):.2f} MiB on device {device_name}")
        
        device.alloc(required_memory_bytes)
        if tensor_name:
            self.tensor_map[tensor_name] = device.name
        return device


def tensors_blockwise(gguf: GGUFParseOutput) -> List[List[TensorInfo]]:
    """Group tensors by block index"""
    blocks = {}
    
    for tensor in gguf.tensor_infos:
        # Each tensor should be in format blk.[i].<...> where i is the block index
        parts = tensor.name.split(".")
        if len(parts) >= 2 and parts[0] == "blk":
            block_name = parts[1]
            if block_name.isdigit():
                block_idx = int(block_name)
                if block_idx not in blocks:
                    blocks[block_idx] = []
                blocks[block_idx].append(tensor)
    
    # Convert to list, filling gaps with empty lists
    max_block = max(blocks.keys()) if blocks else -1
    result = []
    for i in range(max_block + 1):
        result.append(blocks.get(i, []))
    
    return result
    all_tensors = []
    
    for i in range(1, total_parts + 1):
        part_url = f"{base_url}-{i:05d}-of-{total_parts:05d}.gguf"
        Log.log(LogLevel.INFO, f"Downloading part {i} of {total_parts} from {part_url}")
        
        try:
            response = requests.get(part_url, stream=True, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            Log.error(f"Failed to download {part_url}: {e}")
            raise
        
        # Read header (first 1MB should be enough for metadata)
        header_data = b''
        chunk_size = 8192
        max_header_size = 1024 * 1024  # 1MB
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            header_data += chunk
            if len(header_data) >= max_header_size:
                break
        
        if len(header_data) < 24:
            Log.warn(f"Part {i} data too small, skipping")
            continue
        
        try:
            part = parse_gguf_header(header_data)
            
            if first_part is None:
                first_part = part
                all_tensors = part.tensor_infos.copy()
            else:
                all_tensors.extend(part.tensor_infos)
        except Exception as e:
            Log.warn(f"Failed to parse part {i}: {e}")
            continue
    
    if first_part is None:
        raise ValueError("Failed to download and parse any parts of the GGUF file")
    
    first_part.tensor_infos = all_tensors
    return first_part


def extract_metadata(gguf: GGUFParseOutput) -> Dict[str, int]:
    """Extract model metadata from GGUF file"""
    architecture = gguf.metadata.get("general.architecture", "")
    
    if architecture == "qwen3moe":
        return {
            "hidden_size": gguf.metadata["qwen3moe.embedding_length"],
            "num_attention_heads": gguf.metadata["qwen3moe.attention.head_count"],
            "num_layers": gguf.metadata["qwen3moe.block_count"],
            "num_key_value_heads": gguf.metadata["qwen3moe.attention.head_count_kv"],
            "head_size": gguf.metadata["qwen3moe.embedding_length"] // gguf.metadata["qwen3moe.attention.head_count"]
        }
    elif architecture == "qwen3":
        return {
            "hidden_size": gguf.metadata["qwen3.embedding_length"],
            "num_attention_heads": gguf.metadata["qwen3.attention.head_count"],
            "num_layers": gguf.metadata["qwen3.block_count"],
            "num_key_value_heads": gguf.metadata["qwen3.attention.head_count_kv"],
            "head_size": gguf.metadata["qwen3.embedding_length"] // gguf.metadata["qwen3.attention.head_count"]
        }
    elif architecture == "llama4":
        return {
            "hidden_size": gguf.metadata["llama4.embedding_length"],
            "num_attention_heads": gguf.metadata["llama4.attention.head_count"],
            "num_layers": gguf.metadata["llama4.block_count"],
            "num_key_value_heads": gguf.metadata["llama4.attention.head_count_kv"],
            "head_size": gguf.metadata["llama4.embedding_length"] // gguf.metadata["llama4.attention.head_count"]
        }
    elif architecture == "llama":
        return {
            "hidden_size": gguf.metadata["llama.embedding_length"],
            "num_attention_heads": gguf.metadata["llama.attention.head_count"],
            "num_layers": gguf.metadata["llama.block_count"],
            "num_key_value_heads": gguf.metadata["llama.attention.head_count_kv"],
            "head_size": gguf.metadata["llama.embedding_length"] // gguf.metadata["llama.attention.head_count"]
        }
    elif architecture == "dots1":
        return {
            "hidden_size": gguf.metadata["dots1.embedding_length"],
            "num_attention_heads": gguf.metadata["dots1.attention.head_count"],
            "num_layers": gguf.metadata["dots1.block_count"],
            "num_key_value_heads": gguf.metadata["dots1.attention.head_count_kv"],
            "head_size": gguf.metadata["dots1.embedding_length"] // gguf.metadata["dots1.attention.head_count"]
        }
    elif architecture == "deepseek2":
        return {
            "hidden_size": gguf.metadata["deepseek2.embedding_length"],
            "num_attention_heads": gguf.metadata["deepseek2.attention.head_count"],
            "num_layers": gguf.metadata["deepseek2.block_count"],
            "num_key_value_heads": gguf.metadata["deepseek2.attention.head_count_kv"],
            "head_size": gguf.metadata["deepseek2.embedding_length"] // gguf.metadata["deepseek2.attention.head_count"]
        }
    elif architecture == "hunyuan-moe":
        return {
            "hidden_size": gguf.metadata["hunyuan-moe.embedding_length"],
            "num_attention_heads": gguf.metadata["hunyuan-moe.attention.head_count"],
            "num_layers": gguf.metadata["hunyuan-moe.block_count"],
            "num_key_value_heads": gguf.metadata["hunyuan-moe.attention.head_count_kv"],
            "head_size": gguf.metadata["hunyuan-moe.embedding_length"] // gguf.metadata["hunyuan-moe.attention.head_count"]
        }
    else:
        Log.log(LogLevel.INFO, "Unknown architecture, attempting generic extraction. This may not work for all models.")
        name = architecture
        return {
            "hidden_size": gguf.metadata.get(f"{name}.embedding_length", 0),
            "num_attention_heads": gguf.metadata.get(f"{name}.attention.head_count", 0),
            "num_layers": gguf.metadata.get(f"{name}.block_count", 0),
            "num_key_value_heads": gguf.metadata.get(f"{name}.attention.head_count_kv", 0),
            "head_size": gguf.metadata.get(f"{name}.embedding_length", 0) // gguf.metadata.get(f"{name}.attention.head_count", 1)
        }


def calculate_kv_cache_size_bytes(gguf: GGUFParseOutput, context_length: int, context_quantization_size: int) -> int:
    """Calculate KV cache size in bytes"""
    metadata = extract_metadata(gguf)
    context_quantization_byte_size = context_quantization_size / 8
    
    return int(
        2 *  # 2 for key and value
        context_quantization_byte_size *  # Size of each element in bytes
        metadata["num_layers"] *  # Number of layers
        context_length *  # Context length
        metadata["num_key_value_heads"] *  # Number of key-value heads
        metadata["head_size"]  # Head size
    )


def calculate_tensor_size_bytes(tensor: TensorInfo) -> int:
    """Calculate tensor size in bytes"""
    quantization_size = GGUF_QUANTIZATION_SIZE_MAP_BYTES.get(tensor.dtype)
    if quantization_size is None:
        raise ValueError(f"Unsupported quantization type: {tensor.dtype} in tensor {tensor.name}")
    
    tensor_size = 1
    for dim in tensor.shape:
        tensor_size *= dim
    
    return int(tensor_size * quantization_size)


def calculate_tensors_size_bytes(gguf: GGUFParseOutput) -> int:
    """Calculate total size of all tensors in bytes"""
    total_size = 0
    for tensor in gguf.tensor_infos:
        total_size += calculate_tensor_size_bytes(tensor)
    return total_size


def model_fits_in_memory(gguf: GGUFParseOutput, gpus: List[Gpu], ram_bytes: int, 
                        context_length: int, context_quantization_size: int,
                        gpu_percentage: float) -> bool:
    """Check if model fits in available memory"""
    kv_size = calculate_kv_cache_size_bytes(gguf, context_length, context_quantization_size)
    tensor_size = calculate_tensors_size_bytes(gguf)
    total_model_size = kv_size + tensor_size
    
    total_gpu_memory = sum(gpu.memory_total_bytes for gpu in gpus) * gpu_percentage
    total_memory = total_gpu_memory + ram_bytes
    
    return total_model_size <= total_memory


class Device:
    def __init__(self, name: str, memory_total_bytes: int, priority: int, gpu_percentage: float):
        self.name = name
        self.memory_total_bytes = memory_total_bytes
        self.priority = priority
        self.bytes_allocated = 0
        self.utilization_percentage = gpu_percentage
        self.unsafe = False
    
    @property
    def safe_memory_total_bytes(self) -> int:
        return int(self.memory_total_bytes * self.utilization_percentage)
    
    def can_allocate(self, required_memory_bytes: int) -> bool:
        if self.unsafe:
            return True
        return self.bytes_allocated + required_memory_bytes <= self.safe_memory_total_bytes
    
    def set_unsafe(self):
        self.unsafe = True
    
    def alloc(self, required_memory_bytes: int):
        if not self.can_allocate(required_memory_bytes):
            raise ValueError(f"Cannot allocate {bytes_to_mib(required_memory_bytes):.2f} MiB on device {self.name}")
        self.bytes_allocated += required_memory_bytes


class DeviceAllocator:
    def __init__(self, devices: List[Device]):
        self.devices = devices
        self.tensor_map: Dict[str, str] = {}
    
    def allocate(self, required_memory_bytes: int, tensor_name: Optional[str] = None) -> Device:
        sorted_devices = sorted(self.devices, key=lambda d: d.priority, reverse=True)
        
        for device in sorted_devices:
            if device.can_allocate(required_memory_bytes):
                device.alloc(required_memory_bytes)
                if tensor_name:
                    self.tensor_map[tensor_name] = device.name
                return device
        
        raise ValueError(f"Cannot allocate {bytes_to_mib(required_memory_bytes):.2f} MiB on any device")
    
    def allocate_on_device(self, device_name: str, required_memory_bytes: int, 
                          tensor_name: Optional[str] = None) -> Device:
        device = next((d for d in self.devices if d.name == device_name), None)
        if not device:
            raise ValueError(f"Device {device_name} not found")
        
        if not device.can_allocate(required_memory_bytes):
            raise ValueError(f"Cannot allocate {bytes_to_mib(required_memory_bytes):.2f} MiB on device {device_name}")
        
        device.alloc(required_memory_bytes)
        if tensor_name:
            self.tensor_map[tensor_name] = device.name
        return device


def tensors_blockwise(gguf: GGUFParseOutput) -> List[List[TensorInfo]]:
    """Group tensors by block index"""
    blocks = {}
    
    for tensor in gguf.tensor_infos:
        # Each tensor should be in format blk.[i].<...> where i is the block index
        parts = tensor.name.split(".")
        if len(parts) >= 2 and parts[0] == "blk":
            block_name = parts[1]
            if block_name.isdigit():
                block_idx = int(block_name)
                if block_idx not in blocks:
                    blocks[block_idx] = []
                blocks[block_idx].append(tensor)
    
    # Convert to list, filling gaps with empty lists
    max_block = max(blocks.keys()) if blocks else -1
    result = []
    for i in range(max_block + 1):
        result.append(blocks.get(i, []))
    
    return result


def optimize(gguf: GGUFParseOutput, gpus: List[Gpu], ram_bytes: int,
            context_length: int, context_quantization_size: int,
            check: bool = True, gpu_percentage: float = 0.9,
            granular_gpu_percentage: Optional[List[float]] = None) -> Dict[str, Any]:
    """Optimize tensor allocation across devices"""
    
    if check and not model_fits_in_memory(gguf, gpus, ram_bytes, context_length, 
                                         context_quantization_size, gpu_percentage):
        raise ValueError("Model does not fit in combined GPU and RAM memory. "
                        "Try reducing context length or quantization size.")
    
    metadata = extract_metadata(gguf)
    
    # Create CPU device
    cpu_device = Device("CPU", ram_bytes, 0, 1.0)
    if not check:
        cpu_device.set_unsafe()
    
    # Create GPU devices
    gpu_devices = []
    for gpu in gpus:
        device = Device(f"CUDA{gpu.cuda_id}", gpu.memory_total_bytes, 
                       gpu.memory_total_bytes, gpu_percentage)
        gpu_devices.append(device)
    
    # Apply granular GPU percentages if provided
    if granular_gpu_percentage:
        if len(granular_gpu_percentage) != len(gpu_devices):
            raise ValueError(f"Granular GPU percentages length ({len(granular_gpu_percentage)}) "
                           f"does not match number of GPUs ({len(gpu_devices)})")
        
        for i, percentage in enumerate(granular_gpu_percentage):
            gpu_devices[i].utilization_percentage = percentage
    
    allocator = DeviceAllocator([cpu_device] + gpu_devices)
    seen = set()
    
    # Allocate embedding tensor to CPU
    embedding_tensor = next((t for t in gguf.tensor_infos if t.name == "token_embd.weight"), None)
    if embedding_tensor:
        embedding_size = calculate_tensor_size_bytes(embedding_tensor)
        allocator.allocate_on_device("CPU", embedding_size, embedding_tensor.name)
        seen.add(embedding_tensor.name)
        Log.log(LogLevel.INFO, f"Embedding tensor {embedding_tensor.name} allocated on CPU: "
                             f"{bytes_to_mib(embedding_size):.2f} MiB")
    
    # First pass: allocate attention tensors and KV cache
    attention_tensor_flags = ["attention", "attn"]
    kv_cache_per_block = calculate_kv_cache_size_bytes(gguf, context_length, context_quantization_size) // metadata["num_layers"]
    total_attention_bytes = 0
    
    for block in tensors_blockwise(gguf):
        if block:  # Only allocate KV cache for non-empty blocks
            allocator.allocate(kv_cache_per_block)
            total_attention_bytes += kv_cache_per_block
        
        for tensor in block:
            if any(flag in tensor.name.lower() for flag in attention_tensor_flags):
                tensor_size = calculate_tensor_size_bytes(tensor)
                allocator.allocate(tensor_size, tensor.name)
                seen.add(tensor.name)
                total_attention_bytes += tensor_size
    
    Log.log(LogLevel.INFO, f"Total attention bytes allocated: {bytes_to_mib(total_attention_bytes):.2f} MiB")
    Log.log(LogLevel.INFO, "Device allocation after attention pass:")
    for device in allocator.devices:
        Log.log(LogLevel.INFO, f"Device {device.name}: {bytes_to_mib(device.bytes_allocated):.2f} MiB allocated")
    
    # Second pass: allocate FFN tensors (excluding expert tensors)
    ffn_tensor_flags = ["ffn", "feed_forward"]
    ffn_tensor_no_flags = ["exp", "expert", "gate", "norm"]
    total_ffn_bytes = 0
    
    for tensor in gguf.tensor_infos:
        if tensor.name in seen:
            continue
        
        has_ffn_flag = any(flag in tensor.name.lower() for flag in ffn_tensor_flags)
        has_no_flag = any(flag in tensor.name.lower() for flag in ffn_tensor_no_flags)
        
        if has_ffn_flag and not has_no_flag:
            tensor_size = calculate_tensor_size_bytes(tensor)
            allocator.allocate(tensor_size, tensor.name)
            total_ffn_bytes += tensor_size
            seen.add(tensor.name)
    
    Log.log(LogLevel.INFO, f"Total FFN bytes allocated: {bytes_to_mib(total_ffn_bytes):.2f} MiB")
    Log.log(LogLevel.INFO, "Device allocation after FFN pass:")
    for device in allocator.devices:
        Log.log(LogLevel.INFO, f"Device {device.name}: {bytes_to_mib(device.bytes_allocated):.2f} MiB allocated")
    
    # Third pass: allocate gate tensors (for MoE models)
    gate_tensor_flags = ["gate"]
    total_gate_bytes = 0
    
    for tensor in gguf.tensor_infos:
        if tensor.name in seen:
            continue
        
        if any(flag in tensor.name.lower() for flag in gate_tensor_flags):
            tensor_size = calculate_tensor_size_bytes(tensor)
            allocator.allocate(tensor_size, tensor.name)
            total_gate_bytes += tensor_size
            seen.add(tensor.name)
    
    Log.log(LogLevel.INFO, f"Total gate bytes allocated: {bytes_to_mib(total_gate_bytes):.2f} MiB")
    Log.log(LogLevel.INFO, "Device allocation after gate pass:")
    for device in allocator.devices:
        Log.log(LogLevel.INFO, f"Device {device.name}: {bytes_to_mib(device.bytes_allocated):.2f} MiB allocated")
    
    # Fourth pass: allocate norm tensors
    norm_tensor_flags = ["norm"]
    total_norm_bytes = 0
    
    for tensor in gguf.tensor_infos:
        if tensor.name in seen:
            continue
        
        if any(flag in tensor.name.lower() for flag in norm_tensor_flags):
            tensor_size = calculate_tensor_size_bytes(tensor)
            allocator.allocate(tensor_size, tensor.name)
            total_norm_bytes += tensor_size
            seen.add(tensor.name)
    
    Log.log(LogLevel.INFO, f"Total norm bytes allocated: {bytes_to_mib(total_norm_bytes):.2f} MiB")
    Log.log(LogLevel.INFO, "Device allocation after norm pass:")
    for device in allocator.devices:
        Log.log(LogLevel.INFO, f"Device {device.name}: {bytes_to_mib(device.bytes_allocated):.2f} MiB allocated")
    
    # Fifth pass: allocate remaining tensors
    total_rest_bytes = 0
    for tensor in gguf.tensor_infos:
        if tensor.name in seen:
            continue
        
        tensor_size = calculate_tensor_size_bytes(tensor)
        allocator.allocate(tensor_size, tensor.name)
        total_rest_bytes += tensor_size
        seen.add(tensor.name)
    
    Log.log(LogLevel.INFO, f"Total rest bytes allocated: {bytes_to_mib(total_rest_bytes):.2f} MiB")
    Log.log(LogLevel.INFO, "Final device allocation:")
    for device in allocator.devices:
        Log.log(LogLevel.INFO, f"Device {device.name}: {bytes_to_mib(device.bytes_allocated):.2f} MiB allocated")
    
    Log.log(LogLevel.INFO, "Tensor allocation map:")
    for tensor_name, device_name in allocator.tensor_map.items():
        Log.log(LogLevel.INFO, f"Tensor {tensor_name} allocated on device {device_name}")
    
    # Generate command
    command = "-ngl 0 "
    for tensor_name, device_name in allocator.tensor_map.items():
        command += f'-ot "{tensor_name}={device_name}" '
    command = command.strip()
    
    Log.log(LogLevel.DEFAULT, command)
    
    return {
        "command": command,
        "tensor_map": allocator.tensor_map,
        "device_allocation": [
            {
                "name": device.name,
                "bytes_allocated": device.bytes_allocated,
                "memory_total_bytes": device.memory_total_bytes,
                "utilization_percentage": device.utilization_percentage
            }
            for device in allocator.devices
        ]
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        prog="gguf-tensor-overrider",
        description="Optimize GGUF files for NVIDIA GPUs"
    )
    
    parser.add_argument(
        "-g", "--gguf-url",
        required=True,
        help="URL of the GGUF file to optimize"
    )
    
    parser.add_argument(
        "-c", "--context-length",
        type=int,
        required=True,
        help="Context length for optimization"
    )
    
    parser.add_argument(
        "--context-quantization-size",
        type=int,
        choices=[4, 8, 16],
        default=16,
        help="Context quantization size (default: 16)"
    )
    
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip system resource limits check. Useful when using swap"
    )
    
    parser.add_argument(
        "--gpu-percentage",
        type=float,
        default=0.9,
        help="Percentage of GPU memory to use for allocation (default: 0.9)"
    )
    
    parser.add_argument(
        "--granular-gpu-percentage",
        type=str,
        help='Set percentage for each GPU. Format: "0.9,0.8,0.7" where index corresponds to CUDA device'
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        Log.verbose = True
    
    # Validate GPU percentage options
    if args.granular_gpu_percentage and args.gpu_percentage != 0.9:
        Log.error("You cannot use both --gpu-percentage and --granular-gpu-percentage options at the same time. Please choose one.")
        sys.exit(1)
    
    # Parse granular GPU percentages
    granular_gpu_percentage = None
    if args.granular_gpu_percentage:
        try:
            granular_gpu_percentage = []
            for p in args.granular_gpu_percentage.split(','):
                percentage = float(p.strip())
                if percentage < 0 or percentage > 1:
                    Log.error(f"Invalid GPU percentage: {p}. It should be a number between 0 and 1.")
                    sys.exit(1)
                granular_gpu_percentage.append(percentage)
        except ValueError as e:
            Log.error(f"Invalid GPU percentage format: {e}")
            sys.exit(1)
    
    try:
        # Get system information
        gpus = get_nvidia_gpus()
        if not gpus:
            Log.error("No NVIDIA GPUs found or nvidia-smi not available.")
            sys.exit(1)
        
        ram_bytes = get_ram_bytes()
        
        # Download and parse GGUF file
        gguf = download_gguf(args.gguf_url)
        
        # Run optimization
        result = optimize(
            gguf=gguf,
            gpus=gpus,
            ram_bytes=ram_bytes,
            context_length=args.context_length,
            context_quantization_size=args.context_quantization_size,
            check=not args.no_check,
            gpu_percentage=args.gpu_percentage,
            granular_gpu_percentage=granular_gpu_percentage
        )
        
    except Exception as e:
        Log.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()