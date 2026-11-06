"""Block-wise INT4 tensor quantization and sub-byte bit packing utilities for edge communication."""

from io import BytesIO
from typing import Any

import torch


def quantize_tensor(
    tensor: torch.Tensor,
    block_size: int = 64,
    stochastic: bool = False,
) -> dict[str, Any]:
    """Quantize a float tensor into packed 4-bit integers with per-block scaling.

    Pairs of 4-bit signed integers in range [-7, 7] are packed into unsigned 8-bit bytes,
    reducing wire footprint by approximately 8x relative to float32.
    """
    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    orig_numel = tensor.numel()

    if orig_numel == 0:
        return {
            "packed": torch.empty(0, dtype=torch.uint8),
            "scales": torch.empty(0, dtype=torch.float32),
            "shape": orig_shape,
            "numel": 0,
            "dtype": orig_dtype,
        }

    flat = tensor.detach().float().reshape(-1)

    pad_len = (block_size - (orig_numel % block_size)) % block_size
    if pad_len > 0:
        padded = torch.cat([flat, torch.zeros(pad_len, dtype=torch.float32, device=flat.device)])
    else:
        padded = flat

    blocks = padded.reshape(-1, block_size)
    max_vals = torch.amax(torch.abs(blocks), dim=1, keepdim=True)
    scales = max_vals / 7.0

    safe_scales = torch.where(scales == 0.0, torch.ones_like(scales), scales)
    scaled_blocks = blocks / safe_scales

    if stochastic:
        noise = torch.rand_like(scaled_blocks) - 0.5
        quantized = torch.clamp(torch.round(scaled_blocks + noise), -7.0, 7.0)
    else:
        quantized = torch.clamp(torch.round(scaled_blocks), -7.0, 7.0)

    unsigned_vals = (quantized + 7).to(torch.uint8).reshape(-1)

    if unsigned_vals.numel() % 2 != 0:
        unsigned_vals = torch.cat([unsigned_vals, torch.zeros(1, dtype=torch.uint8, device=unsigned_vals.device)])

    low = unsigned_vals[0::2] & 0x0F
    high = (unsigned_vals[1::2] & 0x0F) << 4
    packed = low | high

    return {
        "packed": packed,
        "scales": scales.squeeze(1),
        "shape": orig_shape,
        "numel": orig_numel,
        "dtype": orig_dtype,
        "block_size": block_size,
    }


def dequantize_tensor(quantized_data: dict[str, Any]) -> torch.Tensor:
    """Unpack 4-bit nibbles and restore original float tensor values."""
    orig_shape = quantized_data["shape"]
    orig_numel = quantized_data["numel"]
    orig_dtype = quantized_data.get("dtype", torch.float32)

    if orig_numel == 0:
        return torch.empty(orig_shape, dtype=orig_dtype)

    packed = quantized_data["packed"]
    scales = quantized_data["scales"]
    block_size = quantized_data.get("block_size", 64)

    low = (packed & 0x0F).to(torch.int8) - 7
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 7

    unpacked = torch.empty(packed.numel() * 2, dtype=torch.float32, device=packed.device)
    unpacked[0::2] = low.float()
    unpacked[1::2] = high.float()

    total_padded_numel = scales.numel() * block_size
    unpacked = unpacked[:total_padded_numel]

    blocks = unpacked.reshape(-1, block_size)
    scales_col = scales.unsqueeze(1).to(blocks.device)
    rescaled_blocks = blocks * scales_col

    flat = rescaled_blocks.reshape(-1)[:orig_numel]
    return flat.reshape(orig_shape).to(orig_dtype)


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    block_size: int = 64,
    min_numel: int = 128,
) -> dict[str, Any]:
    """Quantize model state dictionary weights while preserving 1D/scalar tensors."""
    compressed: dict[str, Any] = {}
    for key, tensor in state_dict.items():
        if tensor.dim() > 1 and tensor.numel() >= min_numel:
            compressed[key] = {
                "__quantized__": True,
                "data": quantize_tensor(tensor, block_size=block_size),
            }
        else:
            compressed[key] = {
                "__quantized__": False,
                "data": tensor.detach().cpu(),
            }
    return compressed


def dequantize_state_dict(compressed_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Reconstruct model state dictionary from compressed dictionary."""
    restored: dict[str, torch.Tensor] = {}
    for key, entry in compressed_dict.items():
        if isinstance(entry, dict) and entry.get("__quantized__", False):
            restored[key] = dequantize_tensor(entry["data"])
        elif isinstance(entry, dict) and "data" in entry:
            restored[key] = entry["data"]
        else:
            restored[key] = entry
    return restored


def compress_model_payload(state_dict: dict[str, torch.Tensor], block_size: int = 64) -> bytes:
    """Serialize compressed state dictionary into a byte payload."""
    compressed = quantize_state_dict(state_dict, block_size=block_size)
    buffer = BytesIO()
    torch.save(compressed, buffer)
    return buffer.getvalue()


def decompress_model_payload(payload_bytes: bytes) -> dict[str, torch.Tensor]:
    """Deserialize byte payload and restore model state dictionary."""
    buffer = BytesIO(payload_bytes)
    compressed = torch.load(buffer, map_location="cpu", weights_only=False)
    return dequantize_state_dict(compressed)
