"""Unit tests for FedERA block-wise INT4 tensor quantization and payload compression."""

import unittest
from io import BytesIO

import torch
from torch import nn

from federa.compression import (
    compress_model_payload,
    decompress_model_payload,
    dequantize_state_dict,
    dequantize_tensor,
    quantize_state_dict,
    quantize_tensor,
)


class TestFedERACompression(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_tensor_roundtrip_fidelity(self):
        tensor = torch.randn(64, 128)
        quantized = quantize_tensor(tensor, block_size=64)
        reconstructed = dequantize_tensor(quantized)

        self.assertEqual(reconstructed.shape, tensor.shape)
        cosine_sim = torch.cosine_similarity(tensor.flatten(), reconstructed.flatten(), dim=0).item()
        self.assertGreater(cosine_sim, 0.95)

    def test_zero_tensor_safety(self):
        tensor = torch.zeros(32, 64)
        quantized = quantize_tensor(tensor, block_size=64)
        reconstructed = dequantize_tensor(quantized)

        self.assertFalse(torch.isnan(reconstructed).any())
        self.assertTrue((reconstructed == 0.0).all())

    def test_unaligned_tensor_size(self):
        tensor = torch.randn(77)  # prime length, not divisible by 64
        quantized = quantize_tensor(tensor, block_size=64)
        reconstructed = dequantize_tensor(quantized)

        self.assertEqual(reconstructed.shape, tensor.shape)
        cosine_sim = torch.cosine_similarity(tensor, reconstructed, dim=0).item()
        self.assertGreater(cosine_sim, 0.95)

    def test_empty_tensor(self):
        tensor = torch.empty(0)
        quantized = quantize_tensor(tensor, block_size=64)
        reconstructed = dequantize_tensor(quantized)
        self.assertEqual(reconstructed.numel(), 0)

    def test_stochastic_rounding_unbiasedness(self):
        val = 0.35
        tensor = torch.full((10000,), val)
        quantized = quantize_tensor(tensor, block_size=64, stochastic=True)
        reconstructed = dequantize_tensor(quantized)

        mean_diff = abs(reconstructed.mean().item() - val)
        self.assertLess(mean_diff, 0.05)

    def test_state_dict_quantization(self):
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        state_dict = model.state_dict()
        compressed = quantize_state_dict(state_dict, block_size=64, min_numel=128)

        # Weight tensors with >= 128 elements should be quantized
        self.assertTrue(compressed["0.weight"]["__quantized__"])
        # Biases and batch norm stats should remain unquantized
        self.assertFalse(compressed["0.bias"]["__quantized__"])
        self.assertFalse(compressed["1.running_mean"]["__quantized__"])

        restored = dequantize_state_dict(compressed)
        for key in state_dict:
            self.assertEqual(restored[key].shape, state_dict[key].shape)

    def test_wire_payload_compression_ratio(self):
        model = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 128),
        )
        state_dict = model.state_dict()

        raw_buf = BytesIO()
        torch.save(state_dict, raw_buf)
        raw_size = len(raw_buf.getvalue())

        compressed_payload = compress_model_payload(state_dict, block_size=64)
        compressed_size = len(compressed_payload)

        ratio = raw_size / compressed_size
        self.assertGreaterEqual(ratio, 4.0)

        restored_state = decompress_model_payload(compressed_payload)
        for key in state_dict:
            self.assertEqual(restored_state[key].shape, state_dict[key].shape)


if __name__ == "__main__":
    unittest.main()
