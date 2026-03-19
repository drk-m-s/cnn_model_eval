"""
Unit tests for the CNN Model Performance Evaluator.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.chip import ChipSpec
from src.evaluator import evaluate
from src.layer import BYTES_PER_ELEMENT, LayerProfile
from src.profiler import (
    bytes_per_element,
    compute_conv_macs,
    compute_dram_bytes,
    compute_element_wise_macs,
    compute_gemm_macs,
    compute_matmul_macs,
    compute_pool_macs,
)


class TestBytesPerElement(unittest.TestCase):
    def test_int8(self):
        self.assertEqual(bytes_per_element("int8"), 1.0)

    def test_fp16(self):
        self.assertEqual(bytes_per_element("fp16"), 2.0)

    def test_int4(self):
        self.assertEqual(bytes_per_element("int4"), 0.5)


class TestConvMACs(unittest.TestCase):
    def test_standard_conv_3x3(self):
        """Standard 3x3 Conv: input [1,64,56,56], weight [128,64,3,3], output [1,128,56,56]."""
        macs = compute_conv_macs(
            input_shape=[1, 64, 56, 56],
            weight_shape=[128, 64, 3, 3],
            output_shape=[1, 128, 56, 56],
            groups=1,
        )
        # Expected: 56 * 56 * 128 * 3 * 3 * 64 = 56*56*128*9*64
        expected = 56 * 56 * 128 * 3 * 3 * 64
        self.assertEqual(macs, expected)

    def test_depthwise_conv(self):
        """Depthwise separable conv (groups=C_in): [1,64,56,56], weight [64,1,3,3]."""
        macs = compute_conv_macs(
            input_shape=[1, 64, 56, 56],
            weight_shape=[64, 1, 3, 3],
            output_shape=[1, 64, 56, 56],
            groups=64,
        )
        # Expected: 56 * 56 * 64 * 3 * 3 * 1
        expected = 56 * 56 * 64 * 3 * 3 * 1
        self.assertEqual(macs, expected)

    def test_1x1_conv(self):
        """Pointwise 1x1 conv: [1,256,14,14], weight [512,256,1,1], output [1,512,14,14]."""
        macs = compute_conv_macs(
            input_shape=[1, 256, 14, 14],
            weight_shape=[512, 256, 1, 1],
            output_shape=[1, 512, 14, 14],
            groups=1,
        )
        expected = 14 * 14 * 512 * 1 * 1 * 256
        self.assertEqual(macs, expected)

    def test_conv_with_bias(self):
        """Conv with bias adds C_out * H_out * W_out."""
        macs_no_bias = compute_conv_macs(
            input_shape=[1, 3, 224, 224],
            weight_shape=[64, 3, 7, 7],
            output_shape=[1, 64, 112, 112],
            groups=1,
            has_bias=False,
        )
        macs_with_bias = compute_conv_macs(
            input_shape=[1, 3, 224, 224],
            weight_shape=[64, 3, 7, 7],
            output_shape=[1, 64, 112, 112],
            groups=1,
            has_bias=True,
        )
        self.assertEqual(macs_with_bias - macs_no_bias, 64 * 112 * 112)


class TestGemmMACs(unittest.TestCase):
    def test_fc_1000(self):
        """Fully connected: M=1, K=2048, N=1000."""
        macs = compute_gemm_macs(1, 2048, 1000)
        self.assertEqual(macs, 1 * 2048 * 1000)

    def test_with_bias(self):
        macs = compute_gemm_macs(1, 2048, 1000, has_bias=True)
        self.assertEqual(macs, 1 * 2048 * 1000 + 1 * 1000)


class TestMatMulMACs(unittest.TestCase):
    def test_2d(self):
        macs = compute_matmul_macs([1, 512], [512, 1000])
        self.assertEqual(macs, 1 * 512 * 1000)


class TestPoolMACs(unittest.TestCase):
    def test_maxpool(self):
        macs = compute_pool_macs([1, 64, 56, 56], (3, 3))
        expected = 64 * 56 * 56 * 3 * 3
        self.assertEqual(macs, expected)


class TestLayerProfile(unittest.TestCase):
    def test_ops_double_macs(self):
        lp = LayerProfile(name="test", op_type="Conv", macs=1000)
        self.assertEqual(lp.ops, 2000)

    def test_dram_bytes_int8(self):
        lp = LayerProfile(
            name="conv1",
            op_type="Conv",
            input_shapes=[[1, 3, 224, 224]],
            output_shapes=[[1, 64, 112, 112]],
            weight_params=9408,  # 64*3*7*7
            bias_params=64,
        )
        # int8: act=1 byte, weight=1 byte, bias=4 bytes (INT32)
        read = 3 * 224 * 224 * 1.0 + 9408 * 1.0 + 64 * 4.0
        write = (64 * 112 * 112) * 1.0
        self.assertAlmostEqual(lp.total_dram_bytes("int8"), read + write)

    def test_dram_bytes_fp16(self):
        lp = LayerProfile(
            name="conv1",
            op_type="Conv",
            input_shapes=[[1, 3, 224, 224]],
            output_shapes=[[1, 64, 112, 112]],
            weight_params=9408,
            bias_params=64,
        )
        # fp16: act=2 bytes, weight=2 bytes, bias=4 bytes (FP32)
        read = 3 * 224 * 224 * 2.0 + 9408 * 2.0 + 64 * 4.0
        write = (64 * 112 * 112) * 2.0
        self.assertAlmostEqual(lp.total_dram_bytes("fp16"), read + write)

    def test_fused_layer_zero_dram(self):
        lp = LayerProfile(name="bn1", op_type="BatchNormalization", is_fused=True)
        self.assertEqual(lp.total_dram_bytes("int8"), 0.0)


class TestChipSpec(unittest.TestCase):
    def test_effective_tops_int8(self):
        chip = ChipSpec(name="test", int8_tops=10.0, dram_bandwidth_gbps=50.0,
                        compute_efficiency=0.7, memory_efficiency=0.8)
        # 10 * 1.0 * 0.7 = 7.0
        self.assertAlmostEqual(chip.effective_tops("int8"), 7.0)

    def test_effective_tops_fp16(self):
        chip = ChipSpec(name="test", int8_tops=10.0, dram_bandwidth_gbps=50.0,
                        compute_efficiency=0.7, memory_efficiency=0.8)
        # 10 * 0.5 * 0.7 = 3.5
        self.assertAlmostEqual(chip.effective_tops("fp16"), 3.5)

    def test_effective_tops_int4(self):
        chip = ChipSpec(name="test", int8_tops=10.0, dram_bandwidth_gbps=50.0,
                        compute_efficiency=0.7, memory_efficiency=0.8)
        # 10 * 2.0 * 0.7 = 14.0
        self.assertAlmostEqual(chip.effective_tops("int4"), 14.0)

    def test_effective_bandwidth(self):
        chip = ChipSpec(name="test", int8_tops=10.0, dram_bandwidth_gbps=50.0,
                        compute_efficiency=0.7, memory_efficiency=0.8)
        self.assertAlmostEqual(chip.effective_bandwidth_gbps(), 40.0)

    def test_effective_tops_1d_int8(self):
        chip = ChipSpec(name="test", int8_tops=10.0, dram_bandwidth_gbps=50.0,
                        compute_efficiency=0.7, memory_efficiency=0.8,
                        int8_tops_1d=2.0, dram_bandwidth_gbps_1d=20.0,
                        compute_efficiency_1d=0.5, memory_efficiency_1d=0.6)
        # 2.0 * 1.0 * 0.5 = 1.0
        self.assertAlmostEqual(chip.effective_tops_1d("int8"), 1.0)

    def test_effective_bandwidth_1d(self):
        chip = ChipSpec(name="test", int8_tops=10.0, dram_bandwidth_gbps=50.0,
                        compute_efficiency=0.7, memory_efficiency=0.8,
                        int8_tops_1d=2.0, dram_bandwidth_gbps_1d=20.0,
                        compute_efficiency_1d=0.5, memory_efficiency_1d=0.6)
        # 20.0 * 0.6 = 12.0
        self.assertAlmostEqual(chip.effective_bandwidth_gbps_1d(), 12.0)

    def test_from_json(self):
        data = [{"name": "TestChip", "int8_tops": 8.0, "dram_bandwidth_gbps": 32.0,
                 "int8_tops_1d": 1.0, "dram_bandwidth_gbps_1d": 10.0}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            chip = ChipSpec.from_json(f.name)
        os.unlink(f.name)
        self.assertEqual(chip.name, "TestChip")
        self.assertEqual(chip.int8_tops, 8.0)
        self.assertEqual(chip.int8_tops_1d, 1.0)
        self.assertEqual(chip.dram_bandwidth_gbps_1d, 10.0)

    def test_unsupported_quantization(self):
        chip = ChipSpec(name="test", int8_tops=10.0, dram_bandwidth_gbps=50.0)
        with self.assertRaises(ValueError):
            chip.effective_tops("bf16")
        with self.assertRaises(ValueError):
            chip.effective_tops_1d("bf16")


class TestEvaluator(unittest.TestCase):
    def _make_chip(self):
        return ChipSpec(
            name="TestChip",
            int8_tops=10.0,
            dram_bandwidth_gbps=50.0,
            compute_efficiency=1.0,  # 100% for easy math
            memory_efficiency=1.0,
            int8_tops_1d=1.0,
            dram_bandwidth_gbps_1d=10.0,
            compute_efficiency_1d=1.0,
            memory_efficiency_1d=1.0,
        )

    def test_compute_bound_layer_2d(self):
        """A 2D layer with huge MACs but tiny data should be compute-bound."""
        chip = self._make_chip()
        layers = [
            LayerProfile(
                name="heavy_conv",
                op_type="Conv",
                input_shapes=[[1, 256, 14, 14]],
                output_shapes=[[1, 256, 14, 14]],
                weight_params=256 * 256 * 3 * 3,  # ~590K params
                macs=256 * 256 * 3 * 3 * 14 * 14,  # ~115M MACs
                compute_type="2d",
            ),
        ]
        result = evaluate(chip, layers, "int8", (14, 14), "test")
        self.assertEqual(result.layer_results[0].bottleneck, "compute")
        self.assertEqual(result.layer_results[0].compute_type, "2d")
        self.assertEqual(result.layers_2d, 1)
        self.assertEqual(result.layers_1d, 0)

    def test_memory_bound_layer_1d(self):
        """A 1D layer with tiny MACs but large data should be memory-bound."""
        chip = self._make_chip()
        layers = [
            LayerProfile(
                name="relu",
                op_type="Relu",
                input_shapes=[[1, 512, 28, 28]],
                output_shapes=[[1, 512, 28, 28]],
                weight_params=0,
                macs=512 * 28 * 28,  # ~400K MACs (small)
                compute_type="1d",
            ),
        ]
        result = evaluate(chip, layers, "int8", (28, 28), "test")
        self.assertEqual(result.layer_results[0].bottleneck, "memory")
        self.assertEqual(result.layer_results[0].compute_type, "1d")
        self.assertEqual(result.layers_1d, 1)
        self.assertEqual(result.layers_2d, 0)

    def test_1d_uses_1d_engine_throughput(self):
        """1D layers should use the 1D engine's TOPS/BW, not the 2D engine's."""
        chip = self._make_chip()  # 1D: 1 TOPS, 10 GB/s
        layers = [
            LayerProfile(
                name="sigmoid",
                op_type="Sigmoid",
                input_shapes=[[1, 256, 28, 28]],
                output_shapes=[[1, 256, 28, 28]],
                weight_params=0,
                macs=256 * 28 * 28,
                ops_override=256 * 28 * 28 * 4,  # 4 ops/elem for Sigmoid
                compute_type="1d",
            ),
        ]
        result = evaluate(chip, layers, "int8", (28, 28), "test")
        lr = result.layer_results[0]
        # compute_time = ops / (1 TOPS * 1e12) = (200704*4) / 1e12
        expected_compute = (256 * 28 * 28 * 4) / 1e12
        self.assertAlmostEqual(lr.compute_time_s, expected_compute, places=18)

    def test_2d_uses_2d_engine_throughput(self):
        """2D layers should use the 2D engine's TOPS/BW."""
        chip = self._make_chip()  # 2D: 10 TOPS, 50 GB/s
        macs = 256 * 256 * 3 * 3 * 14 * 14
        layers = [
            LayerProfile(
                name="conv",
                op_type="Conv",
                input_shapes=[[1, 256, 14, 14]],
                output_shapes=[[1, 256, 14, 14]],
                weight_params=256 * 256 * 3 * 3,
                macs=macs,
                compute_type="2d",
            ),
        ]
        result = evaluate(chip, layers, "int8", (14, 14), "test")
        lr = result.layer_results[0]
        # compute_time = ops / (10 TOPS * 1e12) = (macs*2) / 10e12
        expected_compute = (macs * 2) / (10.0 * 1e12)
        self.assertAlmostEqual(lr.compute_time_s, expected_compute, places=18)

    def test_fused_layer_zero_time(self):
        """Fused layers should have zero execution time."""
        chip = self._make_chip()
        layers = [
            LayerProfile(
                name="bn1",
                op_type="BatchNormalization",
                input_shapes=[[1, 64, 56, 56]],
                output_shapes=[[1, 64, 56, 56]],
                macs=0,
                is_fused=True,
            ),
        ]
        result = evaluate(chip, layers, "int8", (56, 56), "test")
        self.assertEqual(result.layer_results[0].layer_time_s, 0.0)
        self.assertEqual(result.layer_results[0].bottleneck, "fused")

    def test_fps_positive(self):
        """FPS should be positive for any non-trivial model."""
        chip = self._make_chip()
        layers = [
            LayerProfile(
                name="conv1",
                op_type="Conv",
                input_shapes=[[1, 3, 224, 224]],
                output_shapes=[[1, 64, 112, 112]],
                weight_params=9408,
                bias_params=64,
                macs=118013952,
                compute_type="2d",
            ),
        ]
        result = evaluate(chip, layers, "int8", (224, 224), "test")
        self.assertGreater(result.fps, 0)
        self.assertGreater(result.total_time_s, 0)

    def test_quantization_impact(self):
        """FP16 should be slower than INT8 (more data, fewer effective TOPS)."""
        chip = self._make_chip()
        layers = [
            LayerProfile(
                name="conv1",
                op_type="Conv",
                input_shapes=[[1, 3, 224, 224]],
                output_shapes=[[1, 64, 112, 112]],
                weight_params=9408,
                bias_params=64,
                macs=118013952,
                compute_type="2d",
            ),
        ]
        result_int8 = evaluate(chip, layers, "int8", (224, 224), "test")
        result_fp16 = evaluate(chip, layers, "fp16", (224, 224), "test")
        # FP16 should be slower → lower FPS
        self.assertGreater(result_int8.fps, result_fp16.fps)

    def test_mixed_2d_1d_layers(self):
        """A model with both 2D and 1D layers should track engine breakdown."""
        chip = self._make_chip()
        layers = [
            LayerProfile(
                name="conv1",
                op_type="Conv",
                input_shapes=[[1, 3, 224, 224]],
                output_shapes=[[1, 64, 112, 112]],
                weight_params=9408,
                bias_params=64,
                macs=118013952,
                compute_type="2d",
            ),
            LayerProfile(
                name="relu1",
                op_type="Relu",
                input_shapes=[[1, 64, 112, 112]],
                output_shapes=[[1, 64, 112, 112]],
                weight_params=0,
                macs=64 * 112 * 112,
                compute_type="1d",
            ),
        ]
        result = evaluate(chip, layers, "int8", (224, 224), "test")
        self.assertEqual(result.layers_2d, 1)
        self.assertEqual(result.layers_1d, 1)
        self.assertGreater(result.total_time_2d_s, 0)
        self.assertGreater(result.total_time_1d_s, 0)
        self.assertAlmostEqual(
            result.total_time_s,
            result.total_time_2d_s + result.total_time_1d_s,
        )


class TestDRAMBytesHelper(unittest.TestCase):
    def test_compute_dram_bytes(self):
        total = compute_dram_bytes(
            input_elements=150528,   # 3*224*224
            output_elements=802816,  # 64*112*112
            weight_elements=9408,    # 64*3*7*7
            bias_elements=64,
            quantization="int8",
        )
        expected = (150528 + 9408 + 64 + 802816) * 1.0
        self.assertAlmostEqual(total, expected)


if __name__ == "__main__":
    unittest.main()
