# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import unittest

import paddle
from paddle.device.cuda.memory_analyzer import MemoryAnalysisTool


class TestCUDAVMMV2Allocator(unittest.TestCase):
    @staticmethod
    def _contains_ptr(blocks, ptr, is_free=None):
        return any(
            address <= ptr < address + size
            and (is_free is None or free == is_free)
            for size, address, free in blocks
        )

    @staticmethod
    def _contains_range(blocks, ptr):
        return any(address <= ptr < address + size for size, address in blocks)

    def test_allocator_facade_creates_vmm_v2_allocator(self):
        flags = paddle.get_flags(
            [
                "FLAGS_allocator_strategy",
                "FLAGS_use_vmm_auto_growth_best_fit_allocator_v2",
                "FLAGS_vmm_v2_large_pool_handle_size_in_mb",
            ]
        )
        self.assertEqual(flags["FLAGS_allocator_strategy"], "auto_growth")
        self.assertTrue(
            flags["FLAGS_use_vmm_auto_growth_best_fit_allocator_v2"]
        )
        self.assertGreater(
            flags["FLAGS_vmm_v2_large_pool_handle_size_in_mb"], 0
        )

        x = paddle.zeros([1024], dtype="float32")
        paddle.device.synchronize()
        self.assertEqual(x.shape, [1024])
        self.assertGreater(paddle.device.cuda.memory_reserved(), 0)

    def test_public_api_allocate_free_path(self):
        # Exercise the real allocator stack instead of constructing allocator
        # internals directly:
        #   AllocatorFacade -> RetryAllocator -> StreamSafeCUDAAllocator
        #   -> VMMAutoGrowthBestFitMultiPoolAllocatorV2.
        # Explicitly dropping tensors covers the stream-safe free path that
        # records VMM v2 remap-safety metadata for later OOM compaction.
        tensors = [
            paddle.zeros([1024 * 1024], dtype="float32") for _ in range(4)
        ]
        paddle.device.synchronize()
        reserved_before_free = paddle.device.cuda.memory_reserved()
        self.assertGreater(reserved_before_free, 0)

        del tensors
        gc.collect()
        paddle.device.synchronize()

        y = paddle.zeros([1024 * 1024], dtype="float32")
        paddle.device.synchronize()
        self.assertEqual(y.shape, [1024 * 1024])

    def test_vmm_v2_block_info(self):
        small = paddle.zeros([1024], dtype="float32")
        large = paddle.zeros([1024 * 1024], dtype="float32")
        paddle.device.synchronize()
        small_ptr = small.data_ptr()
        large_ptr = large.data_ptr()

        small_info = MemoryAnalysisTool.vmm_small_all_block_info()
        large_info = MemoryAnalysisTool.vmm_large_all_block_info()
        all_info = MemoryAnalysisTool.vmm_all_block_info()

        self.assertEqual(len(small_info), 1)
        self.assertEqual(len(large_info), 1)
        self.assertEqual(len(all_info), 2)
        self.assertTrue(self._contains_ptr(small_info[0], small_ptr, False))
        self.assertTrue(self._contains_ptr(large_info[0], large_ptr, False))
        self.assertFalse(self._contains_ptr(small_info[0], large_ptr))
        self.assertFalse(self._contains_ptr(large_info[0], small_ptr))

        del small
        del large
        gc.collect()
        paddle.device.synchronize()

        free_info = MemoryAnalysisTool.vmm_free_block_info()
        self.assertEqual(len(free_info), 2)
        self.assertTrue(
            any(
                address <= small_ptr < address + size
                for pool in free_info
                for size, address in pool
            )
        )
        self.assertTrue(
            any(
                address <= large_ptr < address + size
                for pool in free_info
                for size, address in pool
            )
        )

    def test_vmm_v2_unmapped_block_info(self):
        tensors = [
            paddle.zeros([1024 * 1024], dtype="float32") for _ in range(3)
        ]
        paddle.device.synchronize()
        hole_ptr = tensors[1].data_ptr()

        del tensors[1]
        gc.collect()
        paddle.device.synchronize()
        paddle.device.cuda.empty_cache()

        all_info = MemoryAnalysisTool.vmm_all_block_info()
        unmapped_info = MemoryAnalysisTool.vmm_unmapped_block_info()
        self.assertFalse(
            any(self._contains_ptr(pool, hole_ptr) for pool in all_info)
        )
        self.assertTrue(
            any(self._contains_range(pool, hole_ptr) for pool in unmapped_info)
        )


if __name__ == "__main__":
    unittest.main()
