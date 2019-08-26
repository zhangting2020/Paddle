#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid


def crop(data, offsets, crop_shape):
    def indexOf(shape, index):
        result = []
        for dim in reversed(shape):
            result.append(index % dim)
            index = index / dim
        return result[::-1]

    result = []
    for i, value in enumerate(data.flatten()):
        index = indexOf(data.shape, i)
        selected = True
        if len(index) == len(offsets):
            for j, offset in enumerate(offsets):
                selected = selected and index[j] >= offset and index[
                    j] < crop_shape[j] + offset
            if selected:
                result.append(value)
    return np.array(result).reshape(crop_shape)


class TestCropTensorOp(OpTest):
    def setUp(self):
        self.op_type = "crop_tensor"
        self.crop_by_1D_shape = False
        self.offset_by_input = False
        self.unk_dim_idx = -1
        self.attrs = {}
        self.initTestCase()

        if self.crop_by_1D_shape:
            self.inputs = {
                'X': np.random.random(self.x_shape).astype("float32"),
                'Shape': np.array(self.crop_shape).astype("int32")
            }
        else:
            self.attrs['shape'] = self.crop_shape
            self.inputs = {
                'X': np.random.random(self.x_shape).astype("float32"),
            }
        if self.offset_by_input:
            self.inputs['Offsets'] = np.array(self.offsets).astype('int32')
        else:
            self.attrs['offsets'] = self.offsets

        if self.unk_dim_idx != -1:
            self.crop_shape[self.unk_dim_idx] = self.x_shape[self.unk_dim_idx]
        self.outputs = {
            'Out': crop(self.inputs['X'], self.offsets, self.crop_shape)
        }

    def initTestCase(self):
        self.x_shape = (8, 8)
        self.crop_shape = [2, 2]
        self.offsets = [1, 2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.006)


class TestCase1(TestCropTensorOp):
    def initTestCase(self):
        self.x_shape = (100)
        self.crop_shape = [64]
        self.offsets = [13]


class TestCase2(TestCropTensorOp):
    def initTestCase(self):
        self.x_shape = (12, 24)
        self.crop_shape = [-1, 8]  #only the first dimension (batch) can be -1
        self.offsets = [0, 0]
        self.unk_dim_idx = 0


class TestCase3(TestCropTensorOp):
    def initTestCase(self):
        self.x_shape = (4, 8, 16)
        self.crop_shape = [2, 2, 3]
        self.offsets = [1, 5, 3]
        self.crop_by_1D_shape = True


class TestCase4(TestCropTensorOp):
    def initTestCase(self):
        self.x_shape = (8, 3, 6, 6)
        self.crop_shape = [-1, 3, 4, 4]
        self.offsets = [0, 0, 0, 0]
        self.crop_by_1D_shape = True
        self.unk_dim_idx = 0


class TestCase5(TestCropTensorOp):
    def initTestCase(self):
        self.x_shape = (2, 4, 5, 8, 8)
        self.crop_shape = [1, 1, 2, 4, 4]
        self.offsets = [1, 0, 0, 2, 2]
        self.offset_by_input = True


class TestCase6(TestCropTensorOp):
    def initTestCase(self):
        self.x_shape = (2, 2, 4, 4, 4, 2)
        self.crop_shape = [1, 1, 4, 2, 2, 2]
        self.offsets = [0, 0, 0, 0, 0, 0]
        self.crop_by_1D_shape = True
        self.offset_by_input = True


class TestCropTensorOp_attr_tensor(OpTest):
    def setUp(self):
        self.op_type = "crop_tensor"
        self.mixed_type = False
        self.attrs = {}
        self.initTestCase()

        shape_tensor = []
        for index, ele in enumerate(self.crop_shape):
            shape_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))
        self.inputs = {
            "X": np.random.random(self.x_shape).astype("float32"),
            'ShapeTensor': shape_tensor
        }
        if self.mixed_type:
            self.attrs['shape'] = self.shape_attr
        self.attrs['offsets'] = self.offsets
        self.outputs = {
            'Out': crop(self.inputs['X'], self.offsets, self.crop_shape)
        }

    def initTestCase(self):
        self.x_shape = (8, 8)
        self.crop_shape = (2, 2)
        self.offsets = [1, 2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(["X"], "Out", max_relative_error=0.006)


class TestCropTensorOp_attr_tensor_case1(TestCropTensorOp_attr_tensor):
    def init_data(self):
        self.x_shape = (16, 8, 32)
        self.crop_shape = [2, 2, 3]
        self.offsets = [1, 5, 3]


class TestCropTensorOp_attr_tensor_case2(TestCropTensorOp_attr_tensor):
    def init_data(self):
        self.x_shape = (4, 8, 16, 8)
        self.crop_shape = [2, 2, 3, 4]
        self.offsets = [1, 5, 3, 0]
        self.shape_attr = [-1, -1, 3, 4]
        self.mixed_type = True


# Test python API
class TestCropTensorAPI(OpTest):
    def test_case1(self):
        x1 = fluid.layers.data(name="x1", shape=[6, 5, 8])
        crop_shape1 = (-1, 2, 4, 4)
        out1 = fluid.layers.crop_tensor(x1, shape=crop_shape1)

    def test_case2(self):
        x2 = fluid.layers.data(name="x2", shape=[3, 12, 8])
        crop_shape2 = fluid.layers.data(
            name="crop_shape", shape=[4], append_batch_size=False)
        out2 = fluid.layers.crop_tensor(x2, shape=crop_shape2)

    def test_case3(self):
        x3 = fluid.layers.data(name="x3", shape=[2, 3, 6, 12])
        dim1 = fluid.layers.data(name="dim1", shape=[1])
        dim2 = fluid.layers.data(name="dim2", shape=[1])
        out3 = fluid.layers.crop_tensor(x3, shape=[-1, dim1, dim2, 4])

    def test_case4(self):
        x4 = fluid.layers.data(name="x4", shape=[6, 5, 8])
        crop_shape4 = [1, 2, 4, 4]
        crop_offset = fluid.layers.data(name="crop_offset", shape=[4])
        out4 = fluid.layers.crop_tensor(
            x4, shape=crop_shape4, offsets=crop_offset)


if __name__ == '__main__':
    unittest.main()
