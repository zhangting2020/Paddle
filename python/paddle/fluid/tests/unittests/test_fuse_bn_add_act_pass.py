#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
from paddle.fluid import core

paddle.enable_static()


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class TestFusedBnAddActAPI(unittest.TestCase):
    def setUp(self):
        self.conv_param_attr1 = fluid.ParamAttr(
            name='conv2d_1.weight',
            initializer=fluid.initializer.Xavier(uniform=False),
            learning_rate=0.001)
        self.conv_param_attr2 = fluid.ParamAttr(
            name='conv2d_2.weight',
            initializer=fluid.initializer.Xavier(uniform=False),
            learning_rate=0.001)
        self.bn_param_attr1 = fluid.ParamAttr(
            name='batch_norm_w_1',
            initializer=fluid.initializer.Constant(value=1.0))
        self.bn_bias_attr1 = fluid.ParamAttr(
            name='batch_norm_b_1',
            initializer=fluid.initializer.Constant(value=0.0))
        self.bn_param_attr2 = fluid.ParamAttr(
            name='batch_norm_w_2',
            initializer=fluid.initializer.Constant(value=1.0))
        self.bn_bias_attr2 = fluid.ParamAttr(
            name='batch_norm_b_2',
            initializer=fluid.initializer.Constant(value=0.0))
        self.fc_param_attr = fluid.ParamAttr(
            name='fc.weight',
            initializer=fluid.initializer.Xavier(uniform=False))

    def build_fused_program(self,
                            main_program,
                            startup_program,
                            use_cuda,
                            seed=1):
        with fluid.program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[1, 28, 28], dtype='float32')
            y = fluid.layers.data(name="y", shape=[1], dtype='int64')
            conv1_1 = fluid.layers.conv2d(
                input=x,
                filter_size=3,
                num_filters=32,
                stride=1,
                padding=1,
                act=None,
                param_attr=self.conv_param_attr1,
                bias_attr=False,
                data_format='NHWC')
            conv1_2 = fluid.layers.conv2d(
                input=x,
                filter_size=3,
                num_filters=32,
                stride=1,
                padding=1,
                act=None,
                param_attr=self.conv_param_attr2,
                bias_attr=False,
                data_format='NHWC')
            bn = fluid.layers.batch_norm(
                input=conv1_1,
                param_attr=self.bn_param_attr1,
                bias_attr=self.bn_bias_attr1,
                act=None,
                data_layout='NHWC')
            fused_bn_add_act = fluid.contrib.layers.fused_bn_add_act(
                conv1_2,
                bn,
                param_attr=self.bn_param_attr2,
                bias_attr=self.bn_bias_attr2)
            prediction = fluid.layers.fc(input=fused_bn_add_act,
                                         size=10,
                                         act='softmax',
                                         param_attr=self.fc_param_attr)
            loss = fluid.layers.cross_entropy(input=prediction, label=y)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(learning_rate=0.001)
            sgd = fluid.contrib.mixed_precision.decorate(
                sgd, use_dynamic_loss_scaling=True, init_loss_scaling=128.0)
            sgd.minimize(loss)

        return x, y, loss

    def build_origin_program(self,
                             main_program,
                             startup_program,
                             use_cuda,
                             seed=1):
        with fluid.program_guard(main_program, startup_program):
            x1 = fluid.layers.data(
                name='x1', shape=[1, 28, 28], dtype='float32')
            x2 = fluid.layers.data(
                name='x2', shape=[1, 28, 28], dtype='float32')
            y = fluid.layers.data(name="y", shape=[1], dtype='int64')
            conv1_1 = fluid.layers.conv2d(
                input=x1,
                filter_size=3,
                num_filters=32,
                stride=1,
                padding=1,
                act=None,
                param_attr=self.conv_param_attr1,
                bias_attr=False,
                data_format='NHWC')
            conv1_2 = fluid.layers.conv2d(
                input=x2,
                filter_size=3,
                num_filters=32,
                stride=1,
                padding=1,
                act=None,
                param_attr=self.conv_param_attr2,
                bias_attr=False,
                data_format='NHWC')
            bn1 = fluid.layers.batch_norm(
                input=conv1_1,
                param_attr=self.bn_param_attr1,
                bias_attr=self.bn_bias_attr1,
                act=None,
                data_layout='NHWC')
            bn2 = fluid.layers.batch_norm(
                input=conv1_2,
                param_attr=self.bn_param_attr2,
                bias_attr=self.bn_bias_attr2,
                act=None,
                data_layout='NHWC')
            out = bn1 + bn2
            out = fluid.layers.relu(out)
            prediction = fluid.layers.fc(input=out,
                                         size=10,
                                         act='softmax',
                                         param_attr=self.fc_param_attr)
            loss = fluid.layers.cross_entropy(input=prediction, label=y)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(learning_rate=0.001)
            sgd = fluid.contrib.mixed_precision.decorate(
                sgd, use_dynamic_loss_scaling=True, init_loss_scaling=128.0)
            sgd.minimize(loss)

        return loss

    def check(self, place, use_cuda):
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        iters = 5
        batch_size = 16

        # build_fused_program: turn on fuse_bn_add_act_ops
        main_program = fluid.Program()
        startup_program = fluid.Program()
        loss = self.build_origin_program(main_program, startup_program,
                                         use_cuda)
        build_strategy_fused = fluid.BuildStrategy()
        build_strategy_fused.fuse_bn_add_act_ops = True
        binary_fused = fluid.CompiledProgram(main_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy_fused)
        exe = fluid.Executor(place)
        loss_vals_fused = []
        x1_data = []
        x2_data = []
        y_data = []
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            for _ in range(iters):
                x1 = np.random.random((batch_size, 1, 28, 28)).astype("float32")
                x2 = np.random.random((batch_size, 1, 28, 28)).astype("float32")
                y = np.random.random((batch_size, 1)).astype("int64")
                x1_data.append(x1)
                x2_data.append(x2)
                y_data.append(y)
                loss_v = exe.run(binary_fused,
                                 feed={"x1": x1,
                                       "x2": x2,
                                       "y": y},
                                 fetch_list=[loss])
                loss_vals_fused.append(loss_v[0][0])

        # build_origin_program: turn off fused_bn_act_ops
        build_strategy = fluid.BuildStrategy()
        build_strategy.fuse_bn_add_act_ops = False
        binary = fluid.CompiledProgram(main_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy_fused)
        loss_vals = []
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            for i in range(iters):
                loss_v = exe.run(
                    binary,
                    feed={"x1": x1_data[i],
                          "x2": x2_data[i],
                          "y": y_data[i]},
                    fetch_list=[loss])
                loss_vals.append(loss_v[0][0])

        # check loss
        for i in range(iters):
            self.assertAlmostEqual(loss_vals[i], loss_vals_fused[i], delta=1e-5)

    def test_fuse_bn_add_act(self):
        place = fluid.CUDAPlace(0)
        self.check(place, use_cuda=True)

    def test_fuse_bn_add_act_API(self):
        # build_fused_program: use fused_bn_add_act python API
        main_program = fluid.Program()
        startup_program = fluid.Program()
        place = fluid.CUDAPlace(0)
        x, y, loss = self.build_fused_program(
            main_program, startup_program, use_cuda=True)
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            for _ in range(5):
                x = np.random.random((4, 1, 28, 28)).astype("float32")
                y = np.random.random((4, 1)).astype("int64")
                loss_v = exe.run(main_program,
                                 feed={"x": x,
                                       "y": y},
                                 fetch_list=[loss])


if __name__ == '__main__':
    unittest.main()
