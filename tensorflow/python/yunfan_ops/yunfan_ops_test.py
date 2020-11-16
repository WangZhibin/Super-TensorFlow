# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""unit tests of yunfan ops
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.yunfan_ops import yunfan_ops
from tensorflow.python.platform import test


class YunfanOpsTest(test.TestCase):

  def check_backward(self, func, x):
    if not isinstance(x, (list, tuple)):
      x = [x]
    x_dtype = x[0].dtype.base_dtype

    with self.cached_session() as _:
      variables.global_variables_initializer().run()
      error = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(func, x))
      if x_dtype is dtypes.float32:
        self.assertLess(error, 1e-3)
      elif x_dtype is dtypes.float64:
        self.assertLess(error, 1e-4)


class BatchFMInteractionTest(YunfanOpsTest):

  @staticmethod
  def _ref(x):
    x_shape = x.shape
    assert len(x_shape) == 3
    sum1 = math_ops.reduce_sum(x, axis=1)
    sum2 = math_ops.reduce_sum(x * x, axis=1)
    z1 = (sum1 * sum1 - sum2) * 0.5
    z2 = math_ops.reduce_sum(z1, axis=1, keepdims=True)
    return z2

  @test_util.run_in_graph_and_eager_modes
  def test_forward(self):
    x = np.random.randn(2, 5, 8)
    self.assertAllClose(self._ref(x), yunfan_ops.batch_fm_interaction(x))

  @test_util.deprecated_graph_mode_only
  def test_infer_shape(self):
    x = variables.Variable(np.random.randn(2, 5, 8), dtype=dtypes.float32)
    z = yunfan_ops.batch_fm_interaction(x)
    self.assertEqual(z.shape, [2, 1])

  @test_util.deprecated_graph_mode_only
  def test_backward(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = variables.Variable(np.random.randn(2, 5, 8), dtype=dtype)
      self.check_backward(yunfan_ops.batch_fm_interaction, x)


class BatchFMInteraction2Test(YunfanOpsTest):

  @staticmethod
  def _ref(x):
    x_shape = x.shape
    assert len(x_shape) == 3
    sum1 = math_ops.reduce_sum(x, axis=1)
    sum2 = math_ops.reduce_sum(x * x, axis=1)
    z = (sum1 * sum1 - sum2) * 0.5
    return z

  @test_util.run_in_graph_and_eager_modes
  def test_forward(self):
    x = np.random.randn(2, 5, 8)
    self.assertAllClose(self._ref(x), yunfan_ops.batch_fm_interaction2(x))

  @test_util.deprecated_graph_mode_only
  def test_infer_shape(self):
    x = variables.Variable(np.random.randn(2, 5, 8), dtype=dtypes.float32)
    z = yunfan_ops.batch_fm_interaction2(x)
    self.assertEqual(z.shape, [2, 8])

  @test_util.deprecated_graph_mode_only
  def test_backward(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = variables.Variable(np.random.randn(2, 5, 8), dtype=dtype)
      self.check_backward(yunfan_ops.batch_fm_interaction2, x)


class BatchPairwiseInteractionTest(YunfanOpsTest):

  @staticmethod
  def _ref(x):
    x_shape = x.shape
    assert len(x_shape) == 3
    m = x_shape[1]
    pairs = []
    for i in range(0, m):
      xi = x[:, i, :]
      for j in range(i + 1, m):
        xj = x[:, j, :]
        pairs.append(math_ops.multiply(xi, xj))
    z1 = array_ops.stack(pairs)
    z2 = array_ops.transpose(z1, perm=[1, 0, 2])
    return z2

  @test_util.run_in_graph_and_eager_modes
  def test_forward(self):
    x = np.random.randn(2, 5, 8)
    self.assertAllClose(self._ref(x), yunfan_ops.batch_pairwise_interaction(x))

  @test_util.deprecated_graph_mode_only
  def test_infer_shape(self):
    x = variables.Variable(np.random.randn(2, 5, 8), dtype=dtypes.float32)
    z = yunfan_ops.batch_pairwise_interaction(x)
    self.assertEqual(z.shape, [2, 10, 8])

  @test_util.deprecated_graph_mode_only
  def test_backward(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = variables.Variable(np.random.randn(2, 5, 8), dtype=dtype)
      self.check_backward(yunfan_ops.batch_pairwise_interaction, x)


class BatchPairwiseInteraction2Test(YunfanOpsTest):

  @staticmethod
  def _ref(x, y):
    x_shape = x.shape
    assert len(x_shape) == 3
    m1 = x_shape[1]
    y_shape = y.shape
    assert len(y_shape) == 3
    m2 = y_shape[1]
    pairs = []
    for i in range(0, m1):
      xi = x[:, i, :]
      for j in range(0, m2):
        yj = y[:, j, :]
        pairs.append(math_ops.multiply(xi, yj))
    z1 = array_ops.stack(pairs)
    z2 = array_ops.transpose(z1, perm=[1, 0, 2])
    return z2

  @test_util.run_in_graph_and_eager_modes
  def test_forward(self):
    x = np.random.randn(2, 5, 8)
    y = np.random.randn(2, 7, 8)
    self.assertAllClose(self._ref(x, y),
                        yunfan_ops.batch_pairwise_interaction2(x, y))

  @test_util.deprecated_graph_mode_only
  def test_infer_shape(self):
    x = variables.Variable(np.random.randn(2, 5, 8), dtype=dtypes.float32)
    y = variables.Variable(np.random.randn(2, 7, 8), dtype=dtypes.float32)
    z = yunfan_ops.batch_pairwise_interaction2(x, y)
    self.assertEqual(z.shape, [2, 35, 8])

  @test_util.deprecated_graph_mode_only
  def test_backward(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = variables.Variable(np.random.randn(2, 5, 8), dtype=dtype)
      y = variables.Variable(np.random.randn(2, 7, 8), dtype=dtype)
      self.check_backward(
          lambda _x: yunfan_ops.batch_pairwise_interaction2(_x, y), x)
      self.check_backward(
          lambda _y: yunfan_ops.batch_pairwise_interaction2(x, _y), y)


if __name__ == '__main__':
  test.main()
