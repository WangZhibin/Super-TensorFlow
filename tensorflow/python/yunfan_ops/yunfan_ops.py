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
"""yunfan ops
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.yunfan_ops.ops import gen_yunfan_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export('batch_fm_interaction')
def batch_fm_interaction(x, *args, **kwargs):
  r"""Perform a standard FM second-order interaction on rows of `x_i`.

  Shape of `x` is like `[batch, m, n]`.
  Shape of `x_i` is like `[m, n]`.

  Let `z` denote `batch_fm_interaction(x)`.
  Shape of `z_i` will be `[1]`.
  Shape of `z` will be `[batch, 1]`.

  `z_i` is mathematically:
  $$
  z_i = \sum_{l=0}^n
  \left(\sum_{j=0}^m \sum_{k=j+1}^m x_{ij} \cdot x_{ik}\right)_l
  $$

  For example:

  ```python
  x = tf.constant([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]],
                  dtype=tf.float32)
  tf.batch_fm_interaction(x)
  # ==> [[ 14.],
  #      [212.]]
  ```

  Args:
    x: A `Tensor` with shape `[batch, m, n]`.
    Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with shape `[batch, 1]`. Has the same type as `x`.

  References:
    - [Factorization machines]
      (https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
  """
  return gen_yunfan_ops.batch_fm_interaction(x, *args, **kwargs)


@ops.RegisterGradient("BatchFMInteraction")
def _batch_fm_interaction_grad(op, gz, *args, **kwargs):
  return gen_yunfan_ops.batch_fm_interaction_grad(op.inputs[0], gz, *args,
                                                  **kwargs)


@tf_export('batch_fm_interaction2')
def batch_fm_interaction2(x, *args, **kwargs):
  r"""Perform an extended FM second-order interaction on rows of `x_i`.

  Shape of `x` is like `[batch, m, n]`.
  Shape of `x_i` is like `[m, n]`.

  Let `z` denote batch_fm_interaction2(x).
  Shape of `z_i` will be `[n]`.
  Shape of `z` will be `[batch, n]`.

  `z_i` is mathematically:
  $$
  z_i = \sum_{j=0}^m \sum_{k=j+1}^m x_{ij} \cdot x_{ik}
  $$

  `batch_fm_interaction` is equivalent to `reduce_sum` the result of
  `batch_fm_interaction2`, while `batch_fm_interaction2` preserves more
  information.

  For example:

  ```python
  x = tf.constant([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]],
                  dtype=tf.float32)
  tf.batch_fm_interaction2(x, y)
  # ==> [[ 0.,  4., 10.],
  #      [54., 70., 88.]]
  ```

  Args:
    x: A `Tensor` with shape `[batch, m, n]`.
    Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with shape `[batch, n]`. Has the same type as `x`.

  References:
    - [Factorization machines]
      (https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
  """
  return gen_yunfan_ops.batch_fm_interaction2(x, *args, **kwargs)


@ops.RegisterGradient("BatchFMInteraction2")
def _batch_fm_interaction2_grad(op, gz, *args, **kwargs):
  return gen_yunfan_ops.batch_fm_interaction2_grad(op.inputs[0], gz, *args,
                                                   **kwargs)


@tf_export('batch_pairwise_interaction')
def batch_pairwise_interaction(x, *args, **kwargs):
  r"""Perform a pairwise interaction(multiplication) on rows of `x_i`.

  Shape of `x` is like `[batch, m, n]`.
  Shape of `x_i` is like `[m, n]`.
  The pairwise interaction will be performed on `x_i` and itself.

  Let `z` denote batch_pairwise_interaction(x).
  Shape of `z_i` will be `[m * (m - 1) / 2, n]`.
  Shape of `z` will be `[batch, m * (m - 1) / 2, n]`.

  For example:

  ```python
  x = tf.constant([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                  [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]],
                  dtype=tf.float32)
  tf.batch_pairwise_interaction(x)
  # ==> [[[  0.   4.  10.]
  #       [  0.   7.  16.]
  #       [  0.  10.  22.]
  #       [ 18.  28.  40.]
  #       [ 27.  40.  55.]
  #       [ 54.  70.  88.]]
  #      [[180. 208. 238.]
  #       [216. 247. 280.]
  #       [252. 286. 322.]
  #       [270. 304. 340.]
  #       [315. 352. 391.]
  #       [378. 418. 460.]]]
  ```

  Args:
    x: A `Tensor` with shape `[batch, m, n]`.
    Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with shape `[batch, m * (m - 1) / 2, n]`. Has the same type as `x`.

  References:
    - [Attentional Factorization Machines:
      Learning the Weight of Feature Interactions via Attention Networks]
      (https://arxiv.org/pdf/1708.04617.pdf)
    """
  return gen_yunfan_ops.batch_pairwise_interaction(x, *args, **kwargs)


@ops.RegisterGradient('BatchPairwiseInteraction')
def _batch_pairwise_interaction_grad(op, gz, *args, **kwargs):
  return gen_yunfan_ops.batch_pairwise_interaction_grad(op.inputs[0], gz, *args,
                                                        **kwargs)


@tf_export('batch_pairwise_interaction2')
def batch_pairwise_interaction2(x, y, *args, **kwargs):
  r"""Perform an interaction(multiplication) on rows of `x_i` and rows of `y_i`
  in a batch approach.

  Shape of `x` is like `[batch, m1, n]`.
  Shape of `x_i` is like `[m1, n]`.
  Shape of `y` is like `[batch, m2, n]`.
  Shape of `y_i` is like `[m2, n]`.
  The pairwise interaction will be performed on `x_i` and `y_i`.

  Let `z` denote batch_pairwise_interaction2(x, y).
  Shape of `z_i` will be `[m1 * m2, n]`.
  Shape of `z` will be `[batch, m1 * m2, n]`.

  For example:

  ```python
  x = tf.constant([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]],
                  dtype=tf.float32)
  y = tf.constant([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[9, 10, 11], [12, 13, 14], [15, 16, 17]]],
                  dtype=tf.float32)
  tf.batch_pairwise_interaction2(x, y)
  # ==> [[[  0.   1.   4.]
  #       [  0.   4.  10.]
  #       [  0.   7.  16.]
  #       [  0.   4.  10.]
  #       [  9.  16.  25.]
  #       [ 18.  28.  40.]]
  #      [[ 54.  70.  88.]
  #       [ 72.  91. 112.]
  #       [ 90. 112. 136.]
  #       [ 81. 100. 121.]
  #       [108. 130. 154.]
  #       [135. 160. 187.]]]
  ```

  Args:
    x: A `Tensor` with shape `[batch, m1, n]`.
    Must be one of the following types: `float32`, `float64`.
    y: A `Tensor` with shape `[batch, m2, n]`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with shape `[batch, m1 * m2, n]`. Has the same type as `x`.

  References:
    - [Attentional Factorization Machines:
      Learning the Weight of Feature Interactions via Attention Networks]
      (https://arxiv.org/pdf/1708.04617.pdf)
  """
  return gen_yunfan_ops.batch_pairwise_interaction2(x, y, *args, **kwargs)


@ops.RegisterGradient('BatchPairwiseInteraction2')
def _batch_pairwise_interaction2_grad(op, gz, *args, **kwargs):
  return gen_yunfan_ops.batch_pairwise_interaction2_grad(
      op.inputs[0], op.inputs[1], gz, *args, **kwargs)
