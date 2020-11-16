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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import tempfile
import time

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adadelta
from tensorflow.python.training import adagrad
from tensorflow.python.training import adagrad_da
from tensorflow.python.training import adam
from tensorflow.python.training import ftrl
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import lamb
from tensorflow.python.training import momentum
from tensorflow.python.training import proximal_adagrad
from tensorflow.python.training import proximal_gradient_descent as pgd
from tensorflow.python.training import device_setter
from tensorflow.python.training import rmsprop
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import server_lib
from tensorflow.python.training import training_util
from tensorflow.python.yunfan_ops import dynamic_embedding_ops as deo
from tensorflow.python.yunfan_ops import dynamic_variable_restrict as dvr

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


def select_slot_vars(trainable_wrapper, opt):
  slot_names = opt.get_slot_names()
  slot_vars = list()
  for name in slot_names:
    slot = opt.get_slot(trainable_wrapper, name)
    if slot is not None:
      slot_vars.append(slot.params)
  return slot_vars


def get_size_info(sess, var, policy, slot_vars):
  if isinstance(policy, dvr.TimestampRestrictPolicy):
    policy_status = policy.tstp_var
  elif isinstance(policy, dvr.FrequencyRestrictPolicy):
    policy_status = policy.freq_var

  var_size = sess.run(var.size())
  policy_var_size = sess.run(policy_status.size())
  slot_var_size = sum(sess.run([rec.size() for rec in slot_vars],))
  return var_size, policy_var_size, slot_var_size


def create_graph_and_policy(optimizer, ids, mode):
  var = deo.get_variable('sp_var',
                         key_dtype=ids.dtype,
                         value_dtype=dtypes.float32,
                         initializer=-1.,
                         dim=2)
  embedding_w, trainable_wrapper = deo.embedding_lookup(var,
                                                        ids,
                                                        return_trainable=True)
  loss = math_ops.reduce_sum(embedding_w)
  train_op = optimizer.minimize(loss, var_list=[trainable_wrapper])

  if mode == 'timestamp':
    policy = dvr.TimestampRestrictPolicy(var, optimizer)
  elif mode == 'frequency':
    policy = dvr.FrequencyRestrictPolicy(var, optimizer)
  else:
    raise NotImplementedError
  slot_vars = select_slot_vars(trainable_wrapper, optimizer)
  return slot_vars, policy, var, train_op


def get_multiple_optimizers():
  return [
      adagrad.AdagradOptimizer(0.1),
      adam.AdamOptimizer(0.1),
      ftrl.FtrlOptimizer(0.1),
      momentum.MomentumOptimizer(0.1, 0.1),
      rmsprop.RMSPropOptimizer(0.1)
  ]


def build_distributed_graph():
  batch_size = 4
  shape_0 = [batch_size, 5]
  shape_1 = [batch_size, 6]
  maxval = int(0x7FFF)

  server0 = server_lib.Server.create_local_server()
  server1 = server_lib.Server.create_local_server()
  cluster_def = cluster_pb2.ClusterDef()
  job = cluster_def.job.add()
  job.name = 'worker'
  job.tasks[0] = server0.target[len('grpc://'):]
  job.tasks[1] = server1.target[len('grpc://'):]

  config = config_pb2.ConfigProto(
      cluster_def=cluster_def,
      experimental=config_pb2.ConfigProto.Experimental(
          share_session_state_in_clusterspec_propagation=True,),
  )
  config.allow_soft_placement = False

  with ops.device('/job:worker/task:0'):
    feat_0 = random_ops.random_uniform(shape_0,
                                       maxval=maxval,
                                       dtype=dtypes.int64)
    feat_0 = array_ops.reshape(feat_0, (-1,))

    feat_1 = random_ops.random_uniform(shape_1,
                                       maxval=maxval,
                                       dtype=dtypes.int64)
    feat_1 = array_ops.reshape(feat_1, (-1,))

    var_0 = deo.get_variable(
        name='sp_var_0',
        devices=[
            '/job:worker/task:1',
        ],
        initializer=init_ops.random_normal_initializer(0, 0.005),
    )
    var_1 = deo.get_variable(
        name='sp_var_1',
        devices=[
            '/job:worker/task:1',
        ],
        initializer=init_ops.random_normal_initializer(0, 0.005),
    )
    var_list = [var_0, var_1]

    _, tw_0 = deo.embedding_lookup(
        params=var_0,
        ids=feat_0,
        name='sp_emb_0',
        return_trainable=True,
    )
    _, tw_1 = deo.embedding_lookup(
        params=var_1,
        ids=feat_1,
        name='sp_emb_1',
        return_trainable=True,
    )

    collapse_0 = array_ops.reshape(tw_0, (batch_size, -1))
    collapse_1 = array_ops.reshape(tw_1, (batch_size, -1))

    logits_0 = math_ops.reduce_sum(collapse_0, axis=1)
    logits_1 = math_ops.reduce_sum(collapse_1, axis=1)

    logits = math_ops.add(logits_0, logits_1)
    labels = array_ops.zeros((batch_size,), dtype=dtypes.float32)

    loss = math_ops.reduce_mean(
        nn_impl.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
        ))
    optimizers = get_multiple_optimizers()

    return server0, server1, config, var_list, optimizers, loss


def ps_worker_cluster(ps_num=1, worker_num=1):
  ps_servers, worker_servers = list(), list()
  cluster_def = cluster_pb2.ClusterDef()
  ps_jobs = cluster_def.job.add()
  ps_jobs.name = 'ps'
  worker_jobs = cluster_def.job.add()
  worker_jobs.name = 'worker'

  for i in range(ps_num):
    ps = server_lib.Server.create_local_server()
    ps.start()
    ps_jobs.tasks[i] = ps.target[len('grpc://'):]
    ps_servers.append(ps)

  for i in range(worker_num):
    worker = server_lib.Server.create_local_server()
    worker.start()
    worker_jobs.tasks[i] = worker.target[len('grpc://'):]
    worker_servers.append(worker)
  return ps_servers, worker_servers, cluster_def


default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    inter_op_parallelism_threads=2,
    intra_op_parallelism_threads=2,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


@test_util.deprecated_graph_mode_only
class TimestampRestrictPolicyUpdateTest(test.TestCase):

  def common_single_step_update_verification(self, optimizer):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      ids = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      slot_vars, policy, var, train_op =  \
          create_graph_and_policy(optimizer, ids, 'timestamp')
      update_op = policy.update()
      self.evaluate(variables.global_variables_initializer())
      sess.run([train_op, update_op])
      var_size, tstp_var_size, slot_var_size =  \
          get_size_info(sess, var, policy, slot_vars)

      self.assertAllEqual(var_size, 3)
      self.assertAllEqual(tstp_var_size, 3)
      self.assertAllEqual(slot_var_size, 3 * len(slot_vars))

      keys, _ = sess.run(var.export())
      keys.sort()
      self.assertAllEqual(keys, [1, 2, 3])

      for sv in slot_vars:
        keys, _ = sess.run(sv.export())
        keys.sort()
        self.assertAllEqual(keys, [1, 2, 3])

      keys, _ = sess.run(policy.tstp_var.export())
      keys.sort()
      self.assertAllEqual(keys, [1, 2, 3])

  def test_adadelta_restrictor_update(self):
    opt = adadelta.AdadeltaOptimizer()
    self.common_single_step_update_verification(opt)

  def test_adagrad_restrictor_update(self):
    opt = adagrad.AdagradOptimizer(0.1)
    self.common_single_step_update_verification(opt)

  def test_adagrad_da_restrictor_update(self):
    gstep = training_util.create_global_step()
    opt = adagrad_da.AdagradDAOptimizer(0.1, gstep)
    self.common_single_step_update_verification(opt)

  def test_adam_restrictor_update(self):
    opt = adam.AdamOptimizer(0.1)
    self.common_single_step_update_verification(opt)

  def test_ftrl_restrictor_update(self):
    opt = ftrl.FtrlOptimizer(0.1)
    self.common_single_step_update_verification(opt)

  def test_gradient_descent_restrictor_update(self):
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    self.common_single_step_update_verification(opt)

  def test_lamb_restrictor_update(self):
    opt = lamb.LAMBOptimizer()
    self.common_single_step_update_verification(opt)

  def test_momentum_restrictor_update(self):
    opt = momentum.MomentumOptimizer(0.1, 0.1)
    self.common_single_step_update_verification(opt)

  def test_proximal_adagrad_restrictor_update(self):
    opt = proximal_adagrad.ProximalAdagradOptimizer(0.1)
    self.common_single_step_update_verification(opt)

  def test_pgd_restrictor_update(self):
    opt = pgd.ProximalGradientDescentOptimizer(0.1)
    self.common_single_step_update_verification(opt)

  def test_rmsprop_restrictor_update(self):
    opt = rmsprop.RMSPropOptimizer(0.1)
    self.common_single_step_update_verification(opt)


@test_util.deprecated_graph_mode_only
class TimestampRestrictPolicyRestrictTest(test.TestCase):

  def common_single_step_restrict_verification(self, optimizer):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      ids = array_ops.placeholder(dtypes.int64)
      slot_vars, policy, var, train_op =  \
          create_graph_and_policy(optimizer, ids, 'timestamp')
      update_op = policy.update()
      restrict_op = policy.restrict(threshold=6, factor=1.2)
      restrict_op_oversize = policy.restrict(threshold=100, factor=1.2)

      self.evaluate(variables.global_variables_initializer())
      sess.run([train_op, update_op], feed_dict={ids: [0, 1, 2, 3, 4, 5]})
      time.sleep(2)
      sess.run([train_op, update_op], feed_dict={ids: [3, 4, 5, 6, 7, 8]})

      var_size, tstp_var_size, slot_var_size =  \
          get_size_info(sess, var, policy, slot_vars)
      self.assertAllEqual([var_size, tstp_var_size, slot_var_size],
                          [9, 9, 9 * len(slot_vars)])
      keys, _ = sess.run(var.export())
      keys.sort()
      self.assertAllEqual(keys, [0, 1, 2, 3, 4, 5, 6, 7, 8])
      for sv in slot_vars:
        keys, _ = sess.run(sv.export())
        keys.sort()
        self.assertAllEqual(keys, [0, 1, 2, 3, 4, 5, 6, 7, 8])
      keys, _ = sess.run(policy.tstp_var.export())
      keys.sort()
      self.assertAllEqual(keys, [0, 1, 2, 3, 4, 5, 6, 7, 8])

      sess.run(restrict_op_oversize)
      var_size, tstp_var_size, slot_var_size =  \
          get_size_info(sess, var, policy, slot_vars)
      self.assertAllEqual([var_size, tstp_var_size, slot_var_size],
                          [9, 9, 9 * len(slot_vars)])
      keys, _ = sess.run(var.export())
      keys.sort()
      self.assertAllEqual(keys, [0, 1, 2, 3, 4, 5, 6, 7, 8])
      for sv in slot_vars:
        keys, _ = sess.run(sv.export())
        keys.sort()
        self.assertAllEqual(keys, [0, 1, 2, 3, 4, 5, 6, 7, 8])
      keys, _ = sess.run(policy.tstp_var.export())
      keys.sort()
      self.assertAllEqual(keys, [0, 1, 2, 3, 4, 5, 6, 7, 8])

      sess.run(restrict_op)
      var_size, tstp_var_size, slot_var_size =  \
          get_size_info(sess, var, policy, slot_vars)
      keys, _ = sess.run(var.export())
      keys.sort()
      self.assertAllEqual(keys, [3, 4, 5, 6, 7, 8])
      for sv in slot_vars:
        keys, _ = sess.run(sv.export())
        keys.sort()
        self.assertAllEqual(keys, [3, 4, 5, 6, 7, 8])
      keys, _ = sess.run(policy.tstp_var.export())
      keys.sort()
      self.assertAllEqual(keys, [3, 4, 5, 6, 7, 8])

  def test_adadelta_restrict_on_policy(self):
    opt = adadelta.AdadeltaOptimizer()
    self.common_single_step_restrict_verification(opt)

  def test_adagrad_restrict_on_policy(self):
    opt = adagrad.AdagradOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)

  def test_adagrad_da_restrict_on_policy(self):
    gstep = training_util.create_global_step()
    opt = adagrad_da.AdagradDAOptimizer(0.1, gstep)
    self.common_single_step_restrict_verification(opt)

  def test_adam_restrict_on_policy(self):
    opt = adam.AdamOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)

  def test_ftrl_restrict_on_policy(self):
    opt = ftrl.FtrlOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)

  def test_gradient_descent_restrict_on_policy(self):
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)

  def test_lamb_restrict_on_policy(self):
    opt = lamb.LAMBOptimizer()
    self.common_single_step_restrict_verification(opt)

  def test_momentum_restrict_on_policy(self):
    opt = momentum.MomentumOptimizer(0.1, 0.1)
    self.common_single_step_restrict_verification(opt)

  def test_proximal_adagrad_restrict_on_policy(self):
    opt = proximal_adagrad.ProximalAdagradOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)

  def test_pgd_restrict_on_policy(self):
    opt = pgd.ProximalGradientDescentOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)

  def test_rmsprop_restrict_on_policy(self):
    opt = rmsprop.RMSPropOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)


@test_util.deprecated_graph_mode_only
class FrequencyRestrictPolicyUpdateTest(test.TestCase):

  def common_single_step_update_verification(self, optimizer):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      ids = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      slot_vars, policy, var, train_op =  \
          create_graph_and_policy(optimizer, ids, 'frequency')
      update_op = policy.update()

      self.evaluate(variables.global_variables_initializer())
      sess.run([train_op, update_op])
      var_size, freq_var_size, slot_var_size =  \
          get_size_info(sess, var, policy, slot_vars)

      self.assertAllEqual(var_size, 3)
      self.assertAllEqual(freq_var_size, 3)
      self.assertAllEqual(slot_var_size, 3 * len(slot_vars))

      keys, _ = sess.run(var.export())
      keys.sort()
      self.assertAllEqual(keys, [1, 2, 3])

      for sv in slot_vars:
        keys, _ = sess.run(sv.export())
        keys.sort()
        self.assertAllEqual(keys, [1, 2, 3])

      keys, _ = sess.run(policy.freq_var.export())
      keys.sort()
      self.assertAllEqual(keys, [1, 2, 3])

  def test_adadelta_restrictor_update(self):
    opt = adadelta.AdadeltaOptimizer()
    self.common_single_step_update_verification(opt)

  def test_adagrad_restrictor_update(self):
    opt = adagrad.AdagradOptimizer(0.1)
    self.common_single_step_update_verification(opt)

  def test_adagrad_da_restrictor_update(self):
    gstep = training_util.create_global_step()
    opt = adagrad_da.AdagradDAOptimizer(0.1, gstep)
    self.common_single_step_update_verification(opt)

  def test_adam_restrictor_update(self):
    opt = adam.AdamOptimizer(0.1)
    self.common_single_step_update_verification(opt)

  def test_ftrl_restrictor_update(self):
    opt = ftrl.FtrlOptimizer(0.1)
    self.common_single_step_update_verification(opt)

  def test_gradient_descent_restrictor_update(self):
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    self.common_single_step_update_verification(opt)

  def test_lamb_restrictor_update(self):
    opt = lamb.LAMBOptimizer()
    self.common_single_step_update_verification(opt)

  def test_momentum_restrictor_update(self):
    opt = momentum.MomentumOptimizer(0.1, 0.1)
    self.common_single_step_update_verification(opt)

  def test_proximal_adagrad_restrictor_update(self):
    opt = proximal_adagrad.ProximalAdagradOptimizer(0.1)
    self.common_single_step_update_verification(opt)

  def test_pgd_restrictor_update(self):
    opt = pgd.ProximalGradientDescentOptimizer(0.1)
    self.common_single_step_update_verification(opt)

  def test_rmsprop_restrictor_update(self):
    opt = rmsprop.RMSPropOptimizer(0.1)
    self.common_single_step_update_verification(opt)


@test_util.deprecated_graph_mode_only
class FrequencyRestrictPolicyRestrictTest(test.TestCase):

  def common_single_step_restrict_verification(self, optimizer):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      ids = array_ops.placeholder(dtypes.int64)
      slot_vars, policy, var, train_op =  \
          create_graph_and_policy(optimizer, ids, 'frequency')
      update_op = policy.update()
      restrict_op = policy.restrict(threshold=6, factor=1.2)
      restrict_op_oversize = policy.restrict(threshold=100, factor=1.2)

      self.evaluate(variables.global_variables_initializer())
      sess.run([train_op, update_op], feed_dict={ids: [0, 1, 2, 3, 4, 5]})
      sess.run([train_op, update_op], feed_dict={ids: [3, 4, 5, 6, 7, 8]})

      var_size, freq_var_size, slot_var_size =  \
          get_size_info(sess, var, policy, slot_vars)
      self.assertAllEqual([var_size, freq_var_size, slot_var_size],
                          [9, 9, 9 * len(slot_vars)])
      keys, _ = sess.run(var.export())
      keys.sort()
      self.assertAllEqual(keys, [0, 1, 2, 3, 4, 5, 6, 7, 8])
      for sv in slot_vars:
        keys, _ = sess.run(sv.export())
        keys.sort()
        self.assertAllEqual(keys, [0, 1, 2, 3, 4, 5, 6, 7, 8])
      keys, _ = sess.run(policy.freq_var.export())
      keys.sort()
      self.assertAllEqual(keys, [0, 1, 2, 3, 4, 5, 6, 7, 8])

      sess.run(restrict_op_oversize)
      var_size, freq_var_size, slot_var_size =  \
          get_size_info(sess, var, policy, slot_vars)
      self.assertAllEqual([var_size, freq_var_size, slot_var_size],
                          [9, 9, 9 * len(slot_vars)])
      keys, _ = sess.run(var.export())
      keys.sort()
      self.assertAllEqual(keys, [0, 1, 2, 3, 4, 5, 6, 7, 8])
      for sv in slot_vars:
        keys, _ = sess.run(sv.export())
        keys.sort()
        self.assertAllEqual(keys, [0, 1, 2, 3, 4, 5, 6, 7, 8])
      keys, _ = sess.run(policy.freq_var.export())
      keys.sort()
      self.assertAllEqual(keys, [0, 1, 2, 3, 4, 5, 6, 7, 8])

      sess.run(restrict_op)
      var_size, freq_var_size, slot_var_size =  \
          get_size_info(sess, var, policy, slot_vars)
      keys, _ = sess.run(var.export())
      self.assertTrue(all(x in keys for x in [3, 4, 5]))
      for sv in slot_vars:
        keys, _ = sess.run(sv.export())
        self.assertTrue(all(x in keys for x in [3, 4, 5]))
      keys, _ = sess.run(policy.freq_var.export())
      self.assertTrue(all(x in keys for x in [3, 4, 5]))

  def test_adadelta_restrict_on_policy(self):
    opt = adadelta.AdadeltaOptimizer()
    self.common_single_step_restrict_verification(opt)

  def test_adagrad_restrict_on_policy(self):
    opt = adagrad.AdagradOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)

  def test_adagrad_da_restrict_on_policy(self):
    gstep = training_util.create_global_step()
    opt = adagrad_da.AdagradDAOptimizer(0.1, gstep)
    self.common_single_step_restrict_verification(opt)

  def test_adam_restrict_on_policy(self):
    opt = adam.AdamOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)

  def test_ftrl_restrict_on_policy(self):
    opt = ftrl.FtrlOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)

  def test_gradient_descent_restrict_on_policy(self):
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)

  def test_lamb_restrict_on_policy(self):
    opt = lamb.LAMBOptimizer()
    self.common_single_step_restrict_verification(opt)

  def test_momentum_restrict_on_policy(self):
    opt = momentum.MomentumOptimizer(0.1, 0.1)
    self.common_single_step_restrict_verification(opt)

  def test_proximal_adagrad_restrict_on_policy(self):
    opt = proximal_adagrad.ProximalAdagradOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)

  def test_pgd_restrict_on_policy(self):
    opt = pgd.ProximalGradientDescentOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)

  def test_rmsprop_restrict_on_policy(self):
    opt = rmsprop.RMSPropOptimizer(0.1)
    self.common_single_step_restrict_verification(opt)


class VariableRestrictorTestBase(object):

  def common_run_context(self, var_list, opt_list, name):
    raise NotImplementedError

  def test_init_exception_invalid_policy(self):
    var_list = [
        deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    err = None
    with self.assertRaises(TypeError):
      _ = dvr.VariableRestrictor(var_list=var_list,
                                 optimizer_list=[opt],
                                 policy=None)

  def test_ops_with_var_and_adadelta(self):
    var_list = [
        deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    opt_list = [
        adadelta.AdadeltaOptimizer(),
    ]
    self.common_run_context(var_list, opt_list, name='adadelta_test')

  def test_ops_with_var_and_adagrad(self):
    var_list = [
        deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    opt_list = [
        adagrad.AdagradOptimizer(0.1),
    ]
    self.common_run_context(var_list, opt_list, name='adagrad_test')

  def test_ops_with_var_and_adagrad_da(self):
    var_list = [
        deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    gstep = training_util.create_global_step()
    opt_list = [
        adagrad_da.AdagradDAOptimizer(0.1, gstep),
    ]
    self.common_run_context(var_list, opt_list, name='adagrad_da_test')

  def test_ops_with_var_and_adam(self):
    var_list = [
        deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    opt_list = [
        adam.AdamOptimizer(0.1),
    ]
    self.common_run_context(var_list, opt_list, name='adam_test')

  def test_ops_with_var_and_ftrl(self):
    var_list = [
        deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    opt_list = [
        ftrl.FtrlOptimizer(0.1),
    ]
    self.common_run_context(var_list, opt_list, name='ftrl_test')

  def test_ops_with_var_and_gradient_descent(self):
    var_list = [
        deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    opt_list = [
        gradient_descent.GradientDescentOptimizer(0.1),
    ]
    self.common_run_context(var_list, opt_list, name='gradient_descent_test')

  def test_ops_with_var_and_lamb(self):
    var_list = [
        deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    opt_list = [
        lamb.LAMBOptimizer(),
    ]
    self.common_run_context(var_list, opt_list, name='lamb_test')

  def test_ops_with_var_and_momentum(self):
    var_list = [
        deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    opt_list = [
        momentum.MomentumOptimizer(0.1, 0.1),
    ]
    self.common_run_context(var_list, opt_list, name='momentum_test')

  def test_ops_with_var_and_proximal_adagrad(self):
    var_list = [
        deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    opt_list = [
        proximal_adagrad.ProximalAdagradOptimizer(0.1),
    ]
    self.common_run_context(var_list, opt_list, name='proximal_adagrad_test')

  def test_ops_with_var_and_pgd(self):
    var_list = [
        deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    opt_list = [
        pgd.ProximalGradientDescentOptimizer(0.1),
    ]
    self.common_run_context(var_list, opt_list, name='pgd_test')

  def test_ops_with_var_and_rmsprop(self):
    var_list = [
        deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    opt_list = [
        rmsprop.RMSPropOptimizer(0.1),
    ]
    self.common_run_context(var_list, opt_list, name='rmsprop_test')

  def test_ops_with_vars_and_optimizers(self):
    num_vars = 4
    var_list = [
        deo.get_variable('sp_var_' + str(_i), initializer=0.0, dim=2)
        for _i in range(num_vars)
    ]
    opt_list = get_multiple_optimizers()
    self.common_run_context(var_list, opt_list, name='multiple_optimizer_test')

  def test_ops_with_various_variables_and_optimizers(self):
    key_dtypes = [dtypes.int64]
    n = 0
    var_list = list()
    for _kt in key_dtypes:
      _var = deo.get_variable('sp_var_' + str(n),
                              initializer=0.0,
                              key_dtype=_kt,
                              dim=8)
      var_list.append(_var)
      n += 1
    opt_list = get_multiple_optimizers()
    self.common_run_context(var_list, opt_list, name='cross_context_test')


@test_util.deprecated_graph_mode_only
class VariableRestrictorRestoreFromCheckpointTest(test.TestCase,
                                                  VariableRestrictorTestBase):

  def common_run_context(self, var_list, opt_list, name):
    save_dir = os.path.join(self.get_temp_dir(), 'save_restore')
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), 'restrict')

    batch_size = 2
    sample_length = 3
    emb_domain_list = list()
    tws = list()

    for _v in var_list:
      ids = random_ops.random_uniform((batch_size, sample_length),
                                      maxval=1000000,
                                      dtype=_v.key_dtype)
      ids = array_ops.reshape(ids, (-1,))

      _, tw = deo.embedding_lookup(_v, ids, return_trainable=True)
      tws.append(tw)
      _collapse = array_ops.reshape(tw, (batch_size, -1))
      _logits = math_ops.reduce_sum(_collapse, axis=1)
      _logits = math_ops.cast(_logits, dtypes.float32)
      emb_domain_list.append(_logits)
    logits = math_ops.add_n(emb_domain_list)

    labels = array_ops.zeros((batch_size,), dtype=dtypes.float32)
    loss = math_ops.reduce_mean(
        nn_impl.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
        ))

    _train_ops = list()
    for _opt in opt_list:
      _train_ops.append(_opt.minimize(loss))
    train_op = control_flow_ops.group(_train_ops)

    restrictor = dvr.VariableRestrictor(var_list=var_list,
                                        optimizer_list=opt_list)

    policies = list(itertools.chain(*restrictor.policy_group.values()))
    tstp_vars = [policy.tstp_var for policy in policies]
    slot_vars = list()
    for tw in tws:
      for opt in opt_list:
        slot_vars += select_slot_vars(tw, opt)

    update_op = restrictor.update()

    threshold = int(batch_size * sample_length * 1.5)
    factor = 1.2
    restrict_op = restrictor.restrict(threshold=threshold, factor=factor)
    saver = saver_lib.Saver()

    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      self.evaluate(variables.global_variables_initializer())
      n, MAX_ITER = 0, 1000
      while n < MAX_ITER:
        sess.run([train_op, update_op])
        if all(
            self.evaluate(var.size()) > threshold * factor for var in var_list):
          break

      rt_save_path = saver.save(sess, save_path)
      self.assertAllEqual(rt_save_path, save_path)
      sess.close()

    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      self.evaluate(variables.global_variables_initializer())
      saver.restore(sess, save_path)
      s1 = self.evaluate([var.size() for var in var_list])
      s2 = self.evaluate([tv.size() for tv in tstp_vars])
      s3 = self.evaluate([sv.size() for sv in slot_vars])

      self.assertAllGreater(s1, threshold * factor)
      self.assertAllGreater(s2, threshold * factor)
      if s3:
        self.assertAllGreater(s3, threshold * factor)

      saver.save(sess, save_path)

      sess.run(restrict_op)
      s1 = self.evaluate([var.size() for var in var_list])
      s2 = self.evaluate([tv.size() for tv in tstp_vars])
      s3 = self.evaluate([sv.size() for sv in slot_vars])

      self.assertAllLess(s1, threshold * factor + 1)
      self.assertAllLess(s2, threshold * factor + 1)
      if s3:
        self.assertAllLess(s3, threshold * factor + 1)
      sess.close()


@test_util.deprecated_graph_mode_only
class VariableRestrictorDistributedTest(test.TestCase,
                                        VariableRestrictorTestBase):

  def test_tstp_table_ops_placement_distributed(self):
    server0, server1, config,  \
        var_list, optimizers, loss = build_distributed_graph()

    _ = [_opt.minimize(loss) for _opt in optimizers]
    restrictor = dvr.VariableRestrictor(var_list=var_list,
                                        optimizer_list=optimizers,
                                        policy=dvr.TimestampRestrictPolicy)
    _ = restrictor.update()
    threshold, factor = 125, 1.2
    _ = restrictor.restrict(threshold=threshold, factor=factor, sloppy=False)

    with session.Session(server0.target, config=config) as sess:
      graph = sess.graph
      graph_ops = graph.get_operations()
      op_names = [_op.name for _op in graph_ops]
      for _name in op_names:
        if 'table' in _name.lower():
          op = graph.get_operation_by_name(_name)
          dev_str = op.device
          self.assertAllEqual(dev_str, '/job:worker/task:1')

  def test_freq_table_ops_placement_distributed(self):
    server0, server1, config,  \
        var_list, optimizers, loss = build_distributed_graph()

    _ = [_opt.minimize(loss) for _opt in optimizers]
    restrictor = dvr.VariableRestrictor(var_list=var_list,
                                        optimizer_list=optimizers,
                                        policy=dvr.FrequencyRestrictPolicy)
    _ = restrictor.update()
    threshold, factor = 125, 1.2
    _ = restrictor.restrict(threshold=threshold, factor=factor)

    with session.Session(server0.target, config=config) as sess:
      graph = sess.graph
      graph_ops = graph.get_operations()
      op_names = [_op.name for _op in graph_ops]
      for _name in op_names:
        if 'table' in _name.lower():
          op = graph.get_operation_by_name(_name)
          dev_str = op.device
          self.assertAllEqual(dev_str, '/job:worker/task:1')

  def common_run_context(self, var_list, opt_list, name):
    batch_size = 2
    sample_length = 3
    emb_domain_list = list()
    tws = list()

    cluster = ps_worker_cluster(ps_num=2)
    ps_servers, worker_servers, cluster_def = cluster

    config = config_pb2.ConfigProto(
        cluster_def=cluster_def,
        experimental=config_pb2.ConfigProto.Experimental(
            share_session_state_in_clusterspec_propagation=True,),
        allow_soft_placement=False,
        inter_op_parallelism_threads=2,
        intra_op_parallelism_threads=2,
        gpu_options=config_pb2.GPUOptions(allow_growth=True),
    )

    dev_placement = device_setter.replica_device_setter(
        ps_tasks=2,
        ps_device='/job:ps',
        worker_device='/job:worker',
        cluster=cluster_def,
    )

    with ops.device(dev_placement):
      shared_var_0 = deo.get_variable('distributed_sp_var_0',
                                      initializer=0.0,
                                      devices=['/job:worker/task:0'],
                                      dim=8)
      shared_var_1 = deo.get_variable('distributed_sp_var_1',
                                      initializer=0.0,
                                      devices=['/job:worker/task:0'],
                                      dim=4)
      opt_list = get_multiple_optimizers()

      distributed_var_list = [shared_var_0, shared_var_1]
      for _v in distributed_var_list:
        ids = random_ops.random_uniform((batch_size, sample_length),
                                        maxval=1000000,
                                        dtype=_v.key_dtype)
        ids = array_ops.reshape(ids, (-1,))

        _, tw = deo.embedding_lookup(_v, ids, return_trainable=True)
        tws.append(tw)
        _collapse = array_ops.reshape(tw, (batch_size, -1))
        _logits = math_ops.reduce_sum(_collapse, axis=1)
        _logits = math_ops.cast(_logits, dtypes.float32)
        emb_domain_list.append(_logits)
      logits = math_ops.add_n(emb_domain_list)

      labels = array_ops.zeros((batch_size,), dtype=dtypes.float32)
      loss = math_ops.reduce_mean(
          nn_impl.sigmoid_cross_entropy_with_logits(
              logits=logits,
              labels=labels,
          ))

      _train_ops = list()
      for _opt in opt_list:
        _train_ops.append(_opt.minimize(loss))
      train_op = control_flow_ops.group(_train_ops)

      restrictor = dvr.VariableRestrictor(var_list=distributed_var_list,
                                          optimizer_list=opt_list)
      update_op = restrictor.update()
      threshold = int(batch_size * sample_length * 1.5)
      factor = 1.2
      restrict_op = restrictor.restrict(threshold=threshold, factor=factor)

    policies = list(itertools.chain(*restrictor.policy_group.values()))
    tstp_vars = [policy.tstp_var for policy in policies]
    slot_vars = list()
    for tw in tws:
      for opt in opt_list:
        slot_vars += select_slot_vars(tw, opt)

    with session.Session(worker_servers[0].target, config=config) as sess:
      sess.run(variables.global_variables_initializer())
      n, MAX_ITER = 0, 1000
      while n < MAX_ITER:
        sess.run([train_op, update_op])
        if all(
            sess.run(var.size()) > threshold * factor
            for var in distributed_var_list):
          break

      s1 = sess.run([var.size() for var in distributed_var_list])
      s2 = sess.run([tv.size() for tv in tstp_vars])
      s3 = sess.run([sv.size() for sv in slot_vars])

      self.assertAllGreater(s1, threshold * factor)
      self.assertAllGreater(s2, threshold * factor)
      if s3:
        self.assertAllGreater(s3, threshold * factor)

      sess.run(restrict_op)
      s1 = sess.run([var.size() for var in distributed_var_list])
      s2 = sess.run([tv.size() for tv in tstp_vars])
      s3 = sess.run([sv.size() for sv in slot_vars])

      self.assertAllLess(s1, threshold * factor + 1)
      self.assertAllLess(s2, threshold * factor + 1)
      if s3:
        self.assertAllLess(s3, threshold * factor + 1)
      sess.close()


if __name__ == '__main__':
  os.environ['TF_HASHTABLE_INIT_SIZE'] = str(64 * 1024)
  test.main()
