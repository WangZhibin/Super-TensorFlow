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
"""unit tests of dynamic embedding ops
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import numpy as np
import os
import six
import tempfile

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import training_util
from tensorflow.python.training import warm_starting_util as ws_util
from tensorflow.python.training.tracking import util as tracking_util
from tensorflow.python.yunfan_ops import dynamic_embedding_ops as deo


# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
def _type_converter(tf_type):
  mapper = {
      dtypes.int32: np.int32,
      dtypes.int64: np.int64,
      dtypes.float32: np.float,
      dtypes.float64: np.float64,
      dtypes.string: np.str
  }
  return mapper[tf_type]


def _get_devices():
  return ['/gpu:0' if test_util.is_gpu_available() else '/cpu:0']


def _check_device(op, expexted_device='gpu'):
  return expexted_device.upper() in op.device


def _test_dir(temp_dir, test_name):
  """Create an empty dir to use for tests.

  Args:
    temp_dir: Tmp directory path.
    test_name: Name of the test.

  Returns:
    Absolute path to the test directory.
  """
  test_dir = os.path.join(temp_dir, test_name)
  if os.path.isdir(test_dir):
    for f in glob.glob('%s/*' % test_dir):
      os.remove(f)
  else:
    os.makedirs(test_dir)
  return test_dir


def _get_meta_file(ckpt_dir):
  for fname in os.listdir(ckpt_dir):
    if fname.endswith(".meta"):
      return os.path.join(ckpt_dir, fname)
  else:
    raise ValueError("No meta file found in {}.".format(ckpt_dir))


def _write_checkpoint(test, sess):
  saver = saver_lib.Saver()
  ckpt_prefix = os.path.join(test.get_temp_dir(), "model")
  saver.save(sess, ckpt_prefix, global_step=0)


def _create_dynamic_shape_tensor(max_len=100,
                                 min_len=2,
                                 min_val=0x0000f00000000001,
                                 max_val=0x0000f00000000020,
                                 dtype=np.int64):

  def _func():
    length = np.random.randint(min_len, max_len)
    if dtype == np.str:
      tensor = np.random.randint(min_val, max_val, max_len, dtype=np.int64)
      tensor = np.array(map(str, tensor[0:length]), dtype=dtype)
    else:
      tensor = np.random.randint(min_val, max_val, max_len, dtype=np.int64)
      tensor = np.array(tensor[0:length], dtype=dtype)
    return tensor

  return _func


def _next_run_step_config(keys_type_list=[dtypes.int64]):
  run_id = 0
  for num_shards, key_dtype, value_dtype, init_mode, dim, run_step in \
      itertools.product(
        [2],
        keys_type_list,
        [dtypes.float32],
        ['constant'],
        [1, 10],
        [10]):
    run_id += 1
    yield run_id, num_shards, key_dtype, value_dtype, init_mode, dim, run_step


default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


class TestGraph(object):

  def __init__(self,
               key_dtype,
               value_dtype,
               dim,
               num_shards,
               var_name,
               devar_name,
               run_id,
               x=None):
    self.key_dtype = key_dtype
    self.value_dtype = value_dtype
    self.dim = dim

    # common define
    init_ids = [0, 1, 2]
    init_vals = np.random.rand(3, self.dim)
    raw_ids = [1]
    if x is None:
      self.x = constant_op.constant(np.random.rand(self.dim, len(raw_ids)),
                                    dtype=self.value_dtype)
    else:
      self.x = ops.convert_to_tensor(x, dtype=self.value_dtype)

    # variable graph
    self.var = resource_variable_ops.ResourceVariable(name='t2020-' + var_name +
                                                      str(run_id),
                                                      initial_value=init_vals,
                                                      dtype=self.value_dtype)
    ids = constant_op.constant(raw_ids, dtype=self.key_dtype)
    self.var_lookup = embedding_ops.embedding_lookup([self.var], ids)
    self.var_pred = math_ops.matmul(self.var_lookup, self.x)
    self.var_loss = self.var_pred * self.var_pred
    self.var_opt_op = adam.AdamOptimizer(1.0).minimize(self.var_loss)

    # deo variable graph
    self.devar = deo.get_variable(name='t2020-' + devar_name + str(run_id),
                                  key_dtype=self.key_dtype,
                                  value_dtype=self.value_dtype,
                                  devices=_get_devices() * num_shards,
                                  initializer=1.,
                                  dim=dim)
    self.devar_init_op = self.devar.upsert(
        constant_op.constant(init_ids, dtype=self.key_dtype),
        constant_op.constant(init_vals, dtype=self.value_dtype))
    self.devar_lookup, _ = deo.embedding_lookup([self.devar],
                                                ids,
                                                return_trainable=True)
    self.devar_pred = math_ops.matmul(self.devar_lookup, self.x)
    self.devar_loss = self.devar_pred * self.devar_pred
    self.devar_opt_op = adam.AdamOptimizer(1.0).minimize(self.devar_loss)


@test_util.run_all_in_graph_and_eager_modes
class DynamicEmbeddingVariableSaveRestoreTest(test.TestCase):

  def test_save_restore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(config=default_config, graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0.], [1.], [2.]], dtypes.float32)
      table = deo.Variable(key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           initializer=-1.,
                           name='t1',
                           dim=1)

      save = saver_lib.Saver(var_list=[v0, v1, table])
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

      del table

    with self.session(config=default_config, graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      table = deo.Variable(name="t1",
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           initializer=-1.,
                           dim=1,
                           checkpoint=True)
      self.evaluate(
          table.upsert(constant_op.constant([0, 1], dtypes.int64),
                       constant_op.constant([[12.], [24.]], dtypes.float32)))
      size_op = table.size()
      self.assertAllEqual(2, self.evaluate(size_op))

      save = saver_lib.Saver(var_list=[v0, v1, table])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual([10.0], self.evaluate(v0))
      self.assertEqual([20.0], self.evaluate(v1))

      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([5, 0, 1, 2, 6], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([[-1.], [0.], [1.], [2.], [-1.]],
                          self.evaluate(output))

      del table

  def test_save_restore_only_table(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(config=default_config,
                      graph=ops.Graph(),
                      use_gpu=test_util.is_gpu_available()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = deo.Variable(dtypes.int64,
                           dtypes.int32,
                           name='t1',
                           initializer=default_val,
                           checkpoint=True)

      save = saver_lib.Saver([table])
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)
      del table

    with self.session(config=default_config,
                      graph=ops.Graph(),
                      use_gpu=test_util.is_gpu_available()) as sess:
      default_val = -1
      table = deo.Variable(dtypes.int64,
                           dtypes.int32,
                           name='t1',
                           initializer=default_val,
                           checkpoint=True)
      self.evaluate(
          table.upsert(constant_op.constant([0, 2], dtypes.int64),
                       constant_op.constant([[12], [24]], dtypes.int32)))
      self.assertAllEqual(2, self.evaluate(table.size()))

      save = saver_lib.Saver([table._tables[0]])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.

      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 2, 3, 4], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([[0], [1], [2], [-1], [-1]], self.evaluate(output))
      del table

  def test_training_save_restore(self):
    # embedding_lookup does not work in eager mode when num_shards is more than 1.
    ops.disable_eager_execution()
    keys_type_list = [dtypes.int64] if test_util.is_gpu_available() else [
        dtypes.int64, dtypes.string
    ]
    for run_id, num_shards, key_dtype, value_dtype, _, dim, run_step \
        in _next_run_step_config(keys_type_list):
      save_dir = os.path.join(self.get_temp_dir(), "save_restore")
      save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

      ids = script_ops.py_func(
          _create_dynamic_shape_tensor(dtype=_type_converter(key_dtype)),
          inp=[],
          Tout=key_dtype,
          stateful=True)

      params = deo.get_variable(name="params-test-0915-" + str(run_id),
                                key_dtype=key_dtype,
                                value_dtype=value_dtype,
                                devices=_get_devices() * num_shards,
                                initializer=init_ops.random_normal_initializer(
                                    0.0, 0.01),
                                dim=dim)
      _, var0 = deo.embedding_lookup(params, ids, return_trainable=True)
      loss = lambda: var0 * var0

      params_keys, params_vals = params.export()
      opt = adam.AdamOptimizer(0.3)
      mini = opt.minimize(loss, var_list=[var0])
      opt_slots = [opt.get_slot(var0, _s) for _s in opt.get_slot_names()]
      _saver = saver_lib.Saver([params] + [_s.params for _s in opt_slots])

      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        self.evaluate(variables.global_variables_initializer())
        for _i in range(run_step):
          self.evaluate([mini])
        size_before_saved = self.evaluate(params.size())
        np_params_keys_before_saved = self.evaluate(params_keys)
        np_params_vals_before_saved = self.evaluate(params_vals)
        opt_slots_kv_pairs = [_s.params.export() for _s in opt_slots]
        np_slots_kv_pairs_before_saved = [
            self.evaluate(_kv) for _kv in opt_slots_kv_pairs
        ]
        _saver.save(sess, save_path)

      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        self.evaluate(variables.global_variables_initializer())
        self.assertAllEqual(0, self.evaluate(params.size()))

        _saver.restore(sess, save_path)
        params_keys_restored, params_vals_restored = params.export()
        size_after_restored = self.evaluate(params.size())
        np_params_keys_after_restored = self.evaluate(params_keys_restored)
        np_params_vals_after_restored = self.evaluate(params_vals_restored)

        opt_slots_kv_pairs_restored = [_s.params.export() for _s in opt_slots]
        np_slots_kv_pairs_after_restored = [
            self.evaluate(_kv) for _kv in opt_slots_kv_pairs_restored
        ]
        self.assertAllEqual(size_before_saved, size_after_restored)
        self.assertAllEqual(np.sort(np_params_keys_before_saved),
                            np.sort(np_params_keys_after_restored))
        self.assertAllEqual(np.sort(np_params_vals_before_saved, axis=0),
                            np.sort(np_params_vals_after_restored, axis=0))
        for pairs_before, pairs_after in zip(np_slots_kv_pairs_before_saved,
                                             np_slots_kv_pairs_after_restored):

          self.assertAllEqual(np.sort(pairs_before[0], axis=0),
                              np.sort(pairs_after[0], axis=0))
          self.assertAllEqual(np.sort(pairs_before[1], axis=0),
                              np.sort(pairs_after[1], axis=0))
        if test_util.is_gpu_available():
          self.assertTrue(_check_device(params.tables[0].resource_handle,
                                        "GPU"))

  def test_import_meta_graph_from_checkpoint(self):
    for run_id, num_shards, k_dtype, d_dtype, init_mode, dim, run_step \
      in _next_run_step_config():
      with ops.Graph().as_default() as g:
        with self.session(graph=g,
                          use_gpu=test_util.is_gpu_available(),
                          config=default_config) as sess:
          graph = TestGraph(k_dtype, d_dtype, dim, num_shards, 'var', 'devar',
                            run_id)
          var_loss_name = graph.var_loss.name
          devar_loss_name = graph.devar_loss.name
          self.evaluate(variables.global_variables_initializer())
          sess.run([graph.devar_init_op])
          for _ in range(run_step):
            sess.run([graph.var_opt_op, graph.devar_opt_op])
          prev_var_loss, prev_devar_loss = sess.run(
              [var_loss_name, devar_loss_name])
          prev_devar_names = sorted([
              v.name for v in ops.get_collection(
                  ops.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES)
          ])
          self.assertAllCloseAccordingToType(
              prev_var_loss,
              prev_devar_loss,
              msg="Cond:{},{},{},{},{},{}".format(num_shards, k_dtype, d_dtype,
                                                  init_mode, dim, run_step))
          _write_checkpoint(self, sess)

      with ops.Graph().as_default() as g:
        with self.session(graph=g,
                          use_gpu=test_util.is_gpu_available(),
                          config=default_config) as sess:
          ckpt_dir = self.get_temp_dir()
          saver = saver_lib.import_meta_graph(_get_meta_file(ckpt_dir))
          devar_names = sorted([
              n for n in ops.get_collection(
                  ops.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES)
          ])
          self.assertAllEqual(devar_names, prev_devar_names)
          saver.restore(sess, saver_lib.latest_checkpoint(ckpt_dir))
          prev_var_loss, prev_devar_loss = sess.run(
              [var_loss_name, devar_loss_name])
          var_loss, devar_loss = sess.run([var_loss_name, devar_loss_name])
      self.assertAllCloseAccordingToType(var_loss,
                                         prev_var_loss,
                                         msg="Cond:{},{},{},{},{},{}".format(
                                             num_shards, k_dtype, d_dtype,
                                             init_mode, dim, run_step))
      self.assertAllCloseAccordingToType(devar_loss,
                                         prev_devar_loss,
                                         msg="Cond:{},{},{},{},{},{}".format(
                                             num_shards, k_dtype, d_dtype,
                                             init_mode, dim, run_step))

  def test_fail_to_write_checkpoint_for_loaded_meta_graph(self):
    run_id, num_shards, k_dtype, d_dtype, init_mode, dim, run_step = \
      list(_next_run_step_config())[0]
    with ops.Graph().as_default() as g:
      with self.session(graph=g,
                        use_gpu=test_util.is_gpu_available(),
                        config=default_config) as sess:
        graph = TestGraph(k_dtype, d_dtype, dim, num_shards, 'var', 'devar',
                          run_id)
        self.evaluate(variables.global_variables_initializer())
        sess.run([graph.devar_init_op])
        sess.run([graph.var_opt_op, graph.devar_opt_op])
        _write_checkpoint(self, sess)

    with ops.Graph().as_default() as g:
      with self.session(graph=g,
                        use_gpu=test_util.is_gpu_available(),
                        config=default_config) as sess:
        ckpt_dir = self.get_temp_dir()
        saver = saver_lib.import_meta_graph(_get_meta_file(ckpt_dir))
        saver.restore(sess, saver_lib.latest_checkpoint(ckpt_dir))
        with self.assertRaises(TypeError) as te:
          _write_checkpoint(self, sess)
          self.assertStartsWith(te.exception, "Can't convert Operation")


@test_util.run_all_in_graph_and_eager_modes
class WarmStartingUtilTest(test.TestCase):

  def _add_devar(self, name, dim):
    var = deo.get_variable(name,
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           dim=dim)
    return var

  def _add_and_initialize_devar(self, name, keys, values, dim):
    var = deo.get_variable(name,
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           dim=dim)
    self.evaluate(
        var.upsert(constant_op.constant(keys, dtypes.int64),
                   constant_op.constant(values, dtypes.float32)))
    return var

  def _export_sorted_keys_and_values(self, devar):
    exported_keys, exported_values = devar.export()
    return np.sort(self.evaluate(exported_keys)), \
           np.sort(self.evaluate(exported_values), axis=0)

  def test_basic_devars(self):
    # Save checkpoint from which to warm-start.
    dim1, dim2 = 10, 20
    keys1, keys2 = [0, 1, 2, 3], [4, 5, 6]
    values1, values2 = [[k] * dim1 for k in keys1], [[k] * dim2 for k in keys2]
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        prev_var1 = self._add_and_initialize_devar("old_scope/var1", keys1,
                                                   values1, dim1)
        prev_var2 = self._add_and_initialize_devar("old_scope/var2", keys2,
                                                   values2, dim2)
        _write_checkpoint(self, sess)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        var1 = self._add_devar("new_scope/var1", dim1)
        self.assertAllEqual(0, self.evaluate(var1.size()))
        checkpoint_utils.init_mht_saveable_from_checkpoint(
            self.get_temp_dir(), {
                prev_table.saveable.name: table.saveable
                for prev_table, table in zip(prev_var1.tables, var1.tables)
            })
        self.evaluate(deo.dynamic_embedding_variables_initializer())
        self.assertAllEqual(4, self.evaluate(var1.size()))
        keys, values = self._export_sorted_keys_and_values(var1)
        self.assertAllEqual(keys1, keys)
        self.assertAllEqual(values1, values)

        var2 = self._add_devar("new_scope/var2", dim2)
        self.assertAllEqual(0, self.evaluate(var2.size()))
        checkpoint_utils.init_mht_saveable_from_checkpoint(
            self.get_temp_dir(), {
                prev_table.saveable.name: table.saveable
                for prev_table, table in zip(prev_var2.tables, var2.tables)
            })
        self.evaluate(deo.dynamic_embedding_variables_initializer())
        self.assertAllEqual(3, self.evaluate(var2.size()))
        keys, values = self._export_sorted_keys_and_values(var2)
        self.assertAllEqual(keys2, keys)
        self.assertAllEqual(values2, values)

  def test_both_vars_and_devars(self):
    # Save checkpoint from which to warm-start.
    dim1, dim2 = 10, 20
    keys1, keys2 = [0, 1, 2, 3], [4, 5, 6]
    values1, values2 = [[k] * dim1 for k in keys1], [[k] * dim2 for k in keys2]
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        var = variable_scope.get_variable(
            "v1", shape=[10, 1], initializer=init_ops.ones_initializer())
        self.evaluate(variables.global_variables_initializer())
        prev_int_val = self.evaluate(var)
        self.assertAllEqual(np.ones([10, 1]), prev_int_val)
        devar1 = self._add_and_initialize_devar("devar1", keys1, values1, dim1)
        self.assertAllEqual(4, self.evaluate(devar1.size()))
        devar2 = self._add_and_initialize_devar("devar2", keys2, values2, dim2)
        self.assertAllEqual(3, self.evaluate(devar2.size()))
        _write_checkpoint(self, sess)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        # Initialize with zeros.
        var = variable_scope.get_variable(
            "v1", shape=[10, 1], initializer=init_ops.zeros_initializer())
        devar1 = self._add_devar("devar1", dim1)
        devar2 = self._add_devar("devar2", dim2)
        self.assertAllEqual(0, self.evaluate(devar1.size()))
        self.assertAllEqual(0, self.evaluate(devar2.size()))
        ws_util.warm_start(self.get_temp_dir(),
                           vars_to_warm_start=[var, devar1])
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(deo.dynamic_embedding_variables_initializer())
        # Verify weights were correctly warm-started (init overridden to ones).
        self.assertAllEqual(var.eval(), prev_int_val)
        self.assertAllEqual(4, self.evaluate(devar1.size()))
        keys, values = self._export_sorted_keys_and_values(devar1)
        self.assertAllEqual(keys1, keys)
        self.assertAllEqual(values1, values)
        self.assertAllEqual(0, self.evaluate(devar2.size()))

  def test_list_of_strings(self):
    # Save checkpoint from which to warm-start.
    dim1, dim2 = 10, 20
    keys1, keys2 = [0, 1, 2, 3], [4, 5, 6]
    values1, values2 = [[k] * dim1 for k in keys1], [[k] * dim2 for k in keys2]
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        var = variable_scope.get_variable(
            "v1", shape=[10, 1], initializer=init_ops.ones_initializer())
        self.evaluate(variables.global_variables_initializer())
        prev_int_val = self.evaluate(var)
        devar1 = self._add_and_initialize_devar("devar1", keys1, values1, dim1)
        self.assertAllEqual(4, self.evaluate(devar1.size()))
        devar2 = self._add_and_initialize_devar("devar2", keys2, values2, dim2)
        self.assertAllEqual(3, self.evaluate(devar2.size()))
        _write_checkpoint(self, sess)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        # Initialize with zeros.
        var = variable_scope.get_variable(
            "v1", shape=[10, 1], initializer=init_ops.zeros_initializer())
        devar1 = self._add_devar("devar1", dim1)
        devar2 = self._add_devar("devar2", dim2)
        ws_util.warm_start(self.get_temp_dir(),
                           vars_to_warm_start=["v1", "devar1"])
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(deo.dynamic_embedding_variables_initializer())
        # Verify weights were correctly warm-started (init overridden to ones).
        self.assertAllEqual(var.eval(), prev_int_val)
        self.assertAllEqual(4, self.evaluate(devar1.size()))
        keys, values = self._export_sorted_keys_and_values(devar1)
        self.assertAllEqual(keys1, keys)
        self.assertAllEqual(values1, values)
        self.assertAllEqual(0, self.evaluate(devar2.size()))

  def test_list_of_regexes(self):
    # Save checkpoint from which to warm-start.
    dim = 2
    keys = [0, 1]
    values = [[0, 0], [1, 1]]
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        with variable_scope.variable_scope("outer"):
          self._add_and_initialize_devar("v1", keys, values, dim)
          self._add_and_initialize_devar("v1/Momentum", keys, values, dim)
          self._add_and_initialize_devar("v2", keys, values, dim)
          self._add_and_initialize_devar("v2/Momentum", keys, values, dim)
          _write_checkpoint(self, sess)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        with variable_scope.variable_scope("outer"):
          v1 = self._add_devar("v1", dim)
          v1_momentum = self._add_devar("v1/Momentum", dim)
          v2 = self._add_devar("v2", dim)
          v2_momentum = self._add_devar("v2/Momentum", dim)
          self.assertAllEqual(0, self.evaluate(v1.size()))

          ws_util.warm_start(
              self.get_temp_dir(),
              # This warm-starts both v1 and v1/Momentum, but only
              # v2 (and not v2/Momentum).
              vars_to_warm_start=["outer/v1", "outer/v2$"])
          self.evaluate(deo.dynamic_embedding_variables_initializer())
          # Verify the selection of weights were correctly warm-started (init
          # overridden to ones).
          for v in [v1, v1_momentum, v2]:
            keys, values = self._export_sorted_keys_and_values(v)
            self.assertAllEqual(keys, keys)
            self.assertAllEqual(values, values)
          self.assertAllEqual(0, self.evaluate(v2_momentum.size()))

  def test_use_var_name_to_prev_var_name(self):
    # Save checkpoint from which to warm-start.
    dim = 2
    keys = [0, 1]
    values = [[0, 0], [1, 1]]
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        with variable_scope.variable_scope("old_outer"):
          prev_v1 = self._add_and_initialize_devar("v1", keys, values, dim)
          prev_v2 = self._add_and_initialize_devar("v2", keys, values, dim)
          _write_checkpoint(self, sess)

    # New graph, new session with warm-starting.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
        with variable_scope.variable_scope("new_outer"):
          v1 = self._add_devar("v1", dim)
          v2 = self._add_devar("v2", dim)
          self.assertAllEqual(0, self.evaluate(v1.size()))

          ms_name_to_prev_ms_name = {}
          for v, prev_v in zip([v1, v2], [prev_v1, prev_v2]):
            for table, prev_table in zip(v.tables, prev_v.tables):
              ms_name_to_prev_ms_name[table.saveable.full_name] = \
                prev_table.saveable.full_name

          # Unfound MutableHashTable._Saveable names raises ValueError
          self.assertRaises(ValueError,
                            ws_util.warm_start,
                            self.get_temp_dir(),
                            vars_to_warm_start=["new_outer/v1", "new_outer/v2"])

          # Unused previous MutableHashTable._Saveable names raises ValueError.
          self.assertRaises(ValueError,
                            ws_util.warm_start,
                            self.get_temp_dir(),
                            vars_to_warm_start=["new_outer/v1"],
                            var_name_to_prev_var_name=ms_name_to_prev_ms_name)

          ws_util.warm_start(
              self.get_temp_dir(),
              vars_to_warm_start=["new_outer/v1", "new_outer/v2"],
              var_name_to_prev_var_name=ms_name_to_prev_ms_name)
          self.evaluate(deo.dynamic_embedding_variables_initializer())
          # Verify the selection of weights were correctly warm-started (init
          # overridden to ones).
          for v in [v1, v2]:
            keys, values = self._export_sorted_keys_and_values(v)
            self.assertAllEqual(keys, keys)
            self.assertAllEqual(values, values)

  def test_from_object_based_checkpoint(self):
    dim = 10
    keys = [0, 1, 2, 3]
    values = [[k] * dim for k in keys]
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        with variable_scope.variable_scope("outer"):
          prev_var = self._add_and_initialize_devar("prefix/devar", keys,
                                                    values, dim)
          # Save object-based checkpoint.
          tracking_util.Checkpoint(v=prev_var).save(
              os.path.join(self.get_temp_dir(), "checkpoint"))

    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        with variable_scope.variable_scope("outer"):
          var = self._add_devar("prefix/devar", dim)
          ws_util.warm_start(self.get_temp_dir(),
                             vars_to_warm_start=["outer/prefix/devar"])
          self.evaluate(deo.dynamic_embedding_variables_initializer())
          _keys, _values = self._export_sorted_keys_and_values(var)
          self.assertAllEqual(keys, _keys)
          self.assertAllEqual(values, _values)

  def test_warm_start_optimizers(self):
    extra_run_step = 2
    for run_id, num_shards, k_dtype, d_dtype, init_mode, dim, run_step \
        in _next_run_step_config():
      error_msg = "Cond:{},{},{},{},{},{}".format(num_shards, k_dtype, d_dtype,
                                                  init_mode, dim, run_step)
      with ops.Graph().as_default() as g:
        with self.session(graph=g,
                          use_gpu=test_util.is_gpu_available(),
                          config=default_config) as sess:
          graph = TestGraph(k_dtype, d_dtype, dim, num_shards, 'var', 'devar',
                            run_id)
          self.evaluate(variables.global_variables_initializer())
          sess.run([graph.devar_init_op])
          prev_x = sess.run([graph.x])[0]
          for _ in range(run_step):
            sess.run([graph.var_opt_op, graph.devar_opt_op])
          sess.run([graph.var_loss, graph.devar_loss])
          _write_checkpoint(self, sess)
          for _ in range(extra_run_step):
            sess.run([graph.var_opt_op, graph.devar_opt_op])
          prev_var_loss, prev_devar_loss = sess.run(
              [graph.var_loss, graph.devar_loss])
          self.assertAllCloseAccordingToType(prev_var_loss,
                                             prev_devar_loss,
                                             msg=error_msg)

      with ops.Graph().as_default() as g:
        with self.session(graph=g,
                          use_gpu=test_util.is_gpu_available(),
                          config=default_config) as sess:
          graph = TestGraph(k_dtype, d_dtype, dim, num_shards, 'var', 'devar',
                            run_id, prev_x)
          ws_util.warm_start(self.get_temp_dir(), vars_to_warm_start=['.*'])
          self.evaluate(variables.global_variables_initializer())
          self.evaluate(deo.dynamic_embedding_variables_initializer())
          for _ in range(extra_run_step):
            sess.run([graph.var_opt_op, graph.devar_opt_op])
          var_loss, devar_loss = sess.run([graph.var_loss, graph.devar_loss])
          self.assertAllCloseAccordingToType(var_loss,
                                             prev_var_loss,
                                             msg=error_msg)
          self.assertAllCloseAccordingToType(devar_loss,
                                             prev_devar_loss,
                                             msg=error_msg)

  def test_checkpoint_overwrite_warm_start(self):
    extra_run_step = 2
    ws_ckpt_dir = tempfile.mkdtemp(
        prefix=os.path.join(self.get_temp_dir(), "warm_start"))
    final_ckpt_dir = tempfile.mkdtemp(
        prefix=os.path.join(self.get_temp_dir(), "final"))
    for run_id, num_shards, k_dtype, d_dtype, init_mode, dim, run_step \
        in _next_run_step_config():
      error_msg = "Cond:{},{},{},{},{},{}".format(num_shards, k_dtype, d_dtype,
                                                  init_mode, dim, run_step)
      with ops.Graph().as_default() as g:
        with self.session(graph=g,
                          use_gpu=test_util.is_gpu_available(),
                          config=default_config) as sess:
          training_util.create_global_step()
          graph = TestGraph(k_dtype, d_dtype, dim, num_shards, 'var', 'devar',
                            run_id)
          self.evaluate(variables.global_variables_initializer())
          sess.run([graph.devar_init_op])
          prev_x = sess.run([graph.x])[0]
          for _ in range(run_step):
            sess.run([graph.var_opt_op, graph.devar_opt_op])
          saver_lib.Saver().save(sess, os.path.join(ws_ckpt_dir, "model"))
          prev_ws_var_loss, prev_ws_devar_loss = sess.run(
              [graph.var_loss, graph.devar_loss])
          self.assertAllCloseAccordingToType(prev_ws_var_loss,
                                             prev_ws_devar_loss,
                                             msg=error_msg)
          for _ in range(extra_run_step):
            sess.run([graph.var_opt_op, graph.devar_opt_op])
          saver_lib.Saver().save(sess, os.path.join(final_ckpt_dir, "model"))
          prev_final_var_loss, prev_final_devar_loss = sess.run(
              [graph.var_loss, graph.devar_loss])
          self.assertAllCloseAccordingToType(prev_final_var_loss,
                                             prev_final_devar_loss,
                                             msg=error_msg)

      with ops.Graph().as_default():
        training_util.create_global_step()
        graph = TestGraph(k_dtype, d_dtype, dim, num_shards, 'var', 'devar',
                          run_id, prev_x)
        ws_util.warm_start(ws_ckpt_dir, vars_to_warm_start=['.*'])
        with monitored_session.MonitoredTrainingSession(
            config=default_config, is_chief=True,
            checkpoint_dir=final_ckpt_dir) as sess:
          var_loss, devar_loss = sess.run([graph.var_loss, graph.devar_loss])
          self.assertAllCloseAccordingToType(var_loss,
                                             prev_final_var_loss,
                                             msg=error_msg)
          self.assertAllCloseAccordingToType(devar_loss,
                                             prev_final_devar_loss,
                                             msg=error_msg)


if __name__ == '__main__':
  os.environ['TF_HASHTABLE_INIT_SIZE'] = '1000'
  test.main()
