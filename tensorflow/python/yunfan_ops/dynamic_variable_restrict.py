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
"""Restrictor of variables"""
# pylint: disable=g-bad-name

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.yunfan_ops import dynamic_embedding_ops


@tf_export("dynamic_embedding.RestrictPolicy")
class RestrictPolicy(object):
  """
  RestrictPolicy records the status of variable and variables in slots of
  related optimizers, while provides interfaces for continuously updating
  with the status with training progress, and restricting the key-value pairs
  of variables under record. The status is a set of variables which take
  notes on the size of variables, variables in slots, and the status variables
  themselves. The RestrictPolicy is an abstract class, which can be inherited
  for customization.

  RestrictPolicy holds `create_status`, `update`, and `restrict` methods:
    create_status: Creates a record as status of variable.
    update: updates the status in iteration. The update operation usually
      runs with the training operation.
    restrict: Restrict the status. It's used to constrain the memory usage
      from the over growth of the dynamic_embedding.Variable.
  """

  def __init__(self, var, opt):
    """
    Construct the RestrictPolicy from variable and optimizer.

    Args:
      var: dynamic_ebmedding.Variable.
      opt: A `tf.train.Optimizer`.
    """
    if not isinstance(var, dynamic_embedding_ops.Variable):
      raise TypeError("parameter var type should be"  \
                      "dynamic_embedding_ops.Variable.")

    if not isinstance(opt, optimizer.Optimizer):
      raise TypeError("parameter opt type should be"  \
                      "optimizer.Optimizer.")

    self.var = var
    self.opt = opt
    self.create_status()

  def create_status(self, **kwargs):
    """
    Create status for recording the variable and variables in
    slots of optimizer. Usually the status is implemented with
    an untrainable variable, which has the same key_dtype as
    the target variable.
    """
    raise NotImplementedError

  def update(self, trainable_wrappers=None, **kwargs):
    """
    Update the status. The returned update operation is
    usually run in company with training operation, to keep
    the status following changes of variables and relative
    variables in slots of optimizer.

    Args:
      trainable_wrappers: A list of `tf.dynamic_embedding.TrainableWrapper`
        objects. The variable's status is updated by the trainable_wrappers,
        to avoid consuming unrelated ids when the embedding is shared by
        multiple optimizers, while only part of optimizers work. `NoneType`
        default value update all embeddings.

      **kwargs: Optional keyword arguments.

    Returns:
      An operation to update the status.
    """
    raise NotImplementedError

  def restrict(self, **kwargs):
    """
    Restrict the variable and variables in slots of optimizer, according to
    the instructions defined in this function and records in status. This
    method creates a sub-branch in graph for restricting the size of variable
    and variables in slots.

    Returns:
      An operation to restrict the status.
    """
    raise NotImplementedError


@tf_export("dynamic_embedding.TimestampRestrictPolicy")
class TimestampRestrictPolicy(RestrictPolicy):
  """
  An implementation derived from RestrictPolicy.
  TimestampRestrictPolicy provides feature elimination for variable
  and variables in slots of optimizers based on oldest-out rule. It
  records the timestamp of last appearance for features.
  """

  def __init__(self, var, opt, **kwargs):
    self.tstp_var = None
    self.var_in_slots = []
    self.threshold = 0
    super(TimestampRestrictPolicy, self).__init__(var, opt)

  def create_status(self, **kwargs):
    """
    Create relative timestamp status variables.
    """
    for tw in self.var.trainable_wrappers:
      slots = [
          self.opt.get_slot(tw, name) for name in self.opt.get_slot_names()
      ]
      self.var_in_slots += [x.params for x in slots if x is not None]

    scope = variable_scope.get_variable_scope()
    if scope.name:
      scope_name = scope.name + '/timestamp_status'
    else:
      scope_name = 'timestamp_status'
    status_name = self.var.name + '/' + self.opt.get_name()

    with ops.name_scope(scope_name, "timestamp_status", []) as unique_scope:
      full_name = unique_scope + '/' + status_name
      self.tstp_var = dynamic_embedding_ops.get_variable(
          key_dtype=self.var.key_dtype,
          value_dtype=dtypes.int32,
          dim=1,
          name=full_name,
          devices=self.var.devices,
          partitioner=self.var.partition_fn,
          trainable=False,
      )

  def update(self, trainable_wrappers=None, **kwargs):
    """
    Update the timestamp status variable. The corresponding timestamps
    are updated if the relative features appear in embedding_lookup.

    Args:
      trainable_wrappers: A list of `tf.dynamic_embedding.TrainableWrapper`
        objects. The variable's status is updated by the trainable_wrappers,
        to avoid consuming unrelated ids when the embedding is shared by
        multiple optimizers, and only part of optimizers stay in work.
        `NoneType` default value update all embeddings.

      **kwargs: Optional keyword arguments.

    Returns:
      An operation for updating the timestamp status.
    """
    update_ops = []
    if trainable_wrappers:
      for tw in trainable_wrappers:
        if tw not in self.var.trainable_wrappers:
          raise ValueError('trainable_wrappers must be subset of variable\'s'
                           'relative `tf.dynamic_embedding.TrainableWrapper`'
                           'generated in clan of embedding_lookup functions.')
      chosen_tws = trainable_wrappers
    else:
      chosen_tws = self.var.trainable_wrappers

    for tw in chosen_tws:
      tw_status_ids = array_ops.reshape(tw.ids, (-1,))

      fresh_tstp = array_ops.tile(
          array_ops.reshape(gen_logging_ops.timestamp(), [1]),
          array_ops.reshape(array_ops.size(tw_status_ids), (-1,)),
      )
      fresh_tstp = math_ops.cast(fresh_tstp, dtypes.int32)
      fresh_tstp = array_ops.reshape(fresh_tstp, (-1, 1))
      tw_status_update = self.tstp_var.upsert(tw_status_ids, fresh_tstp)

      update_ops.append(tw_status_update)

    return control_flow_ops.group(update_ops)

  def restrict(self, **kwargs):
    """
    Restrict the variable, variables in slots of optimizer, and the
    status variables themselves, eliminate the oldest features, if the
    variable grows oversized.

    Args:
      **kwargs: Keyword arguments, including
        threshold: int. The threshold for feature number in variable.
          If the variable is sharded into multiple devices, then all
          fractional table take the sharded threshold where threshold
          were divided by the number of shards equally.

        factor - (Optional): Factor of stretching the restriction space,
          which leaving the restriction not triggered if the variable's
          size is less than `threshold * factor`. Default to be 1.0. If
          the variable's size if greater equal to threshold * factor, then
          restrict the variable's size to threshold. Factor's type can be
          `int`, `float`, `tf.int32`, `tf.int64`, `tf.float32`.

    Returns:
      An operation to restrict the size of variable, the
      variables in slots of optimizers, and status variables
      themselves.
    """
    try:
      self.threshold = kwargs['threshold']
    except:
      raise KeyError('restrict method expects parameter `threshold`.')
    if not isinstance(self.threshold, int):
      raise TypeError('threshold must be an integer.')
    if self.threshold < 0:
      raise ValueError('threshold must be greater or equal to zero.')

    factor = kwargs.get('factor', 1.0)
    if isinstance(factor, ops.Tensor):
      if factor.dtype not in (dtypes.int32, dtypes.int64, dtypes.float32):
        raise TypeError(
            'factor expects int, float, tf.int32, tf.int64, or tf.float32')
      factor = math_ops.cast(factor, dtype=dtypes.float32)
    if not isinstance(factor, (int, float)):
      raise TypeError(
          'factor expects int, float, tf.int32, tf.int64, or tf.float32')

    threshold = math_ops.cast(self.threshold, dtype=dtypes.float32) * factor
    threshold = math_ops.cast(threshold, dtype=dtypes.int64)
    condition = math_ops.greater(self.var.size(), threshold)
    restrict_op = control_flow_ops.cond(condition, self._conditional_restrict,
                                        self._conditional_no_op)
    return restrict_op

  def _conditional_no_op(self):
    return control_flow_ops.no_op()

  def _conditional_restrict(self):
    restrict_var_ops = list()
    restrict_status_ops = list()
    restrict_slot_var_ops = list()

    for idx, dev in enumerate(self.tstp_var.devices):
      with ops.device(dev):
        sub_tk, sub_tv = self.tstp_var.tables[idx].export()
        sharded_threshold = int(self.threshold / self.tstp_var.shard_num)
        sub_tv = array_ops.reshape(sub_tv, (-1,))
        first_dim = array_ops.shape(sub_tv)[0]

        k_on_top = math_ops.cast(first_dim - sharded_threshold,
                                 dtype=dtypes.int32)
        k_on_top = math_ops.maximum(k_on_top, 0)
        _, removed_keys_ids = nn_ops.top_k(-sub_tv, k_on_top, sorted=False)
        removed_keys = array_ops.gather(sub_tk, removed_keys_ids)

        restrict_var_ops.append(self.var.tables[idx].remove(removed_keys))
        restrict_status_ops.append(
            self.tstp_var.tables[idx].remove(removed_keys))
        for slot_var in self.var_in_slots:
          restrict_slot_var_ops.append(
              slot_var.tables[idx].remove(removed_keys))

    restrict_var_ops = control_flow_ops.group(restrict_var_ops)
    restrict_slot_var_ops = control_flow_ops.group(restrict_slot_var_ops)
    restrict_status_ops = control_flow_ops.group(restrict_status_ops)

    restrict_op = control_flow_ops.group(
        restrict_var_ops,
        restrict_status_ops,
        restrict_slot_var_ops,
    )
    return restrict_op


@tf_export("dynamic_embedding.FrequencyRestrictPolicy")
class FrequencyRestrictPolicy(RestrictPolicy):
  """
  A status inherts from RestrictPolicy, providing updating and
  restriction for variable by frequency rule.

  When call restrict method, the class will delete values on
  ids by following the lowest-appearance-out rule for every ids
  in record. And when call update method, the record of every
  ids in related trainable_wrappers will be increased by 1.
  """

  def __init__(self, var, opt, **kwargs):
    self.freq_var = None
    self.var_in_slots = []
    self.threshold = 0
    self.default_count = constant_op.constant(0, dtypes.int32)
    super(FrequencyRestrictPolicy, self).__init__(var, opt)

  def create_status(self, **kwargs):
    """
    Create relative frequency status variables.
    """
    for tw in self.var.trainable_wrappers:
      slots = [
          self.opt.get_slot(tw, name) for name in self.opt.get_slot_names()
      ]
      self.var_in_slots += [x.params for x in slots if x is not None]

    scope = variable_scope.get_variable_scope()
    if scope.name:
      scope_name = scope.name + '/frequency_status'
    else:
      scope_name = 'frequency_status'
    status_name = self.var.name + '/' + self.opt.get_name()

    with ops.name_scope(scope_name, "frequency_status", []) as unique_scope:
      full_name = unique_scope + '/' + status_name
      self.freq_var = dynamic_embedding_ops.get_variable(
          key_dtype=self.var.key_dtype,
          value_dtype=dtypes.int32,
          dim=1,
          name=full_name,
          devices=self.var.devices,
          partitioner=self.var.partition_fn,
          initializer=self.default_count,
          trainable=False,
      )

  def update(self, trainable_wrappers=None, **kwargs):
    """
    Update the frequency status. The corresponding frequency
    records will be increased by 1 for every features appears
    in the training.

    Args:
      trainable_wrappers: A list of `tf.dynamic_embedding.TrainableWrapper`
        objects. The variable's status is updated by the trainable_wrappers,
        to avoid consuming unrelated ids when the embedding is shared by
        multiple optimizers, and only part of optimizers work. `NoneType`
        default value update all embeddings.
      **kwargs: Optional keyword arguments.

    Returns:
      An operation for updating the frequency status.
    """
    update_ops = []
    if trainable_wrappers:
      for tw in trainable_wrappers:
        if tw not in self.var.trainable_wrappers:
          raise ValueError('trainable_wrappers must be subset of variable\'s'
                           'relative `tf.dynamic_embedding.TrainableWrapper'
                           'generated in'
                           'clan of embedding_lookup functions.')
      chosen_tws = trainable_wrappers
    else:
      chosen_tws = self.var.trainable_wrappers

    for tw in chosen_tws:
      tw_status_ids = array_ops.reshape(tw.ids, (-1,))
      partition_index =  \
          self.var.partition_fn(tw_status_ids, self.var.shard_num)
      partitioned_ids_list, _ =  \
          dynamic_embedding_ops._partition(tw_status_ids,
                                           partition_index,
                                           self.var.shard_num)

      for idx, dev in enumerate(self.freq_var.devices):
        with ops.device(dev):
          feature_counts =  \
              self.freq_var.tables[idx].lookup(
                  partitioned_ids_list[idx],
                  dynamic_default_values=self.default_count,
              )
          feature_counts += 1

          mht_update =  \
              self.freq_var.tables[idx].insert(
                  partitioned_ids_list[idx],
                  feature_counts,
              )
          update_ops.append(mht_update)

    return control_flow_ops.group(update_ops)

  def restrict(self, **kwargs):
    """
    Restrict the variable, variables in slots of optimizer,
    and the status variables themselves, eliminate the least
    frequent features, if the size of variable grow too large
    for threshold.

    Args:
      **kwargs: keyword arguments, including
        threshold: int. The threshold for feature number in variable.
          If the variable is sharded into multiple devices, then all
          fractional table take the sharded threshold where threshold
          were divided by the number of shards equally.

        factor - (Optional): Factor of stretching the restriction space,
          which leaving the restriction not triggered if the variable's
          size is less than `threshold * factor`. Default to be 1.0. If
          the variable's size if greater equal to threshold * factor, then
          restrict the variable's size to threshold. Factor's type can be
          `int`, `float`, `tf.int32`, `tf.int64`, `tf.float32`.

    Returns:
      An operation to restrict the size of variable, the status, and
        the variables in the slots.
    """
    try:
      self.threshold = kwargs['threshold']
    except:
      raise KeyError('restrict method expects parameter `threshold`.')
    if not isinstance(self.threshold, int):
      raise TypeError('threshold must be an integer.')
    if self.threshold < 0:
      raise ValueError('threshold must be greater or equal to zero.')

    factor = kwargs.get('factor', 1.0)
    if isinstance(factor, ops.Tensor):
      if factor.dtype not in (dtypes.int32, dtypes.int64, dtypes.float32):
        raise TypeError(
            'factor expects int, float, tf.int32, tf.int64, or tf.float32')
      factor = math_ops.cast(factor, dtype=dtypes.float32)
    if not isinstance(factor, (int, float)):
      raise TypeError(
          'factor expects int, float, tf.int32, tf.int64, or tf.float32')

    threshold = math_ops.cast(self.threshold, dtype=dtypes.float32) * factor
    threshold = math_ops.cast(threshold, dtype=dtypes.int64)
    condition = math_ops.greater(self.var.size(), threshold)
    restrict_op = control_flow_ops.cond(condition, self._conditional_restrict,
                                        self._conditional_no_op)
    return restrict_op

  def _conditional_no_op(self):
    return control_flow_ops.no_op()

  def _conditional_restrict(self):
    restrict_var_ops = list()
    restrict_status_ops = list()
    restrict_slot_var_ops = list()

    for idx, dev in enumerate(self.freq_var.devices):
      with ops.device(dev):
        sub_fk, sub_fv = self.freq_var.tables[idx].export()
        sub_fv = array_ops.reshape(sub_fv, (-1,))
        first_dim = array_ops.shape(sub_fv)[0]

        sharded_threshold = int(self.threshold / self.freq_var.shard_num)

        k_on_top = math_ops.cast(first_dim - sharded_threshold,
                                 dtype=dtypes.int32)
        k_on_top = math_ops.maximum(k_on_top, 0)
        _, removed_keys_ids = nn_ops.top_k(-sub_fv, k_on_top, sorted=False)
        removed_keys = array_ops.gather(sub_fk, removed_keys_ids)

        restrict_var_ops.append(self.var.tables[idx].remove(removed_keys))
        restrict_status_ops.append(
            self.freq_var.tables[idx].remove(removed_keys))
        for slot_var in self.var_in_slots:
          restrict_slot_var_ops.append(
              slot_var.tables[idx].remove(removed_keys))

    restrict_var_ops = control_flow_ops.group(restrict_var_ops)
    restrict_slot_var_ops = control_flow_ops.group(restrict_slot_var_ops)
    restrict_status_ops = control_flow_ops.group(restrict_status_ops)

    restrict_op = control_flow_ops.group([
        restrict_var_ops,
        restrict_status_ops,
        restrict_slot_var_ops,
    ])
    return restrict_op


@tf_export("dynamic_embedding.VariableRestrictor")
class VariableRestrictor(object):
  """
  A restrictor for constraining the variable's feature number, with
  keeping recording and eliminating the obsolete features and their
  corresponding weights in variables and the variables in slots of
  optimizers.

  # Example:

  ```python
  # Create a variable
  var = tf.dynamic_embedding.get_variable(
    name = 'var',
    devices = ['/CPU:0'],
    tf.random_normal_initializer(0, 0.01),
    trainable = True,
    dim = 8,
  )
  ...

  opt = tf.train.AdagradOptimizer(0.1)

  # Call the minimize to the loss with optimizer.
  train_op = opt.minimize(loss)

  # Get a VariableRestrictor.
  restrictor = tf.dynamic_embedding.VariableRestrictor(
                   var_list=[var],
                   optimizer_list=[opt,],
                   policy=TimestampRestrictPolicy,
               )

  # Call update to get an operation to update policy status.
  update_op = restrictor.update()

  # Call restrict to have an operation to restrict the policy status.
  threshold = 1000
  restrict_op = restrictor.restrict(threshold=threshold,
                                    factor=1.2)

  with tf.Session() as sess:
    ...

    for step in range(num_iter):

      # Run the update_op with update every time to have
      # policy status keep with training.
      # There is no need to call update_op in inference.
      sess.run([train_op, update_op])

      # If the variable size is too large, run the restrict op.
      # There is no need to call restrict_op in inference.
      if step % 100 == 0:
        sess.run(restrict_op)
      ...

  ```

  """

  def __init__(self,
               var_list=None,
               optimizer_list=None,
               policy=TimestampRestrictPolicy):
    """
    Creates an `VariableRestrictor` object. Every variable in
    var_list and slots in optimizer_list share same policy.

    Args:
      var_list: A list of `tf.dynamic_embedding.Variable` objects.
      optimizer_list: A list of `tf.train.Optimizer` objects.
      policy: A RestrictPolicy class to specify the rules for
        recoding, updating, and restricting the size of the
        variables in var_list. The defaults policy choice is the
        TimestampRestrictPolicy.
    """
    if not issubclass(policy, RestrictPolicy):
      raise TypeError("policy must be subclass of"  \
                      "RestrictPolicy object.")
    self.var_list = var_list
    self.optimizer_list = optimizer_list
    self.policy_group = dict()

    for var in self.var_list:
      if not self.policy_group.get(var, None):
        self.policy_group[var] = []
      for opt in self.optimizer_list:
        self.policy_group[var].append(policy(var, opt))

  def update(self, trainable_wrappers=None, **kwargs):
    """
    Update the status for every variable in var_list.

    Args:
      trainable_wrappers: A list of `tf.dynamic_embedding.TrainableWrapper`
        objects. The variable's status is updated according to the ids of
        trainable_wrappers, to avoid consuming unrelated ids when the
        embedding is shared by multiple optimizers, and only part of optimizers
        work. `NoneType` default value update all embeddings.
      **kwargs: Optional keyword arguments passed to the keyword arguments
        in policy's update method.

    Returns:
      An operation to update the status for every variable.
    """
    update_ops = []
    policies = []

    for pols in self.policy_group.values():
      policies += pols

    for pol in policies:
      update_ops.append(pol.update(**kwargs))

    return control_flow_ops.group(update_ops)

  def restrict(self, **kwargs):
    """
    Restrict the size of variables in var_list.

    Args:
      **kwargs: Optional keyword arguments passed to the method
        policy.restrict(**kwargs). For example, in the `restrict`
        method of `TimestampRestrictPolicy` and `FrequencyRestrictPolicy`
        have parameters `threshold`, `factor`, etc..

    Returns:
      An operation to restrict the variables.
    """
    restrict_ops = []
    policies = []

    for pols in self.policy_group.values():
      policies += pols

    for pol in policies:
      restrict_ops.append(pol.restrict(**kwargs))

    return control_flow_ops.group(restrict_ops)
