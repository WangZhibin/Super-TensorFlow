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

"""Contains MixedPrecisionTrainingPolicy classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import time

from tensorflow.python.platform import tf_logging

@six.add_metaclass(abc.ABCMeta)
class MixedPrecisionTrainingPolicy(object):
  """Policy for dynamically control mixed precision training.
  """
  def __init__(self):
    pass

  @abc.abstractmethod
  def enable_mixed_precision(self, step, **kwargs):
    raise NotImplementedError("AMP schedule must override enable_mixed_precision")


class LossDescendingSpeedPolicy(MixedPrecisionTrainingPolicy):
  """Policy for dynamically control mixed precision training.
  """
  def __init__(self, test_interval=1000, test_step=100):
    assert test_interval > 2 * test_step, (
    	'the test_interval must be twice larger than test_step.'
    )
    self.test_interval = test_interval
    self.test_step = test_step
    self._enable_mixed_precision = True
    self.working = True

  def enable_mixed_precision(self, step, **kwargs):
    if not self.working:
        return self._enable_mixed_precision
    if "loss" not in kwargs:
        raise ValueError("no loss value passed.")
    loss = kwargs["loss"]
    if step % self.test_interval == 0:
        self._start_profiling(loss)
    elif step % self.test_interval == self.test_step:
        self.mixed_duration, self.mixed_loss_diff = self._end_profiling(loss)
        self._enable_mixed_precision = False
        self._start_profiling(loss)
    elif step % self.test_interval == 2 * self.test_step:
        self.full_duration, self.full_loss_diff = self._end_profiling(loss)
        self._check_switch()
    return self._enable_mixed_precision
  
  def _start_profiling(self, loss):
    self._start_time = time.time()
    self._start_loss = loss
  
  def _end_profiling(self, loss):
    return [time.time() - self._start_time, self._start_loss - loss]
  
  def _check_switch(self):
    flag = (
    		self.mixed_loss_diff / self.mixed_duration >=
      	self.full_loss_diff / self.full_duration
    )
    self._enable_mixed_precision = flag
    if not flag:
      self.working = False

  def get_config(self):
    return {
      "test_interval": self.test_interval,
      "test_step": self.test_step
    }


class TimeSchedulePolicy(MixedPrecisionTrainingPolicy):
  """Policy for TimeSchedule mixed precision training.
  """
  def __init__(self, start_step, end_step):
    assert start_step < end_step, (
      'MixedPrecision end_step should be larger than start_step.'
      )
    self.start_step = start_step
    self.end_step = end_step

  def enable_mixed_precision(self, step, **kwargs):
    if step > self.end_step or step <self.start_step:
      tf_logging.debug('disable mixed_precision at step %d' % step)
      return False
    else:
      tf_logging.debug('enable mixed_precision at step %d' % step)
      return True

class TimeAndLrSchedulePolicy(MixedPrecisionTrainingPolicy):
  """Policy for TimeAndLrSchedule mixed precision training.
  """
  def __init__(self, start_step, end_step, hold_step):
    assert start_step < end_step, (
      'MixedPrecision end_step should be larger than start_step.'
      )
    assert hold_step > 0, (
      'MixedPrecision hold_step should be larger than 0.'
      )
    self.start_step = start_step
    self.end_step = end_step
    self.hold_step = hold_step
 
    self.lr_last = 0.
    self.step_last = - hold_step * 2

  def enable_mixed_precision(self, step, **kwargs):
    amp_flag_time = True
    if step > self.end_step or step < self.start_step:
      amp_flag_time = False

    if "lr" not in kwargs:
        raise ValueError("no lr value passed.")
    lr_new = kwargs["lr"] 

    amp_flag_lr = True
    if self.lr_last != lr_new:
      amp_flag_lr = False 
      self.step_last = step
      self.lr_last = lr_new
    elif (step - self.step_last) < self.hold_step: 
      amp_flag_lr = False
    #else:
    #  self.step_last = 0 
  
    enable_mixed_precision = amp_flag_time and amp_flag_lr   
    if enable_mixed_precision:
      tf_logging.debug('enable mixed_precision at step %d' % step)
    else:
      tf_logging.debug('disable mixed_precision at step %d' % step)

    return enable_mixed_precision
 

