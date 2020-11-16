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

import os
from absl.testing import parameterized

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training.experimental import mixed_precision_policy

class MixedPrecisionTest(test.TestCase, parameterized.TestCase):
  @test_util.run_in_graph_and_eager_modes
  def test_time_schedule_policy(self):
    actual_outputs = []
    expected_outputs = [False] * 3 + [True] * 5 + [False] * 2
    policy = mixed_precision_policy.TimeSchedulePolicy(3, 7)
    
    for step in range(10):
      flag = policy.enable_mixed_precision(step)
      actual_outputs.append(flag)
    self.assertEqual(actual_outputs, expected_outputs)


  @test_util.run_in_graph_and_eager_modes
  def test_time_lr_schedule_policy(self):
    actual_outputs = []
    expected_outputs = [False, False, False,  True,  True,
                        False, False,  True,  True,  True,
                        False, False,  True,  True,  True,
                        False, False,  True, False, False]

    policy = mixed_precision_policy.TimeAndLrSchedulePolicy(3, 17, 2)

    lrs = [0.1] * 5 + [0.01] * 5 + [0.001] * 5 + [0.0001] * 5
    for step in range(20):
      #print(step)
      #print(expected_outputs[step])
      flag = policy.enable_mixed_precision(step, lr=lrs[step])
      actual_outputs.append(flag)
    self.assertEqual(actual_outputs, expected_outputs)

  @test_util.run_in_graph_and_eager_modes
  def test_loss_descend_schedule_policy(self):
    actual_outputs = []
    expected_outputs = [True, True,   False, False,  True,
                        True, True,    True,  True,  True,
                        True, True,   False, False, False,
                        False, False, False, False, False,
                        False, False, False, False, False,
                        False, False, False, False, False]

    policy = mixed_precision_policy.LossDescendingSpeedPolicy(10, 2)

    losses = [0.01] * 3 + [0.1] * 3 + [0.001] * 4
    losses += [0.1] * 3 + [0.01] * 3 + [0.001] * 4
    losses += [0.01] * 3 + [0.1] * 3 + [0.001] * 4
    for step in range(30):
      #print(step)
      flag = policy.enable_mixed_precision(step, loss=losses[step])
      actual_outputs.append(flag)
      #print(flag)
    self.assertEqual(actual_outputs, expected_outputs)

if __name__ == '__main__':
  test.main()
