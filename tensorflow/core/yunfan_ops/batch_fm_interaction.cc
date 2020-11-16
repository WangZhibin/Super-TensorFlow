/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "third_party/deepx_core/include/deepx_core/tensor/ll_math.h"

namespace tensorflow {

/************************************************************************/
/* BatchFMInteraction */
/************************************************************************/
template <typename T>
class BatchFMInteractionOp : public OpKernel {
 private:
  using ll_math_t = deepx_core::LLMath<T>;

 public:
  explicit BatchFMInteractionOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    DCHECK_EQ(c->num_inputs(), 1);
    const Tensor& x = c->input(0);

    DCHECK_EQ(x.dims(), 3);
    int64 batch = x.dim_size(0);
    int64 m = x.dim_size(1);
    int64 n = x.dim_size(2);

    Tensor* z = nullptr;
    TensorShape shape;
    shape.AddDim(batch);
    shape.AddDim(1);
    OP_REQUIRES_OK(c, c->allocate_output(0, shape, &z));

    const auto* _x = (const T*)x.tensor_data().data();
    auto* _z = (T*)z->tensor_data().data();
    memset(_z, 0, batch * sizeof(T));
    for (int64 i = 0; i < batch; ++i) {
      for (int64 k = 0; k < n; ++k) {
        T sum1 = 0, sum2 = 0;
        for (int64 j = 0; j < m; ++j) {
          T xjk = _x[j * n + k];
          sum1 += xjk;
          sum2 += xjk * xjk;
        }
        *_z += (sum1 * sum1 - sum2) * (T)0.5;
      }
      _x += m * n;
      _z += 1;
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("BatchFMInteraction").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    BatchFMInteractionOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("BatchFMInteraction").Device(DEVICE_CPU).TypeConstraint<double>("T"),
    BatchFMInteractionOp<double>);

/************************************************************************/
/* BatchFMInteractionGrad */
/************************************************************************/
template <typename T>
class BatchFMInteractionGradOp : public OpKernel {
 private:
  using ll_math_t = deepx_core::LLMath<T>;

 public:
  explicit BatchFMInteractionGradOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    DCHECK_EQ(c->num_inputs(), 2);
    const Tensor& x = c->input(0);
    const Tensor& gz = c->input(1);

    DCHECK_EQ(x.dims(), 3);
    int64 batch = x.dim_size(0);
    int64 m = x.dim_size(1);
    int64 n = x.dim_size(2);
    DCHECK_EQ(gz.dims(), 2);
    DCHECK_EQ(gz.dim_size(0), batch);
    DCHECK_EQ(gz.dim_size(1), 1);

    Tensor* gx = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, x.shape(), &gx));

    const auto* _x = (const T*)x.tensor_data().data();
    const auto* _gz = (const T*)gz.tensor_data().data();
    auto* _gx = (T*)gx->tensor_data().data();
    memset(_gx, 0, batch * m * n * sizeof(T));
    std::vector<T> aux(n);
    T* _aux = aux.data();
    for (int64 i = 0; i < batch; ++i) {
      ll_math_t::sum_row(m, n, 1, _x, 0, _aux);
      for (int64 j = 0; j < m; ++j) {
        for (int64 k = 0; k < n; ++k) {
          _gx[k] += *_gz * (_aux[k] - _x[k]);
        }
        _x += n;
        _gx += n;
      }
      _gz += 1;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BatchFMInteractionGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        BatchFMInteractionGradOp<float>);

REGISTER_KERNEL_BUILDER(Name("BatchFMInteractionGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        BatchFMInteractionGradOp<double>);

}  // namespace tensorflow
