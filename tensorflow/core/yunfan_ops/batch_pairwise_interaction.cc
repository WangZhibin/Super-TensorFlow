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
/* BatchPairwiseInteraction */
/************************************************************************/
template <typename T>
class BatchPairwiseInteractionOp : public OpKernel {
 private:
  using ll_math_t = deepx_core::LLMath<T>;

 public:
  explicit BatchPairwiseInteractionOp(OpKernelConstruction* c) : OpKernel(c) {}

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
    shape.AddDim(m * (m - 1) / 2);
    shape.AddDim(n);
    OP_REQUIRES_OK(c, c->allocate_output(0, shape, &z));

    const auto* _x = (const T*)x.tensor_data().data();
    auto* _z = (T*)z->tensor_data().data();
    for (int64 i = 0; i < batch; ++i) {
      for (int64 j = 0; j < m; ++j) {
        for (int64 k = j + 1; k < m; ++k) {
          ll_math_t::mul(n, _x + j * n, _x + k * n, _z);
          _z += n;
        }
      }
      _x += m * n;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BatchPairwiseInteraction")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        BatchPairwiseInteractionOp<float>);

REGISTER_KERNEL_BUILDER(Name("BatchPairwiseInteraction")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        BatchPairwiseInteractionOp<double>);

/************************************************************************/
/* BatchPairwiseInteractionGrad */
/************************************************************************/
template <typename T>
class BatchPairwiseInteractionGradOp : public OpKernel {
 private:
  using ll_math_t = deepx_core::LLMath<T>;

 public:
  explicit BatchPairwiseInteractionGradOp(OpKernelConstruction* c)
      : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    DCHECK_EQ(c->num_inputs(), 2);
    const Tensor& x = c->input(0);
    const Tensor& gz = c->input(1);

    DCHECK_EQ(x.dims(), 3);
    int64 batch = x.dim_size(0);
    int64 m = x.dim_size(1);
    int64 n = x.dim_size(2);
    DCHECK_EQ(gz.dims(), 3);
    DCHECK_EQ(gz.dim_size(0), batch);
    DCHECK_EQ(gz.dim_size(1), m * (m - 1) / 2);
    DCHECK_EQ(gz.dim_size(2), n);

    Tensor* gx = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, x.shape(), &gx));

    const auto* _gz = (const T*)gz.tensor_data().data();
    const auto* _x = (const T*)x.tensor_data().data();
    auto* _gx = (T*)gx->tensor_data().data();
    memset(_gx, 0, batch * m * n * sizeof(T));
    for (int64 i = 0; i < batch; ++i) {
      for (int64 j = 0; j < m; ++j) {
        for (int64 k = j + 1; k < m; ++k) {
          ll_math_t::xypz(n, _gz, _x + k * n, _gx + j * n);
          ll_math_t::xypz(n, _gz, _x + j * n, _gx + k * n);
          _gz += n;
        }
      }
      _x += m * n;
      _gx += m * n;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BatchPairwiseInteractionGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        BatchPairwiseInteractionGradOp<float>);

REGISTER_KERNEL_BUILDER(Name("BatchPairwiseInteractionGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        BatchPairwiseInteractionGradOp<double>);

}  // namespace tensorflow
