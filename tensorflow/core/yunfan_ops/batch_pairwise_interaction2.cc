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
/* BatchPairwiseInteraction2 */
/************************************************************************/
template <typename T>
class BatchPairwiseInteraction2Op : public OpKernel {
 private:
  using ll_math_t = deepx_core::LLMath<T>;

 public:
  explicit BatchPairwiseInteraction2Op(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    DCHECK_EQ(c->num_inputs(), 2);
    const Tensor& x = c->input(0);
    const Tensor& y = c->input(1);

    DCHECK_EQ(x.dims(), 3);
    int64 batch = x.dim_size(0);
    int64 m1 = x.dim_size(1);
    int64 n = x.dim_size(2);
    DCHECK_EQ(y.dims(), 3);
    DCHECK_EQ(y.dim_size(0), batch);
    int64 m2 = y.dim_size(1);
    DCHECK_EQ(y.dim_size(2), n);

    Tensor* z = nullptr;
    TensorShape shape;
    shape.AddDim(batch);
    shape.AddDim(m1 * m2);
    shape.AddDim(n);
    OP_REQUIRES_OK(c, c->allocate_output(0, shape, &z));

    const auto* _x = (const T*)x.tensor_data().data();
    const auto* _y = (const T*)y.tensor_data().data();
    auto* _z = (T*)z->tensor_data().data();
    for (int64 i = 0; i < batch; ++i) {
      for (int64 j = 0; j < m1; ++j) {
        for (int64 k = 0; k < m2; ++k) {
          ll_math_t::mul(n, _x + j * n, _y + k * n, _z);
          _z += n;
        }
      }
      _x += m1 * n;
      _y += m2 * n;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BatchPairwiseInteraction2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        BatchPairwiseInteraction2Op<float>);

REGISTER_KERNEL_BUILDER(Name("BatchPairwiseInteraction2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        BatchPairwiseInteraction2Op<double>);

/************************************************************************/
/* BatchPairwiseInteraction2Grad */
/************************************************************************/
template <typename T>
class BatchPairwiseInteraction2GradOp : public OpKernel {
 private:
  using ll_math_t = deepx_core::LLMath<T>;

 public:
  explicit BatchPairwiseInteraction2GradOp(OpKernelConstruction* c)
      : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    DCHECK_EQ(c->num_inputs(), 3);
    const Tensor& x = c->input(0);
    const Tensor& y = c->input(1);
    const Tensor& gz = c->input(2);

    DCHECK_EQ(x.dims(), 3);
    int64 batch = x.dim_size(0);
    int64 m1 = x.dim_size(1);
    int64 n = x.dim_size(2);
    DCHECK_EQ(y.dims(), 3);
    DCHECK_EQ(y.dim_size(0), batch);
    int64 m2 = y.dim_size(1);
    DCHECK_EQ(y.dim_size(2), n);
    DCHECK_EQ(gz.dims(), 3);
    DCHECK_EQ(gz.dim_size(0), batch);
    DCHECK_EQ(gz.dim_size(1), m1 * m2);
    DCHECK_EQ(gz.dim_size(2), n);

    Tensor* gx = nullptr;
    Tensor* gy = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, x.shape(), &gx));
    OP_REQUIRES_OK(c, c->allocate_output(1, y.shape(), &gy));

    const auto* _gz = (const T*)gz.tensor_data().data();
    const auto* _x = (const T*)x.tensor_data().data();
    const auto* _y = (const T*)y.tensor_data().data();
    auto* _gx = (T*)gx->tensor_data().data();
    auto* _gy = (T*)gy->tensor_data().data();
    memset(_gx, 0, batch * m1 * n * sizeof(T));
    memset(_gy, 0, batch * m2 * n * sizeof(T));
    for (int64 i = 0; i < batch; ++i) {
      for (int64 j = 0; j < m1; ++j) {
        for (int64 k = 0; k < m2; ++k) {
          ll_math_t::xypz(n, _gz, _y + k * n, _gx + j * n);
          ll_math_t::xypz(n, _gz, _x + j * n, _gy + k * n);
          _gz += n;
        }
      }
      _x += m1 * n;
      _gx += m1 * n;
      _y += m2 * n;
      _gy += m2 * n;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BatchPairwiseInteraction2Grad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        BatchPairwiseInteraction2GradOp<float>);

REGISTER_KERNEL_BUILDER(Name("BatchPairwiseInteraction2Grad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        BatchPairwiseInteraction2GradOp<double>);

}  // namespace tensorflow
