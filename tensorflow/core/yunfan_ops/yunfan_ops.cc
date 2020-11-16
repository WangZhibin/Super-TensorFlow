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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("BatchFMInteraction")
    .Input("x: T")   // [batch, m, n]
    .Output("z: T")  // [batch, 1]
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &x));
      DimensionHandle batch = c->Dim(x, 0);
      c->set_output(0, c->Matrix(batch, 1));
      return Status::OK();
    });

REGISTER_OP("BatchFMInteractionGrad")
    .Input("x: T")
    .Input("gz: T")
    .Output("gx: T")
    .Attr("T: {float, double} = DT_FLOAT");

REGISTER_OP("BatchFMInteraction2")
    .Input("x: T")   // [batch, m, n]
    .Output("z: T")  // [batch, n]
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &x));
      DimensionHandle batch = c->Dim(x, 0);
      DimensionHandle n = c->Dim(x, 2);
      c->set_output(0, c->Matrix(batch, n));
      return Status::OK();
    });

REGISTER_OP("BatchFMInteraction2Grad")
    .Input("x: T")
    .Input("gz: T")
    .Output("gx: T")
    .Attr("T: {float, double} = DT_FLOAT");

REGISTER_OP("BatchPairwiseInteraction")
    .Input("x: T")   // [batch, m, n]
    .Output("z: T")  // [batch, m * (m - 1) / 2, n]
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &x));
      DimensionHandle batch = c->Dim(x, 0);
      DimensionHandle m = c->Dim(x, 1);
      DimensionHandle n = c->Dim(x, 2);

      // mm = m * (m - 1) / 2
      DimensionHandle mm;
      TF_RETURN_IF_ERROR(c->Subtract(m, 1, &mm));
      TF_RETURN_IF_ERROR(c->Multiply(m, mm, &mm));
      TF_RETURN_IF_ERROR(c->Divide(mm, 2, true, &mm));

      std::vector<DimensionHandle> z_dim = {batch, mm, n};
      c->set_output(0, c->MakeShape(z_dim));
      return Status::OK();
    });

REGISTER_OP("BatchPairwiseInteractionGrad")
    .Input("x: T")
    .Input("gz: T")
    .Output("gx: T")
    .Attr("T: {float, double} = DT_FLOAT");

REGISTER_OP("BatchPairwiseInteraction2")
    .Input("x: T")   // [batch, m1, n]
    .Input("y: T")   // [batch, m2, n]
    .Output("z: T")  // [batch, m1 * m2, n]
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x;
      ShapeHandle y;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &y));
      DimensionHandle batch1 = c->Dim(x, 0);
      DimensionHandle batch2 = c->Dim(y, 0);
      DimensionHandle batch;
      TF_RETURN_IF_ERROR(c->Merge(batch1, batch2, &batch));

      DimensionHandle m1 = c->Dim(x, 1);
      DimensionHandle m2 = c->Dim(y, 1);
      DimensionHandle m1_m2;
      TF_RETURN_IF_ERROR(c->Multiply(m1, m2, &m1_m2));

      DimensionHandle n1 = c->Dim(x, 2);
      DimensionHandle n2 = c->Dim(y, 2);
      DimensionHandle n;
      TF_RETURN_IF_ERROR(c->Merge(n1, n2, &n));

      std::vector<DimensionHandle> z_dim = {batch, m1_m2, n};
      c->set_output(0, c->MakeShape(z_dim));
      return Status::OK();
    });

REGISTER_OP("BatchPairwiseInteraction2Grad")
    .Input("x: T")
    .Input("y: T")
    .Input("gz: T")
    .Output("gx: T")
    .Output("gy: T")
    .Attr("T: {float, double} = DT_FLOAT");

}  // namespace tensorflow
