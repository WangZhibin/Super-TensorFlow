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
#if GOOGLE_CUDA
#include "tensorflow/core/kernels/lookup_impl/lookup_table_op_gpu.h"

#define EIGEN_USE_GPU

#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <stdlib.h>
#include <string>
#include <type_traits>
#include <utility>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/initializable_lookup_table.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/nvhash/nv_hashtable.cuh"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {
namespace lookup {

// Lookup table that wraps an cuDF::concurrent_unordered_map.
// Behaves identical to CPU version of MutableHashTableOfTensors.
template <class K, class V>
class MutableHashTableOfTensorsGpu final : public LookupInterface {
 public:
  MutableHashTableOfTensorsGpu(OpKernelContext* ctx, OpKernel* kernel) {
    size_t env_var = 0;
    Status status =
        ReadSizetFromEnvVar("TF_HASHTABLE_INIT_SIZE",
                            16 * 1024 * 1024,  // 16M KV pairs by default
                            &env_var);
    min_size_ = env_var;
    max_size_ = env_var;

    LOG(INFO) << "GPU MutableHashTable init: max size=" << max_size_
              << ", min size=" << min_size_;
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(value_shape_),
        errors::InvalidArgument("Default value must be a vector, got shape ",
                                value_shape_.DebugString()));
    runtime_dim_ = value_shape_.dim_size(0);
    OP_REQUIRES(
        ctx, (runtime_dim_ <= 500),
        errors::InvalidArgument("The dim of MutableHashTable on GPU should be "
                                "less than or equal 500 vs.",
                                runtime_dim_));
    CUDA_CHECK(cudaStreamCreate(&default_stream_));
    CreateTable(max_size_, &table_);
  }
  ~MutableHashTableOfTensorsGpu() {
    CUDA_CHECK(cudaStreamDestroy(default_stream_));
    delete table_;
  }

  void CreateTable(size_t max_size_, TableWrapperBase<K, V>** pptable) {
    // Create the branch code for instantiating TableWrapper with different DIM.
    if (runtime_dim_ <= 50) {
      CreateGpuHashTable0(max_size_, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 100) {
      CreateGpuHashTable1(max_size_, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 150) {
      CreateGpuHashTable2(max_size_, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 200) {
      CreateGpuHashTable3(max_size_, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 250) {
      CreateGpuHashTable4(max_size_, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 300) {
      CreateGpuHashTable5(max_size_, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 350) {
      CreateGpuHashTable6(max_size_, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 400) {
      CreateGpuHashTable7(max_size_, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 450) {
      CreateGpuHashTable8(max_size_, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 500) {
      CreateGpuHashTable9(max_size_, runtime_dim_, pptable);
    }
    assert(*pptable != nullptr &&
           "MutableHashTableOfTensorsGpu::CreateTable fail!");
  }
  size_t size() const override {
    tf_shared_lock l(mu_);
    size_t retv = table_->get_size(default_stream_);
    CUDA_CHECK(cudaStreamSynchronize(default_stream_));
    return retv;
  }

  Status Find(OpKernelContext* ctx, const Tensor& d_keys, Tensor* value,
              const Tensor& default_value) override {
    size_t len = d_keys.flat<K>().size();
    bool* d_status;
    ValueArrayBase<V>* d_default_value;

    auto value_flat = value->flat_inner_dims<V, 2>();
    const auto default_flat = default_value.flat<V>();
    int64 total = value_flat.size();
    int64 default_total = default_flat.size();
    bool is_full_default = (total == default_total);

    cudaStream_t _stream;

    if (len > 0) {
      size_t default_value_num =
          is_full_default ? default_value.shape().dim_size(0) : 1;
      CUDA_CHECK(cudaStreamCreate(&_stream));
      CUDA_CHECK(cudaMalloc((void**)&d_status, sizeof(bool) * len));
      {
        tf_shared_lock l(mu_);
        table_->get((const key_t*)d_keys.tensor_data().data(),
                    (ValueArrayBase<V>*)value->tensor_data().data(), d_status,
                    len, (ValueArrayBase<V>*)default_value.tensor_data().data(),
                    _stream, is_full_default);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaFree(d_status));
      CUDA_CHECK(cudaFree(d_default_value));
      CUDA_CHECK(cudaStreamDestroy(_stream));
    }
    return Status::OK();
  }

  void RehashIfNeeded(cudaStream_t stream) {
    key_t* d_keys;
    ValueArrayBase<V>* d_values;
    size_t* d_dump_counter;
    size_t new_max_size = max_size_;

    size_t total_size = table_->get_size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (total_size >= 0.75 * max_size_) {
      new_max_size = max_size_ * 2;
    }
    if (total_size < 0.25 * max_size_ && max_size_ > min_size_) {
      new_max_size = max_size_ / 2;
    }
    if (new_max_size != max_size_) {  // rehash manually.
      size_t capacity = table_->get_capacity();
      size_t h_dump_counter = 0;
      LOG(INFO) << "GPU start rehash: size=" << total_size
                << ", max_size_=" << max_size_
                << ", load factor=" << std::setprecision(2)
                << (float)total_size / (float)max_size_;
      CUDA_CHECK(cudaMalloc((void**)&d_dump_counter, sizeof(size_t)));
      CUDA_CHECK(cudaMalloc((void**)&d_keys, sizeof(key_t) * capacity));
      CUDA_CHECK(
          cudaMalloc((void**)&d_values, sizeof(V) * runtime_dim_ * capacity));
      table_->dump(d_keys, (ValueArrayBase<V>*)d_values, 0, capacity,
                   d_dump_counter, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      delete table_;
      table_ = NULL;
      CreateTable(new_max_size, &table_);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMemcpy((size_t*)&h_dump_counter, (size_t*)d_dump_counter,
                            sizeof(size_t), cudaMemcpyDefault));
      table_->upsert((const key_t*)d_keys, (const ValueArrayBase<V>*)d_values,
                     h_dump_counter, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(d_keys));
      CUDA_CHECK(cudaFree(d_values));
      CUDA_CHECK(cudaFree(d_dump_counter));
      max_size_ = new_max_size;
      LOG(INFO) << "GPU end rehash: size=" << total_size
                << ", max_size_=" << max_size_
                << ", load factor=" << std::setprecision(2)
                << (float)total_size / (float)max_size_;
    }
  }

  Status Insert(OpKernelContext* ctx, const Tensor& keys,
                const Tensor& values) override {
    size_t len = keys.flat<K>().size();
    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));
    {
      mutex_lock l(mu_);
      RehashIfNeeded(_stream);
      table_->upsert((const key_t*)keys.tensor_data().data(),
                     (const ValueArrayBase<V>*)values.tensor_data().data(), len,
                     _stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    };
    CUDA_CHECK(cudaStreamDestroy(_stream));

    return Status::OK();
  }

  Status Remove(OpKernelContext* ctx, const Tensor& keys) override {
    size_t len = keys.flat<K>().size();
    key_t* d_keys;
    cudaStream_t _stream;

    CUDA_CHECK(cudaStreamCreate(&_stream));
    if (len > 0) {
      CUDA_CHECK(cudaMalloc((void**)&d_keys, sizeof(key_t) * len));
      CUDA_CHECK(cudaMemcpy((void*)d_keys, (void*)keys.tensor_data().data(),
                            sizeof(key_t) * len, cudaMemcpyDefault));
      {
        mutex_lock l(mu_);
        table_->remove((const key_t*)d_keys, len, _stream);
        RehashIfNeeded(_stream);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaStreamDestroy(_stream));
      CUDA_CHECK(cudaFree(d_keys));
    }
    return Status::OK();
  }

  Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values) override {
    size_t len = keys.flat<K>().size();
    key_t* d_keys;
    ValueArrayBase<V>* d_values;
    if (len > 0) {
      CUDA_CHECK(cudaMalloc((void**)&d_keys, sizeof(key_t) * len));
      CUDA_CHECK(cudaMalloc((void**)&d_values, sizeof(V) * runtime_dim_ * len));
      CUDA_CHECK(cudaMemcpy((void*)d_keys, (void*)keys.tensor_data().data(),
                            sizeof(key_t) * len, cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy((void*)d_values, (void*)values.tensor_data().data(),
                            sizeof(V) * runtime_dim_ * len, cudaMemcpyDefault));
      {
        mutex_lock l(mu_);
        table_->upsert((const key_t*)d_keys, (const ValueArrayBase<V>*)d_values,
                       len, default_stream_);
        CUDA_CHECK(cudaStreamSynchronize(default_stream_));
      }
      CUDA_CHECK(cudaFree(d_keys));
      CUDA_CHECK(cudaFree(d_values));
    }
    return Status::OK();
  }

  Status ExportValues(OpKernelContext* ctx) override {
    size_t len = 0;
    int64 size = 0;

    const size_t offset = 0;

    Tensor* keys;
    Tensor* values;

    size_t* d_dump_counter;

    {
      tf_shared_lock l(mu_);
      len = table_->get_capacity();
      size = (int64)table_->get_size(default_stream_);
      CUDA_CHECK(cudaStreamSynchronize(default_stream_));
    }

    CUDA_CHECK(cudaMalloc((void**)&d_dump_counter, sizeof(size_t)));

    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    attr.set_on_host(false);

    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({(size)}), &keys, attr));
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values", TensorShape({size, (int64)runtime_dim_}), &values, attr));
    if (size) {
      tf_shared_lock l(mu_);
      table_->dump((key_t*)keys->flat<K>().data(),
                   (ValueArrayBase<V>*)values->matrix<V>().data(), offset, len,
                   d_dump_counter, default_stream_);
      CUDA_CHECK(cudaStreamSynchronize(default_stream_));
    }
    CUDA_CHECK(cudaFree(d_dump_counter));
    return Status::OK();
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }
  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }
  TensorShape key_shape() const final { return TensorShape(); }
  TensorShape value_shape() const override { return value_shape_; }

 private:
  TensorShape value_shape_;
  size_t max_size_;
  size_t min_size_;
  size_t runtime_dim_;
  cudaStream_t default_stream_;
  mutable mutex mu_;
  TableWrapperBase<K, V>* table_ GUARDED_BY(mu_);
};

}  // namespace lookup

// Table lookup op. Perform the lookup operation on the given table.
class LookupTableFindGpuOp : public OpKernel {
 public:
  explicit LookupTableFindGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    // Input 0 could be a STRING_REF or a RESOURCE
    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& key = ctx->input(1);
    const Tensor& default_value = ctx->input(2);

    TensorShape output_shape = key.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor* out;
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("values", output_shape, &out, attr));

    OP_REQUIRES_OK(ctx, table->Find(ctx, key, out, default_value));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableFind").Device(DEVICE_GPU),
                        LookupTableFindGpuOp);
REGISTER_KERNEL_BUILDER(Name("LookupTableFindV2").Device(DEVICE_GPU),
                        LookupTableFindGpuOp);

// Table insert op.
class LookupTableInsertGpuOp : public OpKernel {
 public:
  explicit LookupTableInsertGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForInsert(keys, values));
    OP_REQUIRES_OK(ctx, table->Insert(ctx, keys, values));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableInsert").Device(DEVICE_GPU),
                        LookupTableInsertGpuOp);
REGISTER_KERNEL_BUILDER(Name("LookupTableInsertV2").Device(DEVICE_GPU),
                        LookupTableInsertGpuOp);

// Table remove op.
class LookupTableRemoveGpuOp : public OpKernel {
 public:
  explicit LookupTableRemoveGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& key = ctx->input(1);
    OP_REQUIRES_OK(ctx, table->CheckKeyTensorForRemove(key));
    OP_REQUIRES_OK(ctx, table->Remove(ctx, key));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableRemoveV2").Device(DEVICE_GPU),
                        LookupTableRemoveGpuOp);

// Op that returns the size of the given table.
class LookupTableSizeGpuOp : public OpKernel {
 public:
  explicit LookupTableSizeGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    Tensor* out;
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_on_host(false);

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("size", TensorShape({}), &out, attr));

    size_t size = table->size();
    const int64* p_size = (const int64*)out->flat<int64>().data();
    CUDA_CHECK(cudaMemcpy((void*)out->tensor_data().data(), (void*)&size,
                          sizeof(size_t), cudaMemcpyDefault));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableSize").Device(DEVICE_GPU),
                        LookupTableSizeGpuOp);
REGISTER_KERNEL_BUILDER(Name("LookupTableSizeV2").Device(DEVICE_GPU),
                        LookupTableSizeGpuOp);

// Op that outputs tensors of all keys and all values.
class LookupTableExportGpuOp : public OpKernel {
 public:
  explicit LookupTableExportGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, table->ExportValues(ctx));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableExport").Device(DEVICE_GPU),
                        LookupTableExportGpuOp);
REGISTER_KERNEL_BUILDER(Name("LookupTableExportV2").Device(DEVICE_GPU),
                        LookupTableExportGpuOp);

// Clear the table and insert data.
class LookupTableImportGpuOp : public OpKernel {
 public:
  explicit LookupTableImportGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForImport(keys, values));
    OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableImport").Device(DEVICE_GPU),
                        LookupTableImportGpuOp);
REGISTER_KERNEL_BUILDER(Name("LookupTableImportV2").Device(DEVICE_GPU),
                        LookupTableImportGpuOp);

// Register the MutableHashTableOfTensors op.
#define REGISTER_KERNEL(key_dtype, value_dtype)                         \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MutableHashTableOfTensors")                                 \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<key_dtype>("key_dtype")                       \
          .TypeConstraint<value_dtype>("value_dtype"),                  \
      LookupTableOp<                                                    \
          lookup::MutableHashTableOfTensorsGpu<key_dtype, value_dtype>, \
          key_dtype, value_dtype>)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MutableHashTableOfTensorsV2")                               \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<key_dtype>("key_dtype")                       \
          .TypeConstraint<value_dtype>("value_dtype"),                  \
      LookupTableOp<                                                    \
          lookup::MutableHashTableOfTensorsGpu<key_dtype, value_dtype>, \
          key_dtype, value_dtype>)

REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(int64, Eigen::half);
REGISTER_KERNEL(int64, int32);
REGISTER_KERNEL(int64, int8);

#undef REGISTER_KERNEL

}  // namespace tensorflow
#endif
