/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/io_ops.cc.

#include <memory>
#include <cstdlib>
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/tdw/tdw_func_wrapper.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class TDWRecordReader : public ReaderBase {
 public:
  TDWRecordReader(const string& node_name,
                  const string& field_indices,
                  Env* env)
      : ReaderBase(strings::StrCat("TDWRecordReader '", node_name, "'")),
        field_indices_(field_indices),
        env_(env) {

            // Get int field indices
       		if (field_indices_.length() > 0) {
      			std::vector<string> str_field_indices = str_util::Split(field_indices_, ',');
      			for (string str: str_field_indices) {
        			int_field_indices_.push_back(atoi(str.c_str()));
      			}
    		}
       }

  Status OnWorkStartedLocked() override {
    no_record_ = 0;
    tdw::TDWClient* tdw_client = tdw::TDWClient::GetInstance();
    if (tdw_client == nullptr) {
      return errors::NotFound("New TDWClient failed");
    }
    tdw::TDWRecordReader* reader = tdw_client->GetRecordReader(current_work());
    if (reader == nullptr) {
      return errors::Internal("Get TDWRecordReader failed");
    }
    reader_.reset(reader);
    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    reader_.reset(nullptr);
    return Status::OK();
  }

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    string raw_record;

    Status s = reader_->ReadNext(&raw_record);
    if (errors::IsOutOfRange(s)) {
      *at_end = true;
      return Status::OK();
    }

    if (!s.ok()) return s;

    *produced = true;
    ++no_record_;
    *key = strings::StrCat(current_work(), ":", no_record_);


    if (int_field_indices_.size() > 0) {
      // Split raw_record to the required record base on field_indices
      std::vector<string> raw_fields = str_util::Split(raw_record, '\01');
      std::vector<string> required_fields;
      for (int i: int_field_indices_) {
        if (i >= raw_fields.size()) {
          return errors::OutOfRange("Field index ", i, " is too big");
        }
        required_fields.push_back(raw_fields[i]);
      }
      *value = str_util::Join(required_fields, "\01");
    } else {
      *value = raw_record;
    }
    return Status::OK();
  }

  Status ResetLocked() override {
    no_record_ = 0;
    reader_.reset(nullptr);
    return ReaderBase::ResetLocked();
  }

 private:
  Env* const env_;
  uint64 no_record_; // The # of record in the current file.
  string field_indices_; // comma-separated, e.g. "0,1,2,3"
  std::vector<int> int_field_indices_;
  std::unique_ptr<tdw::TDWRecordReader> reader_;
};

class TDWRecordReaderOp : public ReaderOpKernel {
 public:
  explicit TDWRecordReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    Env* env = context->env();

    string field_indices;
    OP_REQUIRES_OK(context, context->GetAttr("field_indices", &field_indices));

    SetReaderFactory([this, field_indices, env]() {
      return new TDWRecordReader(name(), field_indices, env);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("TDWRecordReader").Device(DEVICE_CPU),
                        TDWRecordReaderOp);

}  // namespace tensorflow