/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LIB_TDW_TDW_FUNC_WRAPPER_H_
#define TENSORFLOW_LIB_TDW_TDW_FUNC_WRAPPER_H_

#include "tensorflow/core/platform/env.h"

extern "C" {
struct tdw_client_internal;
typedef struct tdw_client_internal* tdw_client_t;

struct tdw_record_reader_internal;
typedef struct tdw_record_reader_internal* tdw_record_reader_t;

struct tdw_record_writer_internal;
typedef struct tdw_record_writer_internal* tdw_record_writer_t;
}


namespace tensorflow {
namespace tdw {

class LibTDW;

class TDWRecordReader {
 public:
  TDWRecordReader(LibTDW* tdw, tdw_record_reader_t reader);
  ~TDWRecordReader();

  Status ReadNext(string* record);

 private:
  LibTDW* tdw_;
  tdw_record_reader_t reader_;
};

class TDWRecordWriter {
 public:
  TDWRecordWriter(LibTDW* tdw, tdw_record_writer_t writer);
  ~TDWRecordWriter();

  // Just for python wrapper
  bool Write(const string& record);
  void Close();

 private:
  LibTDW* tdw_;
  tdw_record_writer_t writer_;
};

class TDWClient {
 public:
  ~TDWClient ();

  // The only method to construct TDWClient
  static TDWClient* New(const string& db, const string& user,
                        const string& password, const string& group);

  static TDWClient* GetInstance() { return instance_; }

  string GetDataPaths(const string& table,
                      const string& pri_parts,
                      const string& sub_parts);
  
  TDWRecordReader* GetRecordReader(const string& fname);

  TDWRecordWriter* GetRecordWriter(const string& table,
                                   const string& pri_part,
                                   const string& sub_part);

 private:
  TDWClient();
  LibTDW* tdw_;
  tdw_client_t client_;
  static TDWClient* instance_;
};

}  // namespace tdw
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_TDW_TDW_FUNC_WRAPPER_H_
