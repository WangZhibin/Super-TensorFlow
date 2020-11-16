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

#include "tensorflow/core/lib/tdw/tdw_func_wrapper.h"

#include <errno.h>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/posix/error.h"
#include "third_party/tdw/tdw.h"

namespace tensorflow {
namespace tdw {

template <typename R, typename... Args>
Status BindFunc(void* handle, const char* name,
                std::function<R(Args...)>* func) {
  void* symbol_ptr = nullptr;
  TF_RETURN_IF_ERROR(
      Env::Default()->GetSymbolFromLibrary(handle, name, &symbol_ptr));
  *func = reinterpret_cast<R (*)(Args...)>(symbol_ptr);
  return Status::OK();
}

class LibTDW {
 public:
  static LibTDW* Load() {
    static LibTDW* lib = []() -> LibTDW* {
      LibTDW* lib = new LibTDW;
      lib->LoadAndBind();
      return lib;
    }();

    return lib;
  }

  // The status, if any, from failure to load.
  Status status() { return status_; }

  std::function<tdw_client_t(const char*, const char*, const char*, const char*)> 
    tdw_new_client_of_group;
  std::function<int(tdw_client_t)> tdw_free_client;
  std::function<tdw_record_reader_t(tdw_client_t, const char*)> 
    tdw_get_record_reader;
  std::function<int(tdw_record_reader_t)> tdw_close_record_reader;
  std::function<int(tdw_record_reader_t, char**)> tdw_read_next;
  std::function<void(char*)> tdw_free_record;
  std::function<int(tdw_client_t, const char*, const char*, const char*, char**)> 
    tdw_get_data_paths;
  std::function<void(char*)> tdw_free_data_paths;
  std::function<tdw_record_writer_t(tdw_client_t,
                                    const char*,
                                    const char*,
                                    const char*)> tdw_get_record_writer;
  std::function<int(tdw_record_writer_t, const char*)> tdw_write_record;
  std::function<int(tdw_record_writer_t)> tdw_close_record_writer;

 private:
  void LoadAndBind() {
    auto TryLoadAndBind = [this](const char* name, void** handle) -> Status {
      TF_RETURN_IF_ERROR(Env::Default()->LoadLibrary(name, handle));
#define BIND_TDW_FUNC(function) \
  TF_RETURN_IF_ERROR(BindFunc(*handle, #function, &function));

      BIND_TDW_FUNC(tdw_new_client_of_group);
      BIND_TDW_FUNC(tdw_free_client);
      BIND_TDW_FUNC(tdw_get_record_reader);
      BIND_TDW_FUNC(tdw_close_record_reader);
      BIND_TDW_FUNC(tdw_read_next);
      BIND_TDW_FUNC(tdw_free_record);
      BIND_TDW_FUNC(tdw_get_data_paths);
      BIND_TDW_FUNC(tdw_free_data_paths);
      BIND_TDW_FUNC(tdw_get_record_writer);
      BIND_TDW_FUNC(tdw_write_record);
      BIND_TDW_FUNC(tdw_close_record_writer);
#undef BIND_TDW_FUNC
      return Status::OK();
    };

    // libtdw.so won't be in the standard locations. Use the path as specified
    // in the libtdw documentation.
    char* tdw_home = getenv("TDW_PLATFORM_API_HOME");
    if (tdw_home == nullptr) {
      status_ = errors::FailedPrecondition(
          "Environment variable TDW_PLATFORM_API_HOME not set");
      return;
    }
    string path = io::JoinPath(tdw_home, "native", "libtdw", "lib", "libtdw.so");
    status_ = TryLoadAndBind(path.c_str(), &handle_);
    return;
  }

  Status status_;
  void* handle_ = nullptr;
};


TDWRecordReader::TDWRecordReader(LibTDW* tdw, tdw_record_reader_t reader)
  : tdw_(tdw), reader_(reader) {}

TDWRecordReader::~TDWRecordReader() {
  tdw_->tdw_close_record_reader(reader_);
}

Status TDWRecordReader::ReadNext(string* record) {
  char* c_record;
  int ret = tdw_->tdw_read_next(reader_, &c_record);
  if (ret < 0) {
    // On error
    return IOError("Read record error", errno);
  }
  if (ret > 0) {
    // On eof
    return errors::OutOfRange("eof");
  }
  int size = strlen(c_record);
  record->resize(size);
  memmove(&(*record)[0], c_record, size);
  tdw_->tdw_free_record(c_record);
  return Status::OK();
}

TDWRecordWriter::TDWRecordWriter(LibTDW* tdw, tdw_record_writer_t writer)
  : tdw_(tdw), writer_(writer) {}

TDWRecordWriter::~TDWRecordWriter() {
  if(writer_ != nullptr){
    tdw_->tdw_close_record_writer(writer_);
    writer_= nullptr;
  }
}

void TDWRecordWriter::Close() {
   if(writer_ != nullptr){
      tdw_->tdw_close_record_writer(writer_);
      writer_= nullptr;
   }  
}

bool TDWRecordWriter::Write(const string& record) {
  int ret = tdw_->tdw_write_record(writer_, record.c_str());
  if (ret == 0) {
    return true;
  } else {
    return false;
  }
}


TDWClient::TDWClient(): tdw_(LibTDW::Load()), client_(nullptr) {}

TDWClient::~TDWClient() {
  if (client_ != nullptr) {
    tdw_->tdw_free_client(client_);
  }
}

TDWClient* TDWClient::New(const string& db, const string& user,
                          const string& password, const string& group) {
  TDWClient* client = new TDWClient();

  Status s = client->tdw_->status();
  if (!s.ok()) {
    LOG(ERROR) << "New TDWClient failed: " << s.error_message();
    delete client;
    return nullptr;
  }

  client->client_ = client->tdw_->tdw_new_client_of_group(
      db.c_str(), user.c_str(), password.c_str(), group.c_str());
  if (client->client_ == nullptr) {
    LOG(ERROR) << "New TDWClient failed: " << strerror(errno);
    delete client;
    return nullptr;
  }

  // Update the instance
  instance_ = client;

  return client;
}

string TDWClient::GetDataPaths(const string& table,
                               const string& pri_parts,
                               const string& sub_parts) {
  string paths;

  Status s = tdw_->status();
  if (!s.ok()) {
    LOG(ERROR) << "TDWClient::GetDataPaths failed: " << s.error_message();
    return paths;
  }

  char* c_paths;
  int ret = tdw_->tdw_get_data_paths(client_, table.c_str(), pri_parts.c_str(),
                                     sub_parts.c_str(), &c_paths);
  LOG(INFO) << "table: " << table << " pri_parts: " << pri_parts << " sub_parts: " << sub_parts << " c_paths: " << c_paths;

  if (ret != 0) {
    LOG(ERROR) << "TDWClient::GetDataPaths failed: " << strerror(errno);
    return paths;
  }
  
  int size = strlen(c_paths);
  paths.resize(size + 1);
  LOG(INFO) << "c_paths size: " << size  << " c_paths: " << c_paths;
  memmove(&(paths[0]), c_paths, size + 1);
  LOG(INFO) << "paths: " << paths  << " c_path: " << c_paths;
  tdw_->tdw_free_data_paths(c_paths);
  LOG(INFO) << "paths: " << paths;
  return paths;
}

TDWRecordReader* TDWClient::GetRecordReader(const string& fname) {
  Status s = tdw_->status();
  if (!s.ok()) {
    LOG(ERROR) << "TDWClient::GetRecordReader failed: " << s.error_message();
    return nullptr;
  }

  tdw_record_reader_t internal_reader = 
    tdw_->tdw_get_record_reader(client_, fname.c_str());
  if (internal_reader == nullptr) {
    LOG(ERROR) << "TDWClient::GetRecordReader failed: " << strerror(errno);
    return nullptr;
  }

  return new TDWRecordReader(tdw_, internal_reader);
}

TDWRecordWriter* TDWClient::GetRecordWriter(const string& table,
                                            const string& pri_part,
                                            const string& sub_part) {
  Status s = tdw_->status();
  if (!s.ok()) {
    LOG(ERROR) << "TDWClient::GetRecordWriter failed: " << s.error_message();
    return nullptr;
  }

  tdw_record_writer_t internal_writer = 
    tdw_->tdw_get_record_writer(client_, table.c_str(), pri_part.c_str(),
                                sub_part.c_str());
  if (internal_writer == nullptr) {
    LOG(ERROR) << "TDWClient::GetRecordWriter failed: " << strerror(errno);
    return nullptr;
  }

  return new TDWRecordWriter(tdw_, internal_writer);
}

TDWClient* TDWClient::instance_ = nullptr;

}  // namespace tdw
}  // namespace tensorflow
