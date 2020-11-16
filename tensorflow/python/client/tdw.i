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

%nothread tensorflow::tdw::TDWRecordWriter::Write;

%include "tensorflow/python/platform/base.i"

%feature("except") tensorflow::tdw::TDWClient::New {
  // Let other threads run while we new a tdw client
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

%feature("except") tensorflow::tdw::TDWClient::GetRecordWriter {
  // Let other threads run while we get a record writer
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

%feature("except") tensorflow::tdw::TDWRecordWriter::Write {
  // Let other threads run while we write
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

%feature("except") tensorflow::tdw::TDWRecordWriter::Close {
  // Let other threads run while we close
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}



%newobject tensorflow::tdw::TDWClient::New;
%newobject tensorflow::tdw::TDWClient::GetRecordWriter;

%{
#include "tensorflow/core/lib/tdw/tdw_func_wrapper.h"
%}

%ignoreall

%unignore tensorflow;
%unignore tensorflow::tdw;
%unignore tensorflow::tdw::TDWClient;
%unignore tensorflow::tdw::TDWClient::~TDWClient;
%unignore tensorflow::tdw::TDWClient::GetDataPaths;
%unignore tensorflow::tdw::TDWClient::GetRecordWriter;
%unignore tensorflow::tdw::TDWClient::New;
%unignore tensorflow::tdw::TDWRecordWriter;
%unignore tensorflow::tdw::TDWRecordWriter::~TDWRecordWriter;
%unignore tensorflow::tdw::TDWRecordWriter::Close;
%unignore tensorflow::tdw::TDWRecordWriter::Write;

%include "tensorflow/core/lib/tdw/tdw_func_wrapper.h"

%unignoreall