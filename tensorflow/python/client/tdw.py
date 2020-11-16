# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""TDW functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


@tf_export('TDWRecordWriter')
class TDWRecordWriter(object):
    """A class for writing tdw .

    @@__init__
    """

    def __init__(self, writer):
        self._writer = writer

    def write(self, record, field_indices=None):
        """Write a record to the tdw table.

        Args:
          record: list, the record to be written
          field_indices: list, (option)the field indices to be written,
                         if not set, write all fields
        """
        str_record = ""
        if field_indices is not None:
            assert len(record) == len(field_indices), "Length of record \
          and field_indices must be equal"
            last_index = 0
            sorted_rec = sorted(zip(field_indices, record))
            for i, field in sorted_rec:
                null_cnt = i - last_index
                if null_cnt == 0:
                    if i == 0:
                        str_record += str(field)
                    else:
                        str_record += "\01" + str(field)
                else:
                    if last_index == 0:
                        str_record += null_cnt * "\\N\01"
                        str_record += str(field)
                    else:
                        str_record += null_cnt * "\01\\N"
                        str_record += "\01" + str(field)
                last_index = i + 1
        else:
            str_record = "\01".join([str(field) for field in record])
        self._writer.Write(compat.as_bytes(str_record))

    def close(self):
        self._writer.Close()


@tf_export('TDWClient')
class TDWClient(object):
    """A class for accessing tdw .

    @@__init__
    """

    def __init__(self, db, user, password, group="tl"):
        """Connect db by user:password and get a client for tdw.

        Args:
          db: The db name you will access.
          user: Your user name .
          password: Your password.
          group: (optional) The cluster name of db.

        Raises:
          IOError: If connect error.
        """

        with errors.raise_exception_on_not_ok_status() as status:
            self._client = pywrap_tensorflow.TDWClient_New(
                compat.as_bytes(db), compat.as_bytes(user),
                compat.as_bytes(password), compat.as_bytes(group))

    def get_data_paths(self, table, pri_parts=None, sub_parts=None):
        """Write a string record to the file.

        Args:
          table: str, the table name
          pri_parts: list, the pri partition names
          sub_parts: list, the sub partition names

        Returns:
          A list of data paths
        """
        str_pri_parts = ",".join(pri_parts) if pri_parts is not None else ""
        str_sub_parts = ",".join(sub_parts) if sub_parts is not None else ""
        return self._client.GetDataPaths(
            compat.as_bytes(table), compat.as_bytes(str_pri_parts),
            compat.as_bytes(str_sub_parts)).decode().split(",")

    def get_record_writer(self, table, pri_part="", sub_part=""):
        """Get a record writer.

        Args:
          table: str, the table name
          pri_part: str, (option)the pri partition name
          sub_part: str, (option)the sub partition name

        Returns:
          A tdw record writer.
        """
        writer = self._client.GetRecordWriter(
            compat.as_bytes(table), compat.as_bytes(pri_part),
            compat.as_bytes(sub_part))
        return TDWRecordWriter(writer)


@tf_export('new_tdw_client')
def new_tdw_client(db, user, password, group="tl"):
    """Connect db by user:password and get a client for tdw.

    Args:
      db: The db name you will access.
      user: Your user name.
      password: Your password.
      group: (optional) The cluster name of db.

    Raises:
      IOError: If connect error.

    Returns:
      A tdw client.
    """
    return TDWClient(db, user, password, group)
    