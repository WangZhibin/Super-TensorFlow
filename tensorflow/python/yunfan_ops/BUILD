load("//tensorflow/python:build_defs.bzl", "tf_gen_op_wrapper_private_py")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility=["//tensorflow:internal"],
    licenses=["notice"],  # Apache 2.0
)

tf_gen_op_wrapper_private_py(name="yunfan_ops_gen")

py_library(
    name="yunfan_ops",
    srcs=["yunfan_ops.py"],
    srcs_version="PY2AND3",
    deps=[
        ":yunfan_ops_gen",
    ],
)

tf_py_test(
    name = "yunfan_ops_test",
    srcs = ["yunfan_ops_test.py"],
    additional_deps = [
        ":yunfan_ops",
        ":dynamic_embedding_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name="dynamic_embedding_ops",
    srcs=["dynamic_embedding_ops.py"],
    srcs_version="PY2AND3",
    deps=[
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/training/tracking:tracking",
    ],
)

tf_py_test(
    name = "dynamic_embedding_ops_test",
    srcs = [
        "dynamic_embedding_ops_test.py",
    ],
    additional_deps = [
        ":dynamic_embedding_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "dynamic_embedding_ops_trainable_test",
    srcs = [
        "dynamic_embedding_ops_trainable_test.py",
    ],
    additional_deps = [
        ":dynamic_embedding_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "dynamic_embedding_ops_save_restore_test",
    srcs = [
        "dynamic_embedding_ops_save_restore_test.py",
    ],
    additional_deps = [
        ":dynamic_embedding_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name="dynamic_variable_restrict",
    srcs=["dynamic_variable_restrict.py"],
    srcs_version="PY2AND3",
    deps=[
        ":dynamic_embedding_ops",
        "//tensorflow/contrib/nn:nn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/eager:context",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
    ],
)

py_library(
    name="dynamic_feature_filter",
    srcs=["dynamic_feature_filter.py"],
    srcs_version="PY2AND3",
    deps=[
        ":dynamic_embedding_ops",
        "//tensorflow/contrib/nn:nn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/eager:context",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
    ],
)

tf_py_test(
    name = "dynamic_variable_test",
    srcs = ["dynamic_embedding_ops_test.py"],
    additional_deps = [
        ":dynamic_embedding_ops",
        ":dynamic_feature_filter",
        ":dynamic_variable_restrict",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "dynamic_feature_filter_test",
    srcs = ["dynamic_feature_filter_test.py"],
    additional_deps = [
        ":dynamic_embedding_ops",
        ":dynamic_feature_filter",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
    ],
)
