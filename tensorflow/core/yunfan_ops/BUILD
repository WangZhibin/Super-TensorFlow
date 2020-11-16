package(
    default_visibility=["//tensorflow:internal"],
    licenses=["notice"],  # Apache 2.0
)

cc_library(
    name="yunfan_ops",
    srcs=[
        "yunfan_ops.cc",
    ],
    deps=[
        "//tensorflow/core:framework",
    ],
    linkstatic=1,
    alwayslink=1,
)

cc_library(
    name="yunfan_kernels",
    srcs=[
        "batch_fm_interaction.cc",
        "batch_fm_interaction2.cc",
        "batch_pairwise_interaction.cc",
        "batch_pairwise_interaction2.cc",
    ],
    deps=[
        "//tensorflow/core:framework",
        "//third_party/deepx_core:deepx_core",
    ],
)
