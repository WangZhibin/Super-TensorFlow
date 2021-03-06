// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_MACRO_H_
#define SAGE2_MACRO_H_

#if defined __cplusplus
#define SAGE2_EXTERN_C extern "C"
#else
#define SAGE2_EXTERN_C extern
#endif

#define SAGE2_C_API SAGE2_EXTERN_C __attribute__((visibility("default")))

#define SAGE2_MAJOR_VERSION 0
#define SAGE2_MINOR_VERSION 2
#define SAGE2_PATCH_LEVEL 1

#endif  // SAGE2_MACRO_H_
