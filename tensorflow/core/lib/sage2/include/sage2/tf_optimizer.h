// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_TF_OPTIMIZER_H_
#define SAGE2_TF_OPTIMIZER_H_

#include <sage2/macro.h>
#include <stdint.h>  // NOLINT

/************************************************************************/
/* ftrl v1 */
/* v1 is designed to accelerate ApplyFtrl::operator() */
/* when learning_rate_power is -0.5 */
/************************************************************************/
// NOLINTNEXTLINE
typedef struct _sage2_tf_ftrl_config_v1_s {
  float alpha;
  float l1;
  float l2;
  float inv_alpha;
} sage2_tf_ftrl_config_v1_s;
SAGE2_C_API void sage2_tf_ftrl_update_v1_ps(
    const sage2_tf_ftrl_config_v1_s* config, uint64_t _n, const float* g,
    float* w, float* n, float* z);

#endif  // SAGE2_TF_OPTIMIZER_H_
