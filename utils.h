/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_UTILS_H_

#include <cstdint>
#include <vector>
#include <limits>
#include <cmath>
#include "delegate_main.h"

namespace vx {
namespace delegate {
namespace utils {

// transpose channel_dim while doing transpose operation.
int32_t TransposeChannelDim(const std::vector<uint32_t>& perm,
                            int32_t channel_dim);

// Convert the perm in TfLite to the perm in vx-delegate when transpose.
std::vector<uint32_t> GetOvxTransposePerm(const std::vector<uint32_t>& perm);

// Convert TfLite axis to OpenVX kind.
inline int32_t ConvertAxis(int32_t axisIn, uint32_t dimNum) {
  return dimNum - (axisIn < 0 ? dimNum + axisIn : axisIn) - 1;
}

template <typename T>
std::vector<T> TransposeVec(const std::vector<T>& input,
                            const std::vector<int>& perm) {
  if (input.size() != perm.size()) {
    return std::vector<T>();
  };

  std::vector<T> output(input.size());
  for (int i = 0; i < perm.size(); i++) {
    output[i] = input[perm[i]];
  }

  return output;
}

inline int32_t CalcWeightSizeForBilinear(int32_t scale) {
  return 2 * scale - scale % 2;
}

inline int32_t CalcPadSizeForBilinear(int32_t scale) { return scale / 2; }

void GenerateWeightsDataForBilinear(float* data,
                                    const std::vector<uint32_t>& weight_shape,
                                    uint32_t scale_w,
                                    uint32_t scale_h);

void GenerateWeightDataForNearest(float* data,
                                  const std::vector<uint32_t>& weight_shape);

template <typename T>
inline void Quantize(const std::vector<float>& data, float scale,
                               int32_t zero_point, std::vector<T>& quant_data) {
  for (const auto& f : data) {
    quant_data.push_back(static_cast<T>(std::max<float>(
        std::numeric_limits<T>::min(),
        std::min<float>(std::numeric_limits<T>::max(),
                        std::round(zero_point + (f / scale))))));
  }
}

}  // namespace utils
}  // namespace delegate
}  // namespace vx

#endif /* TENSORFLOW_LITE_DELEGATES_VX_DELEGAGE_UTILS_H_ */