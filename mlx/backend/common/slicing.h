// Copyright © 2024 Apple Inc.

#pragma once

#include <climits>

#include "mlx/array.h"

namespace mlx::core {

// True when an element offset on this shape does not fit in int32.
// The slice base is applied separately with 64-bit pointer arithmetic.
inline bool offsets_exceed_int32(const Shape& shape, const Strides& strides) {
  int64_t low = 0;
  int64_t high = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    int64_t steps = static_cast<int64_t>(shape[i]) - 1;
    if (steps <= 0 || strides[i] == 0) {
      continue;
    }
    int64_t delta = strides[i] * steps;
    if (strides[i] > 0) {
      high += delta;
    } else {
      low += delta;
    }
  }
  return high > INT32_MAX || low < INT32_MIN;
}

std::tuple<int64_t, Strides> prepare_slice(
    const array& in,
    const Shape& start_indices,
    const Shape& strides);

void shared_buffer_slice(
    const array& in,
    const Strides& out_strides,
    int64_t data_offset,
    size_t data_size,
    array& out);

void slice(
    const array& in,
    array& out,
    const Shape& start_indices,
    const Shape& strides);

} // namespace mlx::core
