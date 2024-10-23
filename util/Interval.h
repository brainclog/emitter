#pragma once
#include "math_constants.h"

class Interval {
public:
  float min, max;

  __device__ Interval() : min(CUDART_INF_F), max(-CUDART_INF_F) {} // Default Interval is empty

  __device__ Interval(float min, float max) : min(min), max(max) {}

  // union of two intervals
  __device__ Interval(const Interval& a, const Interval& b) {
    min = a.min <= b.min ? a.min : b.min;
    max = a.max >= b.max ? a.max : b.max;
  }

  __device__ float size() const {
    return max - min;
  }

  __device__ bool contains(float x) const {
    return min <= x && x <= max;
  }

  __device__ bool surrounds(float x) const {
    return min < x && x < max;
  }

  __device__ float clamp(float x) const {
    if (x < min) return min;
    if (x > max) return max;
    return x;
  }

  __device__ Interval expand(float delta) const {
    auto padding = delta/2;
    return {min - padding, max + padding};
  }

  __device__ static Interval empty() {
    return {+CUDART_INF_F, -CUDART_INF_F};
  }

  __device__ static Interval universe() {
    return {-CUDART_INF_F, +CUDART_INF_F};
  }
};

