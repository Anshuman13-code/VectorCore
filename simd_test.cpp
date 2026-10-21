#include <iostream>
#include <vector>
#include <chrono>
#include <arm_neon.h> // Header for ARM NEON intrinsics

// 1. Plain Scalar (Old)
float l2_scalar(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// 2. ARM NEON SIMD (New)
float l2_neon(const float* a, const float* b, std::size_t dim) {
    // Initialize an accumulator register with four 0.0f values
    float32x4_t acc = vdupq_n_f32(0.0f);

    // Step by 4 elements on each iteration
    for (std::size_t i = 0; i < dim; i += 4) {
        // Load 4 floats from array a and 4 floats from array b
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);

        // Compute diff = va - vb
        float32x4_t diff = vsubq_f32(va, vb);

        // acc = acc + (diff * diff)
        acc = vfmaq_f32(acc, diff, diff);
    }

    // Horizontal add: sum the 4 lanes inside the register into a single float
    return vaddvq_f32(acc);
}

int main() {
    constexpr std::size_t DIM = 128; // 128 is divisible by 4 (32 iterations)
    std::vector<float> vec1(DIM, 1.5f);
    std::vector<float> vec2(DIM, 0.5f);

    // Verify mathematical accuracy
    float res_scalar = l2_scalar(vec1.data(), vec2.data(), DIM);
    float res_neon   = l2_neon(vec1.data(), vec2.data(), DIM);

    std::cout << "Scalar result: " << res_scalar << "\n";
    std::cout << "NEON result  : " << res_neon << "\n";

    // Expected: (1.5 - 0.5)^2 = 1.0 per dimension. Over 128 dims, total = 128.0.
    std::cout << "Expected     : 128.0\n";

    return 0;
}